"""LangGraph Deep Research Agent with Interactive Scoping and ReAct Search.

This agent implements a two-phase approach:
1. Interactive scoping phase: Uses interrupt() for terminal-based conversation to clarify research scope
2. ReAct research phase: Uses create_react_agent with Tavily search to conduct deep research

The agent exports a compiled graph as 'app' for LangGraph Platform deployment.
"""

from typing import Annotated, Dict, Any, Literal
from typing_extensions import TypedDict

from langchain.chat_models import init_chat_model
from langchain_tavily import TavilySearch
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import MemorySaver
import os


class State(TypedDict):
    """State schema for the deep research agent."""
    messages: Annotated[list, add_messages]
    research_scope: dict
    scope_confirmed: bool
    research_brief: str
    final_report: str


def scoping_node(state: State) -> Dict[str, Any]:
    """Interactive scoping node that clarifies research requirements with the user.
    
    Uses interrupt() to conduct back-and-forth terminal conversations until
    the research scope is confirmed by the user.
    """
    # If scope is already confirmed, skip this node
    if state.get("scope_confirmed", False):
        return {"scope_confirmed": True}
    
    # Initialize research scope if not present
    current_scope = state.get("research_scope", {})
    
    # Build clarifying questions based on current scope
    questions = []
    if not current_scope.get("topic"):
        questions.append("What is the main topic or subject you want me to research?")
    if not current_scope.get("focus_areas"):
        questions.append("Are there specific aspects or focus areas within this topic you're most interested in?")
    if not current_scope.get("depth"):
        questions.append("How deep should the research be? (overview, detailed analysis, comprehensive report)")
    if not current_scope.get("sources"):
        questions.append("Do you have preferences for types of sources? (academic, news, industry reports, etc.)")
    if not current_scope.get("timeframe"):
        questions.append("Is there a specific timeframe or recency requirement for the information?")
    
    # If we have basic scope info, ask for confirmation
    if current_scope.get("topic") and len(questions) <= 2:
        confirmation_prompt = f"""
Based on our conversation, here's what I understand about your research request:

Topic: {current_scope.get('topic', 'Not specified')}
Focus Areas: {current_scope.get('focus_areas', 'General overview')}
Depth: {current_scope.get('depth', 'Standard analysis')}
Sources: {current_scope.get('sources', 'Mixed sources')}
Timeframe: {current_scope.get('timeframe', 'Recent information preferred')}

Is this scope correct and complete? Should I proceed with the research, or would you like to modify anything?
Type 'yes' to proceed, or provide additional details/corrections.
"""
        user_response = interrupt(confirmation_prompt)
        
        if user_response and user_response.lower().strip() in ['yes', 'y', 'proceed', 'correct', 'good']:
            # Generate research brief
            brief = f"""
Research Brief:
Topic: {current_scope.get('topic')}
Focus Areas: {current_scope.get('focus_areas', 'General comprehensive analysis')}
Depth Required: {current_scope.get('depth', 'Detailed analysis')}
Preferred Sources: {current_scope.get('sources', 'Authoritative and recent sources')}
Timeframe: {current_scope.get('timeframe', 'Current and recent information')}

Research Objectives:
- Provide comprehensive coverage of the specified topic
- Focus on the identified areas of interest
- Ensure information is current and from reliable sources
- Deliver actionable insights and key findings
"""
            return {
                "scope_confirmed": True,
                "research_brief": brief,
                "research_scope": current_scope
            }
        else:
            # User wants to modify - incorporate their feedback
            if user_response:
                # Simple parsing of user feedback to update scope
                response_lower = user_response.lower()
                if "topic" in response_lower or "subject" in response_lower:
                    current_scope["topic"] = user_response
                elif "focus" in response_lower or "aspect" in response_lower:
                    current_scope["focus_areas"] = user_response
                elif "depth" in response_lower or "detail" in response_lower:
                    current_scope["depth"] = user_response
                elif "source" in response_lower:
                    current_scope["sources"] = user_response
                elif "time" in response_lower or "recent" in response_lower:
                    current_scope["timeframe"] = user_response
                else:
                    # General feedback - add to topic or focus areas
                    if current_scope.get("topic"):
                        current_scope["focus_areas"] = user_response
                    else:
                        current_scope["topic"] = user_response
            
            return {"research_scope": current_scope, "scope_confirmed": False}
    
    # Ask the next clarifying question
    if questions:
        next_question = questions[0]
        user_response = interrupt(f"{next_question}\n\nPlease provide your answer:")
        
        if user_response:
            # Update scope based on the question asked
            if "topic" in next_question.lower():
                current_scope["topic"] = user_response
            elif "focus" in next_question.lower() or "aspect" in next_question.lower():
                current_scope["focus_areas"] = user_response
            elif "depth" in next_question.lower() or "deep" in next_question.lower():
                current_scope["depth"] = user_response
            elif "source" in next_question.lower():
                current_scope["sources"] = user_response
            elif "timeframe" in next_question.lower() or "recency" in next_question.lower():
                current_scope["timeframe"] = user_response
        
        return {"research_scope": current_scope, "scope_confirmed": False}
    
    # Fallback - shouldn't reach here normally
    return {"scope_confirmed": False}


def research_node(state: State) -> Dict[str, Any]:
    """Research node that uses ReAct agent with Tavily search to conduct deep research."""
    
    # Get the research brief
    brief = state.get("research_brief", "")
    if not brief:
        return {"final_report": "Error: No research brief available. Please complete the scoping phase first."}
    
    # Initialize the language model
    try:
        # Try Anthropic first, then OpenAI as fallback
        if os.getenv("ANTHROPIC_API_KEY"):
            llm = init_chat_model("anthropic:claude-3-5-sonnet-latest")
        elif os.getenv("OPENAI_API_KEY"):
            llm = init_chat_model("openai:gpt-4")
        else:
            return {"final_report": "Error: No API key found. Please set ANTHROPIC_API_KEY or OPENAI_API_KEY in your .env file."}
    except Exception as e:
        return {"final_report": f"Error initializing language model: {str(e)}"}
    
    # Initialize Tavily search tool
    try:
        tavily_tool = TavilySearch(max_results=5)
        tools = [tavily_tool]
    except Exception as e:
        return {"final_report": f"Error initializing Tavily search: {str(e)}. Please check your TAVILY_API_KEY in the .env file."}
    
    # Create research prompt incorporating the brief
    research_prompt = f"""You are a deep research agent tasked with conducting comprehensive research based on the following brief:

{brief}

Your task is to:
1. Use the search tool to gather relevant, current information on the specified topic
2. Focus on the identified areas of interest and requirements
3. Synthesize the information into a comprehensive, well-structured report
4. Provide actionable insights and key findings
5. Cite sources and ensure information accuracy

Please conduct thorough research and provide a detailed report that addresses all aspects of the research brief."""

    # Create the ReAct agent
    try:
        react_agent = create_react_agent(
            model=llm,
            tools=tools,
            prompt=research_prompt
        )
        
        # Execute the research
        research_input = {
            "messages": [{"role": "user", "content": f"Please conduct research based on this brief: {brief}"}]
        }
        
        result = react_agent.invoke(research_input)
        
        # Extract the final report from the agent's response
        if result and "messages" in result:
            final_message = result["messages"][-1]
            if hasattr(final_message, 'content'):
                report = final_message.content
            else:
                report = str(final_message)
        else:
            report = "Research completed, but unable to extract report content."
        
        return {"final_report": report}
        
    except Exception as e:
        return {"final_report": f"Error during research execution: {str(e)}"}


def should_continue_scoping(state: State) -> Literal["research_node", "scoping_node"]:
    """Conditional edge function to determine if scoping is complete."""
    if state.get("scope_confirmed", False):
        return "research_node"
    else:
        return "scoping_node"


# Build the graph
builder = StateGraph(State)

# Add nodes
builder.add_node("scoping_node", scoping_node)
builder.add_node("research_node", research_node)

# Add edges
builder.add_edge(START, "scoping_node")
builder.add_conditional_edges(
    "scoping_node",
    should_continue_scoping,
    {
        "scoping_node": "scoping_node",  # Loop back if not confirmed
        "research_node": "research_node"  # Proceed to research if confirmed
    }
)
builder.add_edge("research_node", END)

# Compile the graph with memory for persistence during interrupts
checkpointer = MemorySaver()
app = builder.compile(checkpointer=checkpointer)
