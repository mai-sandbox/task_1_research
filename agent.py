"""
LangGraph Deep Research Agent with Interactive Scoping and ReAct Search

This agent implements a two-phase research workflow:
1. Interactive Scoping Phase: Engages with user via terminal to clarify research requirements
2. ReAct Research Phase: Performs web research using Tavily and generates detailed reports
"""

import json
from typing import Annotated, Dict, Any, List, TypedDict, Literal
from typing_extensions import TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langchain_tavily import TavilySearch
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.types import interrupt, Command


# State definition
class ResearchState(TypedDict):
    """State for the research agent workflow"""
    messages: Annotated[List[BaseMessage], add_messages]
    research_brief: str
    research_phase: Literal["scoping", "researching", "complete"]
    scoping_complete: bool
    final_report: str


# Initialize LLM (prefer Anthropic, fallback to OpenAI)
def get_llm():
    """Initialize the LLM with preference for Anthropic"""
    try:
        return ChatAnthropic(model="claude-3-5-sonnet-20241022", temperature=0.7)
    except:
        try:
            return ChatOpenAI(model="gpt-4o", temperature=0.7)
        except:
            # Fallback to a more basic model if needed
            return ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)


llm = get_llm()


# Scoping Agent Node
def scoping_agent(state: ResearchState) -> Dict[str, Any]:
    """
    Interactive scoping agent that clarifies research requirements with the user.
    Uses interrupt() to pause execution and gather user input via terminal.
    """
    messages = state.get("messages", [])
    research_phase = state.get("research_phase", "scoping")
    
    # If we're just starting, provide initial greeting
    if len(messages) == 1:  # Only the initial user message
        initial_prompt = """I'm your research assistant. I'll help you conduct thorough research on your topic.
        
To ensure I provide the most relevant and comprehensive research, I need to understand your requirements better.
Let me ask you a few clarifying questions about your research needs."""
        
        # Ask the first clarifying question
        scoping_questions = [
            "What is the main topic or question you want me to research?",
            "What specific aspects or angles are you most interested in?",
            "How deep should the research go? (Quick overview vs. comprehensive analysis)",
            "Are there any specific sources or types of information you prefer?",
            "What will you use this research for? (This helps me tailor the format and depth)"
        ]
        
        response = AIMessage(content=f"{initial_prompt}\n\nLet's start with: {scoping_questions[0]}")
        
        # Interrupt to get user input
        user_response = interrupt({"question": scoping_questions[0]})
        
        return {
            "messages": [response, HumanMessage(content=user_response["response"])],
            "research_phase": "scoping"
        }
    
    # Continue the scoping conversation
    scoping_llm = llm.bind_tools([], tool_choice="none")  # No tools during scoping
    
    # Build conversation context
    scoping_prompt = SystemMessage(content="""You are a research scoping assistant. Your job is to:
1. Understand the user's research needs through clarifying questions
2. Build a comprehensive research brief
3. Determine when you have enough information to proceed

Ask clarifying questions one at a time. Focus on:
- Topic clarity and boundaries
- Depth and breadth requirements
- Specific areas of interest
- Preferred sources or methodologies
- Expected output format

When you have enough information, create a structured research brief and indicate you're ready to proceed.""")
    
    conversation = [scoping_prompt] + messages
    response = scoping_llm.invoke(conversation)
    
    # Check if scoping is complete
    completion_check = llm.invoke([
        SystemMessage(content="Determine if the following conversation has gathered enough information for research. Respond with only 'YES' or 'NO'."),
        *messages,
        response
    ])
    
    if "YES" in completion_check.content.upper():
        # Generate the research brief
        brief_prompt = SystemMessage(content="""Based on the conversation, create a structured research brief that includes:
1. Main research topic/question
2. Specific areas to investigate
3. Required depth of analysis
4. Any constraints or preferences
5. Expected deliverables

Format as a clear, actionable brief for the research phase.""")
        
        brief_response = llm.invoke([brief_prompt] + messages + [response])
        
        return {
            "messages": [response, AIMessage(content=f"Great! I have all the information I need. Here's the research brief:\n\n{brief_response.content}\n\nNow proceeding to conduct the research...")],
            "research_brief": brief_response.content,
            "research_phase": "researching",
            "scoping_complete": True
        }
    else:
        # Continue scoping - ask for more information
        user_response = interrupt({"question": "Please provide more details or answer the question above"})
        
        return {
            "messages": [response, HumanMessage(content=user_response["response"])],
            "research_phase": "scoping",
            "scoping_complete": False
        }


# Research Agent Node with Tavily Tool
def research_agent(state: ResearchState) -> Dict[str, Any]:
    """
    ReAct research agent that uses Tavily search to conduct research
    based on the brief from the scoping phase.
    """
    research_brief = state.get("research_brief", "")
    messages = state.get("messages", [])
    
    # Initialize Tavily search tool
    search_tool = TavilySearch(max_results=5)
    tools = [search_tool]
    
    # Create research LLM with tools
    research_llm = llm.bind_tools(tools)
    
    # Create research prompt
    research_prompt = SystemMessage(content=f"""You are a thorough research agent. Based on the following research brief, conduct comprehensive research using the search tool.

RESEARCH BRIEF:
{research_brief}

INSTRUCTIONS:
1. Break down the research into logical search queries
2. Use the search tool to gather information
3. Analyze and synthesize the findings
4. Follow the ReAct pattern: Reason about what to search, Act by searching, Observe the results
5. Continue until you have comprehensive coverage of the topic
6. Generate a detailed, well-structured report

Remember to:
- Search for multiple perspectives
- Verify important facts with multiple sources
- Look for recent and authoritative information
- Cover all aspects mentioned in the brief""")
    
    # Conduct research iterations
    research_messages = [research_prompt]
    max_iterations = 10
    iteration = 0
    
    while iteration < max_iterations:
        # Get LLM response with potential tool calls
        response = research_llm.invoke(research_messages)
        research_messages.append(response)
        
        # Check if there are tool calls
        if hasattr(response, 'tool_calls') and response.tool_calls:
            # Execute tool calls
            for tool_call in response.tool_calls:
                if tool_call["name"] == "tavily_search":
                    search_results = search_tool.invoke(tool_call["args"])
                    # Create tool message with results
                    tool_message = AIMessage(
                        content=f"Search results for '{tool_call['args'].get('query', '')}': {json.dumps(search_results, indent=2)}"
                    )
                    research_messages.append(tool_message)
        else:
            # No more tool calls, research is complete
            break
        
        iteration += 1
    
    # Generate final comprehensive report
    report_prompt = SystemMessage(content=f"""Based on all the research conducted, create a comprehensive, detailed report that addresses the research brief.

RESEARCH BRIEF:
{research_brief}

Structure the report with:
1. Executive Summary
2. Key Findings (organized by topic/theme)
3. Detailed Analysis
4. Supporting Evidence and Sources
5. Conclusions and Recommendations (if applicable)

Make it thorough, well-organized, and actionable.""")
    
    final_report_response = llm.invoke(research_messages + [report_prompt])
    
    return {
        "messages": messages + [AIMessage(content=f"Research completed! Here's your comprehensive report:\n\n{final_report_response.content}")],
        "final_report": final_report_response.content,
        "research_phase": "complete"
    }


# Supervisor logic to determine next step
def route_to_next_agent(state: ResearchState) -> Literal["scoping", "research", "end"]:
    """
    Determines which agent to route to based on the current state.
    """
    research_phase = state.get("research_phase", "scoping")
    scoping_complete = state.get("scoping_complete", False)
    
    if research_phase == "scoping" and not scoping_complete:
        return "scoping"
    elif research_phase == "researching" or scoping_complete:
        return "research"
    else:
        return "end"


# Build the graph
def build_research_graph():
    """
    Builds the LangGraph workflow for the research agent.
    """
    # Create the graph
    workflow = StateGraph(ResearchState)
    
    # Add nodes
    workflow.add_node("scoping", scoping_agent)
    workflow.add_node("research", research_agent)
    
    # Add edges
    workflow.add_edge(START, "scoping")
    
    # Add conditional edges based on routing logic
    workflow.add_conditional_edges(
        "scoping",
        route_to_next_agent,
        {
            "scoping": "scoping",  # Continue scoping
            "research": "research",  # Move to research
            "end": END
        }
    )
    
    workflow.add_edge("research", END)
    
    # Compile the graph
    return workflow.compile()


# Export the compiled graph as 'app'
app = build_research_graph()


# Optional: Helper function for testing
def run_research_agent(initial_query: str):
    """
    Helper function to run the research agent with an initial query.
    """
    initial_state = {
        "messages": [HumanMessage(content=initial_query)],
        "research_phase": "scoping",
        "scoping_complete": False
    }
    
    result = app.invoke(initial_state)
    return result


if __name__ == "__main__":
    # Example usage
    print("LangGraph Deep Research Agent initialized.")
    print("Use app.invoke() with initial state to start research.")
