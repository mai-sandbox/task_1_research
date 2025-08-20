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
    
    This agent:
    1. Engages in back-and-forth conversation to understand research needs
    2. Asks targeted clarifying questions about scope, depth, and requirements
    3. Continues dialogue until confident about the research brief
    4. Returns a structured research brief for the ReAct agent
    """
    messages = state.get("messages", [])
    research_phase = state.get("research_phase", "scoping")
    scoping_complete = state.get("scoping_complete", False)
    
    # Extract only human and AI messages for conversation context
    conversation_messages = [msg for msg in messages if isinstance(msg, (HumanMessage, AIMessage))]
    
    # If we're just starting, provide initial greeting
    if len(conversation_messages) == 1:  # Only the initial user message
        initial_prompt = """Hello! I'm your deep research assistant. I'll help you conduct thorough, comprehensive research on your topic.

To ensure I provide the most relevant and valuable insights, I need to understand your research requirements better. I'll ask you a series of clarifying questions to build a detailed research brief.

Let's start by understanding your main research focus."""
        
        first_question = "What is the main topic, question, or problem you want me to research? Please be as specific as possible."
        
        response_content = f"{initial_prompt}\n\n**Question 1:** {first_question}"
        response = AIMessage(content=response_content)
        
        # Interrupt to get user input
        user_input = interrupt({
            "stage": "initial_scoping",
            "question": first_question,
            "message": "Please provide your response:"
        })
        
        return {
            "messages": [response, HumanMessage(content=user_input.get("response", user_input.get("data", "")))],
            "research_phase": "scoping",
            "scoping_complete": False
        }
    
    # Continue the scoping conversation
    scoping_llm = llm.bind_tools([], tool_choice="none")  # No tools during scoping
    
    # Build conversation context with enhanced prompt
    scoping_prompt = SystemMessage(content="""You are an expert research scoping assistant engaged in an interactive dialogue with the user.

Your role is to:
1. Understand the user's research needs through targeted clarifying questions
2. Build a comprehensive research brief through iterative conversation
3. Ensure all critical aspects are covered before proceeding

Guidelines for your conversation:
- Ask ONE clarifying question at a time
- Build upon previous answers to go deeper
- Be conversational but professional
- Show that you understand their needs by summarizing key points
- Focus on gathering information about:
  * Core research topic and specific questions
  * Scope boundaries (what to include/exclude)
  * Required depth of analysis (surface-level vs deep dive)
  * Specific aspects or angles of interest
  * Time period or geographic focus (if relevant)
  * Preferred sources or methodologies
  * Expected output format and use case
  * Any constraints or special requirements

After gathering sufficient information (typically 4-7 exchanges), indicate you're ready to proceed by saying "I now have a comprehensive understanding of your research needs."

Current conversation stage: You've had {num_exchanges} exchanges with the user.""".format(
        num_exchanges=len(conversation_messages) // 2
    ))
    
    conversation = [scoping_prompt] + conversation_messages
    response = scoping_llm.invoke(conversation)
    
    # Check if scoping is complete by looking for completion indicators
    completion_indicators = [
        "comprehensive understanding",
        "ready to proceed",
        "have all the information",
        "sufficient information",
        "research brief is complete"
    ]
    
    is_complete = any(indicator in response.content.lower() for indicator in completion_indicators)
    
    # Also check if we've had enough exchanges (minimum 3, maximum 10)
    num_exchanges = len(conversation_messages) // 2
    if num_exchanges >= 3 and ("?" not in response.content or is_complete):
        # Time to finalize the research brief
        is_complete = True
    
    if is_complete:
        # Generate the structured research brief
        brief_prompt = SystemMessage(content="""Based on the conversation with the user, create a comprehensive, structured research brief.

The brief should include:

# Research Brief

## 1. Primary Research Question/Topic
[State the main research focus clearly]

## 2. Scope and Boundaries
- What to include:
- What to exclude:
- Geographic/temporal focus (if applicable):

## 3. Specific Areas of Investigation
[List 3-5 key areas to explore]

## 4. Depth of Analysis Required
[Specify the level of detail needed]

## 5. Preferred Sources and Methodologies
[Note any preferences mentioned]

## 6. Constraints and Special Requirements
[List any limitations or special needs]

## 7. Expected Deliverables
[Describe the desired output format and use case]

## 8. Success Criteria
[What would make this research successful for the user]

Make this brief actionable and comprehensive for the research phase.""")
        
        brief_response = llm.invoke([brief_prompt] + conversation_messages + [response])
        
        confirmation_message = f"""Excellent! I now have a comprehensive understanding of your research needs. 

{response.content}

Based on our conversation, I've prepared the following research brief:

{brief_response.content}

I'll now proceed to conduct thorough research based on this brief. The research phase will involve:
1. Systematic web searches using advanced search tools
2. Analysis and synthesis of findings
3. Generation of a comprehensive report

Starting the research phase now..."""
        
        return {
            "messages": [AIMessage(content=confirmation_message)],
            "research_brief": brief_response.content,
            "research_phase": "researching",
            "scoping_complete": True
        }
    else:
        # Continue scoping - ask for more information
        # Extract the question from the response if it exists
        question_prompt = response.content
        if "?" in response.content:
            # Extract the last question asked
            lines = response.content.split('\n')
            question_lines = [line for line in lines if '?' in line]
            if question_lines:
                question_prompt = question_lines[-1]
        
        # Interrupt to get user input for the next question
        user_input = interrupt({
            "stage": "continued_scoping",
            "question": question_prompt,
            "exchange_number": num_exchanges + 1,
            "message": "Please provide your response:"
        })
        
        user_response_content = user_input.get("response", user_input.get("data", ""))
        
        return {
            "messages": [response, HumanMessage(content=user_response_content)],
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

