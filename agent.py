"""
LangGraph Deep Research Agent with User Interaction and ReAct Search

This agent implements a two-phase research system:
1. User Interaction Phase: Clarifies research scope through terminal interaction
2. ReAct Search Phase: Conducts detailed research using web search tools
"""

import os
from typing import Annotated, Dict, Any, List
from typing_extensions import TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_tavily import TavilySearch

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent, ToolNode, tools_condition
from langgraph.types import Command, interrupt
from langgraph.checkpoint.memory import MemorySaver


# Custom State Definition
class ResearchState(TypedDict):
    """State schema for the research agent workflow"""
    messages: Annotated[List[BaseMessage], add_messages]
    research_brief: str
    final_report: str


def user_interaction_agent(state: ResearchState) -> Dict[str, Any]:
    """
    User Interaction Agent Node
    
    This agent engages in a back-and-forth conversation with the user
    to clarify the research scope and requirements. It uses interrupt()
    to pause execution and gather user input.
    """
    messages = state.get("messages", [])
    research_brief = state.get("research_brief", "")
    
    # If we already have a research brief, skip interaction
    if research_brief:
        return {"research_brief": research_brief}
    
    # Initial greeting and context gathering
    if len(messages) == 1:  # First user message
        initial_request = messages[0].content if messages else ""
        
        # Create initial response asking for clarification
        ai_response = AIMessage(content=f"""
I understand you'd like me to conduct research on: "{initial_request}"

To ensure I provide the most relevant and comprehensive research, I'd like to clarify a few things:

1. What specific aspects of this topic are most important to you?
2. Are there any particular sources or types of information you prefer?
3. What will you use this research for (e.g., decision-making, learning, report writing)?
4. Are there any constraints or specific requirements I should be aware of?

Please provide any additional context that would help me conduct better research for you.
""")
        
        # Interrupt to get user response
        user_response = interrupt({
            "question": "Please provide clarification on the research scope",
            "current_context": initial_request
        })
        
        # Process user response and check if we need more clarification
        if user_response and user_response.get("data"):
            clarification = user_response["data"]
            
            # Ask follow-up if needed
            follow_up_response = AIMessage(content=f"""
Thank you for the clarification. Based on your input, I understand:

{clarification}

Is there anything else you'd like to add or modify about the research scope? 
If you're satisfied with the scope, I'll proceed with the research.

Type 'proceed' if you're ready for me to start the research, or provide any additional requirements.
""")
            
            # Get final confirmation
            final_response = interrupt({
                "question": "Final confirmation before research",
                "clarification": clarification
            })
            
            if final_response and final_response.get("data"):
                user_input = final_response["data"].lower()
                
                # Check if user wants to proceed
                if "proceed" in user_input or "yes" in user_input or "start" in user_input or "go" in user_input:
                    # Create comprehensive research brief
                    research_brief = f"""
RESEARCH BRIEF:
===============
Original Request: {initial_request}

User Clarifications:
{clarification}

Research Objectives:
- Conduct comprehensive web research on the specified topic
- Gather relevant, up-to-date information from reliable sources
- Synthesize findings into a detailed, well-structured report
- Include citations and sources for all key information
- Focus on accuracy, relevance, and comprehensiveness
"""
                    
                    confirmation_message = AIMessage(content="Great! I've prepared the research brief. Now conducting detailed research...")
                    
                    return {
                        "messages": [follow_up_response, HumanMessage(content=final_response["data"]), confirmation_message],
                        "research_brief": research_brief
                    }
                else:
                    # User provided more requirements
                    additional_requirements = final_response["data"]
                    research_brief = f"""
RESEARCH BRIEF:
===============
Original Request: {initial_request}

User Clarifications:
{clarification}

Additional Requirements:
{additional_requirements}

Research Objectives:
- Conduct comprehensive web research on the specified topic
- Gather relevant, up-to-date information from reliable sources
- Synthesize findings into a detailed, well-structured report
- Include citations and sources for all key information
- Focus on accuracy, relevance, and comprehensiveness
"""
                    
                    confirmation_message = AIMessage(content="Perfect! I've incorporated all your requirements. Starting the research now...")
                    
                    return {
                        "messages": [follow_up_response, HumanMessage(content=final_response["data"]), confirmation_message],
                        "research_brief": research_brief
                    }
        
        return {
            "messages": [ai_response],
            "research_brief": ""
        }
    
    # Default return if brief already exists
    return {"research_brief": research_brief}


def react_search_agent(state: ResearchState) -> Dict[str, Any]:
    """
    ReAct Search Agent Node
    
    This agent uses the research brief to conduct detailed web searches
    and generate a comprehensive research report.
    """
    research_brief = state.get("research_brief", "")
    
    if not research_brief:
        return {
            "final_report": "Error: No research brief provided. Please restart the research process.",
            "messages": [AIMessage(content="Error: No research brief provided.")]
        }
    
    # Initialize the search tool
    search_tool = TavilySearch(max_results=5)
    
    # Select the LLM (prefer Anthropic, fallback to OpenAI)
    try:
        llm = ChatAnthropic(model="claude-3-5-sonnet-20241022", temperature=0.3)
    except:
        llm = ChatOpenAI(model="gpt-4o", temperature=0.3)
    
    # Create the ReAct agent with search capabilities
    react_agent = create_react_agent(
        model=llm,
        tools=[search_tool],
        prompt=f"""You are a professional research agent with access to web search capabilities.

Your task is to conduct thorough research based on the following brief:

{research_brief}

Guidelines for your research:
1. Start by searching for the most relevant and recent information
2. Use multiple search queries to cover different aspects of the topic
3. Verify information by checking multiple sources when possible
4. Focus on authoritative and credible sources
5. Organize your findings logically and coherently

After gathering sufficient information, compile a detailed research report that:
- Provides a comprehensive overview of the topic
- Includes key findings and insights
- Cites sources for important claims
- Is well-structured and easy to read
- Addresses all aspects mentioned in the research brief

Begin your research now and create a thorough report.""",
        name="react_search_agent"
    )
    
    # Execute the research
    research_messages = [
        SystemMessage(content=f"Research Brief: {research_brief}"),
        HumanMessage(content="Please conduct the research and provide a detailed report based on the brief.")
    ]
    
    try:
        # Run the ReAct agent
        result = react_agent.invoke({"messages": research_messages})
        
        # Extract the final report from the agent's response
        if result and result.get("messages"):
            final_message = result["messages"][-1]
            
            # Create the final report
            final_report = f"""
RESEARCH REPORT
===============

{research_brief}

FINDINGS:
=========

{final_message.content if hasattr(final_message, 'content') else str(final_message)}

---
Report generated using web search and analysis tools.
"""
            
            return {
                "final_report": final_report,
                "messages": [AIMessage(content=final_report)]
            }
    
    except Exception as e:
        error_report = f"Error during research: {str(e)}"
        return {
            "final_report": error_report,
            "messages": [AIMessage(content=error_report)]
        }
    
    return {
        "final_report": "Research completed but no results found.",
        "messages": [AIMessage(content="Research completed but no results found.")]
    }


def should_continue_interaction(state: ResearchState) -> str:
    """
    Conditional edge function to determine if we should continue
    with user interaction or proceed to research.
    """
    research_brief = state.get("research_brief", "")
    
    if research_brief:
        return "react_search"
    else:
        return "user_interaction"


def should_end_or_continue(state: ResearchState) -> str:
    """
    Conditional edge function to determine if we should end
    or continue the workflow.
    """
    final_report = state.get("final_report", "")
    
    if final_report:
        return END
    else:
        return "user_interaction"


# Build the StateGraph
def build_research_graph():
    """
    Constructs the research agent workflow graph.
    """
    # Initialize the graph
    graph = StateGraph(ResearchState)
    
    # Add nodes
    graph.add_node("user_interaction", user_interaction_agent)
    graph.add_node("react_search", react_search_agent)
    
    # Add edges
    graph.add_edge(START, "user_interaction")
    
    # Conditional routing from user_interaction
    graph.add_conditional_edges(
        "user_interaction",
        should_continue_interaction,
        {
            "user_interaction": "user_interaction",
            "react_search": "react_search"
        }
    )
    
    # Conditional routing from react_search
    graph.add_conditional_edges(
        "react_search",
        should_end_or_continue,
        {
            END: END,
            "user_interaction": "user_interaction"
        }
    )
    
    return graph


# Compile the graph with memory
memory = MemorySaver()
graph = build_research_graph()
app = graph.compile(checkpointer=memory)

# Export the compiled graph as 'app' for deployment
__all__ = ["app"]
