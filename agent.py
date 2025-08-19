"""
LangGraph Deep Research Agent with User Interaction and ReAct Search

This agent implements a multi-stage research workflow:
1. Clarify research scope through user interaction
2. Decide when to proceed with research
3. Conduct research using ReAct agent with Tavily search
4. Generate comprehensive final report

Required Environment Variables:
- TAVILY_API_KEY: Get from https://app.tavily.com/
- ANTHROPIC_API_KEY: Get from https://console.anthropic.com/

Usage:
    from agent import app
    
    initial_state = {
        "messages": [HumanMessage("I want to research artificial intelligence")]
    }
    result = app.invoke(initial_state)
"""

import os
from typing import List, Literal, TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent
from tavily import TavilyClient


# Custom State Schema
class ResearchState(TypedDict):
    """State schema for the research agent workflow"""
    messages: Annotated[List[BaseMessage], add_messages]
    research_brief: str
    research_complete: bool
    final_report: str


# Environment variable handling with fallbacks
def get_api_key(key_name: str, service_name: str) -> str:
    """Get API key from environment with helpful error messages"""
    api_key = os.getenv(key_name)
    if not api_key:
        raise ValueError(
            f"Missing {key_name} environment variable. "
            f"Please get your API key from {service_name} and set it as an environment variable."
        )
    return api_key


# Initialize clients with error handling
try:
    TAVILY_API_KEY = get_api_key("TAVILY_API_KEY", "https://app.tavily.com/")
    ANTHROPIC_API_KEY = get_api_key("ANTHROPIC_API_KEY", "https://console.anthropic.com/")
    
    tavily_client = TavilyClient(api_key=TAVILY_API_KEY)
    llm = ChatAnthropic(model="claude-3-5-sonnet-20241022", api_key=ANTHROPIC_API_KEY)
except ValueError as e:
    print(f"Configuration Error: {e}")
    # Set to None to handle gracefully in nodes
    tavily_client = None
    llm = None


# Tavily Search Tool
def tavily_search_tool(query: str) -> str:
    """
    Search the web using Tavily API
    
    Args:
        query: Search query string
        
    Returns:
        Formatted search results as string
    """
    if not tavily_client:
        return "Error: Tavily client not initialized. Please check TAVILY_API_KEY environment variable."
    
    try:
        response = tavily_client.search(
            query=query,
            search_depth="advanced",
            max_results=5,
            include_answer=True,
            include_raw_content=False
        )
        
        # Format the results
        formatted_results = f"Search Results for: {query}\n\n"
        
        if response.get("answer"):
            formatted_results += f"Summary: {response['answer']}\n\n"
        
        formatted_results += "Sources:\n"
        for i, result in enumerate(response.get("results", []), 1):
            formatted_results += f"{i}. {result.get('title', 'No title')}\n"
            formatted_results += f"   URL: {result.get('url', 'No URL')}\n"
            formatted_results += f"   Content: {result.get('content', 'No content')[:200]}...\n\n"
        
        return formatted_results
        
    except Exception as e:
        return f"Error searching with Tavily: {str(e)}"


# Node 1: Clarify Scope
def clarify_scope(state: ResearchState) -> ResearchState:
    """
    Engage with user to clarify research requirements through terminal-based conversation
    """
    if not llm:
        return {
            "messages": [AIMessage("Error: LLM not initialized. Please check ANTHROPIC_API_KEY environment variable.")],
            "research_brief": "",
            "research_complete": False,
            "final_report": ""
        }
    
    messages = state.get("messages", [])
    current_brief = state.get("research_brief", "")
    
    # If we already have a complete research brief, no need to clarify further
    if current_brief and "RESEARCH_BRIEF_COMPLETE" in str(messages[-1].content if messages else ""):
        return {
            "messages": [AIMessage("Research scope has been clarified. Ready to proceed with research.")],
            "research_brief": current_brief,
            "research_complete": False,
            "final_report": ""
        }
    
    # Analyze conversation to determine what information we still need
    conversation_history = "\n".join([f"{msg.type}: {msg.content}" for msg in messages])
    
    # Create a comprehensive prompt for the LLM to act as a research clarification assistant
    clarification_prompt = f"""You are a research assistant helping to clarify the scope of a research project through terminal-based conversation.

Your goal is to gather comprehensive information about:
1. The specific research topic and focus area
2. The depth and scope of research needed (surface-level overview vs deep analysis)
3. The target audience for the research (academic, business, general public, etc.)
4. Any specific aspects, angles, or subtopics to explore
5. Timeline considerations or urgency
6. Preferred format or structure for the final report
7. Any specific sources or types of information to prioritize

Current conversation history:
{conversation_history}

Current research brief: {current_brief if current_brief else "None yet"}

Instructions:
- If this is the first interaction, introduce yourself and ask initial clarifying questions
- If the user has provided some information, ask follow-up questions to fill gaps
- Be conversational and helpful, not robotic
- Ask 2-3 focused questions at a time, don't overwhelm the user
- When you have gathered sufficient information to create a comprehensive research brief, create a detailed brief and end your response with "RESEARCH_BRIEF_COMPLETE"

The research brief should include:
- Research Topic: [Clear, specific topic]
- Research Scope: [Detailed scope and boundaries]
- Target Audience: [Who will read this research]
- Research Depth: [Level of detail required]
- Key Focus Areas: [Specific aspects to emphasize]
- Timeline: [Any urgency or deadline considerations]
- Output Format: [Preferred structure/format]

Respond as the research assistant:"""

    try:
        # Get response from LLM
        response = llm.invoke([HumanMessage(clarification_prompt)])
        
        # Check if the research brief is complete
        if "RESEARCH_BRIEF_COMPLETE" in response.content:
            # Extract the research brief from the response
            response_parts = response.content.split("RESEARCH_BRIEF_COMPLETE")
            brief_content = response_parts[0].strip()
            
            # Look for the research brief in the response
            brief_start = brief_content.find("Research Topic:")
            if brief_start != -1:
                research_brief = brief_content[brief_start:].strip()
            else:
                # If no structured brief found, create one from the conversation
                research_brief = f"""Research Topic: {messages[0].content if messages else 'Research request'}
Research Scope: Comprehensive research based on user requirements
Target Audience: General audience
Research Depth: Detailed analysis
Key Focus Areas: As discussed in conversation
Timeline: Standard research timeline
Output Format: Comprehensive report"""
            
            return {
                "messages": [response],
                "research_brief": research_brief,
                "research_complete": False,
                "final_report": ""
            }
        else:
            # Continue the conversation
            return {
                "messages": [response],
                "research_brief": current_brief,
                "research_complete": False,
                "final_report": ""
            }
            
    except Exception as e:
        return {
            "messages": [AIMessage(f"Error in clarification process: {str(e)}. Please try again.")],
            "research_brief": current_brief,
            "research_complete": False,
            "final_report": ""
        }


# Node 2: Should Proceed Decision
def should_proceed_decision(state: ResearchState) -> Literal["clarify_scope", "react_research"]:
    """
    Decide whether to continue clarifying or proceed with research
    """
    research_brief = state.get("research_brief", "")
    last_message = state["messages"][-1] if state["messages"] else None
    
    # Check if we have a research brief and the last message indicates completion
    if (research_brief and 
        last_message and 
        "RESEARCH_BRIEF_COMPLETE" in last_message.content):
        return "react_research"
    else:
        return "clarify_scope"


# Node 2: Should Proceed (placeholder node - will be implemented in next task)
def should_proceed(state: ResearchState) -> ResearchState:
    """
    Decision node that determines next step in workflow
    This is a placeholder implementation - will be fully implemented in the next task
    """
    return {
        "messages": [AIMessage("Decision phase - to be implemented")],
        "research_brief": state.get("research_brief", ""),
        "research_complete": False,
        "final_report": ""
    }


# Node 3: ReAct Research (placeholder - will be implemented in next task)
def react_research(state: ResearchState) -> ResearchState:
    """
    Conduct research using ReAct agent with Tavily search
    This is a placeholder implementation - will be fully implemented in the next task
    """
    return {
        "messages": [AIMessage("Research phase - to be implemented")],
        "research_brief": state.get("research_brief", ""),
        "research_complete": True,
        "final_report": ""
    }


# Node 4: Generate Report (placeholder - will be implemented in later task)
def generate_report(state: ResearchState) -> ResearchState:
    """
    Generate comprehensive final report
    This is a placeholder implementation - will be fully implemented in a later task
    """
    return {
        "messages": [AIMessage("Report generation - to be implemented")],
        "research_brief": state.get("research_brief", ""),
        "research_complete": state.get("research_complete", False),
        "final_report": "Final report will be generated here"
    }


# Create the StateGraph
def create_research_agent():
    """Create and configure the research agent graph"""
    
    # Initialize the StateGraph with our custom state
    workflow = StateGraph(ResearchState)
    
    # Add nodes
    workflow.add_node("clarify_scope", clarify_scope)
    workflow.add_node("should_proceed", should_proceed)
    workflow.add_node("react_research", react_research)
    workflow.add_node("generate_report", generate_report)
    
    # Add edges
    workflow.add_edge(START, "clarify_scope")
    workflow.add_edge("clarify_scope", "should_proceed")
    workflow.add_conditional_edges(
        "should_proceed",
        should_proceed_decision,
        {
            "clarify_scope": "clarify_scope",
            "react_research": "react_research"
        }
    )
    workflow.add_edge("react_research", "generate_report")
    workflow.add_edge("generate_report", END)
    
    return workflow.compile()


# Export the compiled graph as 'app' (required pattern)
app = create_research_agent()


if __name__ == "__main__":
    # Test the agent with a sample query
    test_state = {
        "messages": [HumanMessage("I want to research the latest developments in quantum computing")],
        "research_brief": "",
        "research_complete": False,
        "final_report": ""
    }
    
    print("Testing the research agent...")
    try:
        result = app.invoke(test_state)
        print("Agent response:", result["messages"][-1].content)
    except Exception as e:
        print(f"Error testing agent: {e}")





