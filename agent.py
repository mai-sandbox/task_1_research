"""
LangGraph Deep Research Agent with User Interaction and ReAct Search

This agent implements a comprehensive research workflow with:
1. User interaction for research scope clarification
2. Research brief generation
3. ReAct agent with Tavily search capabilities
4. Final report compilation
"""

import os
from typing import List, Dict, Any, Optional, TypedDict, Annotated
from typing_extensions import NotRequired

from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent, ToolNode
from langgraph.types import interrupt, Command

from tavily import TavilyClient


# State Schema Definition
class ResearchState(TypedDict):
    """State schema for the research agent workflow"""
    messages: Annotated[List[BaseMessage], add_messages]
    research_scope: NotRequired[str]
    research_brief: NotRequired[str]
    search_results: NotRequired[List[Dict[str, Any]]]
    final_report: NotRequired[str]


# Tavily Search Tool Implementation
@tool
def tavily_search(query: str) -> str:
    """
    Search the web using Tavily API for research purposes.
    
    Args:
        query: The search query to execute
        
    Returns:
        Formatted search results as a string
    """
    try:
        # Get API key from environment
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            return "Error: TAVILY_API_KEY environment variable not set"
        
        # Initialize Tavily client
        tavily_client = TavilyClient(api_key=api_key)
        
        # Perform search
        response = tavily_client.search(
            query=query,
            search_depth="advanced",
            max_results=5,
            include_answer=True,
            include_raw_content=False
        )
        
        # Format results
        formatted_results = []
        
        # Add the answer if available
        if response.get("answer"):
            formatted_results.append(f"**Answer:** {response['answer']}\n")
        
        # Add search results
        if response.get("results"):
            formatted_results.append("**Search Results:**")
            for i, result in enumerate(response["results"], 1):
                title = result.get("title", "No title")
                url = result.get("url", "No URL")
                content = result.get("content", "No content")[:300] + "..."
                
                formatted_results.append(
                    f"{i}. **{title}**\n"
                    f"   URL: {url}\n"
                    f"   Content: {content}\n"
                )
        
        return "\n".join(formatted_results)
        
    except Exception as e:
        return f"Error performing search: {str(e)}"


# Node Implementations
def user_interaction_node(state: ResearchState) -> ResearchState:
    """
    Collect research scope clarification from the user via terminal interaction.
    Uses interrupt() to pause execution and collect user input.
    """
    # Get the last message to understand the research request
    last_message = state["messages"][-1] if state["messages"] else None
    initial_request = last_message.content if last_message else "No initial request provided"
    
    # Create prompt for user clarification
    clarification_prompt = f"""
Research Request: {initial_request}

To provide you with the most relevant and comprehensive research, I need to clarify a few details:

1. What specific aspects of this topic are you most interested in?
2. What is the intended use or audience for this research?
3. Are there any particular time periods, geographic regions, or perspectives you want to focus on?
4. What depth of analysis do you need (overview, detailed analysis, technical deep-dive)?
5. Are there any specific sources or types of information you prefer or want to avoid?

Please provide your clarifications:"""
    
    # Use interrupt to collect user input
    user_clarification = interrupt({
        "prompt": clarification_prompt,
        "type": "research_scope_clarification"
    })
    
    return {
        "research_scope": user_clarification,
        "messages": [AIMessage(content=f"Research scope clarified: {user_clarification}")]
    }


def research_brief_generation_node(state: ResearchState) -> ResearchState:
    """
    Generate a structured research brief based on user clarifications.
    """
    research_scope = state.get("research_scope", "")
    initial_request = state["messages"][0].content if state["messages"] else ""
    
    # Create structured research brief
    research_brief = f"""
# Research Brief

## Original Request
{initial_request}

## Clarified Scope
{research_scope}

## Research Objectives
Based on the clarifications provided, this research will focus on:
- Comprehensive information gathering on the specified topic
- Analysis from multiple perspectives and sources
- Structured presentation of findings
- Actionable insights and conclusions

## Search Strategy
The research will employ systematic web searches using relevant keywords and phrases derived from the scope clarification to ensure comprehensive coverage of the topic.
"""
    
    return {
        "research_brief": research_brief,
        "messages": [AIMessage(content="Research brief generated successfully. Proceeding with detailed research...")]
    }


def report_generation_node(state: ResearchState) -> ResearchState:
    """
    Compile the final research report based on collected information.
    """
    research_brief = state.get("research_brief", "")
    search_results = state.get("search_results", [])
    
    # Extract search information from messages
    search_content = []
    for message in state["messages"]:
        if hasattr(message, 'content') and "Search Results:" in message.content:
            search_content.append(message.content)
    
    # Generate comprehensive report
    final_report = f"""
# Research Report

## Executive Summary
This report presents comprehensive research findings based on the specified scope and objectives.

## Research Brief
{research_brief}

## Key Findings
Based on the research conducted, the following key findings have been identified:

"""
    
    # Add search results summary
    if search_content:
        final_report += "## Research Data\n"
        for i, content in enumerate(search_content, 1):
            final_report += f"### Search Session {i}\n{content}\n\n"
    
    final_report += """
## Conclusions
The research provides valuable insights into the requested topic. The findings are based on current, authoritative sources and provide a comprehensive view of the subject matter.

## Recommendations
Based on the research findings, further investigation may be warranted in specific areas identified during the research process.
"""
    
    return {
        "final_report": final_report,
        "messages": [AIMessage(content=f"Research completed successfully!\n\n{final_report}")]
    }


# Create the main research workflow graph
def create_research_agent():
    """Create and configure the research agent graph"""
    
    # Initialize the state graph
    workflow = StateGraph(ResearchState)
    
    # Add nodes
    workflow.add_node("user_interaction", user_interaction_node)
    workflow.add_node("research_brief", research_brief_generation_node)
    workflow.add_node("report_generation", report_generation_node)
    
    # Create ReAct agent with Tavily search tool
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    react_agent = create_react_agent(
        model=model,
        tools=[tavily_search],
        prompt="You are a research assistant. Use the tavily_search tool to gather comprehensive information based on the research brief. Conduct multiple searches with different keywords to ensure thorough coverage of the topic."
    )
    
    # Add the ReAct agent as a node
    workflow.add_node("react_research", react_agent)
    
    # Define the workflow edges
    workflow.add_edge(START, "user_interaction")
    workflow.add_edge("user_interaction", "research_brief")
    workflow.add_edge("research_brief", "react_research")
    workflow.add_edge("react_research", "report_generation")
    workflow.add_edge("report_generation", END)
    
    # Compile the graph with checkpointer for state persistence
    checkpointer = MemorySaver()
    return workflow.compile(checkpointer=checkpointer)


# Export the compiled graph as 'app' (required by LangGraph deployment)
app = create_research_agent()


# Example usage and testing function
def test_agent():
    """Test function to demonstrate agent usage"""
    from langchain_core.messages import HumanMessage
    
    # Initial state with user message
    initial_state = {
        "messages": [HumanMessage("I want to research the impact of artificial intelligence on healthcare")]
    }
    
    # Configuration with thread ID for session management
    config = {"configurable": {"thread_id": "research_session_1"}}
    
    print("Starting research agent...")
    print("Note: This will pause for user input during the research scope clarification phase.")
    
    try:
        # Run the agent (will pause at interrupt for user input)
        result = app.invoke(initial_state, config=config)
        print("Research completed!")
        print(result)
    except Exception as e:
        print(f"Error running agent: {e}")


if __name__ == "__main__":
    # Only run test if executed directly
    test_agent()
