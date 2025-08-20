"""
LangGraph Deep Research Agent with User Interaction and ReAct Search

This agent implements a two-phase workflow:
1. Interactive scope clarification using terminal input/output
2. ReAct agent with Tavily search tools for comprehensive research
"""

import os
from typing import TypedDict, List, Optional, Annotated
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults, TavilyAnswer
from langchain_core.tools import tool
from tavily import TavilyClient
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
import json


# Custom Tavily tools to match task requirements
@tool
def TavilySearch(query: str) -> str:
    """
    Search the web using Tavily API for comprehensive information.
    
    Args:
        query: The search query to execute
        
    Returns:
        Search results with relevant information
    """
    try:
        client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
        response = client.search(query, max_results=10)
        
        # Format results for better readability
        results = []
        for result in response.get('results', []):
            results.append({
                'title': result.get('title', ''),
                'content': result.get('content', ''),
                'url': result.get('url', ''),
                'score': result.get('score', 0)
            })
        
        return json.dumps(results, indent=2)
    except Exception as e:
        return f"Error performing search: {str(e)}"


@tool
def TavilyExtract(url: str) -> str:
    """
    Extract content from a specific URL using Tavily API.
    
    Args:
        url: The URL to extract content from
        
    Returns:
        Extracted content from the URL
    """
    try:
        client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
        response = client.extract(url)
        
        return json.dumps({
            'url': url,
            'content': response.get('content', ''),
            'title': response.get('title', ''),
            'extracted_at': response.get('extracted_at', '')
        }, indent=2)
    except Exception as e:
        return f"Error extracting content from {url}: {str(e)}"


@tool
def TavilyCrawl(url: str, instructions: str = "Extract all relevant information") -> str:
    """
    Crawl a website starting from a URL using Tavily API.
    
    Args:
        url: The starting URL to crawl
        instructions: Instructions for what to look for during crawling
        
    Returns:
        Crawled content and information
    """
    try:
        client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
        response = client.crawl(url, instructions=instructions)
        
        return json.dumps({
            'starting_url': url,
            'instructions': instructions,
            'results': response.get('results', []),
            'crawled_at': response.get('crawled_at', '')
        }, indent=2)
    except Exception as e:
        return f"Error crawling {url}: {str(e)}"


class ResearchScope(TypedDict):
    """Structured research parameters"""
    topic: str
    depth: str  # "basic", "intermediate", "comprehensive"
    sources: List[str]  # preferred source types
    timeline: str  # research timeframe
    focus_areas: List[str]  # specific areas to focus on
    confirmed: bool  # user confirmation to proceed


class AgentState(TypedDict):
    """State schema for the research agent"""
    messages: Annotated[List[BaseMessage], add_messages]
    research_scope: Optional[ResearchScope]
    research_complete: bool
    final_report: Optional[str]


def clarify_scope(state: AgentState) -> AgentState:
    """
    Interactive scope clarification node that engages with user via terminal
    to understand research requirements and build structured research parameters.
    """
    messages = state.get("messages", [])
    current_scope = state.get("research_scope", {
        "topic": "",
        "depth": "",
        "sources": [],
        "timeline": "",
        "focus_areas": [],
        "confirmed": False
    })
    
    # Get the latest human message
    if messages:
        latest_message = messages[-1]
        if isinstance(latest_message, HumanMessage):
            user_input = latest_message.content.lower()
            
            # Check if user wants to proceed with research
            if any(phrase in user_input for phrase in ["proceed", "start research", "begin", "go ahead", "yes, start"]):
                if current_scope.get("topic") and current_scope.get("depth"):
                    current_scope["confirmed"] = True
                    response = AIMessage(content="Great! I have all the information needed. Starting the research phase...")
                    return {
                        **state,
                        "messages": messages + [response],
                        "research_scope": current_scope,
                        "research_complete": False
                    }
                else:
                    response = AIMessage(content="I still need more information before we can proceed. Let me ask a few more questions.")
            
            # Extract information from user input and ask clarifying questions
            if not current_scope.get("topic"):
                if len(user_input) > 10:  # Assume this is the topic
                    current_scope["topic"] = latest_message.content
                    response = AIMessage(content=f"Great! I understand you want to research: {current_scope['topic']}\n\nWhat level of depth would you like for this research?\n- Basic (overview and key points)\n- Intermediate (detailed analysis)\n- Comprehensive (in-depth investigation with multiple perspectives)")
                else:
                    response = AIMessage(content="I'd be happy to help you with research! What topic would you like me to investigate?")
            
            elif not current_scope.get("depth"):
                if any(word in user_input for word in ["basic", "overview", "simple"]):
                    current_scope["depth"] = "basic"
                elif any(word in user_input for word in ["intermediate", "detailed", "moderate"]):
                    current_scope["depth"] = "intermediate"
                elif any(word in user_input for word in ["comprehensive", "in-depth", "thorough", "complete"]):
                    current_scope["depth"] = "comprehensive"
                else:
                    current_scope["depth"] = "intermediate"  # default
                
                response = AIMessage(content=f"Perfect! I'll conduct a {current_scope['depth']} level research.\n\nWhat types of sources would you prefer? (e.g., academic papers, news articles, official websites, industry reports, etc.)")
            
            elif not current_scope.get("sources"):
                # Extract source preferences
                source_keywords = {
                    "academic": ["academic", "papers", "journals", "scholarly", "research"],
                    "news": ["news", "articles", "journalism", "current"],
                    "official": ["official", "government", "institutional", "authoritative"],
                    "industry": ["industry", "reports", "business", "commercial"],
                    "general": ["general", "web", "online", "any"]
                }
                
                sources = []
                for source_type, keywords in source_keywords.items():
                    if any(keyword in user_input for keyword in keywords):
                        sources.append(source_type)
                
                if not sources:
                    sources = ["general"]  # default
                
                current_scope["sources"] = sources
                response = AIMessage(content=f"Got it! I'll focus on {', '.join(sources)} sources.\n\nWhat timeframe are you interested in? (e.g., recent developments, historical perspective, current year, etc.)")
            
            elif not current_scope.get("timeline"):
                if any(word in user_input for word in ["recent", "current", "latest", "new"]):
                    current_scope["timeline"] = "recent"
                elif any(word in user_input for word in ["historical", "past", "history", "background"]):
                    current_scope["timeline"] = "historical"
                elif any(word in user_input for word in ["all", "comprehensive", "any"]):
                    current_scope["timeline"] = "comprehensive"
                else:
                    current_scope["timeline"] = "recent"  # default
                
                response = AIMessage(content=f"Excellent! I'll focus on {current_scope['timeline']} information.\n\nAre there any specific aspects or focus areas within this topic you'd like me to emphasize? (or type 'none' if you want general coverage)")
            
            elif not current_scope.get("focus_areas"):
                if "none" in user_input or "general" in user_input:
                    current_scope["focus_areas"] = ["general"]
                else:
                    # Extract focus areas from user input
                    focus_areas = [area.strip() for area in latest_message.content.split(",")]
                    current_scope["focus_areas"] = focus_areas if focus_areas else ["general"]
                
                # Provide summary and ask for confirmation
                summary = f"""
Perfect! Here's a summary of your research request:

**Topic:** {current_scope['topic']}
**Depth:** {current_scope['depth']} level research
**Sources:** {', '.join(current_scope['sources'])}
**Timeline:** {current_scope['timeline']} focus
**Focus Areas:** {', '.join(current_scope['focus_areas'])}

Does this look correct? Type 'proceed' to start the research, or let me know what you'd like to change.
"""
                response = AIMessage(content=summary)
            
            else:
                # All information collected, waiting for confirmation
                response = AIMessage(content="I have all the information I need. Type 'proceed' when you're ready to start the research!")
    
    else:
        # Initial greeting
        response = AIMessage(content="Hello! I'm your research assistant. I'll help you conduct comprehensive research on any topic.\n\nTo get started, please tell me what you'd like to research.")
        current_scope = {
            "topic": "",
            "depth": "",
            "sources": [],
            "timeline": "",
            "focus_areas": [],
            "confirmed": False
        }
    
    return {
        **state,
        "messages": messages + [response],
        "research_scope": current_scope,
        "research_complete": False
    }


def create_research_agent():
    """
    Create the ReAct research agent with Tavily search tools
    """
    # Initialize the LLM
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.1,
        api_key=os.getenv("OPENAI_API_KEY")
    )
    
    # Initialize Tavily search tools
    tavily_search = TavilySearchResults(
        max_results=10,
        api_key=os.getenv("TAVILY_API_KEY")
    )
    
    tavily_answer = TavilyAnswer(
        api_key=os.getenv("TAVILY_API_KEY")
    )
    
    # Create the ReAct agent with tools
    research_agent = create_react_agent(
        model=llm,
        tools=[tavily_search, tavily_answer],
        state_modifier="You are a professional research assistant. Use the available search tools to conduct thorough research based on the provided research scope. Synthesize information from multiple sources and provide comprehensive, well-cited reports."
    )
    
    return research_agent


def conduct_research(state: AgentState) -> AgentState:
    """
    ReAct research agent node that uses Tavily search tools to conduct research
    based on the clarified scope and generates a detailed report.
    """
    messages = state.get("messages", [])
    research_scope = state.get("research_scope", {})
    
    if not research_scope or not research_scope.get("confirmed"):
        return state
    
    # Create research agent
    research_agent = create_research_agent()
    
    # Construct research prompt based on scope
    research_prompt = f"""
Please conduct {research_scope.get('depth', 'comprehensive')} research on the following topic:

**Topic:** {research_scope.get('topic', '')}
**Preferred Sources:** {', '.join(research_scope.get('sources', ['general']))}
**Timeline Focus:** {research_scope.get('timeline', 'recent')}
**Focus Areas:** {', '.join(research_scope.get('focus_areas', ['general']))}

Please provide a detailed research report that includes:
1. Executive Summary
2. Key Findings
3. Detailed Analysis
4. Sources and Citations
5. Conclusions and Implications

Use the search tools to gather current, accurate information from multiple sources.
"""
    
    # Execute research using the ReAct agent
    research_input = {
        "messages": [HumanMessage(content=research_prompt)]
    }
    
    try:
        research_result = research_agent.invoke(research_input)
        final_report = research_result["messages"][-1].content
        
        response = AIMessage(content=f"Research completed! Here's your comprehensive report:\n\n{final_report}")
        
        return {
            **state,
            "messages": messages + [response],
            "research_complete": True,
            "final_report": final_report
        }
    
    except Exception as e:
        error_response = AIMessage(content=f"I encountered an error during research: {str(e)}. Please check your API keys and try again.")
        return {
            **state,
            "messages": messages + [error_response],
            "research_complete": False
        }


def should_continue_clarification(state: AgentState) -> str:
    """
    Conditional routing logic to determine whether to continue scope clarification
    or proceed to research phase.
    """
    research_scope = state.get("research_scope", {})
    
    if research_scope and research_scope.get("confirmed"):
        return "research"
    else:
        return "clarify"


def create_research_workflow():
    """
    Create and compile the LangGraph workflow with conditional routing
    """
    # Create the state graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("clarify_scope", clarify_scope)
    workflow.add_node("conduct_research", conduct_research)
    
    # Add edges
    workflow.add_edge(START, "clarify_scope")
    
    # Add conditional routing
    workflow.add_conditional_edges(
        "clarify_scope",
        should_continue_clarification,
        {
            "clarify": "clarify_scope",
            "research": "conduct_research"
        }
    )
    
    workflow.add_edge("conduct_research", END)
    
    # Add memory checkpointer
    memory = MemorySaver()
    
    # Compile the graph
    app = workflow.compile(checkpointer=memory)
    
    return app


# Export the compiled graph as 'app'
app = create_research_workflow()

if __name__ == "__main__":
    # Example usage
    print("Research Agent initialized. Use the 'app' variable to invoke the agent.")
    print("Example:")
    print("result = app.invoke({'messages': [HumanMessage('I want to research artificial intelligence')]})")



