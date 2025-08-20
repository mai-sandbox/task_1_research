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
    Interactive scope clarification node that engages in back-and-forth terminal conversation
    with the user to understand research requirements, asks clarifying questions about topic,
    depth, sources, timeline, and specific focus areas, validates user responses, and updates
    the research_scope state field with structured research parameters when the user confirms
    they're ready to proceed.
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
            user_input = latest_message.content.lower().strip()
            original_input = latest_message.content.strip()
            
            # Handle modification requests
            if any(phrase in user_input for phrase in ["change", "modify", "update", "edit", "different"]):
                return _handle_modification_request(state, messages, current_scope, user_input, original_input)
            
            # Check if user wants to proceed with research
            if any(phrase in user_input for phrase in ["proceed", "start research", "begin", "go ahead", "yes, start", "continue", "ready"]):
                return _handle_proceed_request(state, messages, current_scope)
            
            # Progressive information gathering with validation
            if not current_scope.get("topic"):
                return _collect_topic(state, messages, current_scope, user_input, original_input)
            
            elif not current_scope.get("depth"):
                return _collect_depth(state, messages, current_scope, user_input)
            
            elif not current_scope.get("sources"):
                return _collect_sources(state, messages, current_scope, user_input)
            
            elif not current_scope.get("timeline"):
                return _collect_timeline(state, messages, current_scope, user_input)
            
            elif not current_scope.get("focus_areas"):
                return _collect_focus_areas(state, messages, current_scope, user_input, original_input)
            
            else:
                # All information collected, waiting for confirmation
                return _provide_summary_and_confirmation(state, messages, current_scope)
    
    else:
        # Initial greeting
        response = AIMessage(content="""Hello! I'm your research assistant. I'll help you conduct comprehensive research on any topic.

To get started, please tell me what you'd like to research. For example:
- "I want to research artificial intelligence"
- "Climate change impacts on agriculture"
- "Recent developments in quantum computing"

What topic interests you?""")
        
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


def _handle_modification_request(state: AgentState, messages: List[BaseMessage], current_scope: dict, user_input: str, original_input: str) -> AgentState:
    """Handle user requests to modify previously provided information"""
    if "topic" in user_input:
        current_scope["topic"] = ""
        response = AIMessage(content="Sure! Let's update your research topic. What would you like to research instead?")
    elif "depth" in user_input or "level" in user_input:
        current_scope["depth"] = ""
        response = AIMessage(content="No problem! What level of depth would you prefer?\n- Basic (overview and key points)\n- Intermediate (detailed analysis)\n- Comprehensive (in-depth investigation with multiple perspectives)")
    elif "source" in user_input:
        current_scope["sources"] = []
        response = AIMessage(content="Of course! What types of sources would you prefer? (e.g., academic papers, news articles, official websites, industry reports, etc.)")
    elif "time" in user_input or "timeline" in user_input:
        current_scope["timeline"] = ""
        response = AIMessage(content="Understood! What timeframe are you interested in? (e.g., recent developments, historical perspective, current year, etc.)")
    elif "focus" in user_input or "area" in user_input:
        current_scope["focus_areas"] = []
        response = AIMessage(content="Got it! Are there any specific aspects or focus areas within this topic you'd like me to emphasize? (or type 'none' if you want general coverage)")
    else:
        response = AIMessage(content="I'd be happy to help you modify your research parameters. What specifically would you like to change? (topic, depth, sources, timeline, or focus areas)")
    
    return {
        **state,
        "messages": messages + [response],
        "research_scope": current_scope,
        "research_complete": False
    }


def _handle_proceed_request(state: AgentState, messages: List[BaseMessage], current_scope: dict) -> AgentState:
    """Handle user request to proceed with research"""
    # Validate that we have minimum required information
    if not current_scope.get("topic"):
        response = AIMessage(content="I still need to know what topic you'd like me to research. What would you like me to investigate?")
        return {
            **state,
            "messages": messages + [response],
            "research_scope": current_scope,
            "research_complete": False
        }
    
    # Set defaults for missing optional fields
    if not current_scope.get("depth"):
        current_scope["depth"] = "intermediate"
    if not current_scope.get("sources"):
        current_scope["sources"] = ["general"]
    if not current_scope.get("timeline"):
        current_scope["timeline"] = "recent"
    if not current_scope.get("focus_areas"):
        current_scope["focus_areas"] = ["general"]
    
    current_scope["confirmed"] = True
    response = AIMessage(content="Perfect! I have all the information needed. Starting the comprehensive research phase now...")
    
    return {
        **state,
        "messages": messages + [response],
        "research_scope": current_scope,
        "research_complete": False
    }


def _collect_topic(state: AgentState, messages: List[BaseMessage], current_scope: dict, user_input: str, original_input: str) -> AgentState:
    """Collect and validate the research topic"""
    if len(original_input) < 3:
        response = AIMessage(content="Please provide a more specific research topic. What would you like me to investigate?")
    else:
        # Validate topic is meaningful
        topic = original_input
        current_scope["topic"] = topic
        response = AIMessage(content=f"""Great! I understand you want to research: **{topic}**

What level of depth would you like for this research?
- **Basic**: Overview and key points (quick summary)
- **Intermediate**: Detailed analysis with multiple sources
- **Comprehensive**: In-depth investigation with multiple perspectives and extensive citations

Please choose: basic, intermediate, or comprehensive.""")
    
    return {
        **state,
        "messages": messages + [response],
        "research_scope": current_scope,
        "research_complete": False
    }


def _collect_depth(state: AgentState, messages: List[BaseMessage], current_scope: dict, user_input: str) -> AgentState:
    """Collect and validate the research depth level"""
    if any(word in user_input for word in ["basic", "overview", "simple", "quick", "summary"]):
        current_scope["depth"] = "basic"
    elif any(word in user_input for word in ["intermediate", "detailed", "moderate", "standard"]):
        current_scope["depth"] = "intermediate"
    elif any(word in user_input for word in ["comprehensive", "in-depth", "thorough", "complete", "extensive"]):
        current_scope["depth"] = "comprehensive"
    else:
        # Ask for clarification if unclear
        response = AIMessage(content="I didn't quite understand your preference. Please choose one:\n- **Basic** (quick overview)\n- **Intermediate** (detailed analysis)\n- **Comprehensive** (in-depth investigation)")
        return {
            **state,
            "messages": messages + [response],
            "research_scope": current_scope,
            "research_complete": False
        }
    
    response = AIMessage(content=f"""Perfect! I'll conduct a **{current_scope['depth']}** level research.

What types of sources would you prefer? You can choose multiple:
- **Academic**: Scholarly papers, journals, research studies
- **News**: Current news articles, journalism, media reports
- **Official**: Government websites, institutional sources, authoritative organizations
- **Industry**: Business reports, industry analyses, commercial sources
- **General**: Broad web sources, general information

Please specify your preferences (e.g., "academic and news" or "all sources").""")
    
    return {
        **state,
        "messages": messages + [response],
        "research_scope": current_scope,
        "research_complete": False
    }


def _collect_sources(state: AgentState, messages: List[BaseMessage], current_scope: dict, user_input: str) -> AgentState:
    """Collect and validate source preferences"""
    source_keywords = {
        "academic": ["academic", "papers", "journals", "scholarly", "research", "studies"],
        "news": ["news", "articles", "journalism", "current", "media", "press"],
        "official": ["official", "government", "institutional", "authoritative", "gov"],
        "industry": ["industry", "reports", "business", "commercial", "corporate"],
        "general": ["general", "web", "online", "any", "all", "broad"]
    }
    
    sources = []
    for source_type, keywords in source_keywords.items():
        if any(keyword in user_input for keyword in keywords):
            sources.append(source_type)
    
    if not sources:
        sources = ["general"]  # default
    
    current_scope["sources"] = sources
    response = AIMessage(content=f"""Excellent! I'll focus on **{', '.join(sources)}** sources.

What timeframe are you interested in?
- **Recent**: Latest developments, current information (past 1-2 years)
- **Historical**: Background context, historical perspective
- **Comprehensive**: All relevant timeframes
- **Specific**: A particular time period (please specify)

What's your preference?""")
    
    return {
        **state,
        "messages": messages + [response],
        "research_scope": current_scope,
        "research_complete": False
    }


def _collect_timeline(state: AgentState, messages: List[BaseMessage], current_scope: dict, user_input: str) -> AgentState:
    """Collect and validate timeline preferences"""
    if any(word in user_input for word in ["recent", "current", "latest", "new", "modern"]):
        current_scope["timeline"] = "recent"
    elif any(word in user_input for word in ["historical", "past", "history", "background", "traditional"]):
        current_scope["timeline"] = "historical"
    elif any(word in user_input for word in ["all", "comprehensive", "any", "complete"]):
        current_scope["timeline"] = "comprehensive"
    elif any(word in user_input for word in ["specific", "particular"]) or any(char.isdigit() for char in user_input):
        current_scope["timeline"] = f"specific: {user_input}"
    else:
        current_scope["timeline"] = "recent"  # default
    
    response = AIMessage(content=f"""Great! I'll focus on **{current_scope['timeline']}** information.

Finally, are there any specific aspects or focus areas within this topic you'd like me to emphasize?

For example:
- Specific subtopics or themes
- Particular applications or use cases
- Regional or demographic focus
- Technical vs. non-technical aspects

Please describe your focus areas, or type **'general'** for broad coverage.""")
    
    return {
        **state,
        "messages": messages + [response],
        "research_scope": current_scope,
        "research_complete": False
    }


def _collect_focus_areas(state: AgentState, messages: List[BaseMessage], current_scope: dict, user_input: str, original_input: str) -> AgentState:
    """Collect and validate focus areas"""
    if any(word in user_input for word in ["none", "general", "broad", "all", "everything"]):
        current_scope["focus_areas"] = ["general"]
    else:
        # Extract focus areas from user input
        focus_areas = [area.strip() for area in original_input.replace(",", ";").split(";")]
        focus_areas = [area for area in focus_areas if len(area.strip()) > 2]
        current_scope["focus_areas"] = focus_areas if focus_areas else ["general"]
    
    return _provide_summary_and_confirmation(state, messages, current_scope)


def _provide_summary_and_confirmation(state: AgentState, messages: List[BaseMessage], current_scope: dict) -> AgentState:
    """Provide research summary and ask for final confirmation"""
    summary = f"""
🔍 **Research Request Summary**

**Topic:** {current_scope['topic']}
**Depth:** {current_scope['depth'].title()} level research
**Sources:** {', '.join(current_scope['sources']).title()}
**Timeline:** {current_scope['timeline'].title()} focus
**Focus Areas:** {', '.join(current_scope['focus_areas']).title()}

---

Does this look correct? 

✅ Type **'proceed'** to start the research
🔄 Type **'change [field]'** to modify any aspect (e.g., "change depth")
❓ Ask any questions about the research scope

What would you like to do?"""
    
    response = AIMessage(content=summary)
    
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
    
    # Use the custom Tavily tools as specified in the task requirements
    tools = [TavilySearch, TavilyExtract, TavilyCrawl]
    
    # Create the ReAct agent with tools
    research_agent = create_react_agent(
        model=llm,
        tools=tools,
        state_modifier="You are a professional research assistant. Use the available search tools to conduct thorough research based on the provided research scope. Use TavilySearch for general web searches, TavilyExtract to get detailed content from specific URLs, and TavilyCrawl to comprehensively explore websites. Synthesize information from multiple sources and provide comprehensive, well-cited reports."
    )
    
    return research_agent


def conduct_research(state: AgentState) -> AgentState:
    """
    ReAct research agent node using create_react_agent with Tavily search tools that takes
    the clarified research scope, creates a comprehensive research plan, uses Tavily
    search/extract/crawl tools to gather information from multiple sources, synthesizes
    findings into a detailed report with proper citations, and stores the final report
    in the final_report state field.
    """
    messages = state.get("messages", [])
    research_scope = state.get("research_scope", {})
    
    if not research_scope or not research_scope.get("confirmed"):
        return state
    
    # Notify user that research is starting
    start_message = AIMessage(content="🔍 **Starting comprehensive research phase...**\n\nI'll now use advanced search tools to gather information from multiple sources and create a detailed report based on your requirements.")
    
    # Create research agent
    research_agent = create_research_agent()
    
    # Create comprehensive research plan based on clarified scope
    research_plan = _create_research_plan(research_scope)
    
    # Construct detailed research prompt based on scope and plan
    research_prompt = f"""
You are conducting {research_scope.get('depth', 'comprehensive')} research on the following topic with a structured approach:

**RESEARCH PARAMETERS:**
- **Topic:** {research_scope.get('topic', '')}
- **Depth Level:** {research_scope.get('depth', 'comprehensive')}
- **Preferred Sources:** {', '.join(research_scope.get('sources', ['general']))}
- **Timeline Focus:** {research_scope.get('timeline', 'recent')}
- **Focus Areas:** {', '.join(research_scope.get('focus_areas', ['general']))}

**RESEARCH PLAN:**
{research_plan}

**RESEARCH METHODOLOGY:**
1. **Initial Search Phase**: Use TavilySearch to find relevant sources and get an overview
2. **Deep Dive Phase**: Use TavilyExtract to get detailed content from the most relevant URLs
3. **Comprehensive Exploration**: Use TavilyCrawl to explore key websites thoroughly
4. **Synthesis Phase**: Combine all findings into a comprehensive report

**REQUIRED OUTPUT FORMAT:**
Please provide a detailed research report with the following structure:

# Research Report: {research_scope.get('topic', '')}

## Executive Summary
[Concise overview of key findings and conclusions]

## Research Methodology
[Brief description of sources and methods used]

## Key Findings
[Main discoveries organized by theme or importance]

## Detailed Analysis
[In-depth analysis based on research scope and focus areas]

## Sources and Citations
[All sources used with URLs and brief descriptions]

## Conclusions and Implications
[Summary of insights and their significance]

## Recommendations
[If applicable, actionable recommendations based on findings]

**IMPORTANT INSTRUCTIONS:**
- Use ALL available search tools (TavilySearch, TavilyExtract, TavilyCrawl) to gather comprehensive information
- Prioritize sources that match the specified source preferences: {', '.join(research_scope.get('sources', ['general']))}
- Focus on {research_scope.get('timeline', 'recent')} information as requested
- Pay special attention to: {', '.join(research_scope.get('focus_areas', ['general']))}
- Provide proper citations with URLs for all sources
- Ensure the depth matches the requested level: {research_scope.get('depth', 'comprehensive')}
- Synthesize information from multiple sources rather than just summarizing individual sources
"""
    
    # Execute research using the ReAct agent
    research_input = {
        "messages": [HumanMessage(content=research_prompt)]
    }
    
    try:
        # Execute the research with the ReAct agent
        research_result = research_agent.invoke(research_input)
        final_report = research_result["messages"][-1].content
        
        # Enhance the report with metadata
        enhanced_report = _enhance_report_with_metadata(final_report, research_scope)
        
        # Create completion message
        completion_message = f"""✅ **Research Completed Successfully!**

I've conducted a {research_scope.get('depth', 'comprehensive')} research investigation on "{research_scope.get('topic', '')}" using multiple search tools and sources.

**Research Summary:**
- **Sources Used:** {', '.join(research_scope.get('sources', ['general']))}
- **Timeline Focus:** {research_scope.get('timeline', 'recent')}
- **Focus Areas:** {', '.join(research_scope.get('focus_areas', ['general']))}
- **Tools Utilized:** TavilySearch, TavilyExtract, TavilyCrawl

---

{enhanced_report}"""
        
        response = AIMessage(content=completion_message)
        
        return {
            **state,
            "messages": messages + [start_message, response],
            "research_complete": True,
            "final_report": enhanced_report
        }
    
    except Exception as e:
        error_response = AIMessage(content=f"""❌ **Research Error Encountered**

I encountered an error during the research process: {str(e)}

**Troubleshooting Steps:**
1. Verify that TAVILY_API_KEY is properly set in your environment
2. Verify that OPENAI_API_KEY is properly set in your environment
3. Check your internet connection
4. Ensure you have sufficient API credits

Please check your configuration and try again.""")
        
        return {
            **state,
            "messages": messages + [error_response],
            "research_complete": False
        }


def _create_research_plan(research_scope: dict) -> str:
    """
    Create a comprehensive research plan based on the clarified research scope
    """
    topic = research_scope.get('topic', '')
    depth = research_scope.get('depth', 'comprehensive')
    sources = research_scope.get('sources', ['general'])
    timeline = research_scope.get('timeline', 'recent')
    focus_areas = research_scope.get('focus_areas', ['general'])
    
    # Create depth-specific research strategies
    depth_strategies = {
        'basic': [
            'Find 3-5 key sources for overview',
            'Identify main concepts and definitions',
            'Gather basic statistics or facts',
            'Summarize current status'
        ],
        'intermediate': [
            'Find 8-12 diverse sources across multiple perspectives',
            'Analyze trends and patterns',
            'Compare different viewpoints or approaches',
            'Examine recent developments and changes',
            'Include expert opinions and analysis'
        ],
        'comprehensive': [
            'Gather 15+ sources from multiple source types',
            'Conduct deep analysis of historical context',
            'Examine multiple perspectives and controversies',
            'Analyze implications and future projections',
            'Include case studies and real-world examples',
            'Cross-reference information across sources'
        ]
    }
    
    # Create source-specific search strategies
    source_strategies = []
    if 'academic' in sources:
        source_strategies.append('- Search for scholarly articles, research papers, and academic studies')
    if 'news' in sources:
        source_strategies.append('- Find current news articles and journalistic reports')
    if 'official' in sources:
        source_strategies.append('- Look for government reports, institutional publications, and official statements')
    if 'industry' in sources:
        source_strategies.append('- Gather industry reports, business analyses, and commercial insights')
    if 'general' in sources:
        source_strategies.append('- Use broad web sources and general information repositories')
    
    # Create timeline-specific strategies
    timeline_strategies = {
        'recent': 'Focus on information from the past 1-2 years, emphasizing latest developments',
        'historical': 'Include historical context and background, tracing development over time',
        'comprehensive': 'Cover both historical background and recent developments for complete picture'
    }
    
    # Handle specific timeline formats
    if timeline.startswith('specific:'):
        timeline_strategy = f"Focus on the specified timeframe: {timeline.replace('specific:', '').strip()}"
    else:
        timeline_strategy = timeline_strategies.get(timeline, timeline_strategies['recent'])
    
    # Create focus area strategies
    focus_strategy = "Pay special attention to: " + ", ".join(focus_areas) if focus_areas != ['general'] else "Provide broad coverage of all relevant aspects"
    
    research_plan = f"""
**RESEARCH STRATEGY FOR: {topic}**

**Depth Level: {depth.title()}**
{chr(10).join(f"• {strategy}" for strategy in depth_strategies.get(depth, depth_strategies['comprehensive']))}

**Source Strategy:**
{chr(10).join(source_strategies) if source_strategies else "• Use all available source types"}

**Timeline Strategy:**
• {timeline_strategy}

**Focus Strategy:**
• {focus_strategy}

**Search Execution Plan:**
1. **Broad Search**: Use TavilySearch with general queries to map the landscape
2. **Targeted Extraction**: Use TavilyExtract on the most promising URLs found
3. **Deep Exploration**: Use TavilyCrawl on key websites for comprehensive coverage
4. **Cross-Validation**: Verify information across multiple sources
5. **Synthesis**: Combine findings into coherent, well-cited report
"""
    
    return research_plan


def _enhance_report_with_metadata(report: str, research_scope: dict) -> str:
    """
    Enhance the research report with metadata and formatting
    """
    from datetime import datetime
    
    metadata = f"""
**Research Metadata:**
- **Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
- **Topic:** {research_scope.get('topic', '')}
- **Depth:** {research_scope.get('depth', 'comprehensive').title()}
- **Sources:** {', '.join(research_scope.get('sources', ['general'])).title()}
- **Timeline:** {research_scope.get('timeline', 'recent').title()}
- **Focus Areas:** {', '.join(research_scope.get('focus_areas', ['general'])).title()}

---

"""
    
    return metadata + report


def should_continue_clarification(state: AgentState) -> str:
    """
    Conditional routing logic between the scope clarification and research phases that
    checks the research_complete flag to determine whether to continue clarifying scope
    or proceed to research, handles user confirmation to transition between phases, and
    ensures proper state management throughout the workflow.
    """
    research_scope = state.get("research_scope", {})
    research_complete = state.get("research_complete", False)
    messages = state.get("messages", [])
    
    # If research is already complete, end the workflow
    if research_complete:
        return "end"
    
    # Check if we have a confirmed research scope to proceed to research
    if research_scope and research_scope.get("confirmed"):
        # Validate that we have minimum required information for research
        if _validate_research_scope(research_scope):
            return "research"
        else:
            # Reset confirmation if scope is invalid and return to clarification
            research_scope["confirmed"] = False
            return "clarify"
    
    # Continue with scope clarification if not confirmed or incomplete
    return "clarify"


def _validate_research_scope(research_scope: dict) -> bool:
    """
    Validate that the research scope contains all necessary information
    for conducting research.
    """
    required_fields = ["topic", "depth", "sources", "timeline", "focus_areas"]
    
    # Check that all required fields are present and non-empty
    for field in required_fields:
        value = research_scope.get(field)
        if not value:
            return False
        
        # Special validation for list fields
        if isinstance(value, list) and len(value) == 0:
            return False
    
    # Validate topic is meaningful (not just whitespace)
    topic = research_scope.get("topic", "").strip()
    if len(topic) < 3:
        return False
    
    # Validate depth is one of the expected values
    depth = research_scope.get("depth", "")
    if depth not in ["basic", "intermediate", "comprehensive"]:
        return False
    
    return True


def _check_research_completion(state: AgentState) -> str:
    """
    Check if research has been completed successfully and determine next routing.
    Ensures proper state management after research phase.
    """
    research_complete = state.get("research_complete", False)
    final_report = state.get("final_report")
    
    # If research completed successfully with a report, end the workflow
    if research_complete and final_report:
        return "complete"
    
    # If research failed or incomplete, return to clarification for retry
    # This allows users to modify their scope and try again
    return "retry"


def create_research_workflow():
    """
    Create and compile the LangGraph workflow with enhanced conditional routing
    that ensures proper state management throughout the workflow.
    """
    # Create the state graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("clarify_scope", clarify_scope)
    workflow.add_node("conduct_research", conduct_research)
    
    # Add edges
    workflow.add_edge(START, "clarify_scope")
    
    # Add enhanced conditional routing between scope clarification and research phases
    workflow.add_conditional_edges(
        "clarify_scope",
        should_continue_clarification,
        {
            "clarify": "clarify_scope",  # Continue scope clarification
            "research": "conduct_research",  # Proceed to research phase
            "end": END  # End workflow if research is complete
        }
    )
    
    # Add conditional routing after research completion
    workflow.add_conditional_edges(
        "conduct_research",
        _check_research_completion,
        {
            "complete": END,  # End workflow when research is complete
            "retry": "clarify_scope"  # Return to clarification if research failed
        }
    )
    
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











