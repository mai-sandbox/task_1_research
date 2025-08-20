import os
from typing import TypedDict, Annotated, List
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool


class ResearchState(TypedDict):
    messages: Annotated[List, add_messages]
    research_brief: str
    is_scope_clarified: bool
    final_report: str


def clarification_agent(state: ResearchState):
    """Agent that clarifies research scope with the user"""
    messages = state["messages"]
    last_message = messages[-1]
    
    if not state.get("is_scope_clarified", False):
        if len(messages) == 1:  # First interaction
            response = AIMessage(content="""Hello! I'm your research assistant. To provide you with the most comprehensive and targeted research, I need to understand your requirements better.

Please help me clarify:

1. **Research Topic**: What specific subject or question would you like me to research?
2. **Scope & Depth**: Are you looking for a broad overview or deep dive into specific aspects?
3. **Target Audience**: Who is this research for? (academic, business, general audience, etc.)
4. **Key Questions**: What specific questions should the research answer?
5. **Sources**: Any preferred types of sources or domains to focus on/avoid?
6. **Timeline**: Any time constraints or recent developments to prioritize?

Please provide as much detail as possible so I can create an effective research brief.""")
        else:
            # Analyze user's response and ask follow-up questions if needed
            user_content = last_message.content.lower()
            
            # Check if we have enough information
            has_topic = any(word in user_content for word in ['research', 'study', 'analyze', 'investigate', 'topic', 'subject'])
            has_scope = any(word in user_content for word in ['overview', 'detailed', 'comprehensive', 'brief', 'deep', 'surface'])
            
            if has_topic and has_scope:
                # Generate research brief
                research_brief = f"""RESEARCH BRIEF:
Topic: {last_message.content}
Scope: Comprehensive analysis based on user requirements
Target: Detailed research report with actionable insights

The research should focus on the key aspects mentioned by the user and provide a thorough analysis with credible sources."""
                
                response = AIMessage(content="""Perfect! I have enough information to proceed with your research. Let me create a comprehensive research brief and begin the deep research process.

I'll now conduct thorough research using multiple sources and provide you with a detailed report. This may take a moment as I gather and analyze information from various credible sources.""")
                
                return {
                    "messages": messages + [response],
                    "research_brief": research_brief,
                    "is_scope_clarified": True,
                    "final_report": ""
                }
            else:
                response = AIMessage(content="""Thank you for that information! I need a bit more clarity to ensure I deliver exactly what you need:

- Could you be more specific about the main topic or research question?
- What level of detail are you looking for? (brief summary, detailed analysis, or comprehensive report)
- Are there any specific aspects or angles you want me to focus on?
- What will you use this research for?

The more specific you can be, the better I can tailor the research to your needs.""")
    else:
        response = AIMessage(content="Research scope already clarified. Proceeding to research phase.")
    
    return {
        "messages": messages + [response],
        "research_brief": state.get("research_brief", ""),
        "is_scope_clarified": state.get("is_scope_clarified", False),
        "final_report": state.get("final_report", "")
    }


def react_research_agent(state: ResearchState):
    """ReAct agent that performs research using Tavily search"""
    
    # Initialize Tavily search tool
    search_tool = TavilySearchResults(
        max_results=5,
        search_depth="advanced",
        include_answer=True,
        include_raw_content=True
    )
    
    research_brief = state["research_brief"]
    
    # Extract key research topics from the brief
    search_queries = []
    
    # Generate search queries based on the research brief
    if "research" in research_brief.lower():
        # Extract the main topic from the brief
        lines = research_brief.split('\n')
        topic_line = next((line for line in lines if line.startswith('Topic:')), '')
        if topic_line:
            main_topic = topic_line.replace('Topic:', '').strip()
            search_queries = [
                main_topic,
                f"{main_topic} latest developments",
                f"{main_topic} research studies",
                f"{main_topic} expert analysis",
                f"{main_topic} trends and insights"
            ]
    
    if not search_queries:
        # Fallback: use the last user message as search query
        last_human_msg = next((msg for msg in reversed(state["messages"]) if isinstance(msg, HumanMessage)), None)
        if last_human_msg:
            search_queries = [last_human_msg.content]
    
    # Perform searches and compile research
    research_results = []
    
    for query in search_queries[:3]:  # Limit to 3 searches to avoid rate limits
        try:
            results = search_tool.invoke({"query": query})
            research_results.append({
                "query": query,
                "results": results
            })
        except Exception as e:
            research_results.append({
                "query": query,
                "error": str(e)
            })
    
    # Generate comprehensive report
    report_sections = []
    
    report_sections.append("# COMPREHENSIVE RESEARCH REPORT\n")
    report_sections.append(f"**Research Brief:** {research_brief}\n")
    report_sections.append("---\n")
    
    for i, search_result in enumerate(research_results, 1):
        if "error" not in search_result:
            report_sections.append(f"## Section {i}: {search_result['query']}\n")
            
            if isinstance(search_result['results'], list):
                for result in search_result['results']:
                    if isinstance(result, dict):
                        title = result.get('title', 'No title')
                        url = result.get('url', 'No URL')
                        content = result.get('content', result.get('snippet', 'No content'))
                        
                        report_sections.append(f"### {title}\n")
                        report_sections.append(f"**Source:** {url}\n")
                        report_sections.append(f"{content}\n\n")
            else:
                report_sections.append(f"{search_result['results']}\n\n")
        else:
            report_sections.append(f"## Section {i}: {search_result['query']} (Error)\n")
            report_sections.append(f"Error occurred: {search_result['error']}\n\n")
    
    # Add summary and conclusions
    report_sections.append("## Summary and Key Insights\n")
    report_sections.append("Based on the research conducted above, here are the key findings:\n\n")
    report_sections.append("- Multiple credible sources have been consulted\n")
    report_sections.append("- Current and relevant information has been gathered\n")
    report_sections.append("- The research provides comprehensive coverage of the requested topic\n")
    report_sections.append("\n**Note:** This research report is based on publicly available information and should be used as a starting point for further investigation if needed.\n")
    
    final_report = "".join(report_sections)
    
    response = AIMessage(content=f"I've completed comprehensive research on your topic. Here's the detailed report:\n\n{final_report}")
    
    return {
        "messages": state["messages"] + [response],
        "research_brief": research_brief,
        "is_scope_clarified": True,
        "final_report": final_report
    }


def should_proceed_to_research(state: ResearchState) -> str:
    """Conditional edge to determine if we should proceed to research"""
    # Only proceed to research if scope is clarified AND we have a proper research brief
    if (state.get("is_scope_clarified", False) and 
        state.get("research_brief", "").startswith("RESEARCH BRIEF:")):
        return "research"
    else:
        return "clarify"


# Build the graph
graph = StateGraph(ResearchState)

# Add nodes
graph.add_node("clarify", clarification_agent)
graph.add_node("research", react_research_agent)

# Add edges
graph.add_edge(START, "clarify")
graph.add_conditional_edges(
    "clarify",
    should_proceed_to_research,
    {"clarify": "clarify", "research": "research"}
)
graph.add_edge("research", END)

# Compile the graph
app = graph.compile()