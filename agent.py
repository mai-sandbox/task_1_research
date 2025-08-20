import os
from typing import TypedDict, Annotated, Sequence, Optional
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.tools import tool
from langchain.agents import create_react_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import json


class ResearchState(TypedDict):
    """State for the research agent"""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    research_brief: Optional[str]
    final_report: Optional[str]
    clarification_complete: bool


def clarification_agent(state: ResearchState):
    """Agent that clarifies the research scope with the user"""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
    
    messages = state["messages"]
    
    # Check if this is the first interaction
    if len(messages) == 1:
        # Initial greeting and first question
        prompt = """You are a research assistant helping to clarify the scope of a research project.
        The user wants to conduct research on a topic. Your job is to have a brief conversation to understand:
        1. The specific topic or question they want researched
        2. The depth of analysis needed
        3. Any specific aspects they want to focus on
        4. The intended use of the research
        
        Start by greeting the user and asking about their research topic.
        Keep the conversation focused and aim to gather enough information in 2-3 exchanges.
        """
        response = llm.invoke([SystemMessage(content=prompt)] + messages)
    else:
        # Continue clarification
        prompt = """You are a research assistant clarifying the scope of a research project.
        Based on the conversation so far, either:
        1. Ask a follow-up question to better understand the research needs
        2. If you have enough information, create a research brief summarizing what needs to be researched
        
        If creating a brief, start your response with "RESEARCH BRIEF:" followed by a clear, detailed brief.
        Then add "I'll now begin the deep research based on this brief."
        
        Keep questions concise and focused."""
        
        response = llm.invoke([SystemMessage(content=prompt)] + messages)
    
    # Check if clarification is complete
    clarification_complete = "RESEARCH BRIEF:" in response.content
    research_brief = None
    
    if clarification_complete:
        # Extract the research brief
        brief_start = response.content.index("RESEARCH BRIEF:") + len("RESEARCH BRIEF:")
        brief_end = response.content.find("I'll now begin")
        if brief_end == -1:
            research_brief = response.content[brief_start:].strip()
        else:
            research_brief = response.content[brief_start:brief_end].strip()
    
    return {
        "messages": [response],
        "clarification_complete": clarification_complete,
        "research_brief": research_brief
    }


def research_agent(state: ResearchState):
    """ReAct agent that conducts the actual research using Tavily"""
    llm = ChatOpenAI(model="gpt-4o", temperature=0.2)
    
    # Set up Tavily search tool
    tavily_tool = TavilySearchResults(
        max_results=5,
        search_depth="advanced",
        include_answer=True,
        include_raw_content=False,
        include_domains=[],
        exclude_domains=[]
    )
    
    tools = [tavily_tool]
    
    # Create the research prompt
    research_prompt = f"""You are a deep research agent. Based on the following research brief, 
    conduct thorough research using the available search tools.
    
    Research Brief:
    {state['research_brief']}
    
    Instructions:
    1. Break down the research into multiple search queries to cover different aspects
    2. Search for recent, authoritative information
    3. Synthesize findings into a comprehensive report
    4. Include relevant facts, statistics, and insights
    5. Organize the information logically
    6. Cite sources when possible
    
    Conduct thorough research and provide a detailed, well-structured report."""
    
    # Bind tools to the LLM
    llm_with_tools = llm.bind_tools(tools)
    
    # Initial research message
    messages = [SystemMessage(content=research_prompt)]
    
    # Conduct multiple rounds of research
    max_iterations = 8
    research_findings = []
    
    for i in range(max_iterations):
        # Get LLM response
        response = llm_with_tools.invoke(messages)
        messages.append(response)
        
        # Check if there are tool calls
        if response.tool_calls:
            # Execute tools
            for tool_call in response.tool_calls:
                if tool_call["name"] == "tavily_search_results_json":
                    try:
                        results = tavily_tool.invoke(tool_call["args"])
                        research_findings.append(results)
                        # Add tool results to messages
                        tool_message = AIMessage(
                            content=f"Search results: {json.dumps(results, indent=2)}",
                            name="tavily_search_results_json"
                        )
                        messages.append(tool_message)
                    except Exception as e:
                        messages.append(AIMessage(content=f"Tool error: {str(e)}"))
        
        # Check if research is complete
        if not response.tool_calls and i > 2:
            break
    
    # Generate final comprehensive report
    report_prompt = f"""Based on all the research conducted, create a comprehensive, detailed report.
    
    Research Brief: {state['research_brief']}
    
    Structure your report with:
    1. Executive Summary
    2. Key Findings
    3. Detailed Analysis
    4. Relevant Data and Statistics
    5. Conclusions
    6. Sources/References
    
    Make it thorough, informative, and well-organized."""
    
    final_report = llm.invoke(messages + [SystemMessage(content=report_prompt)])
    
    return {
        "messages": [AIMessage(content=f"Research Complete. Here's the detailed report:\n\n{final_report.content}")],
        "final_report": final_report.content
    }


def should_continue_clarification(state: ResearchState):
    """Determine if clarification should continue or move to research"""
    if state.get("clarification_complete", False):
        return "research"
    return "clarify"


# Build the graph
workflow = StateGraph(ResearchState)

# Add nodes
workflow.add_node("clarify", clarification_agent)
workflow.add_node("research", research_agent)

# Set entry point
workflow.set_entry_point("clarify")

# Add edges
workflow.add_conditional_edges(
    "clarify",
    should_continue_clarification,
    {
        "clarify": "clarify",
        "research": "research"
    }
)

workflow.add_edge("research", END)

# Compile the graph with memory
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)