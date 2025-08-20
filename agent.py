import os
from typing import TypedDict, Annotated, Sequence, Literal
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain.tools import Tool
from langchain_core.prompts import ChatPromptTemplate
import operator


class ResearchState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    research_brief: str
    phase: Literal["clarification", "research", "complete"]
    clarification_complete: bool
    final_report: str


def clarification_node(state: ResearchState):
    """Interactive node for clarifying research requirements with the user"""
    messages = state["messages"]
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
    
    clarification_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a research assistant helping to clarify and scope a research request.
        
Your goal is to:
1. Understand what the user wants to research
2. Clarify the scope, depth, and specific aspects they're interested in
3. Ask follow-up questions if needed
4. When you have enough information, create a clear research brief

Guidelines:
- Be conversational and helpful
- Ask one or two focused questions at a time
- Help the user refine vague requests
- Once you understand the request clearly, summarize it as a research brief

When you're confident you have enough information, start your response with "RESEARCH_BRIEF:" 
followed by a clear, detailed brief for the research phase."""),
        *messages
    ])
    
    response = llm.invoke(clarification_prompt.format_messages())
    
    new_messages = list(state["messages"])
    new_messages.append(response)
    
    # Check if clarification is complete
    clarification_complete = "RESEARCH_BRIEF:" in response.content
    research_brief = ""
    
    if clarification_complete:
        # Extract the research brief
        parts = response.content.split("RESEARCH_BRIEF:")
        if len(parts) > 1:
            research_brief = parts[1].strip()
    
    return {
        "messages": [response],
        "clarification_complete": clarification_complete,
        "research_brief": research_brief,
        "phase": "research" if clarification_complete else "clarification"
    }


def create_research_agent():
    """Create the ReAct research agent with Tavily search tool"""
    
    # Initialize Tavily search tool
    tavily_tool = TavilySearchResults(
        max_results=5,
        search_depth="advanced",
        include_answer=True,
        include_raw_content=False,
        include_images=False,
    )
    
    tools = [tavily_tool]
    
    # Create the research LLM with tool binding
    research_llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.1
    ).bind_tools(tools)
    
    return research_llm, tools


def research_node(state: ResearchState):
    """ReAct agent node that conducts research based on the brief"""
    
    research_llm, tools = create_research_agent()
    
    # Create research prompt with the brief
    research_prompt = f"""You are a deep research agent. Your task is to conduct thorough research based on this brief:

{state['research_brief']}

Instructions:
1. Use the Tavily search tool to gather comprehensive information
2. Search for multiple perspectives and sources
3. Look for recent, authoritative information
4. Synthesize findings into a detailed, well-structured report

Start by searching for relevant information. You may need to conduct multiple searches to cover all aspects of the brief.
Think step by step about what information you need to gather."""
    
    # Add research prompt to messages
    research_messages = [
        SystemMessage(content=research_prompt),
        HumanMessage(content=f"Please research: {state['research_brief']}")
    ]
    
    response = research_llm.invoke(research_messages)
    
    return {
        "messages": [response],
        "phase": "research"
    }


def tool_node_wrapper(state: ResearchState):
    """Wrapper for tool execution"""
    research_llm, tools = create_research_agent()
    tool_node = ToolNode(tools)
    
    # Get the last message which should have tool calls
    last_message = state["messages"][-1]
    
    # Execute tools
    result = tool_node.invoke({"messages": [last_message]})
    
    return {
        "messages": result["messages"],
        "phase": "research"
    }


def synthesis_node(state: ResearchState):
    """Synthesize research findings into a final report"""
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
    
    # Extract research messages (everything after the brief was created)
    research_messages = []
    brief_found = False
    for msg in state["messages"]:
        if brief_found:
            research_messages.append(msg)
        elif isinstance(msg, AIMessage) and "RESEARCH_BRIEF:" in msg.content:
            brief_found = True
    
    synthesis_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a research synthesis expert. Based on the research conducted, create a comprehensive, well-structured report.

The report should:
1. Start with an executive summary
2. Present key findings organized by topic/theme
3. Include relevant details and evidence
4. Provide analysis and insights
5. End with conclusions and recommendations if applicable

Format the report clearly with sections, bullet points, and proper formatting for readability."""),
        *research_messages,
        ("human", "Please synthesize all the research findings into a comprehensive final report.")
    ])
    
    response = llm.invoke(synthesis_prompt.format_messages())
    
    return {
        "messages": [response],
        "final_report": response.content,
        "phase": "complete"
    }


def should_continue_clarification(state: ResearchState):
    """Determine if clarification phase should continue"""
    if state.get("clarification_complete", False):
        return "research"
    return "clarification"


def should_continue_research(state: ResearchState):
    """Determine if research phase should continue"""
    last_message = state["messages"][-1]
    
    # Check if the last message has tool calls
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tools"
    
    # Check if we've done enough research (look for tool messages)
    tool_message_count = sum(1 for msg in state["messages"] if hasattr(msg, 'name') and msg.name == 'tavily_search_results')
    
    if tool_message_count >= 2:  # After at least 2 searches, synthesize
        return "synthesis"
    
    return "research"


# Build the graph
def create_research_graph():
    workflow = StateGraph(ResearchState)
    
    # Add nodes
    workflow.add_node("clarification", clarification_node)
    workflow.add_node("research", research_node)
    workflow.add_node("tools", tool_node_wrapper)
    workflow.add_node("synthesis", synthesis_node)
    
    # Set entry point
    workflow.set_entry_point("clarification")
    
    # Add edges
    workflow.add_conditional_edges(
        "clarification",
        should_continue_clarification,
        {
            "clarification": "clarification",
            "research": "research"
        }
    )
    
    workflow.add_conditional_edges(
        "research",
        should_continue_research,
        {
            "research": "research",
            "tools": "tools",
            "synthesis": "synthesis"
        }
    )
    
    workflow.add_edge("tools", "research")
    workflow.add_edge("synthesis", END)
    
    return workflow.compile()


# Export the compiled graph as 'app'
app = create_research_graph()