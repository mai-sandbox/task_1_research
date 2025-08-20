"""
LangGraph Deep Research Agent

This agent performs deep research through a two-stage process:
1. Clarification stage: Interactive dialogue to understand the research scope
2. Research stage: ReAct agent that conducts research and generates detailed reports
"""

import os
from typing import Annotated, Dict, List, Literal
from typing_extensions import TypedDict

from langchain_anthropic import ChatAnthropic
from langchain_tavily import TavilySearch
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition


# State definition
class ResearchState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    research_brief: str
    stage: Literal["clarification", "research", "completed"]
    clarification_complete: bool
    research_complete: bool


# Initialize LLM and tools
llm = ChatAnthropic(model="claude-3-5-sonnet-20241022", temperature=0)
search_tool = TavilySearch(max_results=5)
tools = [search_tool]
llm_with_tools = llm.bind_tools(tools)


# Clarification Agent Prompts
CLARIFICATION_SYSTEM_PROMPT = """You are a research scope clarification specialist. Your job is to understand exactly what the user wants to research through interactive dialogue.

You should ask clarifying questions to understand:
- The specific topic or subject area
- The depth and scope of research needed
- The intended use case or audience
- Any specific aspects they want to focus on
- Time constraints or deadlines
- Preferred sources or types of information

Keep asking questions until you have a clear, comprehensive understanding of what they need. Be conversational but focused.

When you have gathered enough information, respond with "CLARIFICATION_COMPLETE" followed by a detailed research brief summarizing everything you've learned."""

RESEARCH_SYSTEM_PROMPT = """You are a deep research specialist using the ReAct (Reasoning and Acting) methodology. Your job is to conduct thorough research based on the provided research brief and generate a comprehensive report.

Research Process:
1. Break down the research brief into key research questions
2. Use the search tool to gather information systematically
3. Synthesize findings from multiple sources
4. Think critically about the information quality and relevance
5. Generate a detailed, well-structured report

Your final report should be:
- Comprehensive and well-organized
- Based on credible sources
- Include key findings, insights, and conclusions
- Properly cite sources where relevant
- Written in a clear, professional style

When your research is complete, start your final response with "RESEARCH_COMPLETE" followed by your comprehensive report."""


def clarification_agent(state: ResearchState) -> Dict:
    """Agent that clarifies research requirements through interactive dialogue."""
    
    # Check if we've already completed clarification
    if state.get("clarification_complete", False):
        return {
            "stage": "research",
            "messages": []
        }
    
    # Get the conversation history
    messages = state.get("messages", [])
    
    # Add system prompt for clarification
    system_message = SystemMessage(content=CLARIFICATION_SYSTEM_PROMPT)
    conversation = [system_message] + messages
    
    # Generate response
    response = llm.invoke(conversation)
    
    # Check if clarification is complete
    if "CLARIFICATION_COMPLETE" in response.content:
        # Extract the research brief
        brief_start = response.content.find("CLARIFICATION_COMPLETE") + len("CLARIFICATION_COMPLETE")
        research_brief = response.content[brief_start:].strip()
        
        return {
            "messages": [response],
            "research_brief": research_brief,
            "clarification_complete": True,
            "stage": "research"
        }
    
    return {
        "messages": [response],
        "stage": "clarification"
    }


def research_agent(state: ResearchState) -> Dict:
    """ReAct agent that conducts research using search tools."""
    
    # Check if we've already completed research
    if state.get("research_complete", False):
        return {
            "stage": "completed",
            "messages": []
        }
    
    # Get research brief and messages
    research_brief = state.get("research_brief", "")
    messages = state.get("messages", [])
    
    # Create research prompt with the brief
    research_prompt = f"""
    {RESEARCH_SYSTEM_PROMPT}
    
    RESEARCH BRIEF:
    {research_brief}
    
    Begin your systematic research now. Use the search tool to gather comprehensive information.
    """
    
    # Add system message and start research
    system_message = SystemMessage(content=research_prompt)
    
    # If this is the first time in research stage, start with a research plan
    if not any(msg.content.startswith("Starting research") for msg in messages if isinstance(msg, AIMessage)):
        planning_message = HumanMessage(content="Please start the research process based on the brief provided.")
        conversation = [system_message, planning_message]
    else:
        # Continue with existing conversation
        conversation = [system_message] + messages
    
    # Generate response with tools
    response = llm_with_tools.invoke(conversation)
    
    # Check if research is complete
    if "RESEARCH_COMPLETE" in response.content:
        return {
            "messages": [response],
            "research_complete": True,
            "stage": "completed"
        }
    
    return {
        "messages": [response],
        "stage": "research"
    }


def stage_router(state: ResearchState) -> str:
    """Route to appropriate stage based on current state."""
    stage = state.get("stage", "clarification")
    
    if stage == "clarification" and not state.get("clarification_complete", False):
        return "clarification"
    elif stage == "research" and not state.get("research_complete", False):
        return "research"
    elif stage == "completed":
        return END
    else:
        return "research"  # Default to research if unclear


def should_continue_research(state: ResearchState) -> str:
    """Determine if research should continue or use tools."""
    if state.get("research_complete", False):
        return END
    
    messages = state.get("messages", [])
    if messages and hasattr(messages[-1], "tool_calls") and messages[-1].tool_calls:
        return "tools"
    
    return "continue_research"


def continue_research_node(state: ResearchState) -> Dict:
    """Continue research process after tool usage."""
    messages = state.get("messages", [])
    research_brief = state.get("research_brief", "")
    
    # Add context about continuing research
    system_message = SystemMessage(content=f"""
    {RESEARCH_SYSTEM_PROMPT}
    
    RESEARCH BRIEF: {research_brief}
    
    Continue your research based on the information gathered so far. Use additional searches if needed or compile your final report if you have sufficient information.
    """)
    
    conversation = [system_message] + messages
    response = llm_with_tools.invoke(conversation)
    
    # Check if research is complete
    if "RESEARCH_COMPLETE" in response.content:
        return {
            "messages": [response],
            "research_complete": True,
            "stage": "completed"
        }
    
    return {
        "messages": [response]
    }


# Build the graph
graph_builder = StateGraph(ResearchState)

# Add nodes
graph_builder.add_node("clarification", clarification_agent)
graph_builder.add_node("research", research_agent)
graph_builder.add_node("tools", ToolNode(tools))
graph_builder.add_node("continue_research", continue_research_node)

# Add edges
graph_builder.add_edge(START, "clarification")

# Conditional edges for stage routing
graph_builder.add_conditional_edges(
    "clarification",
    stage_router,
    {
        "clarification": "clarification",
        "research": "research",
        END: END
    }
)

# Research workflow with tool usage
graph_builder.add_conditional_edges(
    "research",
    should_continue_research,
    {
        "tools": "tools",
        "continue_research": "continue_research",
        END: END
    }
)

# After tools, continue research
graph_builder.add_edge("tools", "continue_research")

# Continue research can either use more tools or complete
graph_builder.add_conditional_edges(
    "continue_research",
    should_continue_research,
    {
        "tools": "tools",
        "continue_research": "continue_research",
        END: END
    }
)

# Compile the graph
app = graph_builder.compile()


if __name__ == "__main__":
    # Test the agent
    initial_state = {
        "messages": [HumanMessage("I need help researching renewable energy technologies")],
        "stage": "clarification",
        "clarification_complete": False,
        "research_complete": False,
        "research_brief": ""
    }
    
    print("Deep Research Agent Started")
    print("=" * 50)
    
    # Interactive loop for testing
    state = initial_state
    while state.get("stage") != "completed":
        result = app.invoke(state)
        
        # Print the latest AI message
        if result["messages"]:
            latest_message = result["messages"][-1]
            if isinstance(latest_message, AIMessage):
                print(f"\nAgent: {latest_message.content}")
        
        # Update state
        state = result
        
        # If in clarification stage and not complete, get user input
        if (state.get("stage") == "clarification" and 
            not state.get("clarification_complete", False)):
            user_input = input("\nYou: ")
            if user_input.lower() in ['quit', 'exit']:
                break
            state["messages"].append(HumanMessage(content=user_input))
        elif state.get("stage") == "research":
            # Let research continue automatically
            continue
    
    print("\nResearch completed!")