"""
LangGraph Deep Research Agent with User Interaction and ReAct Search

This agent implements a two-phase workflow:
1. User Interaction Phase: Clarifies research scope through terminal interaction
2. Research Phase: Executes research using ReAct agent with Tavily search
"""

import os
from typing import Annotated, Literal, List
from typing_extensions import TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_tavily import TavilySearch
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt


# State definition
class ResearchState(TypedDict):
    """State for the research agent workflow"""

    messages: Annotated[List[BaseMessage], add_messages]
    research_topic: str
    research_context: str
    research_brief: str
    final_report: str


def clarify_research_scope(state: ResearchState) -> dict:
    """
    Phase 1: Interactive node for clarifying research scope with the user.
    Uses interrupt() to pause execution and gather user input.
    """
    research_topic = state.get("research_topic", "")
    research_context = state.get("research_context", "")
    research_brief = state.get("research_brief", "")

    # If we already have a research brief, skip interaction
    if research_brief:
        return {}

    # Stage 1: Get research topic
    if not research_topic:
        greeting = """Welcome to the Deep Research Agent!

I'll help you conduct thorough research on any topic. First, let me
understand what you'd like to research.

Please describe your research topic and any specific aspects you'd
like me to focus on:"""

        user_topic = interrupt(greeting)
        return {"research_topic": user_topic}

    # Stage 2: Get clarification and context
    if not research_context:
        clarification_prompt = f"""Thank you for sharing your research interest: "{research_topic}"

To ensure I conduct the most relevant research, let me ask a few clarifying questions:

1. What is the main goal or purpose of this research?
2. Are there any specific sources, time periods, or perspectives you'd like me to focus on?
3. What level of detail are you looking for (brief overview vs. comprehensive analysis)?

Please provide any additional context that would help me research this effectively:"""

        user_context = interrupt(clarification_prompt)
        return {"research_context": user_context}

    # Stage 3: Confirm and create research brief
    confirmation_prompt = f"""Based on our discussion, here's my understanding of your research needs:

**Topic:** {research_topic}

**Context and Requirements:** {research_context}

I'll create a research brief and proceed with gathering information.

Would you like to:
1. Proceed with the research as described (press Enter or type '1')
2. Add more specific requirements (type '2')
3. Start over with a different topic (type '3')

Your choice:"""

    user_choice = interrupt(confirmation_prompt)

    if user_choice and user_choice.strip() == "2":
        additional_prompt = "Please provide your additional requirements:"
        additional_input = interrupt(additional_prompt)
        research_context = (
            f"{research_context}\n\nAdditional requirements: {additional_input}"
        )
        return {"research_context": research_context}

    elif user_choice and user_choice.strip() == "3":
        # Reset to start over
        return {"research_topic": "", "research_context": "", "research_brief": ""}

    # Create the research brief
    research_brief = f"""RESEARCH BRIEF:
===============
Topic: {research_topic}

Context and Requirements:
{research_context}

Research Objectives:
- Gather comprehensive and accurate information
- Provide multiple perspectives where relevant
- Include recent and authoritative sources
- Structure findings in a clear, logical manner
- Synthesize information into a detailed report"""

    # Final confirmation
    final_prompt = f"""{research_brief}

I'm ready to begin the research. This will involve:
1. Searching for relevant information using web search
2. Analyzing and synthesizing the findings
3. Creating a detailed report

Press Enter to start the research, or type 'cancel' to stop:"""

    final_confirmation = interrupt(final_prompt)

    if final_confirmation and final_confirmation.strip().lower() == "cancel":
        return {"research_brief": "", "final_report": "Research cancelled by user."}

    return {"research_brief": research_brief}


def execute_research(state: ResearchState) -> dict:
    """
    Phase 2: Execute research using ReAct agent with Tavily search tool.
    """
    research_brief = state.get("research_brief", "")

    if not research_brief:
        return {"final_report": "No research brief provided. Please restart the agent."}

    # Initialize the LLM (prefer OpenAI, fallback to Anthropic)
    try:
        llm = ChatOpenAI(model="gpt-4o", temperature=0.7)
    except Exception:
        try:
            llm = ChatAnthropic(
                model="claude-3-5-sonnet-20241022", temperature=0.7
            )
        except Exception:
            # Fallback to a basic OpenAI model
            llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

    # Initialize Tavily search tool
    tavily_tool = TavilySearch(max_results=5)
    tools = [tavily_tool]

    # Create the research prompt
    research_prompt = f"""You are a thorough research assistant with access to web search capabilities.

{research_brief}

Your task is to:
1. Search for relevant and authoritative information using the Tavily search tool
2. Gather data from multiple sources by making several search queries
3. Synthesize findings into a comprehensive report
4. Ensure accuracy and cite sources where possible

Use the search tool multiple times with different queries to gather comprehensive information.
Focus on finding recent, relevant, and authoritative sources.

After gathering sufficient information, create a detailed research report that:
- Provides a comprehensive overview of the topic
- Includes multiple perspectives and viewpoints
- Cites sources and provides context
- Is well-structured with clear sections
- Addresses all aspects mentioned in the research brief
"""

    # Create the ReAct agent
    react_agent = create_react_agent(model=llm, tools=tools, prompt=research_prompt)

    # Execute the research
    research_request = (
        "Please conduct thorough research based on the brief provided "
        "and create a comprehensive, well-structured report."
    )

    try:
        # Run the ReAct agent
        result = react_agent.invoke(
            {"messages": [HumanMessage(content=research_request)]}
        )

        # Extract the final report from the agent's response
        agent_messages = result.get("messages", [])

        # Find the last AI message with the research results
        final_report = "# Research Report\n\n"
        for msg in reversed(agent_messages):
            if isinstance(msg, AIMessage) and not hasattr(msg, "tool_calls"):
                final_report += msg.content
                break

        if final_report == "# Research Report\n\n":
            # Fallback if no proper response found
            final_report += (
                "Research completed but no detailed findings were generated. "
                "Please try again with a more specific query."
            )

        # Add the research brief to the report for context
        final_report = f"""# Deep Research Report

## Research Brief
{research_brief}

## Research Findings
{final_report}

---
*Report generated by LangGraph Deep Research Agent*
"""

        return {"messages": agent_messages, "final_report": final_report}

    except Exception as e:
        error_report = f"""# Research Report

## Error During Research

An error occurred while conducting the research: {str(e)}

Please ensure:
1. Your Tavily API key is set in the environment (TAVILY_API_KEY)
2. Your OpenAI or Anthropic API key is set
3. You have an active internet connection

You may need to retry the research or adjust your query.
"""
        return {"final_report": error_report}


def route_after_interaction(
    state: ResearchState,
) -> Literal["clarify", "research", "end"]:
    """
    Conditional edge to determine next step after interaction.
    """
    # If we have a final report (including cancellation), end
    if state.get("final_report", ""):
        return "end"

    # If we have a research brief, proceed to research
    if state.get("research_brief", ""):
        return "research"

    # Otherwise, continue clarifying
    return "clarify"


# Build the graph
def build_research_graph():
    """
    Builds the complete research agent graph with two phases.
    """
    # Initialize the graph
    graph_builder = StateGraph(ResearchState)

    # Add nodes
    graph_builder.add_node("clarify", clarify_research_scope)
    graph_builder.add_node("research", execute_research)

    # Add edges
    graph_builder.add_edge(START, "clarify")

    # Conditional routing after clarification
    graph_builder.add_conditional_edges(
        "clarify",
        route_after_interaction,
        {
            "clarify": "clarify",  # Loop back for more interaction
            "research": "research",
            "end": END,
        },
    )

    # After research, end
    graph_builder.add_edge("research", END)

    # Compile with checkpointer for interrupt support
    checkpointer = MemorySaver()
    graph = graph_builder.compile(checkpointer=checkpointer)

    return graph


# Export the compiled graph as 'app'
app = build_research_graph()


# Main execution function for testing
def main():
    """
    Main function to run the research agent interactively.
    """
    print("\n" + "=" * 80)
    print("DEEP RESEARCH AGENT")
    print("=" * 80 + "\n")

    # Initialize state
    initial_state = {
        "messages": [],
        "research_topic": "",
        "research_context": "",
        "research_brief": "",
        "final_report": "",
    }

    # Configuration with thread ID for checkpointing
    config = {"configurable": {"thread_id": "research_session_001"}}

    try:
        # Run the graph
        result = app.invoke(initial_state, config=config)

        # Print the final report
        if result.get("final_report"):
            print("\n" + "=" * 80)
            print("RESEARCH COMPLETE")
            print("=" * 80)
            print(result["final_report"])
            print("=" * 80 + "\n")

    except KeyboardInterrupt:
        print("\n\nResearch session interrupted by user.")
    except Exception as e:
        print(f"\n\nError during research: {str(e)}")


if __name__ == "__main__":
    # Ensure API keys are available
    if not os.getenv("TAVILY_API_KEY"):
        print("Warning: TAVILY_API_KEY not found in environment variables.")
        print("Please set it using: export TAVILY_API_KEY='your-api-key'")

    if not (os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")):
        print("Warning: Neither OPENAI_API_KEY nor ANTHROPIC_API_KEY found.")
        print("Please set at least one using:")
        print("  export OPENAI_API_KEY='your-api-key'")
        print("  export ANTHROPIC_API_KEY='your-api-key'")

    main()
