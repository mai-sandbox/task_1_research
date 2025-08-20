"""
LangGraph Deep Research Agent with User Interaction and ReAct Search

This agent implements a two-stage research system:
1. Clarification Stage: Interactive conversation to refine research scope
2. Research Stage: ReAct agent with Tavily search for detailed research
"""

import os
from typing import Annotated, TypedDict, List, Optional

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_anthropic import ChatAnthropic
from langchain_tavily import TavilySearch
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt, Command


class ResearchState(TypedDict):
    """State schema for the research agent"""

    messages: Annotated[List[BaseMessage], add_messages]
    research_brief: Optional[str]
    final_report: Optional[str]
    clarification_complete: bool


def clarification_node(state: ResearchState) -> dict:
    """
    Interactive node that clarifies research scope with the user.
    Uses interrupt() to pause execution and collect user input.
    """
    messages = state.get("messages", [])

    # Initialize the LLM for clarification
    llm = ChatAnthropic(model="claude-3-5-sonnet-20241022", temperature=0.3)

    # If this is the first interaction, provide initial guidance
    if len(messages) <= 1:
        initial_prompt = """I'm a deep research agent designed to help you conduct thorough
research on any topic.

To provide you with the most comprehensive and relevant research, I need to understand:
1. What specific topic or question you want to research
2. The scope and depth of research needed
3. Any particular aspects or angles you want to focus on
4. The intended use or audience for this research

Let's start by discussing your research needs. What would you like me to research for you?"""

        # Send initial message and wait for user input
        user_response = interrupt(initial_prompt)

        # Add the user's response to messages
        return {
            "messages": [HumanMessage(content=user_response)],
            "clarification_complete": False,
        }

    # Prepare conversation for the LLM to analyze
    system_prompt = """You are a research assistant helping to clarify and refine a research brief.
    Your goal is to understand the user's research needs thoroughly before proceeding.
    
    Analyze the conversation and determine if you have enough information to create a comprehensive research brief.
    If you need more clarification, ask specific, targeted questions.
    If you have sufficient information, create a detailed research brief.
    
    When you have enough information, start your response with "RESEARCH_BRIEF_READY:" followed by the brief."""

    # Get LLM's analysis of the conversation
    llm_messages = [SystemMessage(content=system_prompt)] + messages
    response = llm.invoke(llm_messages)

    # Check if the LLM thinks we have enough information
    if "RESEARCH_BRIEF_READY:" in response.content:
        # Extract the research brief
        brief_start = response.content.index("RESEARCH_BRIEF_READY:") + len(
            "RESEARCH_BRIEF_READY:"
        )
        research_brief = response.content[brief_start:].strip()

        # Confirm with the user
        confirmation_message = f"""I've prepared the following research brief based on our discussion:

{research_brief}

Would you like me to proceed with this research, or would you like to modify anything? 
(Type 'proceed' to continue, or provide any modifications)"""

        user_confirmation = interrupt(confirmation_message)

        if user_confirmation.lower().strip() == "proceed":
            return {
                "messages": [
                    AIMessage(
                        content=f"Research brief confirmed. Proceeding with research..."
                    )
                ],
                "research_brief": research_brief,
                "clarification_complete": True,
            }
        else:
            # User wants modifications
            return {
                "messages": [HumanMessage(content=user_confirmation)],
                "clarification_complete": False,
            }
    else:
        # Need more clarification - send the LLM's questions to the user
        user_response = interrupt(response.content)

        return {
            "messages": [
                AIMessage(content=response.content),
                HumanMessage(content=user_response),
            ],
            "clarification_complete": False,
        }


def research_node(state: ResearchState) -> dict:
    """
    Research node that uses a ReAct agent with Tavily search to conduct research.
    """
    research_brief = state.get("research_brief", "")

    if not research_brief:
        return {
            "final_report": "Error: No research brief provided.",
            "messages": [AIMessage(content="Error: No research brief provided.")],
        }

    # Initialize Tavily search tool
    tavily_tool = TavilySearch(max_results=3)

    # Create a ReAct agent with the search tool
    research_agent = create_react_agent(
        model=ChatAnthropic(model="claude-3-5-sonnet-20241022", temperature=0.3),
        tools=[tavily_tool],
        prompt="""You are an expert research agent with access to web search capabilities.
        
Your task is to conduct thorough, detailed research based on the provided research brief.
Use your search tool to gather comprehensive information from multiple sources.
Synthesize the information into a well-structured, detailed report.

Important guidelines:
1. Search for multiple perspectives and sources
2. Verify facts across different sources when possible
3. Include relevant statistics, examples, and expert opinions
4. Structure your findings clearly with sections and subsections
5. Cite your sources when presenting information
6. Provide a balanced, objective analysis

Research Brief:
{brief}

Conduct thorough research and provide a comprehensive report.""",
    )

    # Execute the research
    research_prompt = f"""Research Brief:
{research_brief}

Please conduct thorough research on this topic and provide a detailed, well-structured report."""

    try:
        # Run the research agent
        result = research_agent.invoke(
            {"messages": [HumanMessage(content=research_prompt)]}
        )

        # Extract the final report from the agent's response
        final_message = result["messages"][-1]
        final_report = (
            final_message.content
            if hasattr(final_message, "content")
            else str(final_message)
        )

        return {
            "final_report": final_report,
            "messages": [
                AIMessage(
                    content=f"Research completed. Here's your detailed report:\n\n{final_report}"
                )
            ],
        }

    except Exception as e:
        error_message = f"An error occurred during research: {str(e)}"
        return {
            "final_report": error_message,
            "messages": [AIMessage(content=error_message)],
        }


def should_continue_clarification(state: ResearchState) -> str:
    """
    Determines whether to continue clarification or move to research.
    """
    if state.get("clarification_complete", False):
        return "research"
    return "clarification"


# Build the graph
def create_research_graph():
    """
    Creates and compiles the research agent graph.
    """
    # Initialize the graph with our state schema
    graph = StateGraph(ResearchState)

    # Add nodes
    graph.add_node("clarification", clarification_node)
    graph.add_node("research", research_node)

    # Add edges
    graph.add_edge(START, "clarification")

    # Add conditional edge from clarification
    graph.add_conditional_edges(
        "clarification",
        should_continue_clarification,
        {
            "clarification": "clarification",  # Loop back for more clarification
            "research": "research",  # Move to research when ready
        },
    )

    # Research node goes to END
    graph.add_edge("research", END)

    # Compile with checkpointer for interrupt support
    checkpointer = InMemorySaver()
    compiled_graph = graph.compile(checkpointer=checkpointer)

    return compiled_graph


# Create and export the compiled graph as 'app'
app = create_research_graph()


# Optional: Add a main function for testing
if __name__ == "__main__":
    import uuid

    # Ensure API keys are set
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("Warning: ANTHROPIC_API_KEY not set. Please set it in your environment.")
    if not os.getenv("TAVILY_API_KEY"):
        print("Warning: TAVILY_API_KEY not set. Please set it in your environment.")

    # Create a config with thread ID for conversation continuity
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    print("Starting Deep Research Agent...")
    print("=" * 50)

    # Initial invocation
    initial_state = {
        "messages": [HumanMessage(content="I need help with research")],
        "clarification_complete": False,
    }

    try:
        # Start the conversation
        for event in app.stream(initial_state, config, stream_mode="updates"):
            if "__interrupt__" in event:
                # Handle interrupt - this is where user input is needed
                interrupt_data = event["__interrupt__"]
                if interrupt_data and len(interrupt_data) > 0:
                    interrupt_value = interrupt_data[0].value
                    print(f"\n{interrupt_value}")
                    print("-" * 50)

                    # Get user input
                    user_input = input("Your response: ")

                    # Resume with user input
                    for resume_event in app.stream(
                        Command(resume=user_input), config, stream_mode="updates"
                    ):
                        if "__interrupt__" in resume_event:
                            # Another interrupt
                            interrupt_data = resume_event["__interrupt__"]
                            if interrupt_data and len(interrupt_data) > 0:
                                interrupt_value = interrupt_data[0].value
                                print(f"\n{interrupt_value}")
                                print("-" * 50)
            else:
                # Regular update
                for node, data in event.items():
                    if node == "research" and "final_report" in data:
                        print("\n" + "=" * 50)
                        print("RESEARCH COMPLETE")
                        print("=" * 50)
                        print(data["final_report"])

    except KeyboardInterrupt:
        print("\n\nResearch agent terminated by user.")
    except Exception as e:
        print(f"\nAn error occurred: {e}")

