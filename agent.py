"""
LangGraph Deep Research Agent with User Interaction and ReAct Search

This agent operates in two phases:
1. Clarification Phase: Interactive conversation with user to clarify research scope
2. Research Phase: ReAct agent with Tavily search to generate detailed reports
"""

import os
from typing import Annotated, TypedDict, Literal
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_anthropic import ChatAnthropic
from langchain_tavily import TavilySearch
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent, ToolNode, tools_condition
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt, Command


# State definition for the entire workflow
class ResearchState(TypedDict):
    """State that flows through both clarification and research phases"""
    messages: Annotated[list[BaseMessage], add_messages]
    research_brief: str
    clarification_complete: bool
    final_report: str


def clarification_node(state: ResearchState) -> dict:
    """
    Phase 1: Interactive clarification with the user
    Uses interrupt() to pause and gather user input multiple times
    """
    messages = state.get("messages", [])
    
    # Initialize the clarification conversation
    if not any(msg.content.startswith("Let me help you clarify") for msg in messages if isinstance(msg, AIMessage)):
        initial_prompt = """Let me help you clarify your research needs. I'll ask you a few questions to better understand:
1. What specific topic or question would you like me to research?
2. What aspects are most important to you?
3. Are there any specific sources or types of information you prefer?

Please share your initial research request."""
        
        # First interrupt to get initial research topic
        user_response = interrupt({
            "question": initial_prompt,
            "phase": "initial_request"
        })
        
        # Add the interaction to messages
        messages.append(AIMessage(content=initial_prompt))
        messages.append(HumanMessage(content=user_response))
        
        # Ask follow-up questions based on initial response
        followup_prompt = f"""Thank you for sharing that. Based on your request about: "{user_response}"

Let me ask a few clarifying questions:
1. What is the intended use or goal of this research?
2. How detailed should the research be?
3. Are there any specific angles or perspectives you want me to focus on?"""
        
        # Second interrupt for clarification
        clarification_response = interrupt({
            "question": followup_prompt,
            "phase": "clarification"
        })
        
        messages.append(AIMessage(content=followup_prompt))
        messages.append(HumanMessage(content=clarification_response))
        
        # Ask for confirmation
        research_brief = f"""Based on our conversation, here's what I understand you need:

**Research Topic**: {user_response}

**Additional Context**: {clarification_response}

I'll conduct a comprehensive search and analysis on this topic, focusing on:
- Current and relevant information
- Multiple perspectives and sources
- Practical insights and applications

Is this correct, or would you like to add/modify anything? (Type 'yes' to proceed or provide additional details)"""
        
        # Final interrupt for confirmation
        confirmation = interrupt({
            "question": research_brief,
            "phase": "confirmation"
        })
        
        messages.append(AIMessage(content=research_brief))
        messages.append(HumanMessage(content=confirmation))
        
        # Check if user confirmed or wants more clarification
        if confirmation.lower() in ['yes', 'y', 'correct', 'proceed', 'go ahead']:
            # Create the final research brief
            final_brief = f"""Research Brief:
Topic: {user_response}
Context and Requirements: {clarification_response}
"""
            messages.append(AIMessage(content="Great! I'll now proceed with the research based on our discussion."))
            
            return {
                "messages": messages,
                "research_brief": final_brief,
                "clarification_complete": True
            }
        else:
            # User wants to modify - ask what to change
            modification_prompt = "I understand you'd like to modify the research brief. Please tell me what you'd like to change or add:"
            
            modification = interrupt({
                "question": modification_prompt,
                "phase": "modification"
            })
            
            messages.append(AIMessage(content=modification_prompt))
            messages.append(HumanMessage(content=modification))
            
            # Update the research brief
            final_brief = f"""Research Brief:
Topic: {user_response}
Context and Requirements: {clarification_response}
Additional modifications: {modification}
"""
            messages.append(AIMessage(content="Perfect! I've updated the research brief. Now proceeding with the research."))
            
            return {
                "messages": messages,
                "research_brief": final_brief,
                "clarification_complete": True
            }
    
    # This shouldn't be reached in normal flow
    return {
        "clarification_complete": True
    }


def research_node(state: ResearchState) -> dict:
    """
    Phase 2: ReAct agent for research using Tavily search
    """
    research_brief = state.get("research_brief", "")
    messages = state.get("messages", [])
    
    # Initialize the research agent with Tavily search tool
    tavily_tool = TavilySearch(max_results=3)
    
    # Use Anthropic model (you can change to OpenAI or other providers)
    model = ChatAnthropic(model="claude-3-5-sonnet-20241022", temperature=0.7)
    
    # Create the ReAct agent with the search tool
    research_agent = create_react_agent(
        model=model,
        tools=[tavily_tool],
        prompt=f"""You are a deep research assistant. Your task is to conduct thorough research based on the following brief and generate a comprehensive report.

{research_brief}

Guidelines for your research:
1. Search for current and relevant information
2. Gather data from multiple sources
3. Synthesize findings into a coherent report
4. Include specific examples and evidence
5. Provide actionable insights when applicable

Use the search tool to gather information, then create a detailed research report."""
    )
    
    # Prepare the research query message
    research_query = f"Please conduct research based on this brief and create a detailed report:\n\n{research_brief}"
    
    # Run the research agent
    research_result = research_agent.invoke({
        "messages": [HumanMessage(content=research_query)]
    })
    
    # Extract the final report from the agent's response
    final_messages = research_result.get("messages", [])
    if final_messages:
        # Get the last AI message which should contain the research report
        for msg in reversed(final_messages):
            if isinstance(msg, AIMessage) and not msg.tool_calls:
                final_report = msg.content
                break
        else:
            final_report = "Research completed but no final report was generated."
    else:
        final_report = "No research results were generated."
    
    # Add the research results to the conversation
    messages.append(AIMessage(content=f"Research Complete. Here's my detailed report:\n\n{final_report}"))
    
    return {
        "messages": messages,
        "final_report": final_report
    }


def should_continue_to_research(state: ResearchState) -> Literal["research", "end"]:
    """Determine if clarification is complete and we should proceed to research"""
    if state.get("clarification_complete", False):
        return "research"
    return "end"


def should_end(state: ResearchState) -> Literal["end"]:
    """Always end after research phase"""
    return "end"


# Build the main graph
def build_graph():
    """Construct the two-phase research agent graph"""
    workflow = StateGraph(ResearchState)
    
    # Add nodes
    workflow.add_node("clarification", clarification_node)
    workflow.add_node("research", research_node)
    
    # Add edges
    workflow.add_edge(START, "clarification")
    workflow.add_conditional_edges(
        "clarification",
        should_continue_to_research,
        {
            "research": "research",
            "end": END
        }
    )
    workflow.add_conditional_edges(
        "research",
        should_end,
        {
            "end": END
        }
    )
    
    # Compile with checkpointer (required for interrupt)
    checkpointer = InMemorySaver()
    return workflow.compile(checkpointer=checkpointer)


# Export the compiled graph as 'app'
app = build_graph()


# Optional: Add a helper function for easy invocation
def run_research_agent(initial_message: str = "I need help with research"):
    """
    Helper function to run the research agent
    
    Args:
        initial_message: The initial message to start the conversation
    
    Returns:
        The final state including messages and research report
    """
    config = {"configurable": {"thread_id": "research-session-1"}}
    
    # Start with an initial message
    initial_state = {
        "messages": [HumanMessage(content=initial_message)],
        "clarification_complete": False,
        "research_brief": "",
        "final_report": ""
    }
    
    # Run the graph
    result = app.invoke(initial_state, config)
    
    return result


if __name__ == "__main__":
    # Example usage
    print("Starting LangGraph Deep Research Agent...")
    print("=" * 50)
    
    # Note: This will require interactive input when run
    # The agent will pause at interrupt points to gather user input
    
    # You can test the agent by running:
    # python agent.py
    
    # Or import and use in another script:
    # from agent import app
    # result = app.invoke({"messages": [HumanMessage("I need research help")]}, config)
