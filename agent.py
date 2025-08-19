"""
LangGraph Deep Research Agent with User Interaction and ReAct Search

This agent implements a two-phase workflow:
1. User Interaction Phase: Clarifies research scope through terminal interaction
2. Research Phase: Executes research using ReAct agent with Tavily search
"""

import os
from typing import Annotated, Literal
from typing_extensions import TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
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
    messages: Annotated[list[BaseMessage], add_messages]
    research_brief: str
    research_complete: bool
    final_report: str
    interaction_count: int


def user_interaction_node(state: ResearchState) -> dict:
    """
    Phase 1: Interactive node for clarifying research scope with the user.
    Uses interrupt() to pause execution and gather user input.
    """
    messages = state.get("messages", [])
    research_brief = state.get("research_brief", "")
    interaction_count = state.get("interaction_count", 0)
    
    # If we already have a research brief, skip interaction
    if research_brief:
        return {}
    
    # Track interaction stages
    if interaction_count == 0:
        # Initial greeting and get topic
        greeting = """Welcome to the Deep Research Agent! 

I'll help you conduct thorough research on any topic. First, let me understand what you'd like to research.

Please describe your research topic and any specific aspects you'd like me to focus on:"""
        
        user_input = interrupt(greeting)
        
        return {
            "messages": [HumanMessage(content=f"Research topic: {user_input}")],
            "interaction_count": 1
        }
    
    elif interaction_count == 1:
        # Get clarification
        # Extract the topic from previous message
        topic = ""
        for msg in messages:
            if isinstance(msg, HumanMessage) and "Research topic:" in msg.content:
                topic = msg.content.replace("Research topic: ", "")
                break
        
        clarification_prompt = f"""Thank you for sharing your research interest: "{topic}"

To ensure I conduct the most relevant research, let me ask a few clarifying questions:

1. What is the main goal or purpose of this research?
2. Are there any specific sources, time periods, or perspectives you'd like me to focus on?
3. What level of detail are you looking for (brief overview vs. comprehensive analysis)?

Please provide any additional context that would help me research this effectively:"""
        
        clarification = interrupt(clarification_prompt)
        
        return {
            "messages": [HumanMessage(content=f"Clarification: {clarification}")],
            "interaction_count": 2
        }
    
    elif interaction_count == 2:
        # Confirm and create research brief
        topic = ""
        clarification = ""
        
        for msg in messages:
            if isinstance(msg, HumanMessage):
                if "Research topic:" in msg.content:
                    topic = msg.content.replace("Research topic: ", "")
                elif "Clarification:" in msg.content:
                    clarification = msg.content.replace("Clarification: ", "")
        
        confirmation_prompt = f"""Based on our discussion, here's my understanding of your research needs:

**Original Topic:** {topic}
**Additional Context:** {clarification}

I'll now create a research brief and proceed with gathering information. 

Would you like to:
1. Proceed with the research as described
2. Add more specific requirements
3. Modify the scope

Please enter 1, 2, or 3 (or just press Enter to proceed):"""
        
        confirmation = interrupt(confirmation_prompt)
        
        if confirmation and confirmation.strip() in ["2", "3"]:
            return {
                "interaction_count": 3
            }
        
        # Create the research brief
        research_brief = f"""RESEARCH BRIEF:
===============
Topic: {topic}

Context and Requirements:
{clarification}

Research Objectives:
- Gather comprehensive and accurate information
- Provide multiple perspectives where relevant
- Include recent and authoritative sources
- Structure findings in a clear, logical manner"""
        
        return {
            "research_brief": research_brief,
            "interaction_count": 4
        }
    
    elif interaction_count == 3:
        # Get additional requirements
        additional_prompt = "Please provide your additional requirements or modifications:"
        additional_input = interrupt(additional_prompt)
        
        # Update the research brief with additional requirements
        topic = ""
        clarification = ""
        
        for msg in messages:
            if isinstance(msg, HumanMessage):
                if "Research topic:" in msg.content:
                    topic = msg.content.replace("Research topic: ", "")
                elif "Clarification:" in msg.content:
                    clarification = msg.content.replace("Clarification: ", "")
        
        clarification = f"{clarification}\n\nAdditional requirements: {additional_input}"
        
        research_brief = f"""RESEARCH BRIEF:
===============
Topic: {topic}

Context and Requirements:
{clarification}

Research Objectives:
- Gather comprehensive and accurate information
- Provide multiple perspectives where relevant
- Include recent and authoritative sources
- Structure findings in a clear, logical manner"""
        
        return {
            "messages": [HumanMessage(content=f"Additional: {additional_input}")],
            "research_brief": research_brief,
            "interaction_count": 4
        }
    
    elif interaction_count == 4:
        # Final confirmation before research
        final_confirmation = f"""Research Brief Created!

I'm ready to begin the research. This will involve:
1. Searching for relevant information using web search
2. Analyzing and synthesizing the findings
3. Creating a detailed report

Press Enter to start the research, or type 'cancel' to stop:"""
        
        start_confirmation = interrupt(final_confirmation)
        
        if start_confirmation and start_confirmation.strip().lower() == 'cancel':
            return {
                "research_complete": True,
                "final_report": "Research cancelled by user."
            }
        
        return {
            "interaction_count": 5
        }
    
    return {}


def research_execution_node(state: ResearchState) -> dict:
    """
    Phase 2: Execute research using ReAct agent with Tavily search tool.
    """
    research_brief = state.get("research_brief", "")
    
    if not research_brief:
        return {
            "research_complete": True,
            "final_report": "No research brief provided."
        }
    
    # Initialize the LLM (prefer OpenAI, fallback to Anthropic)
    try:
        llm = ChatOpenAI(model="gpt-4o", temperature=0.7)
    except:
        try:
            llm = ChatAnthropic(model="claude-3-5-sonnet-20241022", temperature=0.7)
        except:
            # Fallback to a basic OpenAI model
            llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
    
    # Initialize Tavily search tool
    tavily_tool = TavilySearch(max_results=5)
    tools = [tavily_tool]
    
    # Create the ReAct agent with search capabilities
    research_prompt = f"""You are a thorough research assistant with access to web search capabilities.

{research_brief}

Your task is to:
1. Search for relevant and authoritative information
2. Gather data from multiple sources
3. Synthesize findings into a comprehensive report
4. Ensure accuracy and cite sources where possible

Use the search tool multiple times with different queries to gather comprehensive information.
Focus on finding recent, relevant, and authoritative sources.

After gathering sufficient information, create a detailed research report that addresses all aspects of the research brief.
"""
    
    # Create the ReAct agent
    react_agent = create_react_agent(
        model=llm,
        tools=tools,
        prompt=research_prompt,
        version="v2"
    )
    
    # Execute the research
    research_messages = [
        SystemMessage(content=research_prompt),
        HumanMessage(content="Please conduct the research as described in the brief and create a comprehensive report.")
    ]
    
    try:
        # Run the ReAct agent
        result = react_agent.invoke({"messages": research_messages})
        
        # Extract the final report from the agent's response
        agent_messages = result.get("messages", [])
        
        # Find the last AI message with the research results
        final_report = "# Research Report\n\n"
        for msg in reversed(agent_messages):
            if isinstance(msg, AIMessage) and not msg.tool_calls:
                final_report += msg.content
                break
        
        if final_report == "# Research Report\n\n":
            # Fallback if no proper response found
            final_report += "Research completed but no detailed findings were generated. Please try again with a more specific query."
        
        return {
            "messages": state["messages"] + agent_messages,
            "research_complete": True,
            "final_report": final_report
        }
        
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
        return {
            "research_complete": True,
            "final_report": error_report
        }


def should_continue_research(state: ResearchState) -> Literal["research", "interaction", "output"]:
    """
    Conditional edge to determine next step after interaction.
    """
    if state.get("research_complete", False):
        return "output"
    
    if state.get("research_brief", "") and state.get("interaction_count", 0) >= 4:
        return "research"
    
    return "interaction"


def format_output_node(state: ResearchState) -> dict:
    """
    Final node to format and present the research results.
    """
    final_report = state.get("final_report", "No report generated.")
    
    # Don't print here as it interferes with the interrupt flow
    # The report will be in the state for the caller to access
    
    return {"final_report": final_report}


# Build the graph
def build_research_graph():
    """
    Builds the complete research agent graph with two phases.
    """
    # Initialize the graph
    graph_builder = StateGraph(ResearchState)
    
    # Add nodes
    graph_builder.add_node("interaction", user_interaction_node)
    graph_builder.add_node("research", research_execution_node)
    graph_builder.add_node("output", format_output_node)
    
    # Add edges
    graph_builder.add_edge(START, "interaction")
    
    # Conditional routing after interaction
    graph_builder.add_conditional_edges(
        "interaction",
        should_continue_research,
        {
            "research": "research",
            "end": "output",
            "interaction": "interaction"  # Loop back for more interaction if needed
        }
    )
    
    # After research, go to output
    graph_builder.add_edge("research", "output")
    graph_builder.add_edge("output", END)
    
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
    print("\n" + "="*80)
    print("DEEP RESEARCH AGENT")
    print("="*80)
    
    # Initialize state
    initial_state = {
        "messages": [HumanMessage(content="Start research session")],
        "research_brief": "",
        "research_complete": False,
        "final_report": ""
    }
    
    # Configuration with thread ID for checkpointing
    config = {"configurable": {"thread_id": "research_session_001"}}
    
    try:
        # Start the graph execution
        for event in app.stream(initial_state, config=config, stream_mode="updates"):
            # The graph will pause at interrupt points
            # User input is handled through the interrupt() function
            pass
            
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




