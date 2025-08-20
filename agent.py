"""
LangGraph Deep Research Agent with Interactive Scoping and ReAct Search

This agent implements a two-phase workflow:
1. Interactive scoping phase: Engages with user to clarify research scope
2. ReAct research phase: Uses Tavily search to conduct comprehensive research
"""

from typing import Annotated
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage


class ResearchState(TypedDict):
    """
    State schema for the deep research agent workflow.
    
    Fields:
    - messages: Conversation history with automatic message appending
    - research_brief: Clarified research scope and requirements
    - phase: Current workflow phase ('scoping' or 'research')
    - user_confirmed: Whether user has confirmed to proceed with research
    """
    messages: Annotated[list, add_messages]
    research_brief: str
    phase: str
    user_confirmed: bool


def initialize_state(state: ResearchState) -> ResearchState:
    """
    Initialize the state with default values if not provided.
    """
    # Set default values for state fields if they don't exist
    if "research_brief" not in state or state["research_brief"] is None:
        state["research_brief"] = ""
    if "phase" not in state or state["phase"] is None:
        state["phase"] = "scoping"
    if "user_confirmed" not in state or state["user_confirmed"] is None:
        state["user_confirmed"] = False
    
    return state


# Create the StateGraph with our custom state schema
graph_builder = StateGraph(ResearchState)

# Add initialization node to set default state values
graph_builder.add_node("initialize", initialize_state)

# Add edge from START to initialization
graph_builder.add_edge(START, "initialize")

# Placeholder for additional nodes (will be implemented in subsequent tasks)
# - scoping_node: Interactive conversation for research scope clarification
# - routing_node: Conditional logic to determine next step
# - research_node: ReAct agent with Tavily search

# Temporary end connection for the initialization node
# This will be updated when we add the actual workflow nodes
graph_builder.add_edge("initialize", END)

# Compile the graph (will be updated as we add more nodes)
app = graph_builder.compile()
