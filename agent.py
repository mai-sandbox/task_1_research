"""
LangGraph Deep Research Agent with Interactive Scoping and ReAct Search

This agent implements a two-phase workflow:
1. Interactive scoping phase: Engages with user to clarify research scope
2. ReAct research phase: Uses Tavily search to conduct comprehensive research

The agent accepts minimal state with just a 'messages' list containing a HumanMessage
and automatically initializes other required state fields.
"""

from typing import Annotated, Optional
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


def initialize_agent_state(state: ResearchState) -> ResearchState:
    """
    Initialize the agent state with default values for required fields.
    
    This ensures the agent can accept minimal input state containing only 'messages'
    and automatically set up the workflow state fields.
    
    Args:
        state: Input state (may contain only 'messages')
        
    Returns:
        Complete state with all required fields initialized
    """
    # Initialize research_brief if not provided
    if "research_brief" not in state:
        state["research_brief"] = ""
    
    # Initialize phase to 'scoping' if not provided
    if "phase" not in state:
        state["phase"] = "scoping"
    
    # Initialize user_confirmed to False if not provided
    if "user_confirmed" not in state:
        state["user_confirmed"] = False
    
    # Ensure messages exist (should be provided in minimal input)
    if "messages" not in state:
        state["messages"] = []
    
    # Add initial system message if this is the first interaction
    if len(state["messages"]) == 1 and isinstance(state["messages"][0], HumanMessage):
        # Add a welcome message to start the scoping conversation
        welcome_msg = AIMessage(
            content="Hello! I'm your research assistant. I'll help you conduct comprehensive research on any topic. "
                   "First, let me understand exactly what you'd like me to research. "
                   "Could you please provide more details about your research topic and what specific aspects you're most interested in?"
        )
        state["messages"].append(welcome_msg)
    
    return state


# Create the StateGraph with our custom ResearchState schema
graph_builder = StateGraph(ResearchState)

# Add the state initialization node
graph_builder.add_node("initialize_state", initialize_agent_state)

# Connect START to the initialization node
graph_builder.add_edge(START, "initialize_state")

# Placeholder connections for future nodes
# These will be implemented in subsequent tasks:
# - scoping_node: Interactive conversation for research scope clarification
# - routing_node: Conditional logic to determine workflow progression  
# - research_node: ReAct agent with Tavily search capabilities

# Temporary end connection (will be updated when workflow nodes are added)
graph_builder.add_edge("initialize_state", END)

# Compile the graph and export as 'app' for deployment
app = graph_builder.compile()


# Test function to verify the state schema works correctly
def test_minimal_input():
    """
    Test that the agent accepts minimal input state as specified in requirements.
    """
    from langchain_core.messages import HumanMessage
    
    # Test with minimal state containing only messages
    minimal_state = {
        "messages": [HumanMessage("I want to research artificial intelligence")]
    }
    
    # This should work without errors and initialize all required fields
    result = app.invoke(minimal_state)
    
    # Verify all required state fields are present
    assert "messages" in result
    assert "research_brief" in result  
    assert "phase" in result
    assert "user_confirmed" in result
    
    # Verify default values
    assert result["research_brief"] == ""
    assert result["phase"] == "scoping"
    assert result["user_confirmed"] == False
    assert len(result["messages"]) >= 2  # Original message + welcome message
    
    print("✅ State schema test passed - agent accepts minimal input correctly")
    return result


if __name__ == "__main__":
    # Run test when script is executed directly
    test_result = test_minimal_input()
    print(f"Test result: {test_result}")

