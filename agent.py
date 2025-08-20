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
from langgraph.types import interrupt
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


def interactive_scoping_node(state: ResearchState) -> ResearchState:
    """
    Interactive scoping node that engages in back-and-forth conversation 
    with the user to clarify research scope and requirements.
    
    Uses interrupt() to pause execution for human input and asks clarifying
    questions about the research topic. Updates the research_brief field
    with gathered information and sets user_confirmed=True when ready.
    
    Args:
        state: Current research state
        
    Returns:
        Updated state with research_brief and user_confirmed status
    """
    # Get the latest user message
    if not state["messages"]:
        return state
    
    latest_message = state["messages"][-1]
    
    # Check if this is the initial scoping or continuation
    if not state["research_brief"]:
        # Initial scoping - analyze the user's request and ask clarifying questions
        user_content = latest_message.content if hasattr(latest_message, 'content') else str(latest_message)
        
        # Generate clarifying questions based on the initial request
        clarifying_questions = generate_clarifying_questions(user_content)
        
        # Create response asking for more details
        scoping_response = AIMessage(
            content=f"I understand you want to research: {user_content}\n\n"
                   f"To provide you with the most comprehensive and relevant research, "
                   f"I'd like to clarify a few things:\n\n{clarifying_questions}\n\n"
                   f"Please provide more details about these aspects, or if you're satisfied "
                   f"with the current scope, simply type 'proceed' or 'ready' to start the research."
        )
        
        # Add the response to messages
        state["messages"].append(scoping_response)
        
        # Update phase to indicate we're in active scoping
        state["phase"] = "scoping"
        
        # Use interrupt to pause for human input
        interrupt("Please provide more details about your research requirements, or type 'proceed' if ready to start research.")
        
    else:
        # Continuation of scoping - process user's additional input
        user_response = latest_message.content if hasattr(latest_message, 'content') else str(latest_message)
        
        # Check if user wants to proceed with research
        proceed_keywords = ['proceed', 'ready', 'start research', 'go ahead', 'continue', 'yes', 'confirmed']
        if any(keyword in user_response.lower() for keyword in proceed_keywords):
            # User confirmed - finalize research brief and set confirmed flag
            state["user_confirmed"] = True
            state["phase"] = "research"
            
            # Create final confirmation message
            confirmation_msg = AIMessage(
                content="Perfect! I have all the information I need. Let me now conduct comprehensive research "
                       f"on your topic. Here's what I'll be researching:\n\n{state['research_brief']}\n\n"
                       "Starting research now..."
            )
            state["messages"].append(confirmation_msg)
            
        else:
            # User provided more details - update research brief
            if state["research_brief"]:
                state["research_brief"] += f"\n\nAdditional details: {user_response}"
            else:
                state["research_brief"] = user_response
            
            # Ask if they want to add more details or proceed
            follow_up_msg = AIMessage(
                content="Thank you for the additional details! I've updated the research scope:\n\n"
                       f"**Current Research Brief:**\n{state['research_brief']}\n\n"
                       "Would you like to add any more specific requirements, or shall I proceed with "
                       "the research? (Type 'proceed' when ready, or provide more details)"
            )
            state["messages"].append(follow_up_msg)
            
            # Interrupt again for potential additional input
            interrupt("Would you like to add more details or proceed with research?")
    
    return state


def generate_clarifying_questions(user_request: str) -> str:
    """
    Generate relevant clarifying questions based on the user's initial request.
    
    Args:
        user_request: The user's initial research request
        
    Returns:
        Formatted string with clarifying questions
    """
    # Basic clarifying questions that apply to most research topics
    questions = [
        "1. What specific aspects or subtopics are you most interested in?",
        "2. What is the intended use or purpose of this research?",
        "3. Are you looking for recent developments, historical context, or both?",
        "4. Do you need information from specific sources or domains?",
        "5. What level of detail do you need (overview, in-depth analysis, technical details)?"
    ]
    
    # Add topic-specific questions based on keywords
    user_lower = user_request.lower()
    
    if any(tech_word in user_lower for tech_word in ['ai', 'artificial intelligence', 'machine learning', 'technology']):
        questions.append("6. Are you interested in specific AI applications, techniques, or industry impacts?")
    
    if any(business_word in user_lower for business_word in ['business', 'market', 'industry', 'company']):
        questions.append("6. Are you looking for market analysis, competitive landscape, or business strategies?")
    
    if any(health_word in user_lower for health_word in ['health', 'medical', 'healthcare', 'medicine']):
        questions.append("6. Do you need clinical research, treatment options, or general health information?")
    
    return "\n".join(questions)


# Create the StateGraph with our custom ResearchState schema
graph_builder = StateGraph(ResearchState)

# Add the state initialization node
graph_builder.add_node("initialize_state", initialize_agent_state)

# Add the interactive scoping node
graph_builder.add_node("scoping_node", interactive_scoping_node)

# Connect START to the initialization node
graph_builder.add_edge(START, "initialize_state")

# Connect initialization to scoping node
graph_builder.add_edge("initialize_state", "scoping_node")

# Placeholder connections for future nodes
# These will be implemented in subsequent tasks:
# - routing_node: Conditional logic to determine workflow progression  
# - research_node: ReAct agent with Tavily search capabilities

# Temporary end connection from scoping node (will be updated when routing is added)
graph_builder.add_edge("scoping_node", END)

# Compile the graph and export as 'app' for deployment
app = graph_builder.compile()


# Test functions to verify the implementation works correctly
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
    try:
        result = app.invoke(minimal_state)
        print("✅ State schema test passed - agent accepts minimal input correctly")
        return result
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return None


def test_scoping_interaction():
    """
    Test the interactive scoping functionality with simulated user interaction.
    """
    from langchain_core.messages import HumanMessage
    
    print("\n🧪 Testing interactive scoping node...")
    
    # Test initial scoping
    initial_state = {
        "messages": [HumanMessage("I want to research machine learning applications in healthcare")]
    }
    
    try:
        # This should trigger the scoping node and generate clarifying questions
        print("Testing initial scoping interaction...")
        
        # Since we can't actually interrupt in a test, let's test the logic
        test_state = {
            "messages": [HumanMessage("I want to research machine learning applications in healthcare")],
            "research_brief": "",
            "phase": "scoping", 
            "user_confirmed": False
        }
        
        # Test the scoping node function directly
        updated_state = interactive_scoping_node(test_state)
        
        # Verify the scoping node added clarifying questions
        assert len(updated_state["messages"]) > 1
        assert updated_state["phase"] == "scoping"
        assert not updated_state["user_confirmed"]
        
        print("✅ Initial scoping test passed")
        
        # Test user confirmation
        confirmation_state = {
            "messages": [
                HumanMessage("I want to research machine learning applications in healthcare"),
                HumanMessage("proceed")
            ],
            "research_brief": "Machine learning applications in healthcare - focus on diagnostic tools and patient outcomes",
            "phase": "scoping",
            "user_confirmed": False
        }
        
        confirmed_state = interactive_scoping_node(confirmation_state)
        
        # Verify user confirmation was processed
        assert confirmed_state["user_confirmed"] == True
        assert confirmed_state["phase"] == "research"
        
        print("✅ User confirmation test passed")
        print("✅ Interactive scoping node implementation is working correctly")
        
        return True
        
    except Exception as e:
        print(f"❌ Scoping test failed with error: {e}")
        return False


if __name__ == "__main__":
    # Run tests when script is executed directly
    print("🚀 Testing LangGraph Deep Research Agent Implementation")
    print("=" * 60)
    
    # Test 1: Basic state schema and minimal input
    test_result = test_minimal_input()
    
    # Test 2: Interactive scoping functionality  
    scoping_test_passed = test_scoping_interaction()
    
    if test_result and scoping_test_passed:
        print("\n🎉 All tests passed! Interactive scoping node is ready.")
    else:
        print("\n⚠️  Some tests failed. Please check the implementation.")





