"""Test script for the deep research agent"""

from agent import app
from langchain_core.messages import HumanMessage
import asyncio


def test_research_agent():
    """Test the research agent with a sample query"""
    
    # Create initial state with a research request
    initial_state = {
        "messages": [HumanMessage("I want to research the latest developments in quantum computing")],
        "clarification_complete": False,
        "research_brief": None,
        "final_report": None
    }
    
    # Configure with a thread ID for memory
    config = {"configurable": {"thread_id": "test_thread_1"}}
    
    print("Starting Research Agent Test...")
    print("=" * 50)
    
    # Run the agent
    try:
        # First interaction - should ask clarifying questions
        result = app.invoke(initial_state, config)
        print("\n--- Initial Response ---")
        print(result["messages"][-1].content)
        
        # Simulate user providing more details
        follow_up = {
            "messages": [HumanMessage("I'm particularly interested in recent breakthroughs in quantum error correction and quantum supremacy claims from major tech companies. This is for a technical presentation to engineers.")],
        }
        
        print("\n--- User Clarification ---")
        print(follow_up["messages"][0].content)
        
        # Continue conversation
        result = app.invoke(follow_up, config)
        print("\n--- Agent Response ---")
        print(result["messages"][-1].content)
        
        # Check if we need another round or if research has begun
        if not result.get("clarification_complete", False):
            # Provide final clarification
            final_clarification = {
                "messages": [HumanMessage("Yes, please focus on developments from 2023-2024, especially from Google, IBM, and startups.")],
            }
            print("\n--- Final User Input ---")
            print(final_clarification["messages"][0].content)
            
            result = app.invoke(final_clarification, config)
            print("\n--- Final Response ---")
            print(result["messages"][-1].content)
        
        if result.get("final_report"):
            print("\n" + "=" * 50)
            print("FINAL RESEARCH REPORT")
            print("=" * 50)
            print(result["final_report"])
            
    except Exception as e:
        print(f"Error during test: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("Note: Make sure you have set OPENAI_API_KEY and TAVILY_API_KEY environment variables")
    test_research_agent()