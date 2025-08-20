"""
Test script for the deep research agent
"""

from agent import app
from langchain_core.messages import HumanMessage
import uuid

def test_research_agent():
    # Create a thread ID for conversation continuity
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    
    print("=== Deep Research Agent Test ===\n")
    print("Starting clarification phase...")
    print("-" * 50)
    
    # Initial query
    initial_state = {
        "messages": [HumanMessage("I want to research the latest developments in quantum computing")],
        "phase": "clarification",
        "clarification_complete": False
    }
    
    # First interaction
    result = app.invoke(initial_state, config)
    print("\nAgent:", result["messages"][-1].content)
    print("-" * 50)
    
    # Simulate user responses for clarification
    clarification_responses = [
        "I'm particularly interested in quantum error correction and recent breakthroughs from major tech companies",
        "Focus on developments from 2023-2024, especially practical applications and commercial viability",
        "Yes, that covers everything. Please proceed with the research."
    ]
    
    for user_response in clarification_responses:
        print(f"\nUser: {user_response}")
        print("-" * 50)
        
        # Continue conversation
        result = app.invoke(
            {"messages": [HumanMessage(user_response)]},
            config
        )
        
        print("\nAgent:", result["messages"][-1].content)
        print("-" * 50)
        
        # Check if we've moved to research phase
        if result.get("phase") == "research":
            print("\n*** Moving to Research Phase ***\n")
            break
        
        if result.get("phase") == "complete":
            print("\n*** Research Complete ***\n")
            print("Final Report:")
            print(result.get("final_report", "No report generated"))
            break

if __name__ == "__main__":
    test_research_agent()