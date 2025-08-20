#!/usr/bin/env python3
"""
Test script for the LangGraph Deep Research Agent

This script demonstrates how to use the agent programmatically.
"""

import os
from langchain_core.messages import HumanMessage
from agent import app

def test_agent():
    """Test the research agent with a sample query."""
    
    # Check for required environment variables
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("Error: Please set ANTHROPIC_API_KEY environment variable")
        return
    
    if not os.getenv("TAVILY_API_KEY"):
        print("Error: Please set TAVILY_API_KEY environment variable") 
        return
    
    print("🔬 LangGraph Deep Research Agent Test")
    print("=" * 50)
    
    # Test with a simple research topic
    initial_state = {
        "messages": [HumanMessage("I want to research the latest developments in quantum computing")],
        "stage": "clarification",
        "clarification_complete": False,
        "research_complete": False,
        "research_brief": ""
    }
    
    print("Starting research agent...")
    
    try:
        # Run the agent
        result = app.invoke(initial_state)
        
        print("\n📋 Final Result:")
        print("-" * 30)
        
        # Print all messages from the conversation
        for i, message in enumerate(result.get("messages", [])):
            if hasattr(message, 'content'):
                print(f"\nMessage {i+1}:")
                print(f"Type: {type(message).__name__}")
                print(f"Content: {message.content[:500]}{'...' if len(message.content) > 500 else ''}")
        
        print(f"\n📊 Final State:")
        print(f"Stage: {result.get('stage', 'unknown')}")
        print(f"Clarification Complete: {result.get('clarification_complete', False)}")
        print(f"Research Complete: {result.get('research_complete', False)}")
        
        if result.get("research_brief"):
            print(f"\n📝 Research Brief:")
            print(result["research_brief"][:300] + "..." if len(result["research_brief"]) > 300 else result["research_brief"])
            
    except Exception as e:
        print(f"Error running agent: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_agent()