#!/usr/bin/env python3
"""
Test script for the research agent.
Run this to test the agent without terminal interaction.
"""

from agent import app
from langchain_core.messages import HumanMessage

def test_programmatic():
    """Test the agent programmatically without terminal interaction"""
    print("Testing Research Agent (Programmatic Mode)")
    print("="*50)
    
    initial_state = {
        "messages": [HumanMessage("I want to research the latest developments in quantum computing")],
        "research_brief": "",
        "clarification_complete": False,
        "final_report": ""
    }
    
    try:
        result = app.invoke(initial_state)
        print("Initial clarification response:")
        print(result["messages"][-1].content)
        print("\nNote: In production, this would continue with human interaction.")
    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure you have set up your environment variables:")
        print("- OPENAI_API_KEY")
        print("- TAVILY_API_KEY")

def test_direct_research():
    """Test direct research mode (skipping clarification)"""
    print("\nTesting Direct Research Mode")
    print("="*50)
    
    from agent import research_agent
    
    state = {
        "messages": [],
        "research_brief": "Research the latest breakthroughs in quantum computing in 2024, focusing on practical applications and major company developments",
        "clarification_complete": True,
        "final_report": ""
    }
    
    try:
        result = research_agent.research(state)
        print("Research Report Generated:")
        print(result["final_report"])
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--direct":
        test_direct_research()
    else:
        test_programmatic()
        print("\nTo test direct research mode, run: python test_agent.py --direct")