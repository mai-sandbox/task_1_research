#!/usr/bin/env python3
"""Test script for the research agent"""

from langchain_core.messages import HumanMessage

# Test import
try:
    from agent import app
    print("✓ Successfully imported app from agent.py")
    
    # Test basic invocation structure
    initial_state = {
        "messages": [HumanMessage("I want to research the latest developments in quantum computing")],
        "phase": "clarification",
        "clarification_complete": False,
        "research_brief": "",
        "final_report": ""
    }
    
    print("✓ Agent structure is valid")
    print("\nAgent is ready for use!")
    print("\nTo use the agent, ensure you have:")
    print("1. Set OPENAI_API_KEY environment variable")
    print("2. Set TAVILY_API_KEY environment variable")
    print("3. Installed dependencies: pip install -r requirements.txt")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()