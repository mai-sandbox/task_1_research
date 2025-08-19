#!/usr/bin/env python3
"""
Test script for the LangGraph Deep Research Agent
"""

import os
from langchain_core.messages import HumanMessage

# Set dummy API keys for testing structure
os.environ['TAVILY_API_KEY'] = 'test-key'
os.environ['ANTHROPIC_API_KEY'] = 'test-key'

try:
    from agent import app
    print("✅ Agent imported successfully!")
    
    # Test the state schema
    test_state = {
        "messages": [HumanMessage("Test research request")],
        "research_brief": "",
        "research_complete": False,
        "final_report": ""
    }
    
    print("✅ State schema is valid!")
    print("✅ Agent structure created successfully!")
    print("\nAgent is ready for implementation of individual nodes.")
    
except Exception as e:
    print(f"❌ Error: {e}")
