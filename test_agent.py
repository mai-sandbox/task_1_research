#!/usr/bin/env python3
"""
Test script to verify the agent implementation
"""

try:
    from agent import app
    print("✓ Agent imported successfully")
    print("✓ App variable is available:", type(app))
    
    # Test basic structure
    from langchain_core.messages import HumanMessage
    
    # Test state schema
    initial_state = {
        "messages": [HumanMessage("Hello, I want to research artificial intelligence")]
    }
    
    print("✓ Initial state created successfully")
    print("✓ Agent is ready for testing")
    
except ImportError as e:
    print("✗ Import error:", e)
except Exception as e:
    print("✗ Error:", e)
