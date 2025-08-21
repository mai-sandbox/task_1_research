#!/usr/bin/env python3
"""
Test script to verify that all dependencies are properly installed
and the agent can be imported successfully.
"""

try:
    # Test core imports
    import langgraph
    print("✅ langgraph imported successfully")
    
    import tavily
    print("✅ tavily-python imported successfully")
    
    import langchain_core
    print("✅ langchain-core imported successfully")
    
    import langchain_openai
    print("✅ langchain-openai imported successfully")
    
    # Test agent import
    import agent
    print("✅ agent.py imported successfully")
    
    # Verify the app is exported
    if hasattr(agent, 'app'):
        print("✅ agent.app is properly exported")
    else:
        print("❌ agent.app is not found")
    
    print("\n🎉 All dependencies and agent imports successful!")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
except Exception as e:
    print(f"❌ Unexpected error: {e}")
