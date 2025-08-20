#!/usr/bin/env python3
"""
Test script for the Deep Research Agent
Tests the complete workflow from scoping to research
"""

import asyncio
from langchain_core.messages import HumanMessage
from agent import app

def test_basic_flow():
    """Test the basic agent flow with a simple research query"""
    print("=" * 60)
    print("Testing Deep Research Agent")
    print("=" * 60)
    
    # Initial state with a research request
    initial_state = {
        "messages": [HumanMessage(content="I need to research the latest developments in quantum computing")],
        "research_phase": "scoping",
        "scoping_complete": False
    }
    
    print("\n1. Starting with initial query:")
    print(f"   '{initial_state['messages'][0].content}'")
    
    try:
        # Invoke the agent
        print("\n2. Invoking agent (this will trigger scoping phase)...")
        result = app.invoke(initial_state)
        
        # Check the result
        print("\n3. Agent execution completed!")
        
        # Display results
        if "messages" in result:
            print(f"\n4. Total messages in conversation: {len(result['messages'])}")
            
            # Show last message (should be from scoping agent)
            if result['messages']:
                last_message = result['messages'][-1]
                print(f"\n5. Last message from agent:")
                print("-" * 40)
                print(last_message.content[:500] + "..." if len(last_message.content) > 500 else last_message.content)
                print("-" * 40)
        
        if "research_phase" in result:
            print(f"\n6. Current phase: {result['research_phase']}")
        
        if "scoping_complete" in result:
            print(f"7. Scoping complete: {result['scoping_complete']}")
        
        print("\n✅ Test completed successfully!")
        print("\nNote: The agent uses interrupt() for interactive conversation.")
        print("In production, you would resume with user input after each interrupt.")
        
        return result
        
    except Exception as e:
        print(f"\n❌ Error during test: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_graph_structure():
    """Test that the graph is properly structured"""
    print("\n" + "=" * 60)
    print("Testing Graph Structure")
    print("=" * 60)
    
    try:
        # Check graph nodes
        nodes = app.get_graph().nodes
        print(f"\n1. Graph nodes: {list(nodes.keys())}")
        
        # Check if required nodes exist
        required_nodes = ["scoping", "research"]
        for node in required_nodes:
            if node in nodes:
                print(f"   ✅ '{node}' node exists")
            else:
                print(f"   ❌ '{node}' node missing")
        
        # Check graph edges
        edges = app.get_graph().edges
        print(f"\n2. Graph edges: {edges}")
        
        print("\n✅ Graph structure test completed!")
        
    except Exception as e:
        print(f"\n❌ Error checking graph structure: {e}")

def test_state_schema():
    """Test that the state schema is properly defined"""
    print("\n" + "=" * 60)
    print("Testing State Schema")
    print("=" * 60)
    
    try:
        # Get the state schema
        state_schema = app.get_input_schema()
        print(f"\n1. Input schema properties:")
        if hasattr(state_schema, 'schema'):
            schema = state_schema.schema()
            if 'properties' in schema:
                for prop, details in schema['properties'].items():
                    print(f"   - {prop}: {details.get('type', 'unknown type')}")
        
        print("\n✅ State schema test completed!")
        
    except Exception as e:
        print(f"\n❌ Error checking state schema: {e}")

if __name__ == "__main__":
    print("Starting Deep Research Agent Tests")
    print("=" * 80)
    
    # Test graph structure first
    test_graph_structure()
    
    # Test state schema
    test_state_schema()
    
    # Test basic flow
    print("\n" + "=" * 80)
    print("Note: The agent will interrupt for user input during scoping.")
    print("This is expected behavior for interactive conversation.")
    print("=" * 80)
    
    result = test_basic_flow()
    
    print("\n" + "=" * 80)
    print("All tests completed!")
    print("=" * 80)
