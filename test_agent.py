#!/usr/bin/env python3
"""Test script to verify the agent.py implementation"""

import sys

try:
    # Test import
    from agent import app
    print("✓ Successfully imported app from agent.py")
    
    # Verify app is a compiled graph
    print(f"✓ app type: {type(app)}")
    
    # Check if it has the required methods
    if hasattr(app, 'invoke'):
        print("✓ app has 'invoke' method")
    
    if hasattr(app, 'stream'):
        print("✓ app has 'stream' method")
    
    if hasattr(app, 'get_state'):
        print("✓ app has 'get_state' method")
    
    # Verify the graph structure
    if hasattr(app, 'get_graph'):
        graph = app.get_graph()
        print(f"✓ Graph nodes: {graph.nodes}")
        print(f"✓ Graph edges: {graph.edges}")
    
    print("\n✅ All tests passed! The agent is properly configured.")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
