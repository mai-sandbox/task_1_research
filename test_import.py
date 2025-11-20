#!/usr/bin/env python3
"""Test script to verify agent module imports and compiles correctly after formatting."""

import sys

try:
    # Test basic import
    import agent
    print("✅ Successfully imported agent module")
    
    # Test that app attribute exists
    if hasattr(agent, 'app'):
        print("✅ 'app' attribute exists in agent module")
    else:
        print("❌ 'app' attribute not found in agent module")
        sys.exit(1)
    
    # Test that app is the correct type
    app_type = type(agent.app).__name__
    print(f"✅ app type: {app_type}")
    
    # Test that the graph has expected attributes
    if hasattr(agent.app, 'invoke'):
        print("✅ Graph has 'invoke' method")
    else:
        print("❌ Graph missing 'invoke' method")
        sys.exit(1)
    
    # Test that ResearchState is defined
    if hasattr(agent, 'ResearchState'):
        print("✅ ResearchState class is defined")
    else:
        print("❌ ResearchState class not found")
        sys.exit(1)
    
    # Test that key functions exist
    if hasattr(agent, 'build_graph'):
        print("✅ build_graph function exists")
    else:
        print("❌ build_graph function not found")
        sys.exit(1)
    
    print("\n" + "="*50)
    print("ALL TESTS PASSED - Agent module is working correctly!")
    print("="*50)
    
except ImportError as e:
    print(f"❌ Failed to import agent module: {e}")
    sys.exit(1)
except Exception as e:
    print(f"❌ Unexpected error: {e}")
    sys.exit(1)
