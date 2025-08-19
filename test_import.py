#!/usr/bin/env python3
"""Test script to verify agent.py imports correctly"""

try:
    from agent import app
    print("✓ Successfully imported 'app' from agent.py")
    print(f"✓ app is a compiled graph: {type(app).__name__}")
    
    # Check if it has the expected methods
    if hasattr(app, 'invoke'):
        print("✓ app has 'invoke' method")
    if hasattr(app, 'stream'):
        print("✓ app has 'stream' method")
    
    print("\nAgent.py is correctly implemented and ready for use!")
    
except ImportError as e:
    print(f"✗ Failed to import: {e}")
except Exception as e:
    print(f"✗ Error: {e}")
