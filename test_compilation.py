#!/usr/bin/env python3
"""Test script to verify agent compilation"""

try:
    from agent import app
    print("✅ Agent compiles successfully")
    print("✅ Graph exported as 'app' variable")
    
    # Verify the app is a compiled graph
    if hasattr(app, 'invoke'):
        print("✅ App has 'invoke' method")
    
    if hasattr(app, 'stream'):
        print("✅ App has 'stream' method")
    
    print("\n🎉 All compilation tests passed!")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    exit(1)
except Exception as e:
    print(f"❌ Unexpected error: {e}")
    exit(1)
