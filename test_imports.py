#!/usr/bin/env python3
"""Test script to verify all dependencies are installed correctly."""

try:
    import langgraph
    print("✓ langgraph imported successfully")
except ImportError as e:
    print(f"✗ Failed to import langgraph: {e}")

try:
    import langchain_tavily
    print("✓ langchain-tavily imported successfully")
except ImportError as e:
    print(f"✗ Failed to import langchain-tavily: {e}")

try:
    import langchain_anthropic
    print("✓ langchain-anthropic imported successfully")
except ImportError as e:
    print(f"✗ Failed to import langchain-anthropic: {e}")

try:
    import langchain_openai
    print("✓ langchain-openai imported successfully")
except ImportError as e:
    print(f"✗ Failed to import langchain-openai: {e}")

try:
    import typing_extensions
    print("✓ typing-extensions imported successfully")
except ImportError as e:
    print(f"✗ Failed to import typing-extensions: {e}")

print("\nAll required dependencies are installed!")
