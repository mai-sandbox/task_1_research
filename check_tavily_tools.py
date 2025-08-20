#!/usr/bin/env python3
"""
Check available Tavily tools
"""

try:
    # Check what's available in langchain_community.tools.tavily_search
    from langchain_community.tools.tavily_search import TavilySearchResults, TavilyAnswer
    print("✓ Available from langchain_community.tools.tavily_search:")
    print("  - TavilySearchResults")
    print("  - TavilyAnswer")
except ImportError as e:
    print("✗ Error importing from langchain_community.tools.tavily_search:", e)

try:
    # Check if there are other Tavily tools
    import langchain_community.tools.tavily_search as tavily_module
    print("\n✓ All attributes in tavily_search module:")
    for attr in dir(tavily_module):
        if not attr.startswith('_'):
            print(f"  - {attr}")
except Exception as e:
    print("✗ Error checking tavily_search module:", e)

try:
    # Check if there are Tavily tools in other locations
    from tavily import TavilyClient
    print("\n✓ Direct Tavily client available")
    print("  - TavilyClient")
except ImportError as e:
    print("\n✗ Direct Tavily client not available:", e)

try:
    # Check langchain_community.tools for any Tavily-related tools
    import langchain_community.tools as tools_module
    tavily_tools = [attr for attr in dir(tools_module) if 'tavily' in attr.lower()]
    if tavily_tools:
        print(f"\n✓ Found Tavily-related tools in langchain_community.tools: {tavily_tools}")
    else:
        print("\n- No Tavily-related tools found in langchain_community.tools")
except Exception as e:
    print("\n✗ Error checking langchain_community.tools:", e)
