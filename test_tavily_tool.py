#!/usr/bin/env python3
"""
Test script for the Tavily search tool function
"""

import os
from unittest.mock import Mock, patch

# Set dummy API keys for testing structure
os.environ['TAVILY_API_KEY'] = 'test-key'
os.environ['ANTHROPIC_API_KEY'] = 'test-key'

try:
    from agent import tavily_search_tool
    
    print("✅ tavily_search_tool imported successfully!")
    
    # Test 1: Error handling when client is not initialized
    print("\n🧪 Test 1: Error handling with uninitialized client")
    
    # Mock the tavily_client to be None to test error handling
    with patch('agent.tavily_client', None):
        result1 = tavily_search_tool("test query")
        print(f"Result: {result1[:100]}...")
        assert "Error: Tavily client not initialized" in result1
        print("✅ Properly handles uninitialized client")
    
    # Test 2: Mock successful search response
    print("\n🧪 Test 2: Successful search with mocked response")
    
    mock_response = {
        "answer": "This is a test summary of the search results",
        "results": [
            {
                "title": "Test Article 1",
                "url": "https://example.com/article1",
                "content": "This is the content of the first test article with relevant information about the search query."
            },
            {
                "title": "Test Article 2", 
                "url": "https://example.com/article2",
                "content": "This is the content of the second test article providing additional context and details."
            }
        ]
    }
    
    with patch('agent.tavily_client') as mock_client:
        mock_client.search.return_value = mock_response
        result2 = tavily_search_tool("artificial intelligence")
        
        print("✅ Successfully processes mocked search response")
        assert "Search Results for: artificial intelligence" in result2
        assert "Summary: This is a test summary" in result2
        assert "Test Article 1" in result2
        assert "https://example.com/article1" in result2
        print("✅ Formatted output contains all expected elements")
    
    # Test 3: Exception handling
    print("\n🧪 Test 3: Exception handling during search")
    
    with patch('agent.tavily_client') as mock_client:
        mock_client.search.side_effect = Exception("API connection failed")
        result3 = tavily_search_tool("test query")
        
        print(f"Result: {result3}")
        assert "Error searching with Tavily: API connection failed" in result3
        print("✅ Properly handles search exceptions")
    
    # Test 4: Function signature and parameters
    print("\n🧪 Test 4: Function signature validation")
    
    import inspect
    sig = inspect.signature(tavily_search_tool)
    params = list(sig.parameters.keys())
    
    assert len(params) == 1, f"Expected 1 parameter, got {len(params)}"
    assert params[0] == "query", f"Expected parameter 'query', got '{params[0]}'"
    assert sig.return_annotation == str or sig.return_annotation == inspect.Signature.empty
    print("✅ Function signature is correct")
    
    print("\n✅ All Tavily search tool tests passed!")
    print("📋 Tool functionality verified:")
    print("   - Integrates with TavilyClient API")
    print("   - Accepts search query parameter")
    print("   - Returns formatted search results")
    print("   - Includes proper error handling")
    print("   - Suitable for ReAct agent integration")
    
except Exception as e:
    print(f"❌ Error testing Tavily search tool: {e}")
    import traceback
    traceback.print_exc()
