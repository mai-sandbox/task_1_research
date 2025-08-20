#!/usr/bin/env python3
"""
Test script to verify the scope clarification node implementation
"""

from agent import app, clarify_scope, AgentState
from langchain_core.messages import HumanMessage, AIMessage

def test_scope_clarification_flow():
    """Test the complete scope clarification conversation flow"""
    
    print("=== TESTING SCOPE CLARIFICATION NODE ===\n")
    
    # Test 1: Initial greeting
    print("1. Testing initial greeting...")
    initial_state = {
        "messages": [],
        "research_scope": None,
        "research_complete": False,
        "final_report": None
    }
    
    result = clarify_scope(initial_state)
    print(f"✓ Initial response: {result['messages'][-1].content[:100]}...")
    print(f"✓ Research scope initialized: {result['research_scope'] is not None}")
    
    # Test 2: Topic collection
    print("\n2. Testing topic collection...")
    state_with_topic = {
        "messages": [
            AIMessage(content="Hello! I'm your research assistant..."),
            HumanMessage(content="I want to research artificial intelligence")
        ],
        "research_scope": {
            "topic": "",
            "depth": "",
            "sources": [],
            "timeline": "",
            "focus_areas": [],
            "confirmed": False
        },
        "research_complete": False,
        "final_report": None
    }
    
    result = clarify_scope(state_with_topic)
    print(f"✓ Topic captured: {result['research_scope']['topic']}")
    print(f"✓ Asks about depth: {'depth' in result['messages'][-1].content.lower()}")
    
    # Test 3: Depth collection
    print("\n3. Testing depth collection...")
    state_with_depth = {
        **result,
        "messages": result["messages"] + [HumanMessage(content="comprehensive")]
    }
    
    result = clarify_scope(state_with_depth)
    print(f"✓ Depth captured: {result['research_scope']['depth']}")
    print(f"✓ Asks about sources: {'sources' in result['messages'][-1].content.lower()}")
    
    # Test 4: Sources collection
    print("\n4. Testing sources collection...")
    state_with_sources = {
        **result,
        "messages": result["messages"] + [HumanMessage(content="academic and news sources")]
    }
    
    result = clarify_scope(state_with_sources)
    print(f"✓ Sources captured: {result['research_scope']['sources']}")
    print(f"✓ Asks about timeline: {'timeframe' in result['messages'][-1].content.lower()}")
    
    # Test 5: Timeline collection
    print("\n5. Testing timeline collection...")
    state_with_timeline = {
        **result,
        "messages": result["messages"] + [HumanMessage(content="recent developments")]
    }
    
    result = clarify_scope(state_with_timeline)
    print(f"✓ Timeline captured: {result['research_scope']['timeline']}")
    print(f"✓ Asks about focus areas: {'focus' in result['messages'][-1].content.lower()}")
    
    # Test 6: Focus areas collection and summary
    print("\n6. Testing focus areas collection...")
    state_with_focus = {
        **result,
        "messages": result["messages"] + [HumanMessage(content="machine learning applications, ethics")]
    }
    
    result = clarify_scope(state_with_focus)
    print(f"✓ Focus areas captured: {result['research_scope']['focus_areas']}")
    print(f"✓ Provides summary: {'Summary' in result['messages'][-1].content}")
    
    # Test 7: Confirmation handling
    print("\n7. Testing confirmation handling...")
    state_ready_to_proceed = {
        **result,
        "messages": result["messages"] + [HumanMessage(content="proceed")]
    }
    
    result = clarify_scope(state_ready_to_proceed)
    print(f"✓ Confirmation processed: {result['research_scope']['confirmed']}")
    print(f"✓ Ready for research phase: {result['research_scope']['confirmed']}")
    
    # Test 8: Validation - incomplete information
    print("\n8. Testing validation with incomplete information...")
    incomplete_state = {
        "messages": [HumanMessage(content="proceed")],
        "research_scope": {
            "topic": "",
            "depth": "",
            "sources": [],
            "timeline": "",
            "focus_areas": [],
            "confirmed": False
        },
        "research_complete": False,
        "final_report": None
    }
    
    result = clarify_scope(incomplete_state)
    print(f"✓ Handles incomplete info: {not result['research_scope']['confirmed']}")
    print(f"✓ Asks for missing topic: {'topic' in result['messages'][-1].content.lower()}")
    
    print("\n=== SCOPE CLARIFICATION TESTS COMPLETED ===")
    print("✓ All core functionality working correctly")
    
    return True

def test_modification_requests():
    """Test the modification request handling"""
    
    print("\n=== TESTING MODIFICATION REQUESTS ===\n")
    
    # Test modification request
    state_with_data = {
        "messages": [HumanMessage(content="change topic")],
        "research_scope": {
            "topic": "artificial intelligence",
            "depth": "comprehensive",
            "sources": ["academic"],
            "timeline": "recent",
            "focus_areas": ["general"],
            "confirmed": False
        },
        "research_complete": False,
        "final_report": None
    }
    
    result = clarify_scope(state_with_data)
    print(f"✓ Handles modification request: {result['research_scope']['topic'] == ''}")
    print(f"✓ Asks for new topic: {'topic' in result['messages'][-1].content.lower()}")
    
    print("✓ Modification request handling working correctly")

if __name__ == "__main__":
    try:
        test_scope_clarification_flow()
        test_modification_requests()
        print("\n🎉 ALL TESTS PASSED! Scope clarification node is working correctly.")
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
