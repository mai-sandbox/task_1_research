#!/usr/bin/env python3
"""
Test script for the should_proceed decision function
"""

import os
from langchain_core.messages import HumanMessage, AIMessage

# Set dummy API keys for testing structure
os.environ['TAVILY_API_KEY'] = 'test-key'
os.environ['ANTHROPIC_API_KEY'] = 'test-key'

try:
    from agent import should_proceed_decision, should_proceed, ResearchState
    
    print("✅ should_proceed functions imported successfully!")
    
    # Test 1: Empty research brief - should return 'clarify_scope'
    print("\n🧪 Test 1: Empty research brief")
    empty_state = {
        "messages": [HumanMessage("I want to research AI")],
        "research_brief": "",
        "research_complete": False,
        "final_report": ""
    }
    
    result1 = should_proceed_decision(empty_state)
    print(f"Result: {result1}")
    assert result1 == "clarify_scope", f"Expected 'clarify_scope', got {result1}"
    print("✅ Empty brief correctly returns 'clarify_scope'")
    
    # Test 2: Brief with completion marker - should return 'react_research'
    print("\n🧪 Test 2: Brief with completion marker")
    complete_state = {
        "messages": [
            HumanMessage("I want to research AI"),
            AIMessage("Great! Let me ask some questions... RESEARCH_BRIEF_COMPLETE")
        ],
        "research_brief": "Research Topic: AI\nScope: Comprehensive analysis",
        "research_complete": False,
        "final_report": ""
    }
    
    result2 = should_proceed_decision(complete_state)
    print(f"Result: {result2}")
    assert result2 == "react_research", f"Expected 'react_research', got {result2}"
    print("✅ Complete brief correctly returns 'react_research'")
    
    # Test 3: Comprehensive brief without marker - should return 'react_research'
    print("\n🧪 Test 3: Comprehensive brief without completion marker")
    comprehensive_state = {
        "messages": [HumanMessage("I want to research AI")],
        "research_brief": """Research Topic: Artificial Intelligence in Healthcare
Research Scope: Comprehensive analysis of AI applications in medical diagnosis and treatment
Target Audience: Healthcare professionals and technology researchers
Research Depth: Detailed analysis with current developments and future trends
Key Focus Areas: Machine learning in diagnostics, AI-powered treatment recommendations, ethical considerations""",
        "research_complete": False,
        "final_report": ""
    }
    
    result3 = should_proceed_decision(comprehensive_state)
    print(f"Result: {result3}")
    assert result3 == "react_research", f"Expected 'react_research', got {result3}"
    print("✅ Comprehensive brief correctly returns 'react_research'")
    
    # Test 4: Minimal brief - should return 'clarify_scope'
    print("\n🧪 Test 4: Minimal brief")
    minimal_state = {
        "messages": [HumanMessage("I want to research AI")],
        "research_brief": "Research Topic: AI",
        "research_complete": False,
        "final_report": ""
    }
    
    result4 = should_proceed_decision(minimal_state)
    print(f"Result: {result4}")
    assert result4 == "clarify_scope", f"Expected 'clarify_scope', got {result4}"
    print("✅ Minimal brief correctly returns 'clarify_scope'")
    
    # Test 5: Test the should_proceed node function
    print("\n🧪 Test 5: should_proceed node function")
    node_result = should_proceed(comprehensive_state)
    print("✅ should_proceed node function executes successfully")
    assert "messages" in node_result, "Node should return messages"
    assert "research_brief" in node_result, "Node should return research_brief"
    
    print("\n✅ All should_proceed tests passed!")
    print("📋 Decision logic verified:")
    print("   - Empty briefs route to clarification")
    print("   - Completion markers trigger research phase")
    print("   - Comprehensive briefs proceed to research")
    print("   - Minimal briefs continue clarification")
    print("   - Node function maintains state properly")
    
except Exception as e:
    print(f"❌ Error testing should_proceed: {e}")
    import traceback
    traceback.print_exc()
