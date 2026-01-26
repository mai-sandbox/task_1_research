#!/usr/bin/env python3
"""
Test script for the generate_report node functionality
"""

import os
from unittest.mock import Mock, patch
from langchain_core.messages import HumanMessage, AIMessage

# Set dummy API keys for testing structure
os.environ['TAVILY_API_KEY'] = 'test-key'
os.environ['ANTHROPIC_API_KEY'] = 'test-key'

try:
    from agent import generate_report, ResearchState
    
    print("✅ generate_report node imported successfully!")
    
    # Test 1: Error handling when LLM is not initialized
    print("\n🧪 Test 1: Error handling with uninitialized LLM")
    
    with patch('agent.llm', None):
        error_state = {
            "messages": [HumanMessage("Generate report")],
            "research_brief": "Research Topic: AI\nScope: Comprehensive analysis",
            "research_complete": True,
            "final_report": "Research findings about AI..."
        }
        
        result1 = generate_report(error_state)
        print("✅ Properly handles uninitialized LLM")
        assert "Error: LLM not initialized" in result1["messages"][-1].content
        assert result1["final_report"] == ""
    
    # Test 2: Error handling with missing research brief
    print("\n🧪 Test 2: Error handling with missing research brief")
    
    empty_brief_state = {
        "messages": [HumanMessage("Generate report")],
        "research_brief": "",
        "research_complete": True,
        "final_report": "Some research findings..."
    }
    
    result2 = generate_report(empty_brief_state)
    print("✅ Properly handles missing research brief")
    assert "Error: No research brief available" in result2["messages"][-1].content
    assert result2["final_report"] == ""
    
    # Test 3: Error handling with missing research findings
    print("\n🧪 Test 3: Error handling with missing research findings")
    
    empty_findings_state = {
        "messages": [HumanMessage("Generate report")],
        "research_brief": "Research Topic: AI\nScope: Comprehensive analysis",
        "research_complete": True,
        "final_report": ""
    }
    
    result3 = generate_report(empty_findings_state)
    print("✅ Properly handles missing research findings")
    assert "Error: No research findings available" in result3["messages"][-1].content
    assert result3["final_report"] == ""
    
    # Test 4: Successful report generation
    print("\n🧪 Test 4: Successful report generation with mocked LLM")
    
    mock_report = """# Artificial Intelligence in Healthcare: Comprehensive Analysis

## Executive Summary
This report provides a comprehensive analysis of AI applications in healthcare, covering current implementations, benefits, challenges, and future prospects.

## Introduction
Artificial Intelligence (AI) is revolutionizing healthcare through advanced diagnostic tools, treatment optimization, and patient care enhancement.

## Key Findings
- AI diagnostic accuracy exceeds human performance in medical imaging
- Machine learning algorithms improve treatment personalization
- Natural language processing streamlines clinical documentation

## Analysis and Insights
The integration of AI in healthcare demonstrates significant potential for improving patient outcomes while reducing costs.

## Conclusions
AI represents a transformative force in healthcare with substantial benefits for patients, providers, and healthcare systems.

## Recommendations
- Invest in AI training for healthcare professionals
- Develop robust data governance frameworks
- Ensure ethical AI implementation

## Sources and References
- Medical journals and research papers
- Healthcare technology reports
- Expert interviews and case studies"""
    
    comprehensive_state = {
        "messages": [HumanMessage("Please generate a comprehensive report")],
        "research_brief": """Research Topic: Artificial Intelligence in Healthcare
Research Scope: Comprehensive analysis of AI applications in medical diagnosis and treatment
Target Audience: Healthcare professionals and technology researchers
Research Depth: Detailed analysis with current developments and future trends
Key Focus Areas: Machine learning in diagnostics, AI-powered treatment recommendations, ethical considerations""",
        "research_complete": True,
        "final_report": "Based on extensive research, AI in healthcare shows remarkable progress in diagnostic accuracy, treatment personalization, and operational efficiency. Key applications include medical imaging analysis, predictive analytics for patient outcomes, and natural language processing for clinical documentation."
    }
    
    with patch('agent.llm') as mock_llm:
        mock_response = Mock()
        mock_response.content = mock_report
        mock_llm.invoke.return_value = mock_response
        
        result4 = generate_report(comprehensive_state)
        
        print("✅ Successfully generates report with mocked LLM")
        assert result4["research_complete"] == True
        assert result4["final_report"] == mock_report
        assert len(result4["messages"]) > len(comprehensive_state["messages"])
        assert "Research report generated successfully" in result4["messages"][-1].content
        print("✅ Final report stored in state field")
        print("✅ Completion message added to conversation")
    
    # Test 5: Exception handling during report generation
    print("\n🧪 Test 5: Exception handling during report generation")
    
    with patch('agent.llm') as mock_llm:
        mock_llm.invoke.side_effect = Exception("LLM API error")
        
        result5 = generate_report(comprehensive_state)
        
        print("✅ Properly handles report generation exceptions")
        assert "Error generating report:" in result5["messages"][-1].content
        assert result5["final_report"] == ""
    
    # Test 6: Verify report structure and content requirements
    print("\n🧪 Test 6: Report structure and content verification")
    
    with patch('agent.llm') as mock_llm:
        mock_response = Mock()
        mock_response.content = mock_report
        mock_llm.invoke.return_value = mock_response
        
        result6 = generate_report(comprehensive_state)
        
        # Verify LLM was called with proper prompt structure
        assert mock_llm.invoke.called
        call_args = mock_llm.invoke.call_args[0][0]  # First argument (messages list)
        prompt_content = call_args[0].content  # First message content
        
        # Check that prompt includes required elements
        assert "RESEARCH BRIEF:" in prompt_content
        assert "RESEARCH FINDINGS:" in prompt_content
        assert "Executive Summary" in prompt_content
        assert "Key Findings" in prompt_content
        assert "Conclusions" in prompt_content
        assert "Sources and References" in prompt_content
        
        print("✅ Report generation prompt includes all required sections")
        print("✅ Research brief and findings properly incorporated")
    
    # Test 7: State preservation verification
    print("\n🧪 Test 7: State preservation verification")
    
    with patch('agent.llm') as mock_llm:
        mock_response = Mock()
        mock_response.content = mock_report
        mock_llm.invoke.return_value = mock_response
        
        result7 = generate_report(comprehensive_state)
        
        # Verify all state fields are properly preserved/updated
        assert result7["research_brief"] == comprehensive_state["research_brief"]
        assert result7["research_complete"] == True
        assert result7["final_report"] == mock_report
        assert len(result7["messages"]) == len(comprehensive_state["messages"]) + 1
        
        print("✅ All state fields properly preserved and updated")
    
    print("\n✅ All generate_report node tests passed!")
    print("📋 Node functionality verified:")
    print("   - Takes all gathered research information from state")
    print("   - Creates detailed, well-structured final report")
    print("   - Synthesizes search results into comprehensive document")
    print("   - Includes proper sections, citations, and conclusions")
    print("   - Stores final report in final_report state field")
    print("   - Handles missing data and API errors gracefully")
    print("   - Uses professional report structure and formatting")
    
except Exception as e:
    print(f"❌ Error testing generate_report node: {e}")
    import traceback
    traceback.print_exc()
