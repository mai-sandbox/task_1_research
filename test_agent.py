from agent import app
from langchain_core.messages import HumanMessage

def test_research_agent():
    """Test the research agent workflow"""
    print("Testing LangGraph Deep Research Agent")
    print("=" * 50)
    
    # Initial state with user message
    initial_state = {
        "messages": [HumanMessage("I want to research artificial intelligence trends in 2024")]
    }
    
    # Test clarification phase
    print("Phase 1: Research Scope Clarification")
    result = app.invoke(initial_state)
    print(f"Agent Response: {result['messages'][-1].content}")
    print(f"Status: {result['research_status']}")
    
    # Simulate user providing more details
    clarification_state = {
        "messages": result['messages'] + [HumanMessage("I'm interested in business applications of AI, particularly in healthcare and finance. I need this for a strategic planning presentation to executives. Focus on recent developments and market impact.")],
        "research_status": result['research_status']
    }
    
    print("\nPhase 2: User Provides Details")
    result2 = app.invoke(clarification_state)
    print(f"Agent Response: {result2['messages'][-1].content}")
    print(f"Status: {result2['research_status']}")
    
    # Simulate confirming to proceed
    if result2['research_status'] == 'collecting_requirements':
        proceed_state = {
            "messages": result2['messages'] + [HumanMessage("Yes, that looks good. Please proceed with the research.")],
            "research_status": result2['research_status']
        }
        
        print("\nPhase 3: Proceeding to Research")
        final_result = app.invoke(proceed_state)
        print(f"Final Status: {final_result['research_status']}")
        print(f"Report Generated: {'Yes' if final_result.get('detailed_report') else 'No'}")
        
        if final_result.get('detailed_report'):
            print("\nReport Preview (first 500 chars):")
            print(final_result['detailed_report'][:500] + "...")

if __name__ == "__main__":
    test_research_agent()