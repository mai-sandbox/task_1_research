#!/usr/bin/env python3
"""
Test script for the LangGraph research agent.
This demonstrates the two-phase research process:
1. Interactive clarification phase
2. ReAct research phase with Tavily search
"""

import os
import sys
from langchain_core.messages import HumanMessage, AIMessage
from agent import app
from dotenv import load_dotenv

load_dotenv()


def test_research_agent():
    """Test the research agent with a sample research topic"""
    
    print("=" * 60)
    print("LangGraph Deep Research Agent Test")
    print("=" * 60)
    
    # Check for required API keys
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not found in environment variables")
        print("Please create a .env file with your OpenAI API key")
        return
    
    if not os.getenv("TAVILY_API_KEY"):
        print("Error: TAVILY_API_KEY not found in environment variables")
        print("Please create a .env file with your Tavily API key")
        return
    
    print("\nStarting research agent...")
    print("\nPhase 1: Clarification")
    print("-" * 40)
    
    # Initial research request
    initial_request = "I want to research the latest developments in quantum computing"
    print(f"User: {initial_request}\n")
    
    # Start the conversation
    state = {
        "messages": [HumanMessage(content=initial_request)],
        "research_brief": "",
        "clarification_complete": False,
        "research_report": ""
    }
    
    # Run the agent - it will first clarify
    result = app.invoke(state)
    
    # Show clarification response
    if result["messages"]:
        last_message = result["messages"][-1]
        print(f"Agent: {last_message.content}\n")
    
    # Simulate user providing more details
    print("\nPhase 1.1: User provides clarification")
    print("-" * 40)
    
    user_clarification = """Yes, I want to focus on:
    1. Recent breakthroughs in quantum computing hardware (2023-2024)
    2. Major companies and their quantum computing initiatives
    3. Practical applications being developed
    4. Challenges that still need to be overcome
    
    Please proceed with the research."""
    
    print(f"User: {user_clarification}\n")
    
    # Add user's clarification to the conversation
    result["messages"].append(HumanMessage(content=user_clarification))
    
    # Continue the agent
    result = app.invoke(result)
    
    # The agent should now proceed to research phase
    print("\nPhase 2: Research (using Tavily search)")
    print("-" * 40)
    
    if result.get("clarification_complete"):
        print("Research brief created. Conducting research...\n")
        print(f"Research Brief:\n{result.get('research_brief', 'No brief created')}\n")
    
    # Show final research report
    if result.get("research_report"):
        print("\nFinal Research Report:")
        print("=" * 60)
        print(result["research_report"])
    else:
        print("\nResearch phase messages:")
        for msg in result["messages"][-3:]:  # Show last few messages
            if isinstance(msg.content, str):
                print(f"\n{msg.content[:500]}...")  # Truncate for display


def interactive_test():
    """Interactive test where user can have a real conversation"""
    
    print("=" * 60)
    print("Interactive Research Agent Test")
    print("=" * 60)
    print("\nYou can now interact with the research agent.")
    print("Type 'quit' to exit.\n")
    
    # Check for API keys
    if not os.getenv("OPENAI_API_KEY") or not os.getenv("TAVILY_API_KEY"):
        print("Error: Missing API keys. Please set up your .env file.")
        return
    
    # Get initial research topic
    initial_input = input("What would you like to research? > ")
    
    if initial_input.lower() == 'quit':
        return
    
    # Initialize state
    state = {
        "messages": [HumanMessage(content=initial_input)],
        "research_brief": "",
        "clarification_complete": False,
        "research_report": ""
    }
    
    # Clarification loop
    while not state.get("clarification_complete", False):
        # Run the agent
        result = app.invoke(state)
        state = result
        
        # Show agent's response
        if result["messages"]:
            last_message = result["messages"][-1]
            print(f"\nAgent: {last_message.content}\n")
        
        # Check if clarification is complete
        if result.get("clarification_complete"):
            print("\n" + "=" * 40)
            print("Clarification complete! Starting research...")
            print("=" * 40 + "\n")
            break
        
        # Get user input
        user_input = input("Your response > ")
        
        if user_input.lower() == 'quit':
            return
        
        # Add user message to state
        state["messages"].append(HumanMessage(content=user_input))
    
    # Research phase - final invocation will complete the research
    print("Conducting deep research using Tavily search...")
    print("(This may take a moment...)\n")
    
    final_result = app.invoke(state)
    
    # Display the research report
    if final_result.get("research_report"):
        print("\n" + "=" * 60)
        print("RESEARCH REPORT")
        print("=" * 60)
        print(final_result["research_report"])
    else:
        print("\nFinal message from agent:")
        if final_result["messages"]:
            print(final_result["messages"][-1].content)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        interactive_test()
    else:
        test_research_agent()
        print("\n\nTip: Run with --interactive flag for interactive mode:")
        print("  python test_agent.py --interactive")