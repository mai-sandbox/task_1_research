#!/usr/bin/env python3
"""Test script for the research agent."""

from agent import app
from langchain_core.messages import HumanMessage
import asyncio


def test_research_agent():
    """Test the research agent with a sample query."""
    
    print("Starting Research Agent Test...")
    print("-" * 50)
    
    initial_state = {
        "messages": [HumanMessage("I want to research the latest developments in quantum computing")],
        "clarification_complete": False,
        "research_brief": "",
        "final_report": ""
    }
    
    print("Initial Query: 'I want to research the latest developments in quantum computing'")
    print("-" * 50)
    
    config = {"recursion_limit": 25}
    
    for event in app.stream(initial_state, config):
        for node_name, node_output in event.items():
            if node_name == "clarify":
                if node_output.get("messages"):
                    last_message = node_output["messages"][-1]
                    print(f"\nClarification Agent: {last_message.content}\n")
                    
                    if not node_output.get("clarification_complete", False):
                        user_response = input("Your response (or 'done' to proceed with current info): ")
                        if user_response.lower() != 'done':
                            initial_state["messages"].append(HumanMessage(user_response))
                        else:
                            initial_state["messages"].append(
                                HumanMessage("I have all the information I need. Please proceed with the research.")
                            )
                    else:
                        print(f"\nResearch Brief Created:\n{node_output.get('research_brief', '')}\n")
                        
            elif node_name == "research":
                print("\nResearch Agent is working... (this may take a moment)\n")
                if node_output.get("final_report"):
                    print("=" * 50)
                    print("FINAL RESEARCH REPORT")
                    print("=" * 50)
                    print(node_output["final_report"])
                    print("=" * 50)


if __name__ == "__main__":
    test_research_agent()