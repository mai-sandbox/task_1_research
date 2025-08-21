#!/usr/bin/env python3
"""
Simple example showing how to use the research agent
as specified in the requirements.
"""

from langchain_core.messages import HumanMessage
from agent import app
from dotenv import load_dotenv

load_dotenv()


def main():
    # Simple usage as per requirements
    initial_state = {
        "messages": [HumanMessage("I want to research AI safety and alignment")]
    }
    
    # The agent will handle the entire flow:
    # 1. Clarification phase (may require multiple invocations)
    # 2. Research phase with Tavily search
    # 3. Report generation
    
    result = app.invoke(initial_state)
    
    # Display the results
    print("Agent Response:")
    print("-" * 40)
    if result["messages"]:
        for msg in result["messages"]:
            print(f"{msg.content}\n")
    
    # If research is complete, show the report
    if result.get("research_report"):
        print("\nResearch Report:")
        print("=" * 40)
        print(result["research_report"])


if __name__ == "__main__":
    main()