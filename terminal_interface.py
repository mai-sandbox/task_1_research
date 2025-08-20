#!/usr/bin/env python3

import os
from langchain_core.messages import HumanMessage
from agent import app

def main():
    print("🔬 Deep Research Agent")
    print("=" * 50)
    print("Welcome! I'll help you conduct comprehensive research on any topic.")
    print("Let's start by understanding what you'd like to research.\n")
    
    # Get initial research topic
    topic = input("What would you like me to research? ")
    
    if not topic.strip():
        print("Please provide a research topic to get started.")
        return
    
    # Initialize the agent state
    state = {
        "messages": [HumanMessage(content=topic)],
        "research_brief": "",
        "research_complete": False,
        "final_report": ""
    }
    
    print(f"\nProcessing your request: {topic}")
    print("-" * 50)
    
    try:
        # Run the agent
        for step in app.stream(state, {"recursion_limit": 50}):
            for node_name, node_state in step.items():
                if node_name == "clarify":
                    # Handle clarification phase
                    last_message = node_state["messages"][-1]
                    if hasattr(last_message, 'content'):
                        print(f"\nAgent: {last_message.content}")
                        
                        # Check if this is a question that needs user input
                        if "?" in last_message.content and not node_state.get("research_brief"):
                            user_response = input("\nYour response: ")
                            if user_response.strip():
                                # Continue the conversation
                                state["messages"].append(HumanMessage(content=user_response))
                                continue
                
                elif node_name == "research":
                    # Handle research phase
                    print(f"\n🔍 Conducting research based on brief: {node_state['research_brief']}")
                    print("This may take a moment as I gather comprehensive information...")
                    
                    if node_state.get("final_report"):
                        print("\n" + "=" * 80)
                        print("RESEARCH COMPLETE")
                        print("=" * 80)
                        print(node_state["final_report"])
                        print("=" * 80)
                        return
        
        print("\nResearch process completed!")
        
    except KeyboardInterrupt:
        print("\n\nResearch interrupted by user.")
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        print("Please ensure you have:")
        print("1. Set your OPENAI_API_KEY environment variable")
        print("2. Set your TAVILY_API_KEY environment variable")
        print("3. Installed required dependencies: pip install -r requirements.txt")

if __name__ == "__main__":
    main()