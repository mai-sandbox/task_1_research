#!/usr/bin/env python3
"""
Interactive Demo for the LangGraph Deep Research Agent

This script demonstrates the interactive clarification process followed by automated research.
"""

import os
from langchain_core.messages import HumanMessage, AIMessage
from agent import app

def interactive_demo():
    """Run an interactive demo of the research agent."""
    
    # Check for required environment variables
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("❌ Error: Please set ANTHROPIC_API_KEY environment variable")
        print("Example: export ANTHROPIC_API_KEY='your-key-here'")
        return
    
    if not os.getenv("TAVILY_API_KEY"):
        print("❌ Error: Please set TAVILY_API_KEY environment variable")
        print("Example: export TAVILY_API_KEY='your-key-here'")
        return
    
    print("🔬 LangGraph Deep Research Agent - Interactive Demo")
    print("=" * 60)
    print("This agent will:")
    print("1. 💬 Chat with you to clarify your research needs")
    print("2. 🔍 Conduct deep research using search tools")
    print("3. 📊 Generate a comprehensive research report")
    print("=" * 60)
    
    # Get initial research topic
    topic = input("\n🤔 What would you like to research today? ")
    if not topic.strip():
        print("Please enter a research topic!")
        return
    
    print(f"\n🚀 Starting research on: '{topic}'")
    print("💡 The agent will now ask clarifying questions...")
    
    # Initialize state
    state = {
        "messages": [HumanMessage(topic)],
        "stage": "clarification", 
        "clarification_complete": False,
        "research_complete": False,
        "research_brief": ""
    }
    
    conversation_count = 0
    
    try:
        while state.get("stage") != "completed":
            print(f"\n⚡ Processing (stage: {state.get('stage', 'unknown')})...")
            
            # Invoke the agent
            result = app.invoke(state)
            
            # Get the latest AI message
            latest_messages = [msg for msg in result.get("messages", []) 
                             if isinstance(msg, AIMessage) and msg.content.strip()]
            
            if latest_messages:
                latest_message = latest_messages[-1]
                
                # Display agent's response
                print(f"\n🤖 Agent:")
                print("-" * 40)
                print(latest_message.content)
                print("-" * 40)
            
            # Update state
            state = result
            conversation_count += 1
            
            # If still in clarification stage, get user input
            if (state.get("stage") == "clarification" and 
                not state.get("clarification_complete", False)):
                
                print("\n👤 Your turn to respond:")
                user_input = input("You: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'stop']:
                    print("🛑 Demo stopped by user.")
                    break
                    
                if not user_input:
                    print("Please enter a response!")
                    continue
                
                # Add user message to state
                state["messages"].append(HumanMessage(content=user_input))
                
            elif state.get("stage") == "research":
                print("\n🔍 Research phase started - this may take a few moments...")
                print("⏳ The agent is now conducting automated research...")
                
            # Safety check to prevent infinite loops
            if conversation_count > 20:
                print("\n⚠️ Demo stopped - too many iterations")
                break
        
        print("\n✅ Research completed!")
        
        # Display final results
        if state.get("research_brief"):
            print(f"\n📋 Research Brief Generated:")
            print("=" * 50)
            print(state["research_brief"])
            print("=" * 50)
        
        # Display final research report from the last AI message
        final_messages = [msg for msg in state.get("messages", []) 
                         if isinstance(msg, AIMessage) and "RESEARCH_COMPLETE" in msg.content]
        
        if final_messages:
            report = final_messages[-1].content
            # Remove the RESEARCH_COMPLETE marker for clean display
            if "RESEARCH_COMPLETE" in report:
                report = report.split("RESEARCH_COMPLETE", 1)[1].strip()
            
            print(f"\n📊 Final Research Report:")
            print("=" * 50)
            print(report)
            print("=" * 50)
        
        print(f"\n📈 Session Summary:")
        print(f"• Total interactions: {conversation_count}")
        print(f"• Final stage: {state.get('stage', 'unknown')}")
        print(f"• Clarification completed: {state.get('clarification_complete', False)}")
        print(f"• Research completed: {state.get('research_complete', False)}")
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Demo interrupted by user (Ctrl+C)")
    except Exception as e:
        print(f"\n❌ Error during demo: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    interactive_demo()