"""
LangGraph Deep Research Agent with User Interaction and ReAct Search

This agent implements a two-phase research workflow:
1. User interaction phase: Clarifies research scope through terminal interaction
2. ReAct research phase: Conducts research using Tavily search and generates a report
"""

import os
from typing import Annotated, TypedDict, List, Dict, Any
from typing_extensions import NotRequired

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearchResults
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command

# Load environment variables
load_dotenv()

# Initialize the LLM
llm = ChatOpenAI(
    model=os.getenv("MODEL_NAME", "gpt-4o-mini"),
    temperature=float(os.getenv("MODEL_TEMPERATURE", "0.7"))
)

# Initialize Tavily search tool
tavily_search = TavilySearchResults(
    max_results=5,
    search_depth="advanced",
    include_answer=True,
    include_raw_content=False,
    include_images=False,
)


class ResearchState(TypedDict):
    """State schema for the research agent workflow"""
    messages: Annotated[List[BaseMessage], add_messages]
    research_brief: str
    research_report: NotRequired[str]
    clarification_complete: bool


def user_interaction_node(state: ResearchState) -> Dict[str, Any]:
    """
    Node for user interaction to clarify research scope.
    Uses interrupt() to have a back-and-forth conversation with the user.
    """
    messages = state.get("messages", [])
    
    # Initial greeting if this is the first interaction
    if not messages or len(messages) == 1:
        initial_prompt = """
Hello! I'm your Deep Research Assistant. I'll help you conduct thorough research on any topic.

Before we begin, I need to understand exactly what you'd like me to research. 
Let's have a brief conversation to clarify:
1. What specific topic or question would you like researched?
2. What aspects are most important to you?
3. Are there any particular angles or perspectives you want explored?

Please share your research topic and any initial thoughts:
"""
        user_input = interrupt({"message": initial_prompt})
        messages.append(HumanMessage(content=user_input))
    
    # Process the conversation to build understanding
    clarification_messages = []
    research_brief = ""
    max_clarifications = 5
    clarification_count = 0
    
    while clarification_count < max_clarifications:
        # Analyze current understanding
        analysis_prompt = f"""
Based on the user's input, analyze their research needs and determine if we have enough clarity to proceed.

User's latest input: {messages[-1].content if messages else "No input yet"}

Previous conversation context: {[m.content for m in messages[-3:] if isinstance(m, (HumanMessage, AIMessage))]}

If clarification is needed, generate a specific follow-up question.
If we have enough information, summarize the research brief.

Respond in the following format:
CLARITY_STATUS: [NEEDS_CLARIFICATION or READY_TO_PROCEED]
FOLLOW_UP_QUESTION: [Your follow-up question if needed, or "None" if ready]
RESEARCH_BRIEF: [Comprehensive research brief if ready, or "Pending" if not ready]
"""
        
        analysis_response = llm.invoke([SystemMessage(content=analysis_prompt)])
        response_content = analysis_response.content
        
        # Parse the response
        lines = response_content.split('\n')
        clarity_status = "NEEDS_CLARIFICATION"
        follow_up_question = ""
        brief = ""
        
        for line in lines:
            if line.startswith("CLARITY_STATUS:"):
                clarity_status = line.split(":", 1)[1].strip()
            elif line.startswith("FOLLOW_UP_QUESTION:"):
                follow_up_question = line.split(":", 1)[1].strip()
            elif line.startswith("RESEARCH_BRIEF:"):
                brief = line.split(":", 1)[1].strip()
        
        if clarity_status == "READY_TO_PROCEED" and brief != "Pending":
            # We have enough information, confirm with user
            confirmation_prompt = f"""
Great! Based on our conversation, here's what I understand you want researched:

{brief}

Is this correct? Would you like me to proceed with this research brief, or would you like to modify anything?

Type 'proceed' to start the research, or provide any modifications:
"""
            user_confirmation = interrupt({"message": confirmation_prompt})
            
            if user_confirmation.lower().strip() in ['proceed', 'yes', 'correct', 'go ahead', 'start']:
                research_brief = brief
                break
            else:
                # User wants modifications
                messages.append(HumanMessage(content=user_confirmation))
                clarification_count += 1
        else:
            # Need more clarification
            if follow_up_question and follow_up_question != "None":
                user_response = interrupt({"message": follow_up_question})
                messages.append(HumanMessage(content=user_response))
                clarification_count += 1
            else:
                # Shouldn't happen, but handle gracefully
                break
    
    # If we exhausted clarifications without a clear brief, create one from what we have
    if not research_brief:
        fallback_prompt = f"""
Based on our conversation so far, create a research brief from the available information:
{[m.content for m in messages if isinstance(m, HumanMessage)]}
"""
        fallback_response = llm.invoke([SystemMessage(content=fallback_prompt)])
        research_brief = fallback_response.content
    
    return {
        "messages": messages,
        "research_brief": research_brief,
        "clarification_complete": True
    }


def create_research_agent_node(state: ResearchState) -> Dict[str, Any]:
    """
    Node that creates and runs the ReAct research agent with Tavily search.
    """
    research_brief = state.get("research_brief", "")
    
    if not research_brief:
        return {
            "research_report": "Error: No research brief provided. Please restart the conversation.",
            "messages": [AIMessage(content="Error: No research brief was created.")]
        }
    
    # Create the ReAct agent with Tavily search tool
    research_prompt = f"""You are an expert research assistant with access to web search capabilities.

Your task is to conduct thorough research based on the following brief:

{research_brief}

Guidelines for your research:
1. Use the search tool to find relevant, up-to-date information
2. Search for multiple perspectives and sources
3. Verify facts when possible by checking multiple sources
4. Focus on authoritative and credible sources
5. Organize your findings logically

After gathering information, synthesize it into a comprehensive research report that:
- Provides a clear overview of the topic
- Presents key findings and insights
- Includes relevant details and examples
- Cites sources when possible
- Offers balanced perspectives
- Concludes with a summary of the most important points

Begin your research now."""
    
    # Create the ReAct agent
    react_agent = create_react_agent(
        model=llm,
        tools=[tavily_search],
        prompt=research_prompt,
    )
    
    # Run the research agent
    research_messages = [HumanMessage(content="Please conduct the research and provide a detailed report.")]
    
    try:
        # Execute the research
        result = react_agent.invoke({"messages": research_messages})
        
        # Extract the final report from the agent's messages
        final_message = result["messages"][-1]
        
        if hasattr(final_message, 'content'):
            research_report = final_message.content
        else:
            research_report = str(final_message)
        
        # Format the final report
        formatted_report = f"""
# Research Report

## Research Brief
{research_brief}

## Findings
{research_report}

---
*Report generated using advanced web search and analysis*
"""
        
        return {
            "research_report": formatted_report,
            "messages": [AIMessage(content=formatted_report)]
        }
        
    except Exception as e:
        error_message = f"Error during research: {str(e)}"
        return {
            "research_report": error_message,
            "messages": [AIMessage(content=error_message)]
        }


def should_continue_clarification(state: ResearchState) -> str:
    """
    Conditional edge to determine if clarification is complete.
    """
    if state.get("clarification_complete", False):
        return "research"
    return "clarification"


# Build the graph
def build_graph():
    """
    Builds the two-phase research agent graph.
    """
    workflow = StateGraph(ResearchState)
    
    # Add nodes
    workflow.add_node("clarification", user_interaction_node)
    workflow.add_node("research", create_research_agent_node)
    
    # Add edges
    workflow.add_edge(START, "clarification")
    workflow.add_conditional_edges(
        "clarification",
        should_continue_clarification,
        {
            "clarification": "clarification",
            "research": "research"
        }
    )
    workflow.add_edge("research", END)
    
    # Compile with memory for interrupt support
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)


# Create and export the compiled graph as 'app'
app = build_graph()


# Optional: Main function for testing
if __name__ == "__main__":
    import uuid
    
    print("=" * 60)
    print("LangGraph Deep Research Agent")
    print("=" * 60)
    
    # Create a thread ID for this conversation
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    
    # Initial user message to start the conversation
    initial_state = {
        "messages": [HumanMessage(content="I want to research something")],
        "clarification_complete": False
    }
    
    try:
        # Run the graph with interrupts
        print("\nStarting research assistant...\n")
        
        # This will run until the first interrupt
        result = app.invoke(initial_state, config)
        
        # Continue handling interrupts until the research is complete
        while True:
            # Check if we're done
            if "research_report" in result and result["research_report"]:
                print("\n" + "=" * 60)
                print("FINAL RESEARCH REPORT")
                print("=" * 60)
                print(result["research_report"])
                break
            
            # If not done, we must be at an interrupt point
            # Get user input and resume
            user_input = input("\n> ")
            
            # Resume with the user's input
            result = app.invoke(
                Command(resume=user_input),
                config
            )
    
    except KeyboardInterrupt:
        print("\n\nResearch session terminated by user.")
    except Exception as e:
        print(f"\nError: {e}")
