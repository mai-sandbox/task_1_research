"""
LangGraph Deep Research Agent with Interactive Scoping and ReAct Search

This agent operates in two phases:
1. Interactive Scoping: Clarifies research scope through terminal interaction
2. ReAct Research: Performs detailed research using Tavily search and generates reports
"""

import os
from typing import TypedDict, Annotated, Literal, List
from typing_extensions import TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent
from langgraph.types import interrupt, Command


# State definition for the agent
class ResearchState(TypedDict):
    """State schema for the research agent"""
    messages: Annotated[List[BaseMessage], add_messages]
    research_brief: str
    phase: Literal["scoping", "research", "complete"]
    final_report: str


def get_llm():
    """Get the configured LLM model with fallback options"""
    # Try Anthropic first (preferred)
    if os.getenv("ANTHROPIC_API_KEY"):
        return ChatAnthropic(model="claude-3-5-sonnet-20241022", temperature=0.7)
    # Fallback to OpenAI
    elif os.getenv("OPENAI_API_KEY"):
        return ChatOpenAI(model="gpt-4o", temperature=0.7)
    else:
        raise ValueError(
            "No LLM API key found. Please set either ANTHROPIC_API_KEY or OPENAI_API_KEY"
        )


def scoping_node(state: ResearchState) -> dict:
    """
    Interactive scoping node that clarifies research requirements with the user.
    Uses interrupt() to pause execution and gather user input via terminal.
    """
    messages = state.get("messages", [])
    research_brief = state.get("research_brief", "")
    
    # Get the LLM for generating questions
    llm = get_llm()
    
    # Initial message if this is the first interaction
    if not research_brief:
        initial_prompt = """You are a research assistant helping to scope a research project.
        The user has requested: "{}"
        
        Generate 2-3 clarifying questions to better understand:
        1. The specific aspects they want researched
        2. The depth and breadth of information needed
        3. Any particular sources or perspectives they prefer
        
        Format your response as numbered questions.""".format(
            messages[-1].content if messages else "research assistance"
        )
        
        response = llm.invoke([SystemMessage(content=initial_prompt)])
        questions = response.content
        
        # Use interrupt to get user input
        user_response = interrupt({
            "message": "I'll help you with your research. To ensure I provide the most relevant information, "
                      "I have a few clarifying questions:\n\n" + questions + "\n\nPlease provide your answers:",
            "type": "clarification_request"
        })
        
        # Update research brief with initial responses
        research_brief = f"User responses to initial questions:\n{user_response}"
    
    # Check if we need more clarification
    validation_prompt = f"""Based on the following research brief, determine if we have enough information 
    to proceed with detailed research:
    
    {research_brief}
    
    If the scope is clear and specific enough, respond with "READY".
    If you need more clarification, generate 1-2 follow-up questions.
    
    Consider:
    - Is the topic well-defined?
    - Are the research objectives clear?
    - Do we understand the desired output format?"""
    
    validation_response = llm.invoke([SystemMessage(content=validation_prompt)])
    
    if "READY" in validation_response.content.upper():
        # Scope is clear, prepare final research brief
        final_brief_prompt = f"""Create a concise, structured research brief based on this information:
        
        {research_brief}
        
        Format it as:
        RESEARCH BRIEF:
        - Topic: [main topic]
        - Objectives: [key objectives]
        - Scope: [specific areas to cover]
        - Deliverables: [expected output]"""
        
        final_brief = llm.invoke([SystemMessage(content=final_brief_prompt)])
        
        # Confirm with user before proceeding
        confirmation = interrupt({
            "message": f"Great! I've prepared the following research brief:\n\n{final_brief.content}\n\n"
                      "Type 'yes' to proceed with the research, or provide any final adjustments:",
            "type": "confirmation_request"
        })
        
        if confirmation.lower() != "yes":
            research_brief += f"\n\nFinal adjustments: {confirmation}"
            # Regenerate the brief with adjustments
            final_brief = llm.invoke([
                SystemMessage(content=final_brief_prompt + f"\n\nUser adjustments: {confirmation}")
            ])
        
        return {
            "research_brief": final_brief.content,
            "phase": "research",
            "messages": [AIMessage(content=f"Research brief finalized. Starting detailed research...")]
        }
    else:
        # Need more clarification
        follow_up = interrupt({
            "message": f"I need a bit more clarification:\n\n{validation_response.content}\n\nYour response:",
            "type": "follow_up_request"
        })
        
        research_brief += f"\n\nAdditional clarification:\n{follow_up}"
        
        return {
            "research_brief": research_brief,
            "phase": "scoping",
            "messages": [AIMessage(content="Gathering more information...")]
        }


def research_node(state: ResearchState) -> dict:
    """
    Research node that uses ReAct pattern with Tavily search to conduct research.
    This node creates a ReAct agent with search capabilities and generates a report.
    """
    research_brief = state.get("research_brief", "")
    
    # Initialize Tavily search tool
    search_tool = TavilySearch(max_results=5)
    
    # Get the LLM
    llm = get_llm()
    
    # Create the ReAct agent with search capabilities
    research_prompt = f"""You are a deep research agent with access to web search.
    
    Your research brief is:
    {research_brief}
    
    Instructions:
    1. Break down the research into specific queries
    2. Use the search tool to find relevant, credible information
    3. Search for multiple perspectives and recent data
    4. Verify facts by checking multiple sources
    5. Focus on authoritative and recent sources
    
    Conduct thorough research and gather comprehensive information."""
    
    # Create ReAct agent with Tavily tool
    react_agent = create_react_agent(
        model=llm,
        tools=[search_tool],
        prompt=research_prompt
    )
    
    # Execute research
    research_messages = [
        HumanMessage(content=f"Please conduct detailed research based on this brief: {research_brief}")
    ]
    
    research_result = react_agent.invoke({"messages": research_messages})
    
    # Generate final report
    report_prompt = f"""Based on the research conducted, create a comprehensive report.
    
    Research Brief:
    {research_brief}
    
    Research Results:
    {research_result['messages'][-1].content if research_result.get('messages') else 'No results'}
    
    Create a well-structured report with:
    1. Executive Summary
    2. Key Findings (organized by topic)
    3. Detailed Analysis
    4. Sources and References
    5. Recommendations or Conclusions
    
    Make it informative, well-organized, and actionable."""
    
    report_response = llm.invoke([SystemMessage(content=report_prompt)])
    
    return {
        "final_report": report_response.content,
        "phase": "complete",
        "messages": [AIMessage(content="Research completed. Report generated.")]
    }


def route_phase(state: ResearchState) -> Literal["scoping", "research", "complete"]:
    """Route to the appropriate phase based on current state"""
    phase = state.get("phase", "scoping")
    
    if phase == "scoping":
        return "scoping"
    elif phase == "research":
        return "research"
    else:
        return "complete"


def format_output(state: ResearchState) -> dict:
    """Format the final output for the user"""
    final_report = state.get("final_report", "")
    
    if final_report:
        output_message = f"\n{'='*80}\n📊 RESEARCH REPORT\n{'='*80}\n\n{final_report}\n\n{'='*80}"
        return {
            "messages": [AIMessage(content=output_message)]
        }
    return state


# Build the graph
def build_graph():
    """Build and compile the research agent graph"""
    workflow = StateGraph(ResearchState)
    
    # Add nodes
    workflow.add_node("scoping", scoping_node)
    workflow.add_node("research", research_node)
    workflow.add_node("output", format_output)
    
    # Add edges
    workflow.add_edge(START, "scoping")
    
    # Conditional routing based on phase
    workflow.add_conditional_edges(
        "scoping",
        lambda state: "research" if state.get("phase") == "research" else "scoping",
        {
            "scoping": "scoping",
            "research": "research"
        }
    )
    
    workflow.add_edge("research", "output")
    workflow.add_edge("output", END)
    
    # Compile the graph
    return workflow.compile()


# Export the compiled graph as 'app' (required by AGENTS.md)
app = build_graph()


# Optional: CLI interface for testing
if __name__ == "__main__":
    import sys
    from dotenv import load_dotenv
    
    # Load environment variables
    load_dotenv()
    
    print("\n🔬 Deep Research Agent with Interactive Scoping")
    print("=" * 50)
    
    # Get initial research request
    if len(sys.argv) > 1:
        initial_request = " ".join(sys.argv[1:])
    else:
        initial_request = input("\nWhat would you like me to research? ")
    
    # Initialize state
    initial_state = {
        "messages": [HumanMessage(content=initial_request)],
        "phase": "scoping",
        "research_brief": "",
        "final_report": ""
    }
    
    try:
        # Run the agent
        print("\n🤔 Let me clarify a few things to provide the best research...\n")
        
        # Process with interrupts
        current_state = initial_state
        while True:
            result = app.invoke(current_state)
            
            # Check for interrupts
            if "__interrupt__" in result:
                interrupt_data = result["__interrupt__"][0]
                print(interrupt_data.value["message"])
                user_input = input("\n> ")
                
                # Resume with user input
                current_state = result
                result = app.invoke(Command(resume=user_input))
                current_state = result
            else:
                # No more interrupts, display final result
                if result.get("messages"):
                    print(result["messages"][-1].content)
                break
                
    except KeyboardInterrupt:
        print("\n\n❌ Research cancelled by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)
