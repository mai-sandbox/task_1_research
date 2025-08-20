"""
LangGraph Deep Research Agent with User Interaction and ReAct Search

This agent implements a two-phase research workflow:
1. Clarification phase: Interactive terminal conversation to gather research scope
2. Research phase: ReAct agent with Tavily search tool for detailed research
"""

import os
from typing import Annotated, Literal, Optional
from typing_extensions import TypedDict

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent
from tavily import TavilyClient
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize LLM
llm = init_chat_model("anthropic:claude-3-5-sonnet-latest")

# Initialize Tavily client
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))


class State(TypedDict):
    """State schema for the research agent."""
    messages: Annotated[list, add_messages]
    research_brief: Optional[str]
    research_complete: bool


def tavily_search_tool(query: str) -> str:
    """Tavily search tool for the ReAct agent."""
    try:
        response = tavily_client.search(
            query=query,
            search_depth="advanced",
            max_results=5,
            include_answer=True,
            include_raw_content=True
        )
        
        # Format the search results
        formatted_results = []
        
        if response.get("answer"):
            formatted_results.append(f"**Summary Answer:** {response['answer']}\n")
        
        if response.get("results"):
            formatted_results.append("**Detailed Results:**")
            for i, result in enumerate(response["results"][:5], 1):
                title = result.get("title", "No title")
                url = result.get("url", "No URL")
                content = result.get("content", "No content")[:500] + "..." if len(result.get("content", "")) > 500 else result.get("content", "")
                
                formatted_results.append(f"\n{i}. **{title}**")
                formatted_results.append(f"   URL: {url}")
                formatted_results.append(f"   Content: {content}")
        
        return "\n".join(formatted_results)
    
    except Exception as e:
        return f"Error performing search: {str(e)}"


def clarification_node(state: State) -> dict:
    """
    Interactive node that gathers research scope and objectives from the user.
    Uses terminal input() for human-in-the-loop interaction.
    """
    print("\n" + "="*60)
    print("🔍 RESEARCH AGENT - CLARIFICATION PHASE")
    print("="*60)
    print("Let's clarify the scope of your research project.\n")
    
    # Gather research information interactively
    research_components = {}
    
    # Research topic/question
    print("1. What is the main research topic or question you want to explore?")
    research_components["topic"] = input("   Topic: ").strip()
    
    if not research_components["topic"]:
        research_components["topic"] = "General research inquiry"
    
    # Research objectives
    print("\n2. What are your specific research objectives? (What do you want to find out?)")
    research_components["objectives"] = input("   Objectives: ").strip()
    
    # Scope and constraints
    print("\n3. Are there any specific aspects, time periods, or constraints for this research?")
    research_components["constraints"] = input("   Constraints: ").strip()
    
    # Target audience or use case
    print("\n4. Who is the target audience or what will this research be used for?")
    research_components["audience"] = input("   Audience/Use case: ").strip()
    
    # Depth preference
    print("\n5. How detailed should the research be? (brief overview, comprehensive analysis, etc.)")
    research_components["depth"] = input("   Depth preference: ").strip()
    
    # Create research brief
    research_brief = f"""
RESEARCH BRIEF:

Topic: {research_components['topic']}

Objectives: {research_components['objectives'] or 'Comprehensive research on the given topic'}

Constraints: {research_components['constraints'] or 'No specific constraints mentioned'}

Target Audience: {research_components['audience'] or 'General audience'}

Depth: {research_components['depth'] or 'Comprehensive analysis'}

RESEARCH INSTRUCTIONS:
Please conduct thorough research on this topic using web search tools. Provide a detailed report that addresses the objectives while considering the constraints and target audience. Include relevant facts, statistics, recent developments, and credible sources.
"""
    
    # Display the research brief for confirmation
    print("\n" + "="*60)
    print("📋 RESEARCH BRIEF SUMMARY")
    print("="*60)
    print(research_brief)
    
    # Ask for confirmation
    print("\n" + "-"*60)
    while True:
        confirmation = input("Do you want to proceed with this research brief? (yes/no/edit): ").strip().lower()
        
        if confirmation in ['yes', 'y']:
            print("\n✅ Research brief confirmed! Proceeding to research phase...")
            return {
                "messages": [AIMessage(content="Research brief confirmed. Starting research phase.")],
                "research_brief": research_brief,
                "research_complete": True
            }
        elif confirmation in ['no', 'n']:
            print("\n❌ Research cancelled.")
            return {
                "messages": [AIMessage(content="Research cancelled by user.")],
                "research_brief": None,
                "research_complete": False
            }
        elif confirmation in ['edit', 'e']:
            print("\n✏️ Let's refine the research brief...")
            # Allow user to edit specific components
            print("Which aspect would you like to edit?")
            print("1. Topic")
            print("2. Objectives") 
            print("3. Constraints")
            print("4. Audience")
            print("5. Depth")
            
            edit_choice = input("Enter number (1-5): ").strip()
            
            if edit_choice == "1":
                research_components["topic"] = input("New topic: ").strip()
            elif edit_choice == "2":
                research_components["objectives"] = input("New objectives: ").strip()
            elif edit_choice == "3":
                research_components["constraints"] = input("New constraints: ").strip()
            elif edit_choice == "4":
                research_components["audience"] = input("New audience: ").strip()
            elif edit_choice == "5":
                research_components["depth"] = input("New depth preference: ").strip()
            
            # Recreate the research brief with updates
            research_brief = f"""
RESEARCH BRIEF:

Topic: {research_components['topic']}

Objectives: {research_components['objectives'] or 'Comprehensive research on the given topic'}

Constraints: {research_components['constraints'] or 'No specific constraints mentioned'}

Target Audience: {research_components['audience'] or 'General audience'}

Depth: {research_components['depth'] or 'Comprehensive analysis'}

RESEARCH INSTRUCTIONS:
Please conduct thorough research on this topic using web search tools. Provide a detailed report that addresses the objectives while considering the constraints and target audience. Include relevant facts, statistics, recent developments, and credible sources.
"""
            print("\nUpdated research brief:")
            print(research_brief)
        else:
            print("Please enter 'yes', 'no', or 'edit'.")


def research_node(state: State) -> dict:
    """
    Research node that uses create_react_agent with Tavily search tool.
    Conducts detailed research based on the research brief.
    """
    print("\n" + "="*60)
    print("🔬 RESEARCH AGENT - RESEARCH PHASE")
    print("="*60)
    print("Starting detailed research using AI agent with web search capabilities...\n")
    
    # Create ReAct agent with Tavily search tool
    research_agent = create_react_agent(
        model=llm,
        tools=[tavily_search_tool],
        system_prompt=f"""You are a professional research assistant. Your task is to conduct comprehensive research based on the provided research brief.

Use the tavily_search_tool to gather information from the web. Perform multiple searches with different keywords and angles to ensure comprehensive coverage.

Research Brief:
{state.get('research_brief', 'No research brief provided')}

Instructions:
1. Analyze the research brief carefully
2. Identify key search terms and concepts
3. Perform multiple targeted searches
4. Synthesize the information into a comprehensive report
5. Include credible sources and recent information
6. Structure your response clearly with headings and bullet points
7. Provide actionable insights when relevant

Your final response should be a detailed research report that fully addresses the research objectives."""
    )
    
    # Create the research query based on the brief
    research_query = f"Please conduct comprehensive research based on this brief: {state.get('research_brief', 'General research request')}"
    
    # Execute the research agent
    try:
        result = research_agent.invoke({
            "messages": [HumanMessage(content=research_query)]
        })
        
        # Extract the final response
        final_response = result["messages"][-1].content
        
        print("✅ Research completed successfully!")
        print("\n" + "="*60)
        print("📊 RESEARCH REPORT")
        print("="*60)
        print(final_response)
        print("\n" + "="*60)
        
        return {
            "messages": [AIMessage(content=final_response)]
        }
        
    except Exception as e:
        error_message = f"Error during research: {str(e)}"
        print(f"❌ {error_message}")
        return {
            "messages": [AIMessage(content=error_message)]
        }


def route_research(state: State) -> Literal["research_node", END]:
    """
    Routing function that determines whether to proceed to research or end.
    """
    if state.get("research_complete", False) and state.get("research_brief"):
        return "research_node"
    else:
        return END


# Create the StateGraph
graph_builder = StateGraph(State)

# Add nodes
graph_builder.add_node("clarification_node", clarification_node)
graph_builder.add_node("research_node", research_node)

# Add edges
graph_builder.add_edge(START, "clarification_node")
graph_builder.add_conditional_edges(
    "clarification_node",
    route_research,
    {
        "research_node": "research_node",
        END: END
    }
)
graph_builder.add_edge("research_node", END)

# Compile the graph
app = graph_builder.compile()


if __name__ == "__main__":
    # Example usage
    print("🚀 Starting LangGraph Deep Research Agent...")
    
    initial_state = {
        "messages": [HumanMessage("I need help with research.")],
        "research_brief": None,
        "research_complete": False
    }
    
    try:
        result = app.invoke(initial_state)
        print("\n🎉 Research agent execution completed!")
    except KeyboardInterrupt:
        print("\n\n⏹️ Research agent interrupted by user.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
