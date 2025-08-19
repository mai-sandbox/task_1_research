"""
LangGraph Deep Research Agent with User Interaction and ReAct Search

This agent implements a two-phase research workflow:
1. User Interaction Phase: Clarifies research scope through terminal conversation
2. ReAct Research Phase: Conducts research using Tavily search and generates detailed reports
"""

import os
from typing import TypedDict, Literal, Annotated
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from tavily import TavilyClient
from langchain_core.tools import tool


# State Schema
class ResearchState(TypedDict):
    """State schema for the research agent workflow"""
    messages: Annotated[list, add_messages]
    research_brief: str
    final_report: str
    interaction_complete: bool


# Tavily Search Tool
@tool
def tavily_search(query: str) -> str:
    """
    Search the web using Tavily API for comprehensive research results.
    
    Args:
        query: The search query string
        
    Returns:
        Formatted search results with URLs, content, and titles
    """
    try:
        # Initialize Tavily client - API key should be set as environment variable
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            return "Error: TAVILY_API_KEY environment variable not set. Please set your Tavily API key."
        
        client = TavilyClient(api_key=api_key)
        
        # Perform search
        response = client.search(
            query=query,
            search_depth="advanced",
            max_results=5,
            include_answer=True,
            include_raw_content=True
        )
        
        # Format results
        formatted_results = f"Search Query: {query}\n\n"
        
        if response.get("answer"):
            formatted_results += f"Quick Answer: {response['answer']}\n\n"
        
        formatted_results += "Detailed Results:\n"
        for i, result in enumerate(response.get("results", []), 1):
            formatted_results += f"\n{i}. {result.get('title', 'No title')}\n"
            formatted_results += f"   URL: {result.get('url', 'No URL')}\n"
            formatted_results += f"   Content: {result.get('content', 'No content')[:500]}...\n"
        
        return formatted_results
        
    except Exception as e:
        return f"Error performing search: {str(e)}"


def user_interaction_node(state: ResearchState) -> ResearchState:
    """
    Node for interactive research scope clarification with the user.
    Conducts back-and-forth conversation until research scope is clear.
    """
    print("\n" + "="*60)
    print("🔍 RESEARCH SCOPE CLARIFICATION")
    print("="*60)
    
    # Check if this is the first interaction
    if not state.get("research_brief"):
        print("\nWelcome to the Deep Research Agent!")
        print("I'll help you conduct comprehensive research on any topic.")
        print("Let's start by clarifying what you'd like to research.\n")
        
        # Get initial research topic
        topic = input("What topic would you like me to research? ")
        
        # Initialize conversation
        messages = [
            SystemMessage(content="You are a research assistant helping to clarify the scope of a research project. Ask follow-up questions to understand the specific aspects, depth, and focus areas the user wants to explore."),
            HumanMessage(content=f"I want to research: {topic}")
        ]
        
        # Use a simple LLM for conversation (fallback to basic interaction if no API key)
        try:
            llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.1)
            ai_response = llm.invoke(messages)
            print(f"\nResearch Assistant: {ai_response.content}")
            messages.append(ai_response)
        except Exception as e:
            print(f"\nResearch Assistant: I'd like to help you refine your research on '{topic}'. Could you tell me:")
            print("1. What specific aspects are you most interested in?")
            print("2. What's the purpose of this research?")
            print("3. Are there any particular angles or perspectives you want to focus on?")
            messages.append(AIMessage(content="Let me ask some clarifying questions to better understand your research needs."))
    else:
        messages = state["messages"]
    
    # Continue conversation until user is satisfied
    while True:
        print("\n" + "-"*40)
        user_input = input("\nYour response (or type 'done' when ready to proceed): ")
        
        if user_input.lower() in ['done', 'proceed', 'ready', 'go']:
            # Generate research brief from conversation
            conversation_summary = "\n".join([
                f"{'User' if isinstance(msg, HumanMessage) else 'Assistant'}: {msg.content}"
                for msg in messages if isinstance(msg, (HumanMessage, AIMessage))
            ])
            
            research_brief = f"""
RESEARCH BRIEF:
Based on our conversation, here's the research scope:

{conversation_summary}

RESEARCH OBJECTIVES:
- Conduct comprehensive research on the specified topic
- Gather information from multiple reliable sources
- Provide detailed analysis and insights
- Present findings in a structured report format
"""
            
            print(f"\n✅ Research brief generated! Moving to research phase...")
            print(f"Brief: {research_brief[:200]}...")
            
            return {
                "messages": messages + [HumanMessage(content=user_input)],
                "research_brief": research_brief,
                "interaction_complete": True
            }
        
        # Add user input to conversation
        messages.append(HumanMessage(content=user_input))
        
        # Generate AI response for continued clarification
        try:
            llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.1)
            ai_response = llm.invoke(messages + [
                SystemMessage(content="Continue helping the user clarify their research scope. Ask relevant follow-up questions or acknowledge when the scope seems clear.")
            ])
            print(f"\nResearch Assistant: {ai_response.content}")
            messages.append(ai_response)
        except Exception as e:
            print(f"\nResearch Assistant: Thank you for that information. Is there anything else you'd like to specify about your research focus?")
            messages.append(AIMessage(content="Thank you for the additional details."))


def react_research_node(state: ResearchState) -> ResearchState:
    """
    Node that uses ReAct agent with Tavily search to conduct research and generate reports.
    """
    print("\n" + "="*60)
    print("🔬 CONDUCTING RESEARCH")
    print("="*60)
    
    research_brief = state["research_brief"]
    print(f"\nResearch Brief: {research_brief[:200]}...")
    
    try:
        # Create ReAct agent with Tavily search tool
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.1)
        
        # Create the ReAct agent
        react_agent = create_react_agent(
            model=llm,
            tools=[tavily_search],
            system_prompt="""You are a professional research agent. Your task is to:

1. Analyze the research brief provided
2. Conduct comprehensive web searches using the tavily_search tool
3. Gather information from multiple sources
4. Synthesize findings into a detailed, well-structured report

Guidelines:
- Perform multiple searches with different query variations
- Look for diverse perspectives and sources
- Verify information across multiple sources when possible
- Organize findings logically
- Provide citations and source URLs
- Present a balanced and comprehensive analysis

Generate a detailed research report based on your findings."""
        )
        
        # Prepare research prompt
        research_prompt = f"""
Please conduct comprehensive research based on this brief:

{research_brief}

Use the tavily_search tool to gather information from multiple sources. Perform several searches with different query variations to ensure comprehensive coverage. Then synthesize your findings into a detailed research report.
"""
        
        print("\n🔍 Starting research process...")
        
        # Execute the ReAct agent
        result = react_agent.invoke({
            "messages": [HumanMessage(content=research_prompt)]
        })
        
        # Extract the final report from the agent's response
        final_message = result["messages"][-1]
        final_report = final_message.content
        
        print(f"\n✅ Research completed! Generated {len(final_report)} character report.")
        
        return {
            "messages": state["messages"] + result["messages"],
            "research_brief": research_brief,
            "final_report": final_report,
            "interaction_complete": True
        }
        
    except Exception as e:
        error_report = f"""
RESEARCH ERROR REPORT

An error occurred during the research process: {str(e)}

This might be due to:
1. Missing OpenAI API key (OPENAI_API_KEY environment variable)
2. Missing Tavily API key (TAVILY_API_KEY environment variable)
3. Network connectivity issues
4. API rate limits

Please ensure you have set up the required API keys:
- OPENAI_API_KEY for the language model
- TAVILY_API_KEY for web search functionality

Research Brief: {research_brief}
"""
        
        print(f"\n❌ Research failed: {str(e)}")
        
        return {
            "messages": state["messages"] + [AIMessage(content=error_report)],
            "research_brief": research_brief,
            "final_report": error_report,
            "interaction_complete": True
        }


def should_continue_interaction(state: ResearchState) -> Literal["react_research", "user_interaction"]:
    """
    Conditional edge function to determine whether to continue user interaction or proceed to research.
    """
    if state.get("interaction_complete", False):
        return "react_research"
    else:
        return "user_interaction"


def display_final_report(state: ResearchState) -> ResearchState:
    """
    Final node to display the research report to the user.
    """
    print("\n" + "="*60)
    print("📋 FINAL RESEARCH REPORT")
    print("="*60)
    
    final_report = state.get("final_report", "No report generated.")
    
    print(f"\n{final_report}")
    
    print("\n" + "="*60)
    print("🎉 Research Complete!")
    print("="*60)
    
    return state


# Build the StateGraph
def create_research_agent():
    """Create and compile the research agent graph."""
    
    # Initialize the StateGraph
    workflow = StateGraph(ResearchState)
    
    # Add nodes
    workflow.add_node("user_interaction", user_interaction_node)
    workflow.add_node("react_research", react_research_node)
    workflow.add_node("display_report", display_final_report)
    
    # Add edges
    workflow.add_edge(START, "user_interaction")
    workflow.add_conditional_edges(
        "user_interaction",
        should_continue_interaction,
        {
            "user_interaction": "user_interaction",  # Continue interaction
            "react_research": "react_research"       # Proceed to research
        }
    )
    workflow.add_edge("react_research", "display_report")
    workflow.add_edge("display_report", END)
    
    # Compile the graph
    return workflow.compile()


# Export the compiled graph as 'app' (required by LangGraph conventions)
app = create_research_agent()


# Main execution function for testing
def main():
    """Main function to run the research agent."""
    print("🚀 Starting Deep Research Agent...")
    
    # Initial state
    initial_state = {
        "messages": [],
        "research_brief": "",
        "final_report": "",
        "interaction_complete": False
    }
    
    # Run the agent
    try:
        final_state = app.invoke(initial_state)
        return final_state
    except KeyboardInterrupt:
        print("\n\n👋 Research session interrupted by user.")
        return None
    except Exception as e:
        print(f"\n❌ Error running research agent: {str(e)}")
        return None


if __name__ == "__main__":
    main()
