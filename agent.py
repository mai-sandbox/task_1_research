"""
LangGraph Deep Research Agent with Interactive Scoping and ReAct Search

This agent implements a two-phase workflow:
1. Interactive scoping phase: Engages with user to clarify research scope
2. ReAct research phase: Uses Tavily search tool to conduct research and generate detailed report
"""

import os
from typing import TypedDict, List, Annotated, Literal
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from tavily import TavilyClient

# Load environment variables
load_dotenv()

class ResearchState(TypedDict):
    """State for the research agent containing messages and research brief."""
    messages: Annotated[List, add_messages]
    research_brief: str
    phase: Literal["scoping", "research", "complete"]

def create_tavily_search_tool():
    """Create a Tavily search tool for the ReAct agent."""
    tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    
    def tavily_search(query: str) -> str:
        """Search the web using Tavily API.
        
        Args:
            query: The search query to execute
            
        Returns:
            Formatted search results with sources
        """
        try:
            response = tavily_client.search(
                query=query,
                search_depth="advanced",
                include_answer=True,
                max_results=5
            )
            
            # Format the response
            result = f"**Search Query:** {response['query']}\n\n"
            
            if response.get('answer'):
                result += f"**Summary:** {response['answer']}\n\n"
            
            result += "**Sources:**\n"
            for i, source in enumerate(response.get('results', []), 1):
                result += f"{i}. **{source['title']}**\n"
                result += f"   URL: {source['url']}\n"
                result += f"   Content: {source['content']}\n"
                result += f"   Relevance Score: {source.get('score', 'N/A')}\n\n"
            
            return result
            
        except Exception as e:
            return f"Error performing search: {str(e)}"
    
    return tavily_search

def scoping_node(state: ResearchState) -> ResearchState:
    """Interactive scoping node that clarifies research requirements with the user."""
    messages = state["messages"]
    last_message = messages[-1] if messages else None
    
    # Check if this is the initial request
    if not state.get("research_brief") and isinstance(last_message, HumanMessage):
        # Start the scoping conversation
        scoping_questions = [
            "I'll help you conduct thorough research! To provide the best results, I need to understand your research scope better.",
            "",
            "Please help me clarify:",
            "1. What is the main topic or question you want me to research?",
            "2. What specific aspects are you most interested in?",
            "3. Are there any particular time periods, regions, or perspectives you want me to focus on?",
            "4. What level of detail do you need (overview, in-depth analysis, specific data points)?",
            "5. Are there any sources or types of information you want me to prioritize or avoid?",
            "",
            "Please provide as much detail as possible, or answer the questions that are most relevant to your research needs."
        ]
        
        response = AIMessage(content="\n".join(scoping_questions))
        return {
            **state,
            "messages": messages + [response],
            "phase": "scoping"
        }
    
    # If we have a research brief, we're done with scoping
    if state.get("research_brief"):
        return state
    
    # Process user's scoping response
    if isinstance(last_message, HumanMessage) and state.get("phase") == "scoping":
        user_input = last_message.content
        
        # Create a research brief based on user input
        brief_prompt = f"""
        Based on the user's input, create a comprehensive research brief:
        
        User Input: {user_input}
        
        Please confirm if this research brief captures your needs:
        
        **Research Brief:**
        - Main Topic: [Extract the main research topic]
        - Key Questions: [List 3-5 specific questions to investigate]
        - Scope: [Define the boundaries and focus areas]
        - Expected Deliverables: [What kind of report/analysis is needed]
        
        If this looks good, please respond with "PROCEED" to start the research.
        If you'd like to modify anything, please let me know what changes you'd like.
        """
        
        # For now, create a simple brief - in a real implementation, you'd use an LLM here
        research_brief = f"""
        **Research Brief:**
        - Main Topic: {user_input[:200]}...
        - Key Questions: To be determined through comprehensive research
        - Scope: Broad investigation covering multiple perspectives
        - Expected Deliverables: Detailed research report with sources and analysis
        """
        
        confirmation_message = AIMessage(content=f"""
        {research_brief}
        
        Does this research brief capture what you're looking for? 
        
        Please respond with:
        - "PROCEED" if you're ready to start the research
        - Or provide any modifications you'd like me to make to the brief
        """)
        
        return {
            **state,
            "messages": messages + [confirmation_message],
            "research_brief": research_brief,
            "phase": "scoping"
        }
    
    return state

def should_proceed_to_research(state: ResearchState) -> Literal["research", "scoping"]:
    """Determine if we should proceed to research phase or continue scoping."""
    messages = state["messages"]
    last_message = messages[-1] if messages else None
    
    # Check if user confirmed to proceed
    if (isinstance(last_message, HumanMessage) and 
        "PROCEED" in last_message.content.upper() and 
        state.get("research_brief")):
        return "research"
    
    # If we don't have a research brief yet, stay in scoping
    if not state.get("research_brief"):
        return "scoping"
    
    # If we have a brief but no confirmation, stay in scoping
    return "scoping"

def research_node(state: ResearchState) -> ResearchState:
    """Research node that uses ReAct agent with Tavily search to conduct research."""
    messages = state["messages"]
    research_brief = state.get("research_brief", "")
    
    # Create the Tavily search tool
    tavily_search = create_tavily_search_tool()
    
    # Create a ReAct agent with Tavily search tool
    model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")  # Default model
    
    try:
        # Create the ReAct agent
        react_agent = create_react_agent(
            model=model_name,
            tools=[tavily_search],
            state_modifier=SystemMessage(content=f"""
            You are a thorough research assistant. Your task is to conduct comprehensive research based on the following brief:
            
            {research_brief}
            
            Please:
            1. Break down the research into logical components
            2. Use the tavily_search tool to gather information from multiple sources
            3. Synthesize the findings into a comprehensive report
            4. Include proper citations and source references
            5. Provide analysis and insights, not just raw information
            6. Structure your final report clearly with headings and sections
            
            Be thorough and methodical in your research approach.
            """)
        )
        
        # Extract the user's original research request
        original_request = None
        for msg in messages:
            if isinstance(msg, HumanMessage) and "PROCEED" not in msg.content.upper():
                original_request = msg.content
                break
        
        if not original_request:
            original_request = "Please conduct research based on the provided brief."
        
        # Run the ReAct agent
        research_messages = [HumanMessage(content=f"Please conduct comprehensive research on: {original_request}")]
        result = react_agent.invoke({"messages": research_messages})
        
        # Extract the final response
        final_response = result["messages"][-1].content if result["messages"] else "Research completed but no response generated."
        
        completion_message = AIMessage(content=f"""
        # Research Complete
        
        {final_response}
        
        ---
        
        **Research Brief Used:**
        {research_brief}
        
        This concludes your research request. If you need additional information or want to explore any aspect in more detail, please let me know!
        """)
        
        return {
            **state,
            "messages": messages + [completion_message],
            "phase": "complete"
        }
        
    except Exception as e:
        error_message = AIMessage(content=f"""
        I encountered an error while conducting the research: {str(e)}
        
        This might be due to:
        1. Missing API keys (TAVILY_API_KEY, OPENAI_API_KEY)
        2. Network connectivity issues
        3. API rate limits
        
        Please check your environment variables and try again.
        """)
        
        return {
            **state,
            "messages": messages + [error_message],
            "phase": "complete"
        }

# Create the workflow graph
def create_research_workflow():
    """Create the LangGraph workflow for the research agent."""
    workflow = StateGraph(ResearchState)
    
    # Add nodes
    workflow.add_node("scoping", scoping_node)
    workflow.add_node("research", research_node)
    
    # Add edges
    workflow.add_edge(START, "scoping")
    workflow.add_conditional_edges(
        "scoping",
        should_proceed_to_research,
        {
            "scoping": "scoping",
            "research": "research"
        }
    )
    workflow.add_edge("research", END)
    
    # Add memory for persistence
    memory = MemorySaver()
    
    return workflow.compile(checkpointer=memory)

# Create and export the compiled graph
app = create_research_workflow()

if __name__ == "__main__":
    # Test the agent locally
    print("LangGraph Deep Research Agent")
    print("=" * 40)
    
    # Check for required environment variables
    required_vars = ["TAVILY_API_KEY"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"Warning: Missing environment variables: {', '.join(missing_vars)}")
        print("Please set these variables in your .env file or environment.")
    
    # Example usage
    initial_state = {
        "messages": [HumanMessage("I want to research the impact of artificial intelligence on job markets.")],
        "research_brief": "",
        "phase": "scoping"
    }
    
    print("\nExample usage:")
    print("initial_state = {'messages': [HumanMessage('Your research topic here')]}")
    print("result = app.invoke(initial_state)")
