import os
from typing import TypedDict, Annotated, Sequence, Literal
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv

load_dotenv()


class ResearchState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    research_brief: str
    clarification_complete: bool
    research_report: str


def clarification_agent(state: ResearchState) -> ResearchState:
    """Interactive agent that clarifies research scope with the user"""
    messages = state["messages"]
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    # Check if this is the initial request or a follow-up
    if not state.get("research_brief"):
        # Initial clarification
        system_prompt = """You are a research assistant helping to clarify the scope of a research task.
        Your goal is to understand:
        1. The main topic or question to research
        2. The specific aspects or subtopics to focus on
        3. Any constraints or requirements for the research
        4. The desired depth and breadth of the research
        
        Ask clarifying questions to ensure you understand the research scope fully.
        When you have enough information, create a clear research brief and indicate completion."""
        
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            *messages
        ])
        
        return {
            "messages": [response],
            "clarification_complete": False
        }
    else:
        # Process user's response to our clarification questions
        last_user_message = messages[-1].content if messages else ""
        
        # Check if user wants to proceed or needs more clarification
        check_prompt = """Based on the conversation so far, determine if:
        1. The user has provided enough information to proceed with research
        2. The user explicitly wants to proceed (e.g., said "yes", "proceed", "that's all", "go ahead")
        3. More clarification is needed
        
        If ready to proceed, create a comprehensive research brief.
        If not ready, ask follow-up questions."""
        
        response = llm.invoke([
            SystemMessage(content=check_prompt),
            *messages
        ])
        
        # Check if we should proceed to research
        completion_check = llm.invoke([
            SystemMessage(content="Does the following response indicate the research scope is clear and we should proceed? Answer only 'yes' or 'no'."),
            HumanMessage(content=response.content)
        ])
        
        should_proceed = "yes" in completion_check.content.lower()
        
        if should_proceed:
            # Generate the research brief
            brief_prompt = """Based on the entire conversation, create a comprehensive research brief that includes:
            1. Main research topic/question
            2. Key aspects to investigate
            3. Specific requirements or constraints
            4. Expected depth of research
            
            Format this as a clear, actionable brief for the research phase."""
            
            brief_response = llm.invoke([
                SystemMessage(content=brief_prompt),
                *messages
            ])
            
            return {
                "messages": [AIMessage(content="Research brief created. Proceeding with research...")],
                "research_brief": brief_response.content,
                "clarification_complete": True
            }
        else:
            return {
                "messages": [response],
                "clarification_complete": False
            }


def research_agent(state: ResearchState) -> ResearchState:
    """ReAct agent that conducts research using Tavily search"""
    research_brief = state.get("research_brief", "")
    
    if not research_brief:
        return {
            "messages": [AIMessage(content="No research brief provided. Please clarify the research scope first.")],
            "research_report": ""
        }
    
    # Create Tavily search tool
    tavily_tool = TavilySearchResults(
        max_results=5,
        search_depth="advanced",
        include_answer=True,
        include_raw_content=False,
        include_images=False
    )
    
    # Create ReAct agent with Tavily tool
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    
    # Create a custom ReAct agent
    tools = [tavily_tool]
    react_agent = create_react_agent(llm, tools)
    
    # Prepare the research prompt
    research_prompt = f"""You are a research agent conducting deep research based on the following brief:

    {research_brief}
    
    Your task:
    1. Use the Tavily search tool to gather comprehensive information
    2. Search for multiple relevant sources and perspectives
    3. Synthesize the information into a detailed, well-structured report
    4. Include citations and sources where appropriate
    5. Ensure the report addresses all aspects mentioned in the brief
    
    Conduct thorough research and create a comprehensive report."""
    
    # Run the ReAct agent
    result = react_agent.invoke({
        "messages": [HumanMessage(content=research_prompt)]
    })
    
    # Extract the final report from the agent's response
    final_message = result["messages"][-1].content if result["messages"] else "No research conducted."
    
    return {
        "messages": [AIMessage(content=f"Research completed. Here is the detailed report:\n\n{final_message}")],
        "research_report": final_message
    }


def should_continue_clarification(state: ResearchState) -> Literal["clarification", "research", END]:
    """Determine next step based on clarification status"""
    if state.get("clarification_complete", False):
        if state.get("research_report"):
            return END
        return "research"
    return "clarification"


# Build the graph
def create_research_graph():
    graph = StateGraph(ResearchState)
    
    # Add nodes
    graph.add_node("clarification", clarification_agent)
    graph.add_node("research", research_agent)
    
    # Set entry point
    graph.set_entry_point("clarification")
    
    # Add conditional edges
    graph.add_conditional_edges(
        "clarification",
        should_continue_clarification,
        {
            "clarification": "clarification",
            "research": "research",
            END: END
        }
    )
    
    # Research always ends
    graph.add_edge("research", END)
    
    return graph.compile()


# Export the compiled graph as 'app'
app = create_research_graph()