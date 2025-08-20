import os
from typing import Annotated, Literal, TypedDict
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from dotenv import load_dotenv

load_dotenv()


class ResearchState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    research_brief: str
    final_report: str
    clarification_complete: bool


def clarification_agent(state: ResearchState) -> ResearchState:
    """Agent that clarifies research scope with the user."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    messages = state["messages"]
    
    if not state.get("clarification_complete", False):
        system_prompt = SystemMessage(content="""You are a research assistant helping to clarify the scope of a research task.
        Your goal is to have a back-and-forth conversation with the user to understand:
        1. The main research topic/question
        2. Specific aspects they want to focus on
        3. Any constraints or requirements
        4. The desired depth and breadth of research
        
        Ask clarifying questions one at a time. When you have enough information to create a comprehensive research brief,
        respond with "CLARIFICATION_COMPLETE:" followed by a detailed research brief that the research agent can use.
        
        The brief should include:
        - Main research question
        - Key topics to investigate
        - Specific areas of focus
        - Any constraints or requirements mentioned by the user""")
        
        response = llm.invoke([system_prompt] + messages)
        
        if "CLARIFICATION_COMPLETE:" in response.content:
            brief = response.content.split("CLARIFICATION_COMPLETE:")[1].strip()
            return {
                "messages": [response],
                "research_brief": brief,
                "clarification_complete": True
            }
        else:
            return {
                "messages": [response],
                "clarification_complete": False
            }
    
    return state


def research_agent(state: ResearchState) -> ResearchState:
    """ReAct agent that performs research using Tavily search."""
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    
    tavily_api_key = os.getenv("TAVILY_API_KEY")
    if not tavily_api_key:
        raise ValueError("TAVILY_API_KEY environment variable is required")
    
    search_tool = TavilySearchResults(
        max_results=5,
        api_key=tavily_api_key
    )
    
    llm_with_tools = llm.bind_tools([search_tool])
    
    research_brief = state.get("research_brief", "")
    
    system_prompt = SystemMessage(content=f"""You are a deep research agent. Based on the following research brief, 
    conduct thorough research using the search tool available to you.
    
    Research Brief:
    {research_brief}
    
    Instructions:
    1. Break down the research into logical search queries
    2. Use the search tool multiple times to gather comprehensive information
    3. Analyze and synthesize the information
    4. Create a detailed, well-structured report
    
    Your goal is to provide a comprehensive research report that addresses all aspects of the brief.""")
    
    research_messages = [system_prompt, HumanMessage(content="Please conduct the research based on the brief provided.")]
    
    max_iterations = 10
    for _ in range(max_iterations):
        response = llm_with_tools.invoke(research_messages)
        research_messages.append(response)
        
        if response.tool_calls:
            for tool_call in response.tool_calls:
                if tool_call["name"] == "tavily_search":
                    results = search_tool.invoke(tool_call["args"])
                    research_messages.append(
                        AIMessage(
                            content=str(results),
                            name="tavily_search"
                        )
                    )
        else:
            break
    
    report_prompt = SystemMessage(content="""Based on all the research conducted, create a comprehensive, 
    well-structured report. The report should:
    1. Have a clear executive summary
    2. Be organized into logical sections
    3. Include key findings and insights
    4. Cite sources where appropriate
    5. Provide actionable conclusions or recommendations if relevant""")
    
    final_report = llm.invoke(research_messages + [report_prompt])
    
    return {
        "messages": state["messages"] + [final_report],
        "final_report": final_report.content
    }


def should_continue_clarification(state: ResearchState) -> Literal["clarify", "research"]:
    """Determine if clarification is complete."""
    if state.get("clarification_complete", False):
        return "research"
    return "clarify"


def build_graph():
    """Build the main orchestrator graph."""
    workflow = StateGraph(ResearchState)
    
    workflow.add_node("clarify", clarification_agent)
    workflow.add_node("research", research_agent)
    
    workflow.set_entry_point("clarify")
    
    workflow.add_conditional_edges(
        "clarify",
        should_continue_clarification,
        {
            "clarify": "clarify",
            "research": "research"
        }
    )
    
    workflow.add_edge("research", END)
    
    return workflow.compile()


app = build_graph()