"""
LangGraph Deep Research Agent
Two-phase agent: 
1. Clarification phase - interactive terminal conversation
2. Research phase - ReAct agent with Tavily search
"""

import os
import json
from typing import TypedDict, List, Annotated, Literal
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

# State definition
class ResearchState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    research_brief: str
    phase: Literal["clarification", "research", "complete"]
    clarification_complete: bool
    final_report: str

# Initialize models
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
research_llm = ChatOpenAI(model="gpt-4o", temperature=0)

# Initialize Tavily search tool
tavily_tool = TavilySearchResults(
    max_results=5,
    search_depth="advanced",
    include_answer=True,
    include_raw_content=True,
    include_images=False,
)

def clarification_node(state: ResearchState):
    """
    Handles the clarification phase - engages with user to understand research scope
    """
    messages = state["messages"]
    
    # Check if this is the first interaction
    if len(messages) == 1:
        # Initial greeting and request for clarification
        response = llm.invoke([
            SystemMessage(content="""You are a research assistant helping to clarify the scope of a research project.
            Your goal is to understand:
            1. The main research topic/question
            2. Specific aspects or subtopics to focus on
            3. Any constraints or preferences
            4. The desired depth and format of the final report
            
            Ask clarifying questions one at a time. Be conversational but focused.
            When you have enough information to create a clear research brief, respond with:
            "I have all the information I need. Let me create your research brief."
            
            Start by greeting the user and asking about their research topic."""),
            *messages
        ])
        
        return {
            "messages": [response],
            "phase": "clarification",
            "clarification_complete": False
        }
    
    # Continue clarification conversation
    last_human_message = None
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            last_human_message = msg
            break
    
    # Check if user wants to proceed or needs more clarification
    check_complete = llm.invoke([
        SystemMessage(content="""Based on the conversation so far, determine if you have enough information 
        to create a comprehensive research brief. You should have:
        - Clear understanding of the main topic
        - Specific areas to investigate
        - Any constraints or preferences
        
        If you have enough information, respond with exactly: "RESEARCH_READY"
        If you need more clarification, respond with exactly: "NEED_MORE_INFO" """),
        *messages
    ])
    
    if "RESEARCH_READY" in check_complete.content:
        # Generate research brief
        brief_response = llm.invoke([
            SystemMessage(content="""Based on the conversation, create a detailed research brief that will guide 
            the research phase. Include:
            - Main research question/topic
            - Key areas to investigate
            - Specific subtopics or angles
            - Any constraints mentioned
            - Expected depth and format
            
            Format this as a clear, actionable brief for conducting research."""),
            *messages
        ])
        
        confirmation_message = AIMessage(content=f"""Perfect! I have all the information I need. Here's the research brief I'll be working with:

{brief_response.content}

Now proceeding to conduct the research...""")
        
        return {
            "messages": [confirmation_message],
            "research_brief": brief_response.content,
            "phase": "research",
            "clarification_complete": True
        }
    else:
        # Continue clarification
        response = llm.invoke([
            SystemMessage(content="""Continue the clarification conversation. Ask focused questions to understand:
            - The main research topic/question
            - Specific aspects or subtopics to focus on
            - Any constraints or preferences
            - The desired depth and format
            
            Ask one clear question at a time. When you have enough information, let the user know."""),
            *messages
        ])
        
        return {
            "messages": [response],
            "phase": "clarification",
            "clarification_complete": False
        }

def research_node(state: ResearchState):
    """
    Conducts the actual research using ReAct agent with Tavily search
    """
    research_brief = state["research_brief"]
    
    # Create a ReAct agent with Tavily tool
    react_agent = create_react_agent(
        research_llm,
        tools=[tavily_tool],
        state_modifier=SystemMessage(content=f"""You are a thorough research agent. 
        Your task is to conduct comprehensive research based on this brief:
        
        {research_brief}
        
        Guidelines:
        - Search for multiple perspectives and sources
        - Verify information across sources
        - Look for recent and authoritative information
        - Cover all aspects mentioned in the brief
        - Be thorough but focused
        
        Use the search tool multiple times to gather comprehensive information.
        Think step by step about what to search for to fully address the research brief.""")
    )
    
    # Execute research
    research_messages = [
        HumanMessage(content=f"""Please conduct thorough research based on this brief:

{research_brief}

Search for relevant information, explore different aspects, and gather comprehensive data.
After gathering sufficient information, synthesize your findings into a detailed report.""")
    ]
    
    research_result = react_agent.invoke({"messages": research_messages})
    
    # Generate final report
    final_report_response = research_llm.invoke([
        SystemMessage(content="""You are a research report writer. Based on the research conducted, 
        create a comprehensive, well-structured report. Include:
        
        1. Executive Summary
        2. Main Findings (organized by topic/theme)
        3. Detailed Analysis
        4. Key Insights and Patterns
        5. Conclusions
        6. Sources and References
        
        Make it detailed, informative, and well-organized. Use clear headings and bullet points where appropriate."""),
        *research_result["messages"]
    ])
    
    return {
        "messages": [AIMessage(content=f"Research completed! Here's your comprehensive report:\n\n{final_report_response.content}")],
        "final_report": final_report_response.content,
        "phase": "complete"
    }

def should_continue_clarification(state: ResearchState):
    """
    Determines whether to continue clarification or move to research
    """
    if state.get("clarification_complete", False):
        return "research"
    return "clarification"

def should_end(state: ResearchState):
    """
    Determines whether the workflow should end
    """
    if state.get("phase") == "complete":
        return END
    return "continue"

# Build the graph
workflow = StateGraph(ResearchState)

# Add nodes
workflow.add_node("clarification", clarification_node)
workflow.add_node("research", research_node)

# Set entry point
workflow.set_entry_point("clarification")

# Add edges
workflow.add_conditional_edges(
    "clarification",
    should_continue_clarification,
    {
        "clarification": "clarification",
        "research": "research"
    }
)

workflow.add_conditional_edges(
    "research",
    should_end,
    {
        END: END,
        "continue": "research"
    }
)

# Compile with memory for conversation state
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)