from typing import Annotated, TypedDict, List, Optional
from langgraph import StateGraph, END
from langgraph.graph import MessagesState
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import Tool
import json
import os


class ResearchState(MessagesState):
    research_brief: Optional[str] = None
    research_status: str = "collecting_requirements"  # collecting_requirements, researching, completed
    detailed_report: Optional[str] = None


def clarification_node(state: ResearchState):
    """Interactive node to clarify research scope with the user"""
    messages = state["messages"]
    last_message = messages[-1]
    
    if state["research_status"] == "collecting_requirements":
        if isinstance(last_message, HumanMessage):
            # First interaction - ask clarifying questions
            if len(messages) == 1:
                clarifying_questions = """I'll help you conduct deep research. To provide the most valuable insights, I need to understand your requirements better:

1. What is the main topic or question you want me to research?
2. What specific aspects are most important to you?
3. What is the intended use of this research (academic, business, personal, etc.)?
4. Are there any particular sources or perspectives you'd like me to focus on or avoid?
5. What level of detail do you need in the final report?

Please provide as much detail as possible about your research needs."""
                
                return {
                    "messages": messages + [AIMessage(content=clarifying_questions)],
                    "research_status": "collecting_requirements"
                }
            
            # Continue clarification until we have enough information
            user_response = last_message.content.lower()
            
            # Simple heuristic to determine if we have enough information
            if any(word in user_response for word in ["yes", "good", "enough", "proceed", "start", "ready"]) and len(messages) > 3:
                # Generate research brief
                brief_prompt = f"""Based on the conversation, create a detailed research brief. Extract:
- Main research topic/question
- Key areas to investigate
- Research scope and depth required
- Target audience/use case

Conversation history: {[msg.content for msg in messages if isinstance(msg, HumanMessage)]}

Create a structured research brief that will guide the research agent."""
                
                llm = ChatOpenAI(model="gpt-4", temperature=0)
                brief_response = llm.invoke([SystemMessage(content=brief_prompt)])
                
                return {
                    "messages": messages + [AIMessage(content="Perfect! I now have enough information to begin the research. Let me compile a detailed research brief and start the investigation.")],
                    "research_brief": brief_response.content,
                    "research_status": "researching"
                }
            
            # Ask for more specific information
            follow_up = """Thank you for that information. I need a bit more detail to ensure I conduct the most effective research:

- Could you be more specific about what exactly you want to learn?
- Are there particular aspects or angles you're most interested in?
- What will you use this research for?

Once I have these details, I'll begin the comprehensive research process."""
            
            return {
                "messages": messages + [AIMessage(content=follow_up)],
                "research_status": "collecting_requirements"
            }
    
    return state


def research_node(state: ResearchState):
    """ReAct agent node that conducts research using Tavily"""
    if state["research_status"] != "researching":
        return state
    
    research_brief = state["research_brief"]
    
    # Initialize Tavily search tool
    tavily_tool = TavilySearchResults(
        max_results=5,
        search_depth="advanced",
        include_answer=True,
        include_raw_content=True
    )
    
    # Create ReAct agent with search capability
    llm = ChatOpenAI(model="gpt-4", temperature=0.1)
    
    react_prompt = f"""You are a research agent conducting comprehensive research based on this brief:

{research_brief}

Your task is to:
1. Break down the research into key areas to investigate
2. Use the search tool to gather comprehensive information
3. Analyze and synthesize findings
4. Identify key insights, trends, and important details
5. Create a detailed, well-structured report

Think step by step and use the search tool multiple times to gather thorough information from different angles.

Available tool: tavily_search - Use this to search for information on the internet.
"""
    
    # Simple ReAct implementation
    research_messages = [SystemMessage(content=react_prompt)]
    search_queries = []
    search_results = []
    
    # Generate initial search queries based on research brief
    query_generation_prompt = f"""Based on this research brief, generate 3-5 specific search queries that will help gather comprehensive information:

{research_brief}

Return only the search queries, one per line."""
    
    query_response = llm.invoke([SystemMessage(content=query_generation_prompt)])
    queries = [q.strip() for q in query_response.content.split('\n') if q.strip()]
    
    # Conduct searches
    for query in queries:
        try:
            results = tavily_tool.invoke({"query": query})
            search_queries.append(query)
            search_results.append(results)
        except Exception as e:
            print(f"Search error for query '{query}': {e}")
    
    # Generate comprehensive report
    report_prompt = f"""Based on the research brief and search results, create a comprehensive research report.

Research Brief:
{research_brief}

Search Queries and Results:
"""
    
    for i, (query, results) in enumerate(zip(search_queries, search_results)):
        report_prompt += f"\n\nSearch Query {i+1}: {query}\nResults: {results}\n"
    
    report_prompt += """

Create a detailed report with:
1. Executive Summary
2. Key Findings
3. Detailed Analysis by topic area
4. Supporting Evidence
5. Conclusions and Implications
6. Sources and References

Make the report comprehensive, well-structured, and actionable."""
    
    report_response = llm.invoke([SystemMessage(content=report_prompt)])
    
    return {
        "messages": state["messages"] + [AIMessage(content="Research completed! Here's your detailed report:\n\n" + report_response.content)],
        "detailed_report": report_response.content,
        "research_status": "completed"
    }


def should_continue_clarification(state: ResearchState):
    """Router function to determine next step"""
    if state["research_status"] == "collecting_requirements":
        return "clarification"
    elif state["research_status"] == "researching":
        return "research"
    else:
        return END


# Create the graph
workflow = StateGraph(ResearchState)

# Add nodes
workflow.add_node("clarification", clarification_node)
workflow.add_node("research", research_node)

# Set entry point
workflow.set_entry_point("clarification")

# Add conditional routing
workflow.add_conditional_edges(
    "clarification",
    should_continue_clarification,
    {
        "clarification": "clarification",
        "research": "research",
        END: END
    }
)

workflow.add_edge("research", END)

# Compile the graph
app = workflow.compile()