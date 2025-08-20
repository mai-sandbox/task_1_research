"""
LangGraph Deep Research Agent with User Interaction and ReAct Search

This agent operates in two phases:
1. Clarification Phase: Interactive conversation with user to clarify research scope
2. Research Phase: ReAct agent with Tavily search to conduct research and generate report
"""

import os
from typing import Annotated, TypedDict, Literal, List, Dict, Any

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_tavily import TavilySearchResults
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent, ToolNode
from langgraph.types import interrupt


# State definition
class ResearchState(TypedDict):
    """State schema for the research agent workflow"""
    messages: Annotated[List[BaseMessage], add_messages]
    research_brief: str
    clarification_complete: bool
    research_complete: bool
    final_report: str


def get_llm():
    """Get the LLM instance based on available API keys"""
    # Try OpenAI first
    if os.getenv("OPENAI_API_KEY"):
        return ChatOpenAI(model="gpt-4o", temperature=0.7)
    # Fallback to Anthropic
    elif os.getenv("ANTHROPIC_API_KEY"):
        return ChatAnthropic(model="claude-3-5-sonnet-20241022", temperature=0.7)
    else:
        raise ValueError(
            "No LLM API key found. Please set either OPENAI_API_KEY or ANTHROPIC_API_KEY"
        )


def format_messages_for_analysis(messages: List[BaseMessage]) -> str:
    """Format messages for analysis by the LLM"""
    formatted = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            formatted.append(f"User: {msg.content}")
        elif isinstance(msg, AIMessage):
            formatted.append(f"Assistant: {msg.content}")
    return "\n".join(formatted)


def clarification_agent(state: ResearchState) -> Dict[str, Any]:
    """
    Clarification agent node that interacts with user to clarify research scope.
    Uses interrupt() to pause execution and get user input.
    """
    messages = state.get("messages", [])
    research_brief = state.get("research_brief", "")
    clarification_complete = state.get("clarification_complete", False)
    
    # If clarification is already complete, pass through
    if clarification_complete:
        return {}
    
    # Get the LLM
    llm = get_llm()
    
    # If this is the first interaction or we only have the initial user message
    if len(messages) <= 1:
        initial_prompt = """Hello! I'm your research assistant. I'll help you conduct thorough research on any topic.

To ensure I provide the most relevant and comprehensive research, I'd like to clarify a few things about your research needs:

1. What specific topic or question would you like me to research?
2. What aspects of this topic are most important to you?
3. Are there any particular sources or types of information you prefer?
4. What's the intended use or audience for this research?

Please share your research topic and any specific requirements you have."""
        
        # Interrupt to get user input
        print("\n" + initial_prompt)
        user_response = interrupt({"query": "Please provide your research topic and requirements"})
        
        # Add the interaction to messages
        return {
            "messages": [
                AIMessage(content=initial_prompt),
                HumanMessage(content=user_response["data"])
            ]
        }
    
    # Analyze the conversation to determine if we have enough information
    # Check if we have enough information to proceed
    analysis_prompt = f"""Based on the following conversation, determine if we have enough information to create a comprehensive research brief.

Conversation:
{format_messages_for_analysis(messages)}

If we have enough information, respond with "READY" followed by a comprehensive research brief.
If we need more clarification, respond with "CLARIFY" followed by specific questions to ask.

Format your response as:
STATUS: [READY or CLARIFY]
CONTENT: [Research brief if READY, or follow-up questions if CLARIFY]"""
    
    analysis_response = llm.invoke([SystemMessage(content=analysis_prompt)])
    response_text = analysis_response.content
    
    # Parse the response
    lines = response_text.strip().split('\n')
    status = "CLARIFY"
    content = ""
    
    for line in lines:
        if line.startswith("STATUS:"):
            status = line.replace("STATUS:", "").strip()
        elif line.startswith("CONTENT:"):
            content = line.replace("CONTENT:", "").strip()
        elif content:
            content += "\n" + line
    
    if status == "READY":
        # We have enough information, create the research brief
        confirmation_message = f"""Great! Based on our conversation, I've prepared the following research brief:

{content}

I'll now proceed with conducting comprehensive research on this topic. This will involve:
1. Searching for relevant and authoritative sources
2. Analyzing and synthesizing the information
3. Creating a detailed research report

Type 'yes' to proceed with this research brief, or provide any modifications you'd like me to make."""
        
        # Get final confirmation
        print("\n" + confirmation_message)
        user_confirmation = interrupt({"query": "Confirm to proceed or provide modifications"})
        
        if user_confirmation["data"].lower().strip() in ["yes", "y", "proceed", "go ahead", "continue"]:
            return {
                "messages": [
                    AIMessage(content=confirmation_message),
                    HumanMessage(content=user_confirmation["data"])
                ],
                "research_brief": content,
                "clarification_complete": True
            }
        else:
            # User wants modifications
            return {
                "messages": [
                    AIMessage(content=confirmation_message),
                    HumanMessage(content=user_confirmation["data"])
                ]
            }
    else:
        # Need more clarification
        clarification_request = content if content else "Could you provide more details about your research needs?"
        
        # Interrupt to get more information
        print("\n" + clarification_request)
        user_response = interrupt({"query": "Please provide additional information"})
        
        return {
            "messages": [
                AIMessage(content=clarification_request),
                HumanMessage(content=user_response["data"])
            ]
        }


def research_agent_node(state: ResearchState) -> Dict[str, Any]:
    """
    Research agent node that uses create_react_agent with Tavily search
    to conduct research based on the brief and generate a detailed report.
    """
    research_brief = state.get("research_brief", "")
    research_complete = state.get("research_complete", False)
    
    # If research is already complete, pass through
    if research_complete:
        return {}
    
    if not research_brief:
        return {
            "messages": [AIMessage(content="No research brief provided. Please clarify your research needs first.")],
            "research_complete": False
        }
    
    print("\n🔍 Starting research phase...")
    print(f"Research Brief: {research_brief}\n")
    
    # Initialize Tavily search tool
    tavily_tool = TavilySearchResults(
        max_results=5,
        search_depth="advanced",
        include_answer=True,
        include_raw_content=False,
        include_images=False,
        name="tavily_search"
    )
    
    # Get the LLM
    llm = get_llm()
    
    # Create a research prompt
    research_prompt = f"""You are an expert research analyst. Your task is to conduct comprehensive research based on the following brief:

{research_brief}

Instructions:
1. Use the Tavily search tool to find relevant, authoritative, and recent information
2. Search for multiple perspectives and sources
3. Verify facts by cross-referencing multiple sources
4. Focus on credible and authoritative sources
5. Gather both general context and specific details

Conduct thorough research using multiple search queries to cover all aspects of the research brief.
After gathering sufficient information, synthesize your findings into a comprehensive report that includes:

1. **Executive Summary** - Brief overview of key findings
2. **Key Findings** - Main discoveries and insights
3. **Detailed Analysis** - In-depth examination of the topic
4. **Sources and References** - List of sources consulted
5. **Conclusions and Recommendations** - Final thoughts and actionable insights

Format the report in a clear, professional manner with proper sections and bullet points where appropriate."""
    
    # Create the ReAct agent
    react_agent = create_react_agent(
        model=llm,
        tools=[tavily_tool],
        prompt=research_prompt
    )
    
    # Execute the research
    research_messages = [HumanMessage(content=f"Please conduct comprehensive research on the following topic and create a detailed report:\n\n{research_brief}")]
    
    try:
        print("🔄 Conducting research using Tavily search...")
        
        # Run the ReAct agent
        result = react_agent.invoke({"messages": research_messages})
        
        # Extract the final report from the agent's response
        if result.get("messages"):
            final_message = result["messages"][-1]
            final_report = final_message.content if hasattr(final_message, 'content') else str(final_message)
        else:
            final_report = "Research completed but no report was generated."
        
        print("\n✅ Research completed successfully!")
        
        return {
            "messages": [AIMessage(content=f"Research completed successfully. Here's your detailed report:\n\n{final_report}")],
            "final_report": final_report,
            "research_complete": True
        }
        
    except Exception as e:
        error_message = f"An error occurred during research: {str(e)}"
        print(f"\n❌ Error: {error_message}")
        return {
            "messages": [AIMessage(content=error_message)],
            "research_complete": False
        }


def should_continue_clarification(state: ResearchState) -> Literal["clarification", "research"]:
    """
    Conditional edge function to determine next step after clarification.
    Routes to research if clarification is complete, otherwise continues clarification.
    """
    clarification_complete = state.get("clarification_complete", False)
    
    if clarification_complete:
        return "research"
    else:
        return "clarification"


# Build the graph
def build_graph():
    """Build and compile the LangGraph workflow"""
    workflow = StateGraph(ResearchState)
    
    # Add nodes
    workflow.add_node("clarification", clarification_agent)
    workflow.add_node("research", research_agent_node)
    
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
    
    # Compile the graph
    return workflow.compile()


# Export the compiled graph as 'app'
app = build_graph()


# Optional: Add a main function for testing
if __name__ == "__main__":
    from dotenv import load_dotenv
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.types import Command
    
    # Load environment variables
    load_dotenv()
    
    print("Starting LangGraph Deep Research Agent...")
    print("=" * 50)
    
    # Create a checkpointer for persistence (required for interrupt)
    memory = MemorySaver()
    
    # Recompile the graph with checkpointer
    workflow = StateGraph(ResearchState)
    workflow.add_node("clarification", clarification_agent)
    workflow.add_node("research", research_agent_node)
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
    app_with_memory = workflow.compile(checkpointer=memory)
    
    # Initial state
    initial_state = {
        "messages": [HumanMessage(content="I need help with research")],
        "research_brief": "",
        "clarification_complete": False,
        "research_complete": False,
        "final_report": ""
    }
    
    # Configuration with thread ID for persistence
    config = {"configurable": {"thread_id": "research_session_1"}}
    
    try:
        # Run the agent with streaming
        for event in app_with_memory.stream(initial_state, config, stream_mode="updates"):
            if "__interrupt__" in event:
                # Handle interrupt - get user input
                interrupt_data = event["__interrupt__"]
                if isinstance(interrupt_data, tuple) and len(interrupt_data) > 0:
                    interrupt_value = interrupt_data[0].value
                    query = interrupt_value.get("query", "Please provide input")
                    
                    # Get user input from terminal
                    user_input = input(f"\n💬 {query}: ")
                    
                    # Resume with user input using Command
                    command = Command(resume={"data": user_input})
                    for update in app_with_memory.stream(command, config, stream_mode="updates"):
                        if "research" in update and update["research"].get("final_report"):
                            print("\n" + "=" * 70)
                            print("📊 FINAL RESEARCH REPORT")
                            print("=" * 70)
                            print(update["research"]["final_report"])
                            print("=" * 70)
            else:
                # Process normal events
                for node, data in event.items():
                    if node == "research" and data.get("final_report"):
                        print("\n" + "=" * 70)
                        print("📊 FINAL RESEARCH REPORT")
                        print("=" * 70)
                        print(data["final_report"])
                        print("=" * 70)
    
    except KeyboardInterrupt:
        print("\n\n❌ Research agent terminated by user.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()











