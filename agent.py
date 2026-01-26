"""
LangGraph Deep Research Agent with User Interaction and ReAct Search

This agent operates in two phases:
1. Clarification Phase: Interactive conversation with user to clarify research scope
2. Research Phase: ReAct agent with Tavily search to generate detailed reports
"""

import os
from typing import Annotated, TypedDict, Literal, Optional
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_anthropic import ChatAnthropic
from langchain_tavily import TavilySearch
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt


# State definition for the entire workflow
class ResearchState(TypedDict):
    """State that flows through both clarification and research phases"""
    messages: Annotated[list[BaseMessage], add_messages]
    research_brief: Optional[str]
    clarification_complete: Optional[bool]
    final_report: Optional[str]


def clarification_node(state: ResearchState) -> dict:
    """
    Phase 1: Interactive clarification with the user
    Uses interrupt() to pause and gather user input multiple times
    Validates responses and builds a comprehensive research brief
    """
    messages = state.get("messages", [])
    
    # Initialize the clarification conversation
    if not any(msg.content.startswith("Let me help you clarify") for msg in messages if isinstance(msg, AIMessage)):
        initial_prompt = """Let me help you clarify your research needs. I'll ask you a few questions to better understand:
1. What specific topic or question would you like me to research?
2. What aspects are most important to you?
3. Are there any specific sources or types of information you prefer?

Please share your initial research request."""
        
        # First interrupt to get initial research topic
        user_response = interrupt({
            "question": initial_prompt,
            "phase": "initial_request"
        })
        
        # Validate initial response
        if not user_response or len(user_response.strip()) < 5:
            validation_prompt = "I need more information to help you. Could you please provide a more detailed research request?"
            user_response = interrupt({
                "question": validation_prompt,
                "phase": "validation_initial"
            })
            messages.append(AIMessage(content=validation_prompt))
            messages.append(HumanMessage(content=user_response))
        
        # Add the interaction to messages
        messages.append(AIMessage(content=initial_prompt))
        messages.append(HumanMessage(content=user_response))
        
        # Ask follow-up questions based on initial response
        followup_prompt = f"""Thank you for sharing that. Based on your request about: "{user_response}"

Let me ask a few clarifying questions to ensure I understand your needs:
1. What is the intended use or goal of this research?
2. How detailed should the research be (brief overview vs. comprehensive analysis)?
3. Are there any specific angles, perspectives, or sources you want me to focus on?
4. What timeframe or recency of information is most relevant?

Please provide as much detail as you're comfortable sharing."""
        
        # Second interrupt for clarification
        clarification_response = interrupt({
            "question": followup_prompt,
            "phase": "clarification"
        })
        
        # Validate clarification response
        if not clarification_response or len(clarification_response.strip()) < 10:
            validation_prompt = "To provide the best research, I need a bit more context. Could you elaborate on any of the questions above?"
            clarification_response = interrupt({
                "question": validation_prompt,
                "phase": "validation_clarification"
            })
            messages.append(AIMessage(content=validation_prompt))
            messages.append(HumanMessage(content=clarification_response))
        
        messages.append(AIMessage(content=followup_prompt))
        messages.append(HumanMessage(content=clarification_response))
        
        # Build comprehensive research brief
        research_brief = f"""Based on our conversation, here's what I understand you need:

**Research Topic**: {user_response}

**Additional Context and Requirements**: 
{clarification_response}

**Research Approach**:
I'll conduct a comprehensive search and analysis focusing on:
- Current and relevant information from reliable sources
- Multiple perspectives and viewpoints on the topic
- Practical insights, data, and applications
- Evidence-based findings and expert opinions
- Actionable recommendations where applicable

Is this correct, or would you like to add/modify anything? 
(Type 'yes' or 'proceed' to continue, or provide additional details to refine the research scope)"""
        
        # Final interrupt for confirmation
        confirmation = interrupt({
            "question": research_brief,
            "phase": "confirmation"
        })
        
        messages.append(AIMessage(content=research_brief))
        messages.append(HumanMessage(content=confirmation))
        
        # Check if user confirmed or wants more clarification
        confirmation_lower = confirmation.lower().strip()
        if confirmation_lower in ['yes', 'y', 'correct', 'proceed', 'go ahead', 'ok', 'okay', 'confirm', 'confirmed']:
            # Create the final comprehensive research brief
            final_brief = f"""Research Brief:
Topic: {user_response}

Context and Requirements:
{clarification_response}

Research Objectives:
- Provide current and accurate information
- Include multiple perspectives and sources
- Focus on practical applications and insights
- Generate actionable recommendations
"""
            messages.append(AIMessage(content="Excellent! I have a clear understanding of your research needs. I'll now proceed with conducting thorough research based on our discussion."))
            
            return {
                "messages": messages,
                "research_brief": final_brief,
                "clarification_complete": True
            }
        else:
            # User wants to modify - ask what to change
            modification_prompt = """I understand you'd like to modify or add to the research brief. Please tell me:
1. What specific changes would you like to make?
2. Any additional requirements or constraints?
3. Any areas you want me to emphasize or de-emphasize?

Please provide your modifications:"""
            
            modification = interrupt({
                "question": modification_prompt,
                "phase": "modification"
            })
            
            messages.append(AIMessage(content=modification_prompt))
            messages.append(HumanMessage(content=modification))
            
            # Validate modification response
            if not modification or len(modification.strip()) < 5:
                final_check = "Would you like to proceed with the research as originally outlined? (yes/no)"
                final_response = interrupt({
                    "question": final_check,
                    "phase": "final_check"
                })
                
                if final_response.lower().strip() in ['yes', 'y', 'proceed']:
                    modification = "No modifications needed"
                else:
                    modification = "Research scope to be kept general based on initial request"
            
            # Update the research brief with modifications
            final_brief = f"""Research Brief:
Topic: {user_response}

Context and Requirements:
{clarification_response}

Additional Modifications:
{modification}

Research Objectives:
- Provide current and accurate information
- Include multiple perspectives and sources
- Focus on practical applications and insights
- Generate actionable recommendations
- Incorporate specified modifications
"""
            messages.append(AIMessage(content="Perfect! I've updated the research brief with your modifications. Now proceeding with the comprehensive research."))
            
            return {
                "messages": messages,
                "research_brief": final_brief,
                "clarification_complete": True
            }
    
    # This shouldn't be reached in normal flow
    return {
        "clarification_complete": True
    }


def research_node(state: ResearchState) -> dict:
    """
    Phase 2: ReAct agent for research using Tavily search
    Uses create_react_agent() which internally handles ToolNode and tools_condition
    """
    research_brief = state.get("research_brief", "")
    messages = state.get("messages", [])
    
    # Initialize the Tavily search tool with more results for comprehensive research
    tavily_tool = TavilySearch(
        max_results=5,  # Increased for more comprehensive results
        name="tavily_search",
        description="Search the web for current information on any topic"
    )
    
    # Use Anthropic model (you can change to OpenAI or other providers)
    model = ChatAnthropic(
        model="claude-3-5-sonnet-20241022",
        temperature=0.7,
        max_tokens=4000  # Allow for longer responses
    )
    
    # Create a comprehensive system prompt for the research agent
    system_prompt = f"""You are an expert deep research assistant specializing in comprehensive analysis and report generation.

RESEARCH BRIEF:
{research_brief}

YOUR MISSION:
Conduct thorough, multi-faceted research and produce a detailed, professional report.

RESEARCH METHODOLOGY:
1. **Information Gathering Phase**:
   - Use the Tavily search tool multiple times (at least 2-3 searches) with different query angles
   - Search for recent data, statistics, expert opinions, and case studies
   - Look for contrasting viewpoints and comprehensive coverage

2. **Analysis Phase**:
   - Synthesize information from multiple sources
   - Identify patterns, trends, and key insights
   - Evaluate the credibility and relevance of sources

3. **Report Generation Phase**:
   - Structure findings in a clear, logical format
   - Include specific data points, examples, and evidence
   - Provide actionable recommendations

REPORT STRUCTURE REQUIREMENTS:
- **Executive Summary**: Brief overview of key findings
- **Introduction**: Context and scope of research
- **Main Findings**: Detailed analysis organized by themes/topics
- **Data & Evidence**: Specific statistics, quotes, and examples
- **Insights & Analysis**: Your expert interpretation of the findings
- **Recommendations**: Actionable next steps based on research
- **Conclusion**: Summary of most important points
- **Sources**: Key references used

QUALITY STANDARDS:
- Be specific and detailed, not generic
- Include recent and relevant information (prioritize last 2 years)
- Cite specific sources when possible
- Balance comprehensiveness with clarity
- Focus on practical, actionable insights"""
    
    # Create the ReAct agent with the search tool
    # create_react_agent internally uses ToolNode for tool execution and tools_condition for routing
    research_agent = create_react_agent(
        model=model,
        tools=[tavily_tool],
        prompt=system_prompt,
        # The agent will automatically handle tool calling and routing
    )
    
    # Prepare a detailed research query message
    research_query = f"""Based on the research brief provided, please conduct comprehensive research using the Tavily search tool.

Make multiple searches from different angles to ensure thorough coverage:
1. First, search for general information and current state
2. Then, search for specific data, statistics, or case studies
3. Finally, search for expert opinions, best practices, or future trends

After gathering sufficient information, synthesize your findings into a detailed, well-structured report following the format specified in your instructions.

Research Brief to investigate:
{research_brief}

Begin your research now."""
    
    # Run the research agent
    try:
        research_result = research_agent.invoke({
            "messages": [HumanMessage(content=research_query)]
        })
        
        # Extract the final report from the agent's response
        final_messages = research_result.get("messages", [])
        final_report = ""
        
        if final_messages:
            # Get the last AI message which should contain the research report
            for msg in reversed(final_messages):
                if isinstance(msg, AIMessage) and not msg.tool_calls:
                    final_report = msg.content
                    break
            
            if not final_report:
                # Fallback: compile all non-tool messages
                report_parts = []
                for msg in final_messages:
                    if isinstance(msg, AIMessage) and not msg.tool_calls:
                        report_parts.append(msg.content)
                final_report = "\n\n".join(report_parts) if report_parts else "Research completed but no final report was generated."
        else:
            final_report = "No research results were generated."
            
    except Exception as e:
        final_report = f"An error occurred during research: {str(e)}\n\nPlease try again or refine your research request."
    
    # Add the research results to the conversation with formatting
    report_message = f"""## 📊 Research Complete

{final_report}

---
*This report was generated based on the research brief developed during our clarification conversation.*"""
    
    messages.append(AIMessage(content=report_message))
    
    return {
        "messages": messages,
        "final_report": final_report
    }


def should_continue_to_research(state: ResearchState) -> Literal["research", "end"]:
    """Determine if clarification is complete and we should proceed to research"""
    if state.get("clarification_complete", False):
        return "research"
    return "end"


def should_end(state: ResearchState) -> Literal["end"]:
    """Always end after research phase"""
    return "end"


# Build the main graph
def build_graph():
    """Construct the two-phase research agent graph"""
    workflow = StateGraph(ResearchState)
    
    # Add nodes
    workflow.add_node("clarification", clarification_node)
    workflow.add_node("research", research_node)
    
    # Add edges
    workflow.add_edge(START, "clarification")
    workflow.add_conditional_edges(
        "clarification",
        should_continue_to_research,
        {
            "research": "research",
            "end": END
        }
    )
    workflow.add_conditional_edges(
        "research",
        should_end,
        {
            "end": END
        }
    )
    
    # Compile with checkpointer (required for interrupt)
    checkpointer = InMemorySaver()
    return workflow.compile(checkpointer=checkpointer)


# Export the compiled graph as 'app'
app = build_graph()


# Optional: Add a helper function for easy invocation
def run_research_agent(initial_message: str = "I need help with research"):
    """
    Helper function to run the research agent
    
    Args:
        initial_message: The initial message to start the conversation
    
    Returns:
        The final state including messages and research report
    """
    config = {"configurable": {"thread_id": "research-session-1"}}
    
    # Start with an initial message
    initial_state = {
        "messages": [HumanMessage(content=initial_message)],
        "clarification_complete": False,
        "research_brief": "",
        "final_report": ""
    }
    
    # Run the graph
    result = app.invoke(initial_state, config)
    
    return result


if __name__ == "__main__":
    # Example usage
    print("Starting LangGraph Deep Research Agent...")
    print("=" * 50)
    
    # Note: This will require interactive input when run
    # The agent will pause at interrupt points to gather user input
    
    # You can test the agent by running:
    # python agent.py
    
    # Or import and use in another script:
    # from agent import app
    # result = app.invoke({"messages": [HumanMessage("I need research help")]}, config)





