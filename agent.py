"""
LangGraph Deep Research Agent with Interactive Scoping and ReAct Search

This agent implements a two-phase research workflow:
1. Interactive Scoping Phase: Engages with user via terminal to clarify research requirements
2. ReAct Research Phase: Performs web research using Tavily and generates detailed reports
"""

import json
from typing import Annotated, Dict, Any, List, TypedDict, Literal
from typing_extensions import TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langchain_tavily import TavilySearch
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.types import interrupt, Command


# State definition
class ResearchState(TypedDict):
    """State for the research agent workflow"""
    messages: Annotated[List[BaseMessage], add_messages]
    research_brief: str
    research_phase: Literal["scoping", "researching", "complete"]
    scoping_complete: bool
    final_report: str


# Initialize LLM (prefer Anthropic, fallback to OpenAI)
def get_llm():
    """Initialize the LLM with preference for Anthropic"""
    try:
        return ChatAnthropic(model="claude-3-5-sonnet-20241022", temperature=0.7)
    except:
        try:
            return ChatOpenAI(model="gpt-4o", temperature=0.7)
        except:
            # Fallback to a more basic model if needed
            return ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)


llm = get_llm()


# Scoping Agent Node
def scoping_agent(state: ResearchState) -> Dict[str, Any]:
    """
    Interactive scoping agent that clarifies research requirements with the user.
    Uses interrupt() to pause execution and gather user input via terminal.
    
    This agent:
    1. Engages in back-and-forth conversation to understand research needs
    2. Asks targeted clarifying questions about scope, depth, and requirements
    3. Continues dialogue until confident about the research brief
    4. Returns a structured research brief for the ReAct agent
    """
    messages = state.get("messages", [])
    research_phase = state.get("research_phase", "scoping")
    scoping_complete = state.get("scoping_complete", False)
    
    # Extract only human and AI messages for conversation context
    conversation_messages = [msg for msg in messages if isinstance(msg, (HumanMessage, AIMessage))]
    
    # If we're just starting, provide initial greeting
    if len(conversation_messages) == 1:  # Only the initial user message
        initial_prompt = """Hello! I'm your deep research assistant. I'll help you conduct thorough, comprehensive research on your topic.

To ensure I provide the most relevant and valuable insights, I need to understand your research requirements better. I'll ask you a series of clarifying questions to build a detailed research brief.

Let's start by understanding your main research focus."""
        
        first_question = "What is the main topic, question, or problem you want me to research? Please be as specific as possible."
        
        response_content = f"{initial_prompt}\n\n**Question 1:** {first_question}"
        response = AIMessage(content=response_content)
        
        # Interrupt to get user input
        user_input = interrupt({
            "stage": "initial_scoping",
            "question": first_question,
            "message": "Please provide your response:"
        })
        
        return {
            "messages": [response, HumanMessage(content=user_input.get("response", user_input.get("data", "")))],
            "research_phase": "scoping",
            "scoping_complete": False
        }
    
    # Continue the scoping conversation
    scoping_llm = llm.bind_tools([], tool_choice="none")  # No tools during scoping
    
    # Build conversation context with enhanced prompt
    scoping_prompt = SystemMessage(content="""You are an expert research scoping assistant engaged in an interactive dialogue with the user.

Your role is to:
1. Understand the user's research needs through targeted clarifying questions
2. Build a comprehensive research brief through iterative conversation
3. Ensure all critical aspects are covered before proceeding

Guidelines for your conversation:
- Ask ONE clarifying question at a time
- Build upon previous answers to go deeper
- Be conversational but professional
- Show that you understand their needs by summarizing key points
- Focus on gathering information about:
  * Core research topic and specific questions
  * Scope boundaries (what to include/exclude)
  * Required depth of analysis (surface-level vs deep dive)
  * Specific aspects or angles of interest
  * Time period or geographic focus (if relevant)
  * Preferred sources or methodologies
  * Expected output format and use case
  * Any constraints or special requirements

After gathering sufficient information (typically 4-7 exchanges), indicate you're ready to proceed by saying "I now have a comprehensive understanding of your research needs."

Current conversation stage: You've had {num_exchanges} exchanges with the user.""".format(
        num_exchanges=len(conversation_messages) // 2
    ))
    
    conversation = [scoping_prompt] + conversation_messages
    response = scoping_llm.invoke(conversation)
    
    # Check if scoping is complete by looking for completion indicators
    completion_indicators = [
        "comprehensive understanding",
        "ready to proceed",
        "have all the information",
        "sufficient information",
        "research brief is complete"
    ]
    
    is_complete = any(indicator in response.content.lower() for indicator in completion_indicators)
    
    # Also check if we've had enough exchanges (minimum 3, maximum 10)
    num_exchanges = len(conversation_messages) // 2
    if num_exchanges >= 3 and ("?" not in response.content or is_complete):
        # Time to finalize the research brief
        is_complete = True
    
    if is_complete:
        # Generate the structured research brief
        brief_prompt = SystemMessage(content="""Based on the conversation with the user, create a comprehensive, structured research brief.

The brief should include:

# Research Brief

## 1. Primary Research Question/Topic
[State the main research focus clearly]

## 2. Scope and Boundaries
- What to include:
- What to exclude:
- Geographic/temporal focus (if applicable):

## 3. Specific Areas of Investigation
[List 3-5 key areas to explore]

## 4. Depth of Analysis Required
[Specify the level of detail needed]

## 5. Preferred Sources and Methodologies
[Note any preferences mentioned]

## 6. Constraints and Special Requirements
[List any limitations or special needs]

## 7. Expected Deliverables
[Describe the desired output format and use case]

## 8. Success Criteria
[What would make this research successful for the user]

Make this brief actionable and comprehensive for the research phase.""")
        
        brief_response = llm.invoke([brief_prompt] + conversation_messages + [response])
        
        confirmation_message = f"""Excellent! I now have a comprehensive understanding of your research needs. 

{response.content}

Based on our conversation, I've prepared the following research brief:

{brief_response.content}

I'll now proceed to conduct thorough research based on this brief. The research phase will involve:
1. Systematic web searches using advanced search tools
2. Analysis and synthesis of findings
3. Generation of a comprehensive report

Starting the research phase now..."""
        
        return {
            "messages": [AIMessage(content=confirmation_message)],
            "research_brief": brief_response.content,
            "research_phase": "researching",
            "scoping_complete": True
        }
    else:
        # Continue scoping - ask for more information
        # Extract the question from the response if it exists
        question_prompt = response.content
        if "?" in response.content:
            # Extract the last question asked
            lines = response.content.split('\n')
            question_lines = [line for line in lines if '?' in line]
            if question_lines:
                question_prompt = question_lines[-1]
        
        # Interrupt to get user input for the next question
        user_input = interrupt({
            "stage": "continued_scoping",
            "question": question_prompt,
            "exchange_number": num_exchanges + 1,
            "message": "Please provide your response:"
        })
        
        user_response_content = user_input.get("response", user_input.get("data", ""))
        
        return {
            "messages": [response, HumanMessage(content=user_response_content)],
            "research_phase": "scoping",
            "scoping_complete": False
        }


# Research Agent Node with Tavily Tool
def research_agent(state: ResearchState) -> Dict[str, Any]:
    """
    ReAct research agent that uses Tavily search to conduct comprehensive research
    based on the brief from the scoping phase.
    
    This agent:
    1. Receives the structured research brief from scoping phase
    2. Uses TavilySearch tool with max_results parameter for web research
    3. Follows ReAct pattern (Reason-Act-Observe) to iteratively search and analyze
    4. Generates comprehensive detailed report based on findings
    5. Returns final report to user
    """
    research_brief = state.get("research_brief", "")
    messages = state.get("messages", [])
    
    # Initialize Tavily search tool with configurable max_results
    search_tool = TavilySearch(max_results=5)
    tools = [search_tool]
    
    # Create research LLM with tools
    research_llm = llm.bind_tools(tools)
    
    # Create enhanced research prompt with explicit ReAct pattern
    research_prompt = SystemMessage(content=f"""You are an expert research agent following the ReAct (Reason-Act-Observe) pattern.

RESEARCH BRIEF:
{research_brief}

YOUR TASK:
Conduct thorough, comprehensive research based on the brief above using the Tavily search tool.

REACT PATTERN TO FOLLOW:
For each research iteration:
1. REASON: Think about what information you need and why
2. ACT: Use the tavily_search tool with specific, well-crafted queries
3. OBSERVE: Analyze the search results and determine what's still needed

RESEARCH STRATEGY:
1. Start with broad searches to understand the landscape
2. Then dive deeper into specific aspects mentioned in the brief
3. Look for:
   - Current trends and recent developments
   - Multiple perspectives and viewpoints
   - Statistical data and concrete examples
   - Expert opinions and authoritative sources
   - Potential challenges or controversies
   - Best practices and recommendations

SEARCH GUIDELINES:
- Craft specific, targeted search queries
- Use different query formulations to find diverse sources
- Search for recent information (add year if relevant)
- Verify important claims with multiple sources
- Look for both primary and secondary sources

Continue researching until you have comprehensive coverage of all aspects in the brief.
After each search, explicitly state your REASONING for the next step.""")
    
    # Conduct research iterations with explicit ReAct pattern
    research_messages = [research_prompt]
    max_iterations = 15  # Increased for thorough research
    iteration = 0
    search_count = 0
    
    # Track search queries and findings for better coverage
    searched_queries = []
    key_findings = []
    
    while iteration < max_iterations:
        # Get LLM response with potential tool calls
        response = research_llm.invoke(research_messages)
        research_messages.append(response)
        
        # Check if there are tool calls
        if hasattr(response, 'tool_calls') and response.tool_calls:
            # Execute tool calls
            for tool_call in response.tool_calls:
                if tool_call["name"] == "tavily_search":
                    query = tool_call["args"].get("query", "")
                    searched_queries.append(query)
                    search_count += 1
                    
                    # Execute search
                    try:
                        search_results = search_tool.invoke(tool_call["args"])
                        
                        # Format results for better readability
                        formatted_results = f"\n=== SEARCH #{search_count} ===\nQuery: '{query}'\n"
                        
                        if isinstance(search_results, list):
                            formatted_results += f"Found {len(search_results)} results:\n"
                            for idx, result in enumerate(search_results, 1):
                                if isinstance(result, dict):
                                    formatted_results += f"\n{idx}. {result.get('title', 'No title')}\n"
                                    formatted_results += f"   URL: {result.get('url', 'No URL')}\n"
                                    formatted_results += f"   Content: {result.get('content', 'No content')[:500]}...\n"
                                else:
                                    formatted_results += f"\n{idx}. {str(result)[:500]}...\n"
                        else:
                            formatted_results += f"Results: {json.dumps(search_results, indent=2)}\n"
                        
                        # Create tool message with results
                        tool_message = ToolMessage(
                            content=formatted_results,
                            tool_call_id=tool_call.get("id", f"search_{search_count}")
                        )
                        research_messages.append(tool_message)
                        
                        # Track key findings
                        if isinstance(search_results, list):
                            for result in search_results[:3]:  # Top 3 results
                                if isinstance(result, dict):
                                    key_findings.append({
                                        "query": query,
                                        "title": result.get("title", ""),
                                        "url": result.get("url", ""),
                                        "snippet": result.get("content", "")[:200]
                                    })
                    
                    except Exception as e:
                        error_message = ToolMessage(
                            content=f"Search error for query '{query}': {str(e)}",
                            tool_call_id=tool_call.get("id", f"search_error_{search_count}")
                        )
                        research_messages.append(error_message)
        else:
            # Check if the agent has indicated completion
            if response.content and any(phrase in response.content.lower() for phrase in 
                ["research complete", "research is complete", "finished researching", 
                 "comprehensive coverage", "all aspects covered"]):
                break
            
            # If no tool calls and not complete, prompt for continuation
            if iteration < 5:  # Minimum iterations for thorough research
                continuation_prompt = AIMessage(content="""Please continue your research. Remember to:
- Follow the ReAct pattern: REASON about what's needed, ACT with searches, OBSERVE results
- Ensure you've covered all aspects mentioned in the research brief
- Use the tavily_search tool to gather more information""")
                research_messages.append(continuation_prompt)
            else:
                # Sufficient iterations done, can complete if no more searches needed
                break
        
        iteration += 1
    
    # Add summary of research process
    research_summary = f"""
=== RESEARCH PROCESS SUMMARY ===
Total searches conducted: {search_count}
Search queries used: {', '.join(searched_queries[:10])}{'...' if len(searched_queries) > 10 else ''}
Key sources found: {len(key_findings)}
Research iterations: {iteration}
"""
    research_messages.append(AIMessage(content=research_summary))
    
    # Generate final comprehensive report with enhanced structure
    report_prompt = SystemMessage(content=f"""Based on all the research conducted, create a comprehensive, detailed report that fully addresses the research brief.

RESEARCH BRIEF TO ADDRESS:
{research_brief}

RESEARCH CONDUCTED:
- {search_count} searches performed
- {len(key_findings)} key sources identified
- Queries covered: {', '.join(set(searched_queries)[:15])}

REPORT REQUIREMENTS:
Create a thorough, professional report with the following structure:

# COMPREHENSIVE RESEARCH REPORT

## 1. EXECUTIVE SUMMARY
- Brief overview of the research topic
- Key findings at a glance
- Main conclusions and recommendations

## 2. RESEARCH METHODOLOGY
- Search strategies employed
- Sources consulted
- Scope and limitations

## 3. KEY FINDINGS
Organize by the main themes/topics from the research brief:
- Present findings with supporting evidence
- Include relevant statistics, quotes, and examples
- Cite sources with URLs where available

## 4. DETAILED ANALYSIS
For each major aspect in the research brief:
- In-depth exploration of the topic
- Multiple perspectives and viewpoints
- Current trends and developments
- Challenges and opportunities
- Best practices and recommendations

## 5. SUPPORTING EVIDENCE
- Data and statistics
- Expert opinions
- Case studies or examples
- Relevant quotes and citations

## 6. CONCLUSIONS
- Synthesis of key insights
- Answers to the main research questions
- Implications and significance

## 7. RECOMMENDATIONS (if applicable)
- Actionable next steps
- Areas for further investigation
- Strategic considerations

## 8. SOURCES AND REFERENCES
- List of all sources consulted
- URLs for further reading

Make the report:
- Comprehensive and thorough
- Well-organized and easy to navigate
- Evidence-based with proper citations
- Actionable and practical
- Professional in tone and presentation""")
    
    # Generate the final report
    final_report_response = llm.invoke(research_messages + [report_prompt])
    
    # Create final message for user
    completion_message = f"""✅ **Research Completed Successfully!**

I've conducted comprehensive research based on your requirements, performing {search_count} targeted searches across multiple sources.

{final_report_response.content}

---
*Research conducted using {search_count} Tavily searches across {iteration} iterations following the ReAct pattern.*"""
    
    return {
        "messages": [AIMessage(content=completion_message)],
        "final_report": final_report_response.content,
        "research_phase": "complete"
    }


# Supervisor logic to determine next step
def route_to_next_agent(state: ResearchState) -> Literal["scoping", "research", "end"]:
    """
    Supervisor routing logic that determines which agent to route to based on the current state.
    
    This function:
    1. Checks the research phase and scoping completion status
    2. Routes to appropriate agent or ends the workflow
    3. Ensures proper handoff between scoping and research agents
    4. Preserves message history across phase transitions
    """
    research_phase = state.get("research_phase", "scoping")
    scoping_complete = state.get("scoping_complete", False)
    
    # Routing logic with clear phase transitions
    if research_phase == "scoping" and not scoping_complete:
        # Continue in scoping phase until complete
        return "scoping"
    elif research_phase == "researching" or scoping_complete:
        # Transition to research phase when scoping is complete
        return "research"
    elif research_phase == "complete":
        # End workflow when research is complete
        return "end"
    else:
        # Default to end if phase is unknown
        return "end"


# Command-based routing for interrupt handling
def handle_interrupt_command(state: ResearchState) -> Command:
    """
    Handles interrupt commands for resuming after user input.
    
    Returns a Command object to direct workflow after interrupt.
    """
    research_phase = state.get("research_phase", "scoping")
    scoping_complete = state.get("scoping_complete", False)
    
    if research_phase == "scoping" and not scoping_complete:
        # Continue scoping after interrupt
        return Command(goto="scoping")
    elif scoping_complete or research_phase == "researching":
        # Move to research after scoping is complete
        return Command(goto="research")
    else:
        # End workflow
        return Command(goto=END)


# Build the graph with enhanced state management
def build_research_graph():
    """
    Builds the LangGraph workflow for the research agent with proper state management.
    
    Features:
    1. StateGraph with START and conditional edges
    2. Proper handoff between scoping and research agents
    3. Command objects support for resuming after interrupt
    4. Message history preservation across phases
    """
    # Create the graph with typed state
    workflow = StateGraph(ResearchState)
    
    # Add nodes for each agent
    workflow.add_node("scoping", scoping_agent)
    workflow.add_node("research", research_agent)
    
    # Set entry point - always start with scoping
    workflow.add_edge(START, "scoping")
    
    # Add conditional edges for dynamic routing
    workflow.add_conditional_edges(
        "scoping",
        route_to_next_agent,
        {
            "scoping": "scoping",  # Loop back for continued scoping
            "research": "research",  # Transition to research phase
            "end": END  # End if needed
        }
    )
    
    # Research always leads to end
    workflow.add_edge("research", END)
    
    # Compile the graph with interrupt support
    compiled_graph = workflow.compile()
    
    # The compiled graph automatically handles:
    # - State persistence across nodes
    # - Message history preservation
    # - Interrupt/resume functionality
    # - Command-based navigation
    
    return compiled_graph


# Export the compiled graph as 'app'
app = build_research_graph()


# Optional: Helper function for testing
def run_research_agent(initial_query: str):
    """
    Helper function to run the research agent with an initial query.
    """
    initial_state = {
        "messages": [HumanMessage(content=initial_query)],
        "research_phase": "scoping",
        "scoping_complete": False
    }
    
    result = app.invoke(initial_state)
    return result


if __name__ == "__main__":
    # Example usage
    print("LangGraph Deep Research Agent initialized.")
    print("Use app.invoke() with initial state to start research.")



