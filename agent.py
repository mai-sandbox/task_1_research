import os
from typing import TypedDict, Annotated, Sequence, Literal
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages
from dotenv import load_dotenv

load_dotenv()


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    research_brief: str
    final_report: str
    clarification_complete: bool


class ClarificationAgent:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
    
    def clarify(self, state: AgentState) -> AgentState:
        messages = state["messages"]
        
        system_prompt = SystemMessage(content="""You are a research assistant helping to clarify the scope of a research project.
        Your goal is to have a brief conversation with the user to understand:
        1. The specific topic they want researched
        2. The depth and breadth of research needed
        3. Any specific aspects or questions they want answered
        4. Any constraints or preferences they have
        
        Ask clarifying questions one at a time. Be concise and focused.
        When you have enough information to create a clear research brief, respond with:
        "CLARIFICATION_COMPLETE: [Your research brief here]"
        
        The research brief should be a clear, actionable summary of what needs to be researched.""")
        
        last_message = messages[-1]
        
        if isinstance(last_message, HumanMessage):
            if len(messages) == 1:
                response = self.llm.invoke([
                    system_prompt,
                    HumanMessage(content=f"The user wants to research: {last_message.content}\n\nAsk your first clarifying question.")
                ])
            else:
                response = self.llm.invoke([system_prompt] + list(messages))
        else:
            response = self.llm.invoke([system_prompt] + list(messages))
        
        response_text = response.content
        new_messages = list(messages) + [AIMessage(content=response_text)]
        
        if "CLARIFICATION_COMPLETE:" in response_text:
            brief = response_text.split("CLARIFICATION_COMPLETE:")[1].strip()
            return {
                "messages": new_messages,
                "research_brief": brief,
                "clarification_complete": True,
                "final_report": ""
            }
        
        return {
            "messages": new_messages,
            "research_brief": state.get("research_brief", ""),
            "clarification_complete": False,
            "final_report": ""
        }


class ResearchAgent:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
        self.search_tool = TavilySearchResults(
            max_results=5,
            search_depth="advanced",
            include_answer=True,
            include_raw_content=False
        )
        self.llm_with_tools = self.llm.bind_tools([self.search_tool])
    
    def research(self, state: AgentState) -> AgentState:
        research_brief = state["research_brief"]
        
        system_prompt = SystemMessage(content=f"""You are an expert research agent with access to web search.
        
        Research Brief: {research_brief}
        
        Your task:
        1. Break down the research brief into key questions
        2. Use the Tavily search tool to find relevant information
        3. Synthesize findings into a comprehensive report
        4. Ensure all aspects of the brief are addressed
        
        Be thorough but focused. Use multiple searches if needed to cover different aspects.""")
        
        research_messages = [
            system_prompt,
            HumanMessage(content=f"Please conduct thorough research based on this brief: {research_brief}")
        ]
        
        response = self.llm_with_tools.invoke(research_messages)
        research_messages.append(response)
        
        while response.tool_calls:
            tool_results = []
            for tool_call in response.tool_calls:
                if tool_call["name"] == "tavily_search_results_json":
                    result = self.search_tool.invoke(tool_call["args"])
                    tool_results.append({
                        "tool_call_id": tool_call["id"],
                        "output": str(result)
                    })
            
            research_messages.append(AIMessage(content="", tool_calls=response.tool_calls))
            research_messages.append(AIMessage(content=str(tool_results)))
            
            response = self.llm_with_tools.invoke(research_messages)
            research_messages.append(response)
        
        final_report_prompt = SystemMessage(content="""Based on all the research conducted, create a detailed, well-structured report.
        Include:
        - Executive Summary
        - Key Findings (organized by topic/theme)
        - Detailed Analysis
        - Sources and References
        - Conclusion and Recommendations (if applicable)
        
        Make it comprehensive yet readable.""")
        
        final_response = self.llm.invoke(research_messages + [final_report_prompt])
        
        return {
            "messages": state["messages"] + [AIMessage(content=f"Research Complete. Brief: {research_brief}")],
            "research_brief": research_brief,
            "clarification_complete": True,
            "final_report": final_response.content
        }


def human_input(state: AgentState) -> AgentState:
    if state.get("clarification_complete", False):
        return state
    
    user_input = input("\n> ")
    new_messages = list(state["messages"]) + [HumanMessage(content=user_input)]
    
    return {
        "messages": new_messages,
        "research_brief": state.get("research_brief", ""),
        "clarification_complete": state.get("clarification_complete", False),
        "final_report": state.get("final_report", "")
    }


def should_continue_clarification(state: AgentState) -> Literal["human_input", "research"]:
    if state.get("clarification_complete", False):
        return "research"
    return "human_input"


def format_output(state: AgentState) -> AgentState:
    print("\n" + "="*80)
    print("RESEARCH REPORT")
    print("="*80)
    print(state["final_report"])
    print("="*80 + "\n")
    return state


clarification_agent = ClarificationAgent()
research_agent = ResearchAgent()

graph = StateGraph(AgentState)

graph.add_node("clarification", clarification_agent.clarify)
graph.add_node("human_input", human_input)
graph.add_node("research", research_agent.research)
graph.add_node("output", format_output)

graph.set_entry_point("clarification")

graph.add_conditional_edges(
    "clarification",
    should_continue_clarification,
    {
        "human_input": "human_input",
        "research": "research"
    }
)

graph.add_edge("human_input", "clarification")
graph.add_edge("research", "output")
graph.add_edge("output", END)

app = graph.compile()