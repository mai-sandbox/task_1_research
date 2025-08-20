# LangGraph Deep Research Agent

A sophisticated research agent built with LangGraph that conducts comprehensive research through a two-stage process: interactive clarification followed by automated deep research using Tavily search.

## 🚀 Features

- **Interactive Clarification**: Engages in back-and-forth dialogue to understand research scope
- **Deep Research**: Uses ReAct methodology with Tavily search for comprehensive information gathering  
- **Comprehensive Reports**: Generates detailed, well-structured research reports
- **LangGraph Architecture**: Built with proper state management and conditional routing

## 🏗️ Architecture

The agent consists of two main components:

1. **Clarification Agent**: Interactive dialogue to define research scope and requirements
2. **ReAct Research Agent**: Systematic research using search tools and report generation

### Workflow

```
User Input → Clarification Agent ↔ User (Interactive Q&A)
     ↓
Research Brief Generated
     ↓  
ReAct Agent → Search Tools → Comprehensive Report
```

## 📋 Prerequisites

- Python 3.8+
- Anthropic API key (for Claude)
- Tavily API key (for search)

## 🛠️ Installation

1. Clone or download the project files
2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
export ANTHROPIC_API_KEY="your-anthropic-key-here"
export TAVILY_API_KEY="your-tavily-key-here"  
```

## 🎯 Usage

### Interactive Demo (Recommended)
```bash
python interactive_demo.py
```
This provides a full interactive experience showing the clarification dialogue followed by automated research.

### Programmatic Usage
```python
from agent import app
from langchain_core.messages import HumanMessage

# Initialize state
initial_state = {
    "messages": [HumanMessage("Research renewable energy technologies")],
    "stage": "clarification",
    "clarification_complete": False,
    "research_complete": False,
    "research_brief": ""
}

# Run agent
result = app.invoke(initial_state)
```

### Test Script
```bash
python test_agent.py
```
Simple test to verify the agent is working correctly.

## 📁 File Structure

```
task_1_research/
├── agent.py              # Main agent implementation
├── langgraph.json        # LangGraph configuration  
├── requirements.txt      # Python dependencies
├── interactive_demo.py   # Interactive demonstration
├── test_agent.py        # Basic test script
└── README.md            # This file
```

## 🔧 Configuration

The agent is configured through `langgraph.json`:

```json
{
  "dependencies": [
    "langchain-anthropic",
    "langchain-tavily", 
    "langgraph"
  ],
  "graphs": {
    "agent": "./agent.py:app"
  },
  "env": [
    "ANTHROPIC_API_KEY",
    "TAVILY_API_KEY"
  ]
}
```

## 🎭 Agent Behavior

### Clarification Stage
- Asks targeted questions about research topic, scope, audience, etc.
- Continues until it has sufficient information
- Generates a detailed research brief

### Research Stage  
- Uses ReAct methodology (Reasoning + Acting)
- Conducts systematic searches using Tavily
- Synthesizes information from multiple sources
- Generates comprehensive, well-structured reports

## 🔍 Example Workflow

1. **User**: "I want to research artificial intelligence"

2. **Agent**: "I'd be happy to help you research artificial intelligence! To provide you with the most relevant and comprehensive research, I need to understand your specific needs better.

   Could you tell me:
   - What specific aspects of AI are you most interested in? (e.g., machine learning, natural language processing, computer vision, AI ethics, etc.)
   - What's the intended use for this research? (academic paper, business decision, personal learning, etc.)
   - Are there particular applications or industries you want to focus on?"

3. **User**: "I'm interested in AI applications in healthcare for a business presentation"

4. **Agent**: [Continues clarification dialogue...]

5. **Agent**: "CLARIFICATION_COMPLETE - Based on our discussion, I'll research AI applications in healthcare focusing on current implementations, market trends, key players, challenges, and opportunities for business applications..."

6. **Agent**: [Conducts automated research using search tools]

7. **Agent**: "RESEARCH_COMPLETE" + [Comprehensive research report]

## 🔬 Technical Details

### State Management
The agent uses a `ResearchState` TypedDict to maintain:
- Conversation messages
- Research brief
- Current stage (clarification/research/completed)
- Completion flags

### Tool Integration
- **Tavily Search**: Web search with up to 5 results per query
- **LLM with Tools**: Claude model with tool-calling capabilities
- **ToolNode**: LangGraph prebuilt tool execution node

### Graph Structure
- Conditional routing between clarification and research stages
- Tool usage workflow with proper state management
- Automatic completion detection and termination

## 🎯 Use Cases

- Academic research planning
- Business intelligence gathering  
- Market research and analysis
- Technical documentation research
- Competitive analysis
- Industry trend analysis

## 🚨 Error Handling

The agent includes error handling for:
- Missing API keys
- Network issues during search
- Invalid state transitions
- Tool execution failures

## 🔄 Extending the Agent

To customize the agent:

1. **Modify prompts** in `CLARIFICATION_SYSTEM_PROMPT` and `RESEARCH_SYSTEM_PROMPT`
2. **Add new tools** by extending the `tools` list
3. **Customize routing logic** in the conditional edge functions
4. **Adjust search parameters** in the Tavily tool configuration

## 📄 License

This project is provided as-is for educational and development purposes.

## 🤝 Contributing

Feel free to submit issues, feature requests, or improvements!