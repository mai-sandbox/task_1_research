# LangGraph Deep Research Agent

A two-phase research agent built with LangGraph that first clarifies research scope through interactive conversation, then conducts deep research using ReAct pattern with Tavily search.

## Features

- **Interactive Clarification Phase**: Engages in back-and-forth conversation to understand research requirements
- **ReAct Research Phase**: Uses reasoning and action pattern with Tavily search for comprehensive research
- **Structured Workflow**: Seamlessly transitions from clarification to research
- **Detailed Reporting**: Generates comprehensive research reports with sources

## Setup

1. **Install dependencies:**
   ```bash
   ./setup.sh
   ```

2. **Configure API keys in `.env`:**
   ```
   OPENAI_API_KEY=your_openai_api_key
   TAVILY_API_KEY=your_tavily_api_key
   ```
   
   Get your Tavily API key from: https://tavily.com/

## Usage

### Basic Usage

```python
from agent import app
from langchain_core.messages import HumanMessage

initial_state = {
    "messages": [HumanMessage("Research quantum computing advances")]
}
result = app.invoke(initial_state)
```

### Interactive Mode

Run the interactive test for a full conversation experience:

```bash
python test_agent.py --interactive
```

### Automated Test

Run the automated test to see the full workflow:

```bash
python test_agent.py
```

## Architecture

The agent consists of two main components:

1. **Clarification Agent**: 
   - Interacts with user to understand research scope
   - Asks clarifying questions
   - Creates a comprehensive research brief

2. **Research Agent**:
   - Uses ReAct pattern with Tavily search
   - Conducts multiple searches based on the brief
   - Synthesizes information into detailed report

## Files

- `agent.py` - Main agent implementation
- `langgraph.json` - LangGraph configuration
- `test_agent.py` - Test script with interactive mode
- `example_usage.py` - Simple usage example
- `requirements.txt` - Python dependencies
- `.env.example` - Environment variables template

## Deployment

The agent is deployment-ready with the included `langgraph.json` configuration. Deploy using LangGraph's deployment tools.

## Requirements

- Python 3.11+
- OpenAI API key
- Tavily API key