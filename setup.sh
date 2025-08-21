#!/bin/bash

echo "Setting up LangGraph Research Agent..."
echo "======================================"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install requirements
echo "Installing dependencies..."
pip install -r requirements.txt

# Check for .env file
if [ ! -f ".env" ]; then
    echo ""
    echo "Creating .env file from template..."
    cp .env.example .env
    echo ""
    echo "⚠️  IMPORTANT: Please edit .env file and add your API keys:"
    echo "   - OPENAI_API_KEY"
    echo "   - TAVILY_API_KEY"
    echo ""
    echo "You can get a Tavily API key from: https://tavily.com/"
fi

echo ""
echo "✅ Setup complete!"
echo ""
echo "To test the agent:"
echo "  1. Ensure your API keys are set in .env"
echo "  2. Run: python test_agent.py"
echo "  3. For interactive mode: python test_agent.py --interactive"
echo ""
echo "To use in your code:"
echo "  from agent import app"
echo "  from langchain_core.messages import HumanMessage"
echo "  result = app.invoke({'messages': [HumanMessage('Your research topic')]})"