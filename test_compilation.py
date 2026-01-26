#!/usr/bin/env python3
"""Test script to verify agent compilation and structure"""

import ast
import sys
from pathlib import Path

def check_syntax():
    """Check if agent.py has valid Python syntax"""
    agent_file = Path("agent.py")
    if not agent_file.exists():
        print("❌ agent.py file not found")
        return False
    
    try:
        with open(agent_file, 'r') as f:
            code = f.read()
        ast.parse(code)
        print("✅ agent.py has valid Python syntax")
        return True
    except SyntaxError as e:
        print(f"❌ Syntax error in agent.py: {e}")
        return False

def check_structure():
    """Check if agent.py has the required structure"""
    agent_file = Path("agent.py")
    with open(agent_file, 'r') as f:
        code = f.read()
    
    # Check for required components
    checks = [
        ("app = build_graph()" in code or "app = workflow.compile()" in code, "Graph exported as 'app' variable"),
        ("class ResearchState(TypedDict):" in code, "ResearchState TypedDict defined"),
        ("def clarification_agent" in code, "clarification_agent function defined"),
        ("def research_agent_node" in code, "research_agent_node function defined"),
        ("def should_continue_clarification" in code, "should_continue_clarification function defined"),
        ("def build_graph" in code, "build_graph function defined"),
        ("interrupt(" in code, "interrupt() function used for user interaction"),
        ("create_react_agent" in code, "create_react_agent used for research"),
        ("TavilySearchResults" in code, "TavilySearchResults tool configured"),
        ("format_messages_for_analysis" in code, "format_messages_for_analysis function defined"),
    ]
    
    all_passed = True
    for check, description in checks:
        if check:
            print(f"✅ {description}")
        else:
            print(f"❌ {description}")
            all_passed = False
    
    return all_passed

def check_function_order():
    """Check if format_messages_for_analysis is defined before it's used"""
    agent_file = Path("agent.py")
    with open(agent_file, 'r') as f:
        lines = f.readlines()
    
    # Find definition and usage lines
    def_line = None
    first_use_line = None
    
    for i, line in enumerate(lines, 1):
        if "def format_messages_for_analysis" in line:
            def_line = i
        if "format_messages_for_analysis(messages)" in line and first_use_line is None:
            first_use_line = i
    
    if def_line and first_use_line:
        if def_line < first_use_line:
            print(f"✅ format_messages_for_analysis defined (line {def_line}) before first use (line {first_use_line})")
            return True
        else:
            print(f"❌ format_messages_for_analysis used (line {first_use_line}) before definition (line {def_line})")
            return False
    elif def_line is None:
        print("❌ format_messages_for_analysis function not found")
        return False
    else:
        print("✅ format_messages_for_analysis defined but not used (or used differently)")
        return True

def check_imports():
    """Check that unused imports have been removed"""
    agent_file = Path("agent.py")
    with open(agent_file, 'r') as f:
        code = f.read()
    
    # Check for imports that should NOT be present
    unused_imports = [
        ("from langchain_core.tools import tool", "Unused 'tool' import"),
        ("ToolNode", "Unused 'ToolNode' import (should only have create_react_agent)"),
    ]
    
    all_clean = True
    for import_str, description in unused_imports:
        if import_str in code and "create_react_agent, ToolNode" in code:
            print(f"❌ {description} still present")
            all_clean = False
        else:
            print(f"✅ {description} removed")
    
    return all_clean

def main():
    print("=" * 60)
    print("Testing LangGraph Deep Research Agent Compilation")
    print("=" * 60)
    print()
    
    # Run all checks
    syntax_ok = check_syntax()
    print()
    
    if syntax_ok:
        structure_ok = check_structure()
        print()
        
        function_order_ok = check_function_order()
        print()
        
        imports_clean = check_imports()
        print()
        
        if structure_ok and function_order_ok and imports_clean:
            print("=" * 60)
            print("🎉 All compilation and structure tests passed!")
            print("=" * 60)
            print("\nNote: Runtime dependencies not installed.")
            print("To test with dependencies, run: pip install -r requirements.txt")
            return 0
        else:
            print("=" * 60)
            print("❌ Some tests failed. Please review the issues above.")
            print("=" * 60)
            return 1
    else:
        print("=" * 60)
        print("❌ Syntax error detected. Fix syntax before proceeding.")
        print("=" * 60)
        return 1

if __name__ == "__main__":
    sys.exit(main())

