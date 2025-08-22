#!/usr/bin/env python3
"""
Validate the langgraph.json configuration file
"""

import json
import os

def validate_langgraph_config():
    """Validate the langgraph.json configuration meets all requirements"""
    
    try:
        with open('langgraph.json', 'r') as f:
            config = json.load(f)
        
        print("✓ langgraph.json is valid JSON")
        
        # Check required dependencies
        required_deps = ['langgraph', 'tavily-python', 'langchain', 'langchain-community', 'langchain-openai']
        actual_deps = config.get('dependencies', [])
        
        print("\n=== DEPENDENCY VALIDATION ===")
        for dep in required_deps:
            if dep in actual_deps:
                print(f"✓ {dep}")
            else:
                print(f"✗ Missing: {dep}")
        
        # Check environment variables
        required_env_vars = ['TAVILY_API_KEY', 'OPENAI_API_KEY']
        actual_env_vars = config.get('environment_variables', [])
        
        print("\n=== ENVIRONMENT VARIABLES VALIDATION ===")
        for env_var in required_env_vars:
            if env_var in actual_env_vars:
                print(f"✓ {env_var}")
            else:
                print(f"✗ Missing: {env_var}")
        
        # Check graph configuration
        print("\n=== GRAPH CONFIGURATION VALIDATION ===")
        graphs = config.get('graphs', {})
        if 'agent' in graphs and graphs['agent'] == './agent.py:app':
            print("✓ Graph configuration: agent -> ./agent.py:app")
        else:
            print("✗ Invalid graph configuration")
        
        # Check other required fields
        print("\n=== OTHER CONFIGURATION VALIDATION ===")
        if config.get('env') == '.env':
            print("✓ Environment file: .env")
        else:
            print("✗ Missing or invalid env file configuration")
            
        if config.get('python_version'):
            print(f"✓ Python version: {config.get('python_version')}")
        else:
            print("✗ Missing Python version")
        
        print("\n✓ langgraph.json configuration validation complete!")
        return True
        
    except json.JSONDecodeError as e:
        print(f"✗ Invalid JSON: {e}")
        return False
    except FileNotFoundError:
        print("✗ langgraph.json file not found")
        return False
    except Exception as e:
        print(f"✗ Error validating configuration: {e}")
        return False

if __name__ == "__main__":
    validate_langgraph_config()
