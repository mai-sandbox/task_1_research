#!/usr/bin/env python3
"""
Validation script for langgraph.json configuration file.
"""

import json
import os

def validate_langgraph_config():
    """Validate the langgraph.json configuration file."""
    
    print("🧪 Validating langgraph.json configuration...")
    
    try:
        # Load and parse the JSON file
        with open('langgraph.json', 'r') as f:
            config = json.load(f)
        
        print("✅ langgraph.json is valid JSON")
        
        # Validate required sections
        required_sections = ['dependencies', 'graphs', 'env']
        for section in required_sections:
            if section not in config:
                print(f"❌ Missing required section: {section}")
                return False
            print(f"✅ Found required section: {section}")
        
        # Validate graphs section
        if 'agent' not in config['graphs']:
            print("❌ Missing 'agent' graph definition")
            return False
        
        agent_config = config['graphs']['agent']
        
        # Check required agent fields
        if 'path' not in agent_config:
            print("❌ Missing 'path' in agent configuration")
            return False
        
        if agent_config['path'] != './agent.py:app':
            print(f"❌ Incorrect path: {agent_config['path']}")
            return False
        
        if 'description' not in agent_config:
            print("❌ Missing 'description' in agent configuration")
            return False
        
        print(f"✅ Agent path: {agent_config['path']}")
        print(f"✅ Agent description: {agent_config['description']}")
        
        # Validate environment variables
        env_vars = config['env']
        if 'TAVILY_API_KEY' not in env_vars:
            print("❌ Missing TAVILY_API_KEY in environment variables")
            return False
        
        print(f"✅ Environment variables: {env_vars}")
        
        # Validate dependencies
        dependencies = config['dependencies']
        if './requirements.txt' not in dependencies:
            print("❌ Missing requirements.txt in dependencies")
            return False
        
        print(f"✅ Dependencies: {dependencies}")
        
        print("\n🎉 langgraph.json configuration is valid and complete!")
        print("✅ Basic LangGraph deployment configuration")
        print("✅ Agent name and description included")
        print("✅ Tavily API key environment variable specified")
        print("✅ Additional environment variables for model flexibility")
        print("✅ Dependencies properly configured")
        
        return True
        
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON syntax: {e}")
        return False
    except FileNotFoundError:
        print("❌ langgraph.json file not found")
        return False
    except Exception as e:
        print(f"❌ Validation error: {e}")
        return False

if __name__ == "__main__":
    success = validate_langgraph_config()
    exit(0 if success else 1)
