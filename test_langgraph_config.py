#!/usr/bin/env python3
"""
Test script for the langgraph.json configuration file
"""

import json
import os

try:
    # Test 1: Verify langgraph.json file exists and is valid JSON
    print("🧪 Test 1: Configuration file validation")
    
    config_path = "/home/daytona/task_1_research/langgraph.json"
    assert os.path.exists(config_path), "langgraph.json file does not exist"
    print("✅ langgraph.json file exists")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    print("✅ langgraph.json is valid JSON")
    
    # Test 2: Verify required dependencies are included
    print("\n🧪 Test 2: Dependencies validation")
    
    assert "dependencies" in config, "Missing 'dependencies' field"
    dependencies = config["dependencies"]
    
    required_deps = ["langgraph", "tavily-python", "langchain-anthropic", "langchain-core"]
    for dep in required_deps:
        assert dep in dependencies, f"Missing required dependency: {dep}"
        print(f"✅ Required dependency '{dep}' included")
    
    # Test 3: Verify graph configuration
    print("\n🧪 Test 3: Graph configuration validation")
    
    assert "graphs" in config, "Missing 'graphs' field"
    graphs = config["graphs"]
    
    assert "agent" in graphs, "Missing 'agent' graph configuration"
    assert graphs["agent"] == "./agent.py:app", "Incorrect agent graph path"
    print("✅ Agent graph properly configured to export from ./agent.py:app")
    
    # Test 4: Verify environment variables configuration
    print("\n🧪 Test 4: Environment variables validation")
    
    assert "env" in config, "Missing 'env' field"
    env_config = config["env"]
    
    # Check TAVILY_API_KEY configuration
    assert "TAVILY_API_KEY" in env_config, "Missing TAVILY_API_KEY environment variable"
    tavily_config = env_config["TAVILY_API_KEY"]
    assert "description" in tavily_config, "Missing description for TAVILY_API_KEY"
    assert "required" in tavily_config, "Missing required flag for TAVILY_API_KEY"
    assert tavily_config["required"] == True, "TAVILY_API_KEY should be required"
    assert "https://app.tavily.com/" in tavily_config["description"], "Missing Tavily URL in description"
    print("✅ TAVILY_API_KEY properly configured with description and required flag")
    
    # Check ANTHROPIC_API_KEY configuration
    assert "ANTHROPIC_API_KEY" in env_config, "Missing ANTHROPIC_API_KEY environment variable"
    anthropic_config = env_config["ANTHROPIC_API_KEY"]
    assert "description" in anthropic_config, "Missing description for ANTHROPIC_API_KEY"
    assert "required" in anthropic_config, "Missing required flag for ANTHROPIC_API_KEY"
    assert anthropic_config["required"] == True, "ANTHROPIC_API_KEY should be required"
    assert "https://console.anthropic.com/" in anthropic_config["description"], "Missing Anthropic URL in description"
    print("✅ ANTHROPIC_API_KEY properly configured with description and required flag")
    
    # Test 5: Verify deployment settings
    print("\n🧪 Test 5: Deployment settings validation")
    
    assert "python_version" in config, "Missing python_version field"
    assert config["python_version"] == "3.11", "Python version should be 3.11"
    print("✅ Python version properly set to 3.11")
    
    assert "dockerfile_lines" in config, "Missing dockerfile_lines field"
    assert isinstance(config["dockerfile_lines"], list), "dockerfile_lines should be a list"
    print("✅ Dockerfile lines field properly configured")
    
    # Test 6: Verify agent.py export compatibility
    print("\n🧪 Test 6: Agent export compatibility validation")
    
    # Check that agent.py exists and exports 'app'
    agent_path = "/home/daytona/task_1_research/agent.py"
    assert os.path.exists(agent_path), "agent.py file does not exist"
    
    with open(agent_path, 'r') as f:
        agent_content = f.read()
    
    assert "app = " in agent_content, "agent.py does not export 'app' variable"
    assert "compile()" in agent_content, "agent.py does not compile the graph"
    print("✅ agent.py properly exports compiled graph as 'app'")
    
    # Test 7: Configuration completeness check
    print("\n🧪 Test 7: Configuration completeness validation")
    
    expected_fields = ["dependencies", "graphs", "env", "dockerfile_lines", "python_version"]
    for field in expected_fields:
        assert field in config, f"Missing required configuration field: {field}"
    print("✅ All required configuration fields present")
    
    # Test 8: JSON structure validation
    print("\n🧪 Test 8: JSON structure validation")
    
    # Verify the JSON can be serialized back (no circular references, etc.)
    json_str = json.dumps(config, indent=2)
    reparsed_config = json.loads(json_str)
    assert reparsed_config == config, "Configuration cannot be properly serialized/deserialized"
    print("✅ JSON structure is valid and serializable")
    
    print("\n✅ All langgraph.json configuration tests passed!")
    print("📋 Configuration file verified:")
    print("   - Valid JSON structure with all required fields")
    print("   - All required dependencies included")
    print("   - Proper graph export configuration")
    print("   - Environment variables with descriptions and URLs")
    print("   - Deployment-ready settings")
    print("   - Compatible with agent.py export pattern")
    print("   - Ready for LangGraph deployment")
    
except Exception as e:
    print(f"❌ Error validating langgraph.json configuration: {e}")
    import traceback
    traceback.print_exc()
