#!/usr/bin/env python3
"""Simple script to fix trailing whitespace in agent.py"""

with open('agent.py', 'r') as f:
    content = f.read()

# Remove trailing whitespace and ensure single newline at end
content = content.rstrip() + '\n'

with open('agent.py', 'w') as f:
    f.write(content)

print("Fixed trailing whitespace in agent.py")
