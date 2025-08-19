#!/usr/bin/env python3
"""Remove trailing blank lines from agent.py"""

with open('agent.py', 'r') as f:
    lines = f.readlines()

# Remove trailing blank lines
while lines and lines[-1].strip() == '':
    lines.pop()

# Write back without adding extra newline
with open('agent.py', 'w') as f:
    f.writelines(lines)

print("Fixed trailing blank lines in agent.py")
