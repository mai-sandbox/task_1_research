#!/usr/bin/env python3
"""Check for formatting issues in agent.py"""

with open('agent.py', 'r') as f:
    lines = f.readlines()

issues = []

for i, line in enumerate(lines, 1):
    # Check for trailing whitespace
    if line.rstrip() != line.rstrip('\n'):
        issues.append(f"Line {i}: trailing whitespace")
    
    # Check for blank lines with whitespace
    if line.strip() == '' and line != '\n':
        issues.append(f"Line {i}: blank line with whitespace")

if issues:
    print("Formatting issues found:")
    for issue in issues:
        print(f"  {issue}")
else:
    print("No formatting issues found")
