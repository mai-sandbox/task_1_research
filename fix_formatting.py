#!/usr/bin/env python3
"""Fix formatting issues in agent.py"""

with open('agent.py', 'r') as f:
    lines = f.readlines()

# Clean up each line
cleaned_lines = []
for line in lines:
    # Remove trailing whitespace but preserve newline
    cleaned_line = line.rstrip() + '\n' if line.strip() or line == '\n' else '\n'
    cleaned_lines.append(cleaned_line)

# Write back the cleaned content
with open('agent.py', 'w') as f:
    f.writelines(cleaned_lines)

print("Formatting issues fixed!")
print("- Removed trailing whitespace from all lines")
print("- Cleaned up blank lines with whitespace")
