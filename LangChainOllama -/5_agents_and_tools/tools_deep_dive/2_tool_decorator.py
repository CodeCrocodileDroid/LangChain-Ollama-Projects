"""
Minimal @tool decorator agent - No complex imports
"""

import re


# Mock @tool decorator if not available
def tool(func=None, *, args_schema=None):
    def decorator(f):
        f.is_tool = True
        f.args_schema = args_schema
        f.name = f.__name__
        f.description = f.__doc__ or f.__name__
        return f

    return decorator if func is None else decorator(func)


# Define tools with our decorator
@tool()
def greet_user(name: str) -> str:
    """Greets the user by name."""
    return f"Hello, {name}!"


@tool()
def reverse_string(text: str) -> str:
    """Reverses the given string."""
    return text[::-1]


@tool()
def concatenate_strings(a: str, b: str) -> str:
    """Concatenates two strings."""
    return a + b


# Simple agent
tools = [greet_user, reverse_string, concatenate_strings]


def run_agent(command: str) -> str:
    """Simple agent logic"""
    cmd_lower = command.lower()

    if "greet" in cmd_lower:
        name = re.sub(r"greet\s*", "", cmd_lower, flags=re.IGNORECASE).strip()
        return greet_user(name or "there")

    elif "reverse" in cmd_lower:
        matches = re.findall(r"['\"]([^'\"]+)['\"]", command)
        text = matches[0] if matches else "hello"
        return reverse_string(text)

    elif "concatenate" in cmd_lower or "join" in cmd_lower:
        matches = re.findall(r"['\"]([^'\"]+)['\"]", command)
        if len(matches) >= 2:
            return concatenate_strings(matches[0], matches[1])
        return "Need two strings in quotes"

    return f"Try: greet name, reverse 'text', or concatenate 'str1' 'str2'"


# Test
print("Simple @tool Agent")
print("Type 'exit' to quit\n")

tests = [
    "Greet Alice",
    "Reverse 'hello'",
    "Concatenate 'hello' 'world'"
]

for test in tests:
    print(f"Test: {test}")
    print(f"Result: {run_agent(test)}\n")

# Interactive
while True:
    try:
        cmd = input("You: ").strip()
        if cmd.lower() == "exit":
            break
        print(f"Agent: {run_agent(cmd)}")
    except KeyboardInterrupt:
        print("\nGoodbye!")
        break