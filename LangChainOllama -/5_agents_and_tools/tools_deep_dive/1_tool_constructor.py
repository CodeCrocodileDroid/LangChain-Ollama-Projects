"""
Minimal Working Agent - No Complex Imports
"""

import re


# Tool functions
def greet_user(name: str) -> str:
    return f"Hello, {name}!"


def reverse_string(text: str) -> str:
    return text[::-1]


def concatenate_strings(a: str, b: str) -> str:
    return a + b


# Simple agent logic
def process_command(command: str) -> str:
    command_lower = command.lower()

    if "greet" in command_lower:
        # Extract name
        name = command_lower.replace("greet", "").strip()
        if not name:
            name = "there"
        return greet_user(name.title())

    elif "reverse" in command_lower:
        # Extract text in quotes
        matches = re.findall(r"['\"]([^'\"]+)['\"]", command)
        if matches:
            return reverse_string(matches[0])
        else:
            # Try to extract after "reverse"
            parts = command_lower.split("reverse")
            if len(parts) > 1:
                text = parts[1].strip()
                return reverse_string(text)
            return reverse_string("hello")

    elif "concatenate" in command_lower or "join" in command_lower:
        matches = re.findall(r"['\"]([^'\"]+)['\"]", command)
        if len(matches) >= 2:
            return concatenate_strings(matches[0], matches[1])
        return "Please provide two strings in quotes"

    else:
        return f"I received: {command}. Try: greet [name], reverse 'text', or concatenate 'str1' 'str2'"


# Main loop
print("Simple Agent - Type 'exit' to quit")
print("Commands: greet [name], reverse 'text', concatenate 'str1' 'str2'")

while True:
    try:
        user_input = input("\nYou: ").strip()
        if user_input.lower() == "exit":
            break
        response = process_command(user_input)
        print(f"Agent: {response}")
    except KeyboardInterrupt:
        print("\nGoodbye!")
        break
    except Exception as e:
        print(f"Error: {e}")