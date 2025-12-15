# Absolute simplest version that should work
from langchain_ollama import ChatOllama
import datetime

# Direct approach without complex agents
llm = ChatOllama(model="qwen2.5:0.5b")

def simple_agent(query):
    # Check what the user wants
    if "time" in query.lower():
        now = datetime.datetime.now()
        return f"The current time is {now.strftime('%I:%M %p')}"
    elif any(op in query for op in ['+', '-', '*', '/', 'plus', 'minus', 'times', 'divided']):
        # Try to extract numbers
        import re
        numbers = re.findall(r'\d+', query)
        if len(numbers) >= 2:
            a, b = int(numbers[0]), int(numbers[1])
            if '+' in query or 'plus' in query:
                return f"{a} + {b} = {a + b}"
            elif '-' in query or 'minus' in query:
                return f"{a} - {b} = {a - b}"
            elif '*' in query or 'times' in query:
                return f"{a} * {b} = {a * b}"
            elif '/' in query or 'divided' in query:
                return f"{a} / {b} = {a / b}"
        return "I can help with calculations. Please provide numbers."
    else:
        # Use LLM for other queries
        return llm.invoke(query).content

# Test
print(simple_agent("What time is it?"))
print(simple_agent("What is 15 + 27?"))