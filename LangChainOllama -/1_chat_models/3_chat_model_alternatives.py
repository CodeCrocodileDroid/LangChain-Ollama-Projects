from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

# Setup messages
messages = [
    SystemMessage(content="Solve the following math problems"),
    HumanMessage(content="What is 81 divided by 9?"),
]

# Use ONLY the model you know exists
model = ChatOllama(model="qwen2.5:0.5b")

# Invoke the model with messages
result = model.invoke(messages)
print(f"Answer from Ollama: {result.content}")

# Optional: Try another question with the same model
messages2 = [
    SystemMessage(content="Solve math problems"),
    HumanMessage(content="What is 10 times 5?"),
]

result2 = model.invoke(messages2)
print(f"\nSecond answer: {result2.content}")