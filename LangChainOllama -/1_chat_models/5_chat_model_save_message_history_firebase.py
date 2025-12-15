from langchain_ollama import ChatOllama
from langchain_core.messages import AIMessage, HumanMessage

# Initialize Chat Model
model = ChatOllama(model="qwen2.5:0.5b")

# Simple chat history stored in memory
chat_history = []

print("Start chatting with the AI. Type 'exit' to quit.")

while True:
    human_input = input("User: ")
    if human_input.lower() == "exit":
        break

    # Add user message to history
    chat_history.append(HumanMessage(content=human_input))

    # Get AI response using history
    ai_response = model.invoke(chat_history)

    # Add AI response to history
    chat_history.append(AIMessage(content=ai_response.content))

    print(f"AI: {ai_response.content}")

    # Show current history length
    print(f"[History: {len(chat_history)} messages]")

print("\n---- Final Chat History ----")
for i, msg in enumerate(chat_history):
    role = "Human" if isinstance(msg, HumanMessage) else "AI"
    print(f"{i + 1}. {role}: {msg.content}")