from langchain_ollama import ChatOllama
import datetime

# Ultra-simple chat with manual memory
llm = ChatOllama(model="qwen2.5:0.5b")
conversation_history = []


def get_time():
    return datetime.datetime.now().strftime("%I:%M %p")


print("🤖 Simple Chat Agent (type 'exit' to quit)")
print("-" * 40)

while True:
    user_input = input("\nYou: ")
    if user_input.lower() == 'exit':
        break

    # Check for specific commands
    if "time" in user_input.lower():
        response = f"The current time is {get_time()}"
    else:
        # Build context from manual history (last 3 exchanges)
        history_text = ""
        for i in range(max(0, len(conversation_history) - 6), len(conversation_history), 2):
            if i + 1 < len(conversation_history):
                history_text += f"User: {conversation_history[i]}\n"
                history_text += f"Assistant: {conversation_history[i + 1]}\n\n"

        prompt = f"{history_text}User: {user_input}\n\nAssistant:"

        # Get response
        try:
            llm_response = llm.invoke(prompt)
            response = llm_response.content
        except Exception as e:
            response = f"Error: {e}"
            print("Make sure Ollama is running: ollama serve")

    print(f"\nBot: {response}")

    # Update manual history
    conversation_history.append(user_input)
    conversation_history.append(response)