import ollama

# Create and invoke the model with a message
response = ollama.chat(
    model='gemma3:1b',
    messages=[
        {'role': 'user', 'content': "What is 881 multiplied by 91?"}
    ]
)

result = response['message']

print("Full result:")
print(result)
print("\nContent only:")
print(result['content'])