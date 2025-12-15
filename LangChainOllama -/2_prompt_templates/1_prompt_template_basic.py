from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage

# PART 1: Create a ChatPromptTemplate using a template string
template = "Tell me a joke about {topic}."
prompt_template = ChatPromptTemplate.from_template(template)

print("-----Prompt from Template-----")
prompt = prompt_template.invoke({"topic": "cats"})
print(prompt)
print()

# PART 2: Prompt with Multiple Placeholders
template_multiple = """You are a helpful assistant.
Human: Tell me a {adjective} story about a {animal}.
Assistant:"""
prompt_multiple = ChatPromptTemplate.from_template(template_multiple)
prompt = prompt_multiple.invoke({"adjective": "funny", "animal": "panda"})
print("----- Prompt with Multiple Placeholders -----")
print(prompt)
print()

# PART 3: Prompt with System and Human Messages (Using Tuples)
messages = [
    ("system", "You are a comedian who tells jokes about {topic}."),
    ("human", "Tell me {joke_count} jokes."),
]
prompt_template = ChatPromptTemplate.from_messages(messages)
prompt = prompt_template.invoke({"topic": "lawyers", "joke_count": 3})
print("----- Prompt with System and Human Messages (Tuple) -----")
print(prompt)
print()

# Extra Information about Part 3.
# This does work:
messages = [
    ("system", "You are a comedian who tells jokes about {topic}."),
    HumanMessage(content="Tell me 3 jokes."),
]
prompt_template = ChatPromptTemplate.from_messages(messages)
prompt = prompt_template.invoke({"topic": "lawyers"})
print("----- Prompt with System Message (Tuple) + HumanMessage -----")
print(prompt)
print()

# This does NOT work:
try:
    messages = [
        ("system", "You are a comedian who tells jokes about {topic}."),
        HumanMessage(content="Tell me {joke_count} jokes."),
    ]
    prompt_template = ChatPromptTemplate.from_messages(messages)
    prompt = prompt_template.invoke({"topic": "lawyers", "joke_count": 3})
    print("----- This should fail -----")
    print(prompt)
except Exception as e:
    print("----- This correctly fails -----")
    print(f"Error: {e}")