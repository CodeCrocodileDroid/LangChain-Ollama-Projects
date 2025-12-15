import os
from langchain_huggingface import HuggingFaceEmbeddings  # Updated import
from langchain_chroma import Chroma  # Updated import
from langchain_ollama import ChatOllama  # For generating answers
from langchain_core.prompts import ChatPromptTemplate

# Define the persistent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
persistent_directory = os.path.join(current_dir, "db", "chroma_db")

# Define the embedding model
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Load the existing vector store
db = Chroma(persist_directory=persistent_directory,
            embedding_function=embeddings)

# Define the user's question
query = "Who is Odysseus' wife?"

# Retrieve relevant documents
retriever = db.as_retriever(search_kwargs={"k": 3})
relevant_docs = retriever.invoke(query)

# Combine retrieved documents into context
context = "\n\n".join([doc.page_content for doc in relevant_docs])

# Create a prompt template for answer generation
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant answering questions based on the provided context."),
    ("human", "Context: {context}\n\nQuestion: {query}\n\nAnswer:")
])

# Initialize a chat model (using Ollama)
llm = ChatOllama(model="qwen2.5:0.5b")

# Create a chain: prompt → LLM
chain = prompt_template | llm

# Generate answer
result = chain.invoke({"context": context, "query": query})

print("=== RAG ANSWER ===")
print(f"Question: {query}")
print(f"Answer: {result.content}")
print("\n=== Supporting Documents ===")
for i, doc in enumerate(relevant_docs, 1):
    print(f"\nDocument {i} (first 200 chars):")
    print(doc.page_content[:200] + "...")