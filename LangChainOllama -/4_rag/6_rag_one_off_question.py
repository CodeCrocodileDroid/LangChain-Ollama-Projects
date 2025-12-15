import os

from langchain_community.vectorstores import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama

# Define the persistent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
persistent_directory = os.path.join(
    current_dir, "db", "chroma_db_with_metadata")

# Define the embedding model - using HuggingFace (free, local)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Load the existing vector store with the embedding function
try:
    db = Chroma(persist_directory=persistent_directory,
                embedding_function=embeddings)
    print(f"✅ Loaded vector store from: {persistent_directory}")
except Exception as e:
    print(f"❌ Error loading vector store: {e}")
    print("Make sure the vector store was created with the same embedding model!")
    exit(1)

# Define the user's question
query = "How can I learn more about LangChain?"

# Retrieve relevant documents based on the query
retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3},  # Get more documents for better context
)
relevant_docs = retriever.invoke(query)

# Display the relevant results with metadata
print("\n--- Relevant Documents ---")
if relevant_docs:
    for i, doc in enumerate(relevant_docs, 1):
        source = doc.metadata.get('source', 'Unknown')
        print(f"Document {i} (from {source}):")
        print(f"{doc.page_content[:200]}...\n")  # Show first 200 chars
else:
    print("No relevant documents found.")
    print("Make sure your vector store contains relevant information.")
    exit(0)

# Combine the query and the relevant document contents
combined_input = (
        "You are a helpful assistant. Answer the question based ONLY on the provided documents.\n\n"
        "QUESTION: " + query + "\n\n"
                               "DOCUMENTS:\n" + "\n\n---\n\n".join([doc.page_content for doc in relevant_docs]) + "\n\n"
                                                                                                                  "INSTRUCTIONS:\n"
                                                                                                                  "1. Answer using ONLY information from the documents above.\n"
                                                                                                                  "2. If the answer is not in the documents, say 'I cannot find this information in the provided documents.'\n"
                                                                                                                  "3. Keep your answer clear and concise.\n\n"
                                                                                                                  "ANSWER:"
)

# Create a ChatOllama model (free, local)
model = ChatOllama(model="qwen2.5:0.5b")

# Define the messages for the model
messages = [
    SystemMessage(content="You are a helpful assistant that answers questions based only on provided documents."),
    HumanMessage(content=combined_input),
]

print("\n--- Generating Response ---")
try:
    # Invoke the model with the combined input
    result = model.invoke(messages)

    print("\n--- Generated Response ---")
    print("Content only:")
    print(result.content)

    # Optional: Show what sources were used
    if relevant_docs:
        print("\n--- Sources Used ---")
        for i, doc in enumerate(relevant_docs, 1):
            print(f"{i}. From: {doc.metadata.get('source', 'Unknown')}")

except Exception as e:
    print(f"❌ Error generating response: {e}")
    print("Make sure Ollama is running: ollama serve")
    print("And you have the model: ollama pull qwen2.5:0.5b")