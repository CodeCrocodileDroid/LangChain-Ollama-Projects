import os

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# Define the persistent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
db_dir = os.path.join(current_dir, "db")
persistent_directory = os.path.join(db_dir, "chroma_db_with_metadata")

# Define the embedding model - using HuggingFace (free, local)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


# Function to query a vector store with different search types and parameters
def query_vector_store(query, search_type, search_kwargs):
    if os.path.exists(persistent_directory):
        print(f"\n--- Querying Vector Store ---")
        db = Chroma(
            persist_directory=persistent_directory,
            embedding_function=embeddings,
        )
        retriever = db.as_retriever(
            search_type=search_type,
            search_kwargs=search_kwargs,
        )
        relevant_docs = retriever.invoke(query)
        # Display the relevant results with metadata
        print(f"\n--- Relevant Documents ---")
        if relevant_docs:
            for i, doc in enumerate(relevant_docs, 1):
                print(f"Document {i}:\n{doc.page_content}\n")
                if doc.metadata:
                    print(f"Source: {doc.metadata.get('source', 'Unknown')}\n")
        else:
            print("No relevant documents found.\n")
    else:
        print(f"Vector store does not exist.")
        print(f"Create it first by running your vector store creation script.")


# Define the user's question
query = "How did Juliet die?"

# Showcase different retrieval methods

# 1. Similarity Search
print("\n=== 1. Similarity Search ===")
print("Retrieves top k most similar documents")
query_vector_store(query, "similarity", {"k": 3})

# 2. Max Marginal Relevance (MMR)
print("\n=== 2. Max Marginal Relevance (MMR) ===")
print("Balances relevance and diversity")
query_vector_store(query, "mmr", {"k": 3, "fetch_k": 20, "lambda_mult": 0.5})

# Note: HuggingFace embeddings may not support similarity_score_threshold well
# 3. Similarity Search with filter
print("\n=== 3. Similarity Search with Source Filter ===")
print("Filters results by source file")
query_vector_store(query, "similarity", {
    "k": 3,
    "filter": {"source": "romeo_and_juliet.txt"}  # Example filter
})

print("Querying demonstrations with different search types completed.")