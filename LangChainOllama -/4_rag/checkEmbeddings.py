import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
persistent_directory = os.path.join(current_dir, "db", "chroma_db")

# Try to get collection info. You might need chromadb installed.
try:
    import chromadb
    client = chromadb.PersistentClient(path=persistent_directory)
    collection = client.get_collection(name="langchain") # Default name, adjust if you changed it
    print(f"Collection info: {collection}")
    # If the 'metadata' method is available, you might find dimensionality there.
except Exception as e:
    print(f"Could not inspect database directly. Error: {e}")
    print("\nTrying a simpler test:")
    # Test the dimension of the new embedding model
    from langchain_ollama import OllamaEmbeddings
    new_embeddings = OllamaEmbeddings(model="nomic-embed-text")
    test_vector = new_embeddings.embed_query("test query")
    print(f"New model ('nomic-embed-text') creates vectors with dimension: {len(test_vector)}")