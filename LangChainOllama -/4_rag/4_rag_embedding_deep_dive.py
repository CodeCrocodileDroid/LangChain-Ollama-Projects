import os

# Updated imports for the modular packages
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
# Use the new, dedicated packages for vector stores and embeddings
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Define the directory containing the text file and the persistent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "books", "odyssey.txt")
db_dir = os.path.join(current_dir, "db")

# Check if the text file exists
if not os.path.exists(file_path):
    raise FileNotFoundError(
        f"The file {file_path} does not exist. Please check the path."
    )

# Read the text content from the file
loader = TextLoader(file_path, encoding='utf-8')  # Specify encoding to avoid errors
documents = loader.load()

# Split the document into chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

# Display information about the split documents
print("\n--- Document Chunks Information ---")
print(f"Number of document chunks: {len(docs)}")
print(f"Sample chunk:\n{docs[0].page_content[:300]}...\n")  # Show a preview


# Function to create and persist vector store
def create_vector_store(docs, embeddings, store_name):
    persistent_directory = os.path.join(db_dir, store_name)
    if not os.path.exists(persistent_directory):
        print(f"\n--- Creating vector store {store_name} ---")
        Chroma.from_documents(
            docs, embeddings, persist_directory=persistent_directory)
        print(f"--- Finished creating vector store {store_name} ---")
    else:
        print(
            f"Vector store {store_name} already exists. No need to initialize.")


# Hugging Face Transformers ONLY (no OpenAI)
print("\n--- Using Hugging Face Transformers ---")
# Using the smaller, faster model (still very capable)
huggingface_embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"  # Changed to smaller model
)
create_vector_store(docs, huggingface_embeddings, "chroma_db_huggingface")

print("Embedding demonstration for Hugging Face completed.")


# Function to query a vector store
def query_vector_store(store_name, query, embedding_function):
    persistent_directory = os.path.join(db_dir, store_name)
    if os.path.exists(persistent_directory):
        print(f"\n--- Querying the Vector Store {store_name} ---")
        db = Chroma(
            persist_directory=persistent_directory,
            embedding_function=embedding_function,
        )
        # Removed 'score_threshold' as it's not consistently supported across all backends.
        retriever = db.as_retriever(search_kwargs={"k": 3})
        relevant_docs = retriever.invoke(query)
        # Display the relevant results with metadata
        print(f"\n--- Relevant Documents for {store_name} ---")
        if relevant_docs:
            for i, doc in enumerate(relevant_docs, 1):
                print(f"Document {i}:\n{doc.page_content}\n")
                if doc.metadata:
                    print(f"Source: {doc.metadata.get('source', 'Unknown')}\n")
        else:
            print("No documents found.\n")
    else:
        print(f"Vector store {store_name} does not exist.")


# Define the user's question
query = "Who is Odysseus' wife?"

# Query the HuggingFace vector store
query_vector_store("chroma_db_huggingface", query, huggingface_embeddings)

print("Querying demonstration completed.")