import os

from langchain_text_splitters import CharacterTextSplitter  # Updated import
from langchain_community.document_loaders import WebBaseLoader  # Free alternative to FireCrawl
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings  # Free embeddings

# Define the persistent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
db_dir = os.path.join(current_dir, "db")
persistent_directory = os.path.join(db_dir, "chroma_db_web")


def create_vector_store():
    """Scrape the website, split the content, create embeddings, and persist the vector store."""
    # Step 1: Scrape the website using WebBaseLoader (free alternative to FireCrawl)
    print("Begin scraping the website...")

    # You can scrape multiple URLs for better coverage
    urls = [
        "https://www.apple.com/",
        "https://www.apple.com/apple-intelligence/",  # Direct link to Apple Intelligence
        "https://www.apple.com/newsroom/",  # Apple news/announcements
    ]

    loader = WebBaseLoader(urls)
    docs = loader.load()
    print(f"Finished scraping. Loaded {len(docs)} documents.")

    # Step 2: Split the scraped content into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    split_docs = text_splitter.split_documents(docs)

    # Display information about the split documents
    print("\n--- Document Chunks Information ---")
    print(f"Number of document chunks: {len(split_docs)}")
    if split_docs:
        print(f"Sample chunk (first 300 chars):\n{split_docs[0].page_content[:300]}...\n")

    # Step 3: Create embeddings for the document chunks (free, local)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Step 4: Create and persist the vector store with the embeddings
    print(f"\n--- Creating vector store in {persistent_directory} ---")
    db = Chroma.from_documents(
        split_docs, embeddings, persist_directory=persistent_directory
    )
    print(f"--- Finished creating vector store in {persistent_directory} ---")

    return db


# Check if the Chroma vector store already exists
if not os.path.exists(persistent_directory):
    db = create_vector_store()
else:
    print(f"Vector store {persistent_directory} already exists. Loading...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma(persist_directory=persistent_directory,
                embedding_function=embeddings)


# Step 5: Query the vector store
def query_vector_store(query):
    """Query the vector store with the specified question."""
    # Create a retriever for querying the vector store
    retriever = db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5},  # Get more results for better context
    )

    # Retrieve relevant documents based on the query
    relevant_docs = retriever.invoke(query)

    # Display the relevant results with metadata
    print(f"\n--- Relevant Documents for '{query}' ---")
    if relevant_docs:
        for i, doc in enumerate(relevant_docs, 1):
            source = doc.metadata.get('source', 'Unknown')
            print(f"\nDocument {i} (from {source}):")
            print(f"{doc.page_content[:400]}...")  # Show first 400 chars
    else:
        print("No relevant documents found.")
        print("Try scraping different URLs or check your query.")


# Define user questions
queries = [
    "What is Apple Intelligence?",
    "What new features does Apple Intelligence have?",
    "How does Apple Intelligence work?",
]

# Query the vector store with multiple questions
for query in queries:
    query_vector_store(query)
    print("\n" + "=" * 60)