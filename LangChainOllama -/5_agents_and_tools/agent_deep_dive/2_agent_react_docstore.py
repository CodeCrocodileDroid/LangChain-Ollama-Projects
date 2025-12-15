"""
Local Qwen RAG System - Compatible with your all-MiniLM-L6-v2 vector store
Free, local, and properly configured for 384-dimensional embeddings
"""

import os
import sys
from typing import List, Dict, Any

# Load environment variables
from dotenv import load_dotenv

load_dotenv()

try:
    # Use HuggingFace embeddings (same as your original vector store)
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_chroma import Chroma  # Updated Chroma import
    from langchain_ollama import OllamaLLM  # Updated Ollama import
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.messages import AIMessage, HumanMessage
    from langchain_core.documents import Document
except ImportError as e:
    print(f"Missing packages: {e}")
    print("\nInstalling required packages...")
    import subprocess

    subprocess.check_call([sys.executable, "-m", "pip", "install",
                           "langchain-chroma", "langchain-ollama",
                           "langchain-community", "langchain-core",
                           "chromadb", "python-dotenv", "sentence-transformers"])
    print("Please restart the script.")
    sys.exit(1)


def setup_rag_system():
    """Initialize RAG system compatible with your 384-dim vector store"""

    print("=" * 60)
    print("Setting up Local Qwen RAG System")
    print("=" * 60)

    # Load the existing Chroma vector store (same as your working script)
    print("\n1. Loading vector store...")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    db_dir = os.path.join(current_dir, "..", "..", "4_rag", "db")
    persistent_directory = os.path.join(db_dir, "chroma_db_with_metadata")

    if not os.path.exists(persistent_directory):
        raise FileNotFoundError(f"Vector store not found at: {persistent_directory}")

    # Use EXACTLY the same embeddings as your working script
    print("2. Loading HuggingFace embeddings (all-MiniLM-L6-v2, 384-dim)...")
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},  # Use CPU for compatibility
            encode_kwargs={'normalize_embeddings': True}
        )
        print("✓ HuggingFace embeddings loaded successfully")
    except Exception as e:
        print(f"Error loading embeddings: {e}")
        print("\nTrying alternative...")
        # Simpler fallback
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Load Chroma vector store
    print("3. Connecting to Chroma database...")
    try:
        vectorstore = Chroma(
            persist_directory=persistent_directory,
            embedding_function=embeddings
        )

        # Test the connection
        test_docs = vectorstore.similarity_search("test", k=1)
        print(f"✓ Chroma loaded successfully ({len(test_docs)} test docs found)")
    except Exception as e:
        print(f"Error loading Chroma: {e}")

        # Fallback to community import if needed
        try:
            from langchain_community.vectorstores import Chroma as ChromaCommunity
            print("Trying community Chroma import...")
            vectorstore = ChromaCommunity(
                persist_directory=persistent_directory,
                embedding_function=embeddings
            )
            print("✓ Chroma (community) loaded successfully")
        except Exception as e2:
            raise Exception(f"Both Chroma imports failed: {e2}")

    # Initialize Qwen LLM
    print("4. Loading Qwen LLM (qwen2.5:0.5b)...")
    try:
        llm = OllamaLLM(
            model="qwen2.5:0.5b",
            temperature=0.7,
            num_predict=512,
            num_ctx=2048
        )

        # Quick test of the LLM
        test_response = llm.invoke("Say 'hello' in one word.")
        print(f"✓ Qwen LLM loaded successfully (test: {test_response.strip()})")
    except Exception as e:
        print(f"Error loading LLM: {e}")
        print("\nPlease ensure:")
        print("1. Ollama is running: ollama serve")
        print("2. Model is pulled: ollama pull qwen2.5:0.5b")
        raise

    # Create retriever (EXACTLY like your working script)
    print("5. Creating document retriever...")
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": 4,  # Retrieve 4 documents (you used 3 in your script)
            "score_threshold": 0.3  # Optional: minimum similarity score
        }
    )

    # Test the retriever
    print("6. Testing retriever with sample query...")
    try:
        test_query = "test"
        test_docs = retriever.invoke(test_query)
        print(f"✓ Retriever works ({len(test_docs)} documents retrieved)")
    except Exception as e:
        print(f"⚠ Retriever test warning: {e}")

    # Define prompts
    print("7. Setting up prompts...")

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful AI assistant. Answer the question using ONLY the provided context.

        Follow these rules:
        1. If the context contains the answer, provide a clear answer based on it
        2. If the context doesn't contain enough information, say "I don't have enough information to answer that based on the provided documents."
        3. Keep answers concise (2-3 sentences maximum)
        4. Do NOT make up information not in the context

        Context: {context}

        Question: {question}

        Answer:"""),
    ])

    print("\n" + "=" * 60)
    print("RAG System Ready!")
    print("Configuration:")
    print(f"  • Embeddings: HuggingFace all-MiniLM-L6-v2 (384-dim)")
    print(f"  • LLM: qwen2.5:0.5b via Ollama")
    print(f"  • Vector store: {persistent_directory}")
    print("=" * 60)

    return {
        "llm": llm,
        "vectorstore": vectorstore,
        "retriever": retriever,
        "qa_prompt": qa_prompt,
        "embeddings": embeddings
    }


def simple_rag_query(rag_system, question: str) -> Dict[str, Any]:
    """Simple RAG query function"""

    # Retrieve relevant documents
    docs = rag_system["retriever"].invoke(question)

    # Format context from documents
    if docs:
        context = "\n\n".join([
            f"Document {i + 1}:\n{doc.page_content}"
            for i, doc in enumerate(docs)
        ])
    else:
        context = "No relevant documents found."

    # Create prompt messages
    messages = rag_system["qa_prompt"].format_messages(
        context=context,
        question=question
    )

    # Get answer from LLM
    answer = rag_system["llm"].invoke(messages)

    return {
        "question": question,
        "answer": answer.strip(),
        "sources": docs,
        "context": context,
        "num_sources": len(docs)
    }


def chat_interface(rag_system):
    """Interactive chat interface"""

    chat_history = []

    print("\n" + "=" * 50)
    print("Chat Interface - Type 'exit' to quit")
    print("=" * 50)
    print("Commands: exit, clear, history, sources, test")
    print("-" * 50)

    while True:
        try:
            user_input = input("\nYou: ").strip()

            if not user_input:
                continue

            # Handle commands
            if user_input.lower() == "exit":
                print("\nGoodbye!")
                break

            elif user_input.lower() == "clear":
                chat_history.clear()
                print("Chat history cleared.")
                continue

            elif user_input.lower() == "history":
                if not chat_history:
                    print("No chat history.")
                else:
                    print("\nChat History (last 5 exchanges):")
                    start = max(0, len(chat_history) - 10)
                    for i in range(start, len(chat_history), 2):
                        if i + 1 < len(chat_history):
                            print(f"\nQ: {chat_history[i].content[:100]}...")
                            print(f"A: {chat_history[i + 1].content[:100]}...")
                continue

            elif user_input.lower() == "sources":
                if chat_history and isinstance(chat_history[-1], AIMessage):
                    # Show sources from last answer
                    last_question = chat_history[-2].content if len(chat_history) >= 2 else ""
                    if last_question:
                        docs = rag_system["retriever"].invoke(last_question)
                        print(f"\nSources for last question:")
                        for i, doc in enumerate(docs):
                            print(f"\n--- Source {i + 1} ---")
                            print(f"Content: {doc.page_content[:150]}...")
                            if hasattr(doc, 'metadata') and doc.metadata:
                                print(f"Metadata: {list(doc.metadata.items())[:2]}")
                else:
                    print("No previous answer to show sources for.")
                continue

            elif user_input.lower() == "test":
                # Test with a simple query
                print("\nRunning test query: 'What is this document collection about?'")
                result = simple_rag_query(rag_system, "What is this document collection about?")
                print(f"Test Answer: {result['answer'][:150]}...")
                continue

            # Process user question
            print("Thinking...", end=" ", flush=True)

            result = simple_rag_query(rag_system, user_input)

            print(f"\nAI: {result['answer']}")

            # Show source count
            if result['num_sources'] > 0:
                print(f"[Based on {result['num_sources']} relevant document(s)]")

            # Update chat history
            chat_history.append(HumanMessage(content=user_input))
            chat_history.append(AIMessage(content=result['answer']))

            # Option to show detailed sources
            if result['num_sources'] > 0:
                show_sources = input("\nShow source details? (y/n): ").lower()
                if show_sources == 'y':
                    print(f"\n{'=' * 40}")
                    print(f"SOURCES ({result['num_sources']} found)")
                    print('=' * 40)
                    for i, doc in enumerate(result['sources']):
                        print(f"\n📄 Source {i + 1}:")
                        print(f"Content: {doc.page_content[:200]}...")
                        if hasattr(doc, 'metadata') and doc.metadata:
                            for key, value in list(doc.metadata.items())[:3]:
                                print(f"  {key}: {value}")
                        print("-" * 40)

            print("\n" + "-" * 40)

        except KeyboardInterrupt:
            print("\n\nInterrupted. Type 'exit' to quit or ask another question.")
        except Exception as e:
            print(f"\nError: {e}")
            print("Please try again.")


def test_vector_store():
    """Test the vector store directly (like your working script)"""
    print("\n" + "=" * 60)
    print("Testing Vector Store Directly")
    print("=" * 60)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    db_dir = os.path.join(current_dir, "..", "..", "4_rag", "db")
    persistent_directory = os.path.join(db_dir, "chroma_db_with_metadata")

    if not os.path.exists(persistent_directory):
        print("Vector store not found!")
        return False

    try:
        # Exact same code as your working script
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from langchain_community.vectorstores import Chroma

        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        db = Chroma(
            persist_directory=persistent_directory,
            embedding_function=embeddings
        )

        test_query = "How did Juliet die?"
        retriever = db.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )

        relevant_docs = retriever.invoke(test_query)

        print(f"\nTest query: '{test_query}'")
        print(f"Found {len(relevant_docs)} relevant documents")

        if relevant_docs:
            for i, doc in enumerate(relevant_docs, 1):
                print(f"\n--- Document {i} ---")
                print(f"Content: {doc.page_content[:150]}...")
                if hasattr(doc, 'metadata'):
                    print(f"Source: {doc.metadata.get('source', 'Unknown')}")

        print("\n" + "=" * 60)
        print("Vector store test PASSED! ✅")
        print("=" * 60)
        return True

    except Exception as e:
        print(f"\nVector store test FAILED: {e}")
        print("=" * 60)
        return False


def main():
    """Main function"""
    try:
        print("Starting RAG System...\n")

        # First, test the vector store directly
        if not test_vector_store():
            print("\nVector store test failed. Cannot continue.")
            return

        # Setup the full RAG system
        rag_system = setup_rag_system()

        # Start the chat interface
        chat_interface(rag_system)

    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nPlease ensure your vector store exists at:")
        print(".../4_rag/db/chroma_db_with_metadata")

    except Exception as e:
        print(f"\nFatal error: {e}")
        print("\nTroubleshooting:")
        print("1. Install missing packages:")
        print("   pip install langchain-community langchain-chroma langchain-ollama")
        print("2. Ensure Ollama is running: ollama serve")
        print("3. Pull Qwen model: ollama pull qwen2.5:0.5b")
        print("4. For embeddings, sentence-transformers is needed:")
        print("   pip install sentence-transformers")


if __name__ == "__main__":
    main()