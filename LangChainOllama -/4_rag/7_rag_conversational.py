import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

# Simple RAG without complex chains
current_dir = os.path.dirname(os.path.abspath(__file__))
persistent_directory = os.path.join(current_dir, "db", "chroma_db_with_metadata")

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)
llm = ChatOllama(model="qwen2.5:0.5b")


def simple_rag_chat():
    print("Simple RAG chat - Type 'exit' to quit")
    chat_history = []

    while True:
        query = input("\nYou: ")
        if query.lower() == 'exit':
            break

        # Retrieve documents
        retriever = db.as_retriever(search_kwargs={"k": 3})
        docs = retriever.invoke(query)

        # Build context
        context = "\n\n".join([doc.page_content for doc in docs])

        # Create prompt with chat history
        history_text = "\n".join([f"You: {msg.content}" if hasattr(msg, 'content') else str(msg)
                                  for msg in chat_history[-4:]])  # Last 2 exchanges

        prompt = f"""Previous conversation:
{history_text}

Context from documents:
{context}

Question: {query}

Answer based only on the context above:"""

        # Get response
        response = llm.invoke(prompt)
        print(f"\nAI: {response.content}")

        # Update history
        chat_history.append(f"You: {query}")
        chat_history.append(f"AI: {response.content}")


if __name__ == "__main__":
    simple_rag_chat()