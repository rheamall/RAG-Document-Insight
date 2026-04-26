import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()


def load_vectorstore() -> FAISS:
    """Load the FAISS index from disk."""
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=os.getenv("OPENAI_API_KEY")
    )
    vectorstore = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )
    return vectorstore


def retrieve(query: str, k: int = 5) -> list:
    """
    Given a user question, find the k most relevant chunks.
    """
    vectorstore = load_vectorstore()
    results = vectorstore.similarity_search(query, k=k)
    return results


if __name__ == "__main__":
    query = "What are the main conduct of business rules for firms?"
    print(f"\nQuery: {query}\n")
    results = retrieve(query)
    for i, doc in enumerate(results):
        print(f"--- Chunk {i + 1} (Page {doc.metadata.get('page', 'unknown')}) ---")
        print(doc.page_content)
        print()