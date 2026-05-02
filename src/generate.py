import os
from dotenv import load_dotenv
from openai import OpenAI
from src.query import retrieve

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def generate_answer(query: str, k: int = 5) -> dict:
    """
    full RAG pipeline:
    1. retrieving relevant chunks for the query
    2. building a prompt with those chunks as context
    3. sending to OpenAI and getting a grounded answer
    """

    # step 1: retrieving relevant chunks
    chunks = retrieve(query, k=k)

    # step 2: building context string from chunks
    context = ""
    for i, chunk in enumerate(chunks):
        page = chunk.metadata.get('page', 'unknown')
        context += f"[Chunk {i+1}, Page {page}]:\n{chunk.page_content}\n\n"

    # step 3: building the prompt
    system_prompt = """You are a financial regulatory expert assistant specialising in FCA conduct of business rules.

    Your job is to answer questions based ONLY on the context provided below.
    Do not use any outside knowledge.
    
    Rules:
    - If the answer is in the context, answer clearly and cite the page number.
    - If the answer is not in the context, say exactly: "I cannot find this information in the provided document."
    - Never make up information that is not in the context.
    - Be concise and precise."""

    user_prompt = f"""Context: {context}
        Question: {query}
        Answer:"""

    # step 4: calling OpenAI
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": system_prompt},
                  {"role": "user", "content": user_prompt}],
        temperature=0  # 0 = deterministic, no creativity, important for factual QA
    )

    answer = response.choices[0].message.content

    # step 5: returning a structured result
    return {
        "query": query,
        "answer": answer,
        "sources": [
            {
                "page": chunk.metadata.get('page', 'unknown'),
                "content": chunk.page_content[:200]  # first 200 chars as preview
            }
            for chunk in chunks
        ],
        "tokens_used": response.usage.total_tokens
    }


if __name__ == "__main__":
    query = "Where in COBS are the disclosure requirements for costs and charges defined?"
    result = generate_answer(query)

    print(f"\nQuestion: {result['query']}")
    print(f"\nAnswer:\n{result['answer']}")
    print(f"\nSources:")
    for source in result['sources']:
        print(f"  - Page {source['page']}: {source['content'][:100]}...")
    print(f"\nTokens used: {result['tokens_used']}")