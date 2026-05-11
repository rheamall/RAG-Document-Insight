import os
import json
from dotenv import load_dotenv
from openai import OpenAI
from src.query import retrieve

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# FUNCTION SCHEMA: defining the structure that the LLM must return
functions = [
    {
        "name": "return_rag_answer",
        "description": "Return a structured answer grounded in the retrieved document context.",
        "parameters": {
            "type": "object",
            "properties": {
                "answer": {
                    "type": "string",
                    "description": "A clear, concise answer to the user's question based only on the provided context."
                },
                "confidence": {
                    "type": "string",
                    "enum": ["high", "medium", "low"],
                    "description": "How confident the answer is based on the quality and relevance of retrieved chunks."
                },
                "source_pages": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "List of page numbers from which the answer was derived."
                },
                "caveat": {
                    "type": "string",
                    "description": "Any important limitations, caveats, or warnings about the answer. If none, return an empty string."
                }
            },
            "required": ["answer", "confidence", "source_pages", "caveat"]
        }
    }
]


def generate_answer(query: str, k: int = 5) -> dict:
    """
    full RAG pipeline:
    1. retrieving relevant chunks for the query
    2. building a prompt with those chunks as context
    3. calling OpenAI with function calling to enforce structured output
    4. returning parsed JSON response
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
    - If the answer is clearly in the context, answer precisely and set confidence to 'high'.
    - If the answer is partially in the context, answer what you can and set confidence to 'medium'.
    - If the answer is not in the context, set answer to "I cannot find this information in the provided document." and confidence to 'low'.
    - Always cite the page numbers your answer comes from in source_pages
    - Never make up information that is not in the context.
    - Be concise and precise."""

    user_prompt = f"""Context: {context}
    Question: {query}"""

    # step 4: calling OpenAI with function calling
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": system_prompt},
                  {"role": "user", "content": user_prompt}],
        functions=functions,
        function_call={"name": "return_rag_answer"},  #forcing this specific function
        temperature=0  # 0 = deterministic, no creativity, important for factual QA
    )

    # step 5: returning a structured result
    function_args = response.choices[0].message.function_call.arguments
    structured_output = json.loads(function_args)
    structured_output["tokens_used"] = response.usage.total_tokens

    return structured_output


if __name__ == "__main__":
    queries = [
        "What is the fair, clear and not misleading rule?",
        "Where in COBS are the disclosure requirements for costs and charges defined?",
        "What are the rules on best execution?"
    ]

    for query in queries:
        print(f"\n{'=' * 60}")
        print(f"Question: {query}")
        try:
            result = generate_answer(query)
            print(f"Answer: {result['answer']}")
            print(f"Confidence: {result['confidence']}")
            print(f"Source pages: {result['source_pages']}")
            print(f"Caveat: {result['caveat']}")
            print(f"Tokens used: {result['tokens_used']}")
        except Exception as e:
            print(f"Failed to process query: {query}")
            print(f"Error: {e}")