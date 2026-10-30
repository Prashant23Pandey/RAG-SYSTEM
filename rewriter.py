import os
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
REWRITER_MODEL = os.environ.get("REWRITER_MODEL", "llama3.2")


def get_rewriter_llm():
    return ChatOllama(
        model=REWRITER_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0,
    )


rewrite_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a query optimization expert. Your task is to take a user's "
            "original question and rewrite it into a more effective search query. "
            "The rewritten query should be optimized for finding relevant information "
            "on the web or in a document store.\n\n"
            "Guidelines:\n"
            "- Keep the core intent of the original question\n"
            "- Make the query more specific and searchable\n"
            "- Remove ambiguous or unnecessary words\n"
            "- Use keywords that are likely to appear in relevant documents\n"
            "- Return ONLY the rewritten query, nothing else",
        ),
        (
            "human",
            "Original question: {question}\n\nRewrite this as an optimized search query:",
        ),
    ]
)


def rewrite_query(question: str) -> str:
    llm = get_rewriter_llm()
    chain = rewrite_prompt | llm | StrOutputParser()

    try:
        result = chain.invoke({"question": question})
        return result.strip()
    except Exception as e:
        print(f"Rewrite error: {e}")
        return question


if __name__ == "__main__":
    test_queries = [
        "What's that thing with the graphs and agents?",
        "How does the memory thing work in langchain?",
        "Tell me about that embedding database everyone uses",
    ]

    for query in test_queries:
        print(f"Original:  {query}")
        rewritten = rewrite_query(query)
        print(f"Rewritten: {rewritten}")
        print()
