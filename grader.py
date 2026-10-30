import os
import json
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
GRADING_MODEL = os.environ.get("GRADING_MODEL", "llama3.2")


def get_grader_llm():
    return ChatOllama(
        model=GRADING_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0,
    )


grade_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a document relevance grader. You will be given a user question "
            "and a document chunk. Your task is to determine whether the document "
            "contains information relevant to answering the question.\n\n"
            "Respond with a JSON object containing a single key 'score' with value "
            "'yes' if the document is relevant, or 'no' if it is not relevant.\n\n"
            "Only output the JSON, no additional text.",
        ),
        (
            "human",
            "User Question: {question}\n\nDocument:\n{document}",
        ),
    ]
)


def grade_document(question: str, document: str) -> dict:
    llm = get_grader_llm()
    chain = grade_prompt | llm | JsonOutputParser()

    try:
        result = chain.invoke({"question": question, "document": document})
        if isinstance(result, dict) and "score" in result:
            return result
        return {"score": "no"}
    except Exception as e:
        print(f"Grading error: {e}")
        return {"score": "no"}


def grade_documents(question: str, documents: list) -> list:
    graded = []
    for doc in documents:
        result = grade_document(question, doc.page_content)
        graded.append(
            {
                "document": doc,
                "relevant": result.get("score", "no").lower() == "yes",
            }
        )
    return graded


def filter_relevant_documents(question: str, documents: list) -> list:
    graded = grade_documents(question, documents)
    return [item["document"] for item in graded if item["relevant"]]


if __name__ == "__main__":
    test_question = "What is LangGraph used for?"
    test_doc = (
        "LangGraph is a library for building stateful, multi-actor applications "
        "with LLMs, built on top of LangChain."
    )

    print(f"Question: {test_question}")
    print(f"Document: {test_doc[:100]}...")
    result = grade_document(test_question, test_doc)
    print(f"Grade: {result}")

    test_irrelevant_doc = (
        "The capital of France is Paris, which is known for the Eiffel Tower."
    )
    print(f"\nIrrelevant Document: {test_irrelevant_doc}")
    result2 = grade_document(test_question, test_irrelevant_doc)
    print(f"Grade: {result2}")
