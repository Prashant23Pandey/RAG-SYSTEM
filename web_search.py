from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_core.documents import Document


def web_search(query: str, max_results: int = 5) -> list[Document]:
    wrapper = DuckDuckGoSearchAPIWrapper(max_results=max_results)

    try:
        results = wrapper.results(query)
        documents = []
        for result in results:
            title = result.get("title", "")
            snippet = result.get("body", "")
            url = result.get("link", "")

            content = f"{title}\n\n{snippet}"
            doc = Document(
                page_content=content,
                metadata={"source": url, "title": title},
            )
            documents.append(doc)
        return documents
    except Exception as e:
        print(f"Search error: {e}")
        return []


def web_search_snippets(query: str, max_results: int = 5) -> list[str]:
    wrapper = DuckDuckGoSearchAPIWrapper(max_results=max_results)

    try:
        results = wrapper.results(query)
        snippets = []
        for result in results:
            title = result.get("title", "")
            body = result.get("body", "")
            snippets.append(f"{title}: {body}")
        return snippets
    except Exception as e:
        print(f"Search error: {e}")
        return []


if __name__ == "__main__":
    test_query = "What is LangGraph and how does it work?"
    print(f"Search Query: {test_query}\n")

    docs = web_search(test_query, max_results=3)
    print(f"Found {len(docs)} results:\n")
    for i, doc in enumerate(docs):
        print(f"Result {i + 1}:")
        print(f"  Source: {doc.metadata.get('source', 'N/A')}")
        print(f"  Content: {doc.page_content[:150]}...")
        print()
