import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

CHROMA_DIR = os.environ.get("CHROMA_DIR", os.path.join(os.path.dirname(__file__), "chroma_db"))
COLLECTION_NAME = "rag_documents"

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "nomic-embed-text")
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def get_embeddings():
    return OllamaEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=OLLAMA_BASE_URL,
    )


def load_documents(doc_dir: str):
    loader = DirectoryLoader(
        doc_dir,
        glob="**/*.txt",
        loader_cls=TextLoader,
        show_progress=True,
    )
    return loader.load()


def split_documents(docs, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(docs)


def create_vectorstore(documents, persist_dir=CHROMA_DIR):
    embeddings = get_embeddings()
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=persist_dir,
        collection_name=COLLECTION_NAME,
    )
    return vectorstore


def get_vectorstore(persist_dir=CHROMA_DIR):
    embeddings = get_embeddings()
    return Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
    )


def query_vectorstore(query: str, k: int = 4, persist_dir=CHROMA_DIR):
    vectorstore = get_vectorstore(persist_dir)
    results = vectorstore.similarity_search_with_score(query, k=k)
    docs = [doc for doc, _score in results]
    return docs


def query_vectorstore_with_scores(query: str, k: int = 4, persist_dir=CHROMA_DIR):
    vectorstore = get_vectorstore(persist_dir)
    return vectorstore.similarity_search_with_score(query, k=k)


def ingest_directory(doc_dir: str, persist_dir=CHROMA_DIR):
    print(f"Loading documents from: {doc_dir}")
    docs = load_documents(doc_dir)
    print(f"Loaded {len(docs)} document(s)")

    chunks = split_documents(docs)
    print(f"Split into {len(chunks)} chunks")

    vectorstore = create_vectorstore(chunks, persist_dir)
    print(f"Vectorstore created at: {persist_dir}")
    return vectorstore


if __name__ == "__main__":
    sample_dir = os.path.join(os.path.dirname(__file__), "sample_docs")
    os.makedirs(sample_dir, exist_ok=True)

    sample_file = os.path.join(sample_dir, "sample.txt")
    if not os.path.exists(sample_file):
        with open(sample_file, "w", encoding="utf-8") as f:
            f.write(
                "LangGraph is a library for building stateful, multi-actor applications "
                "with LLMs, built on top of LangChain. It extends LangChain expression "
                "language with the ability to coordinate multiple chains (or actors) across "
                "multiple steps of computation in a cyclic manner. This is particularly "
                "useful for building agent workflows where different components need to "
                "make decisions based on the current state of the system.\n\n"
                "ChromaDB is an open-source embedding database that makes it easy to build "
                "AI applications with embeddings. It provides fast, accurate vector search "
                "capabilities. ChromaDB supports various embedding functions and can be "
                "used as a persistent vector store for retrieval augmented generation (RAG) "
                "applications.\n\n"
                "Ollama allows you to run large language models locally on your machine. "
                "It supports models like Llama 3.2, Mistral, and many others. Ollama "
                "provides a simple API for interacting with these models and also offers "
                "embedding models like nomic-embed-text for generating text embeddings."
            )

    ingest_directory(sample_dir)

    print("\n--- Test Query ---")
    results = query_vectorstore("What is LangGraph used for?", k=2)
    for i, doc in enumerate(results):
        print(f"\nResult {i + 1}:")
        print(f"  Source: {doc.metadata.get('source', 'N/A')}")
        print(f"  Content: {doc.page_content[:200]}...")
