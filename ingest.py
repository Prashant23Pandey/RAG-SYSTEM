"""
Intelligent Document Ingestion & Semantic Chunking ETL.

Supports: PDF, Markdown, DOCX, HTML, TXT
Features:
  - Multi-format parsing via unstructured / markitdown
  - Semantic chunking based on embedding distance
  - Rich metadata enrichment (source, section, timestamp, tenant)
  - Upsert into ChromaDB
"""

import os
import re
import sys
import hashlib
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger("rag.ingest")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "nomic-embed-text")
CHROMA_DIR = os.environ.get(
    "CHROMA_DIR", os.path.join(os.path.dirname(__file__), "chroma_db")
)
COLLECTION_NAME = "rag_documents"

SEMANTIC_CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "500"))
SEMANTIC_CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "50"))
SEMANTIC_THRESHOLD = float(os.environ.get("SEMANTIC_THRESHOLD", "0.5"))

SUPPORTED_EXTENSIONS = {".pdf", ".md", ".markdown", ".docx", ".html", ".htm", ".txt"}


# ---------------------------------------------------------------------------
# Document Parsers
# ---------------------------------------------------------------------------
def _parse_txt(path: str) -> list[Document]:
    """Parse plain text files."""
    from langchain_community.document_loaders import TextLoader

    loader = TextLoader(path, encoding="utf-8")
    return loader.load()


def _parse_pdf(path: str) -> list[Document]:
    """Parse PDF files using PDFMiner."""
    from langchain_community.document_loaders import PDFMinerLoader

    loader = PDFMinerLoader(path)
    return loader.load()


def _parse_markdown(path: str) -> list[Document]:
    """Parse Markdown files, preserving headers as metadata."""
    try:
        from langchain_community.document_loaders import UnstructuredMarkdownLoader

        loader = UnstructuredMarkdownLoader(path)
        docs = loader.load()
    except ImportError:
        from langchain_community.document_loaders import TextLoader

        loader = TextLoader(path, encoding="utf-8")
        docs = loader.load()

    # Extract first heading as section title
    for doc in docs:
        match = re.search(r"^#\s+(.+)$", doc.page_content, re.MULTILINE)
        if match:
            doc.metadata["section_title"] = match.group(1).strip()

    return docs


def _parse_html(path: str) -> list[Document]:
    """Parse HTML files."""
    from langchain_community.document_loaders import UnstructuredHTMLLoader

    loader = UnstructuredHTMLLoader(path)
    return loader.load()


def _parse_docx(path: str) -> list[Document]:
    """Parse DOCX files."""
    from langchain_community.document_loaders import Docx2txtLoader

    loader = Docx2txtLoader(path)
    return loader.load()


PARSERS = {
    ".txt": _parse_txt,
    ".pdf": _parse_pdf,
    ".md": _parse_markdown,
    ".markdown": _parse_markdown,
    ".html": _parse_html,
    ".htm": _parse_html,
    ".docx": _parse_docx,
}


def parse_file(path: str) -> list[Document]:
    """
    Parse a file based on its extension.

    Returns:
        List of Document objects with basic metadata populated.
    """
    ext = Path(path).suffix.lower()
    parser = PARSERS.get(ext)
    if parser is None:
        raise ValueError(f"Unsupported file type: {ext}")

    docs = parser(path)

    # Enrich metadata
    stat = os.stat(path)
    created_ts = datetime.fromtimestamp(stat.st_ctime, tz=timezone.utc).isoformat()
    modified_ts = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat()
    file_hash = hashlib.md5(open(path, "rb").read()).hexdigest()

    for doc in docs:
        doc.metadata.update(
            {
                "source": os.path.abspath(path),
                "filename": os.path.basename(path),
                "file_type": ext,
                "file_hash": file_hash,
                "created_at": created_ts,
                "modified_at": modified_ts,
                "ingested_at": datetime.now(timezone.utc).isoformat(),
            }
        )

    return docs


# ---------------------------------------------------------------------------
# Semantic Chunking
# ---------------------------------------------------------------------------
def semantic_chunk_documents(
    docs: list[Document],
    chunk_size: int = SEMANTIC_CHUNK_SIZE,
    chunk_overlap: int = SEMANTIC_CHUNK_OVERLAP,
) -> list[Document]:
    """
    Split documents using semantic-aware chunking.

    Uses RecursiveCharacterTextSplitter with semantically meaningful
    separators to preserve paragraph and section boundaries.
    Falls back to embedding-distance-based splitting when available.
    """
    # Try SemanticChunker if available
    try:
        from langchain_experimental.text_splitter import SemanticChunker
        from langchain_community.embeddings import OllamaEmbeddings

        embeddings = OllamaEmbeddings(
            model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL
        )
        chunker = SemanticChunker(
            embeddings,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=85,
            min_chunk_size=100,
        )
        chunks = chunker.split_documents(docs)
        logger.info("SemanticChunker produced %d chunks", len(chunks))
        return chunks
    except ImportError:
        logger.debug(
            "langchain_experimental not available – using RecursiveCharacterTextSplitter"
        )
    except Exception as e:
        logger.warning("SemanticChunker failed (%s) – falling back", e)

    # Fallback: RecursiveCharacterTextSplitter with semantic separators
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n\n", "\n\n", "\n", ". ", " ", ""],
        keep_separator=True,
    )
    chunks = splitter.split_documents(docs)
    logger.info("RecursiveCharacterTextSplitter produced %d chunks", len(chunks))
    return chunks


# ---------------------------------------------------------------------------
# Section Title Extraction
# ---------------------------------------------------------------------------
def _extract_section_title(text: str) -> str:
    """Extract the first heading from text content."""
    for line in text.split("\n"):
        stripped = line.strip()
        if stripped.startswith("#"):
            return stripped.lstrip("#").strip()
    return ""


def _enrich_section_metadata(chunks: list[Document]) -> list[Document]:
    """Add section_title metadata to each chunk."""
    current_section = ""
    for chunk in chunks:
        title = chunk.metadata.get("section_title") or _extract_section_title(
            chunk.page_content
        )
        if title:
            current_section = title
        chunk.metadata["section_title"] = current_section
    return chunks


# ---------------------------------------------------------------------------
# ChromaDB Upsert
# ---------------------------------------------------------------------------
def _get_embeddings():
    return OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)


def upsert_to_chromadb(
    chunks: list[Document],
    persist_dir: str = CHROMA_DIR,
    collection_name: str = COLLECTION_NAME,
) -> int:
    """
    Upsert document chunks into ChromaDB.

    Uses file_hash + chunk index as deduplication key.
    Returns the number of chunks added.
    """
    if not chunks:
        return 0

    embeddings = _get_embeddings()
    vs = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings,
        collection_name=collection_name,
    )

    # Generate deterministic IDs for deduplication
    ids = []
    for i, chunk in enumerate(chunks):
        src = chunk.metadata.get("source", "unknown")
        h = chunk.metadata.get("file_hash", "nohash")
        ids.append(f"{h}_{i}")

    # Batch add (Chroma handles upsert semantics by ID)
    batch_size = 100
    total_added = 0
    for start in range(0, len(chunks), batch_size):
        end = min(start + batch_size, len(chunks))
        batch_docs = chunks[start:end]
        batch_ids = ids[start:end]
        vs.add_documents(batch_docs, ids=batch_ids)
        total_added += len(batch_docs)

    logger.info("Upserted %d chunks to ChromaDB at %s", total_added, persist_dir)
    return total_added


# ---------------------------------------------------------------------------
# Main Ingestion Pipeline
# ---------------------------------------------------------------------------
def ingest_file(
    file_path: str,
    persist_dir: str = CHROMA_DIR,
    tenant_id: str = "default",
) -> dict:
    """
    Full ETL pipeline for a single file.

    Returns:
        dict with stats: filename, chunks, added, metadata
    """
    path = os.path.abspath(file_path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"File not found: {path}")

    ext = Path(path).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {ext}")

    logger.info("Ingesting: %s", path)

    # Parse
    docs = parse_file(path)
    logger.info("Parsed %d document(s) from %s", len(docs), path)

    # Add tenant_id
    for doc in docs:
        doc.metadata["tenant_id"] = tenant_id

    # Semantic chunking
    chunks = semantic_chunk_documents(docs)
    logger.info("Split into %d chunks", len(chunks))

    # Enrich section metadata
    chunks = _enrich_section_metadata(chunks)

    # Upsert
    added = upsert_to_chromadb(chunks, persist_dir)

    return {
        "filename": os.path.basename(path),
        "chunks_total": len(chunks),
        "chunks_added": added,
        "source": path,
        "metadata": chunks[0].metadata if chunks else {},
    }


def ingest_directory(
    dir_path: str,
    persist_dir: str = CHROMA_DIR,
    tenant_id: str = "default",
    recursive: bool = True,
) -> list[dict]:
    """
    Ingest all supported files from a directory.

    Returns:
        List of per-file ingestion results.
    """
    results = []
    dir_path = os.path.abspath(dir_path)

    pattern = "**/*" if recursive else "*"
    for ext in SUPPORTED_EXTENSIONS:
        for file_path in Path(dir_path).glob(f"{pattern}{ext}"):
            try:
                result = ingest_file(str(file_path), persist_dir, tenant_id)
                results.append(result)
                print(
                    f"  OK: {result['filename']} "
                    f"({result['chunks_added']} chunks)"
                )
            except Exception as e:
                logger.error("Failed to ingest %s: %s", file_path, e)
                print(f"  FAIL: {file_path} – {e}")

    total_chunks = sum(r["chunks_added"] for r in results)
    print(f"\nIngestion complete: {len(results)} file(s), {total_chunks} chunks total")
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    import argparse

    parser = argparse.ArgumentParser(description="Document Ingestion ETL")
    parser.add_argument("path", help="File or directory to ingest")
    parser.add_argument(
        "--persist-dir", default=CHROMA_DIR, help="ChromaDB persist directory"
    )
    parser.add_argument(
        "--tenant-id", default="default", help="Tenant ID for metadata"
    )
    parser.add_argument("--recursive", action="store_true", default=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    target = os.path.abspath(args.path)
    if os.path.isdir(target):
        ingest_directory(target, args.persist_dir, args.tenant_id, args.recursive)
    elif os.path.isfile(target):
        result = ingest_file(target, args.persist_dir, args.tenant_id)
        print(f"Ingested: {result['filename']} – {result['chunks_added']} chunks")
    else:
        print(f"Error: {target} is not a valid file or directory")
        sys.exit(1)


if __name__ == "__main__":
    main()
