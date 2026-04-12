"""
ChromaDB client wrapper for TariffIQ.
Provides singleton connection to ChromaDB for RAG collections.
"""

import logging
import os

import chromadb

logger = logging.getLogger(__name__)

_client = None


def get_chromadb_client():
    """Get or create ChromaDB client singleton."""
    global _client

    if _client is not None:
        return _client

    # For local testing: use in-memory client
    # For production: use HTTP client pointing to chromadb service
    chroma_mode = os.environ.get("CHROMA_MODE", "local")

    if chroma_mode == "http":
        chroma_host = os.environ.get("CHROMA_HOST", "localhost")
        chroma_port = int(os.environ.get("CHROMA_PORT", "8000"))
        logger.info(f"Connecting to ChromaDB HTTP client at {chroma_host}:{chroma_port}")
        _client = chromadb.HttpClient(host=chroma_host, port=chroma_port)
    else:
        logger.info("Using ChromaDB in-memory (ephemeral) client")
        _client = chromadb.Client()

    return _client
