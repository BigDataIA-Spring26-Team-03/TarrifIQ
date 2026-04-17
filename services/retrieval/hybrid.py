"""
HybridRetriever — Dense + Sparse + RRF Fusion for TariffIQ

Combines ChromaDB vector search with BM25 keyword search using
Reciprocal Rank Fusion to merge results.

Collections:
- policy_notices: USTR/CBP/USITC Federal Register chunks
- hts_descriptions: HTS product descriptions
"""

import os
import re
import logging
from typing import Optional, List, Dict
from collections import defaultdict

import chromadb
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)


def get_chroma_client() -> chromadb.HttpClient:
    """Get ChromaDB HttpClient (same pattern as chromadb_init.py)."""
    host = os.environ.get("CHROMADB_HOST", os.environ.get("CHROMA_HOST", "chromadb"))
    port = int(os.environ.get("CHROMADB_PORT", os.environ.get("CHROMA_PORT", 8000)))
    return chromadb.HttpClient(host=host, port=port)


def tokenize(text: str) -> List[str]:
    """Simple tokenizer: lowercase, remove punctuation, split."""
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)  # Remove punctuation
    return text.split()


class HybridRetriever:
    """Hybrid retriever combining dense vector search and BM25 sparse search."""

    def __init__(
        self,
        persist_dir: str = "./chroma_data",
        dense_weight: float = 0.6,
        sparse_weight: float = 0.4,
        rrf_k: int = 60
    ):
        """
        Initialize HybridRetriever.

        Args:
            persist_dir: ChromaDB persistence directory (not used for HttpClient)
            dense_weight: Weight for dense search results in RRF
            sparse_weight: Weight for sparse search results in RRF
            rrf_k: RRF parameter (constant added to ranks)
        """
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.rrf_k = rrf_k

        # Connect to ChromaDB
        self.chroma = get_chroma_client()

        # BM25 components
        self.corpus = []  # List of document texts
        self.doc_ids = []  # List of document IDs (chunk_id)
        self.doc_metadata = []  # List of metadata dicts
        self.bm25 = None  # BM25Okapi index

        # Load BM25 corpus
        self._reload_bm25()

        logger.info(
            "HybridRetriever initialized",
            corpus_size=len(self.corpus),
            dense_weight=dense_weight,
            sparse_weight=sparse_weight,
        )

    def _reload_bm25(self) -> None:
        """Load documents from ChromaDB and build BM25 index."""
        try:
            collection = self.chroma.get_collection("policy_notices")
        except Exception as e:
            raise RuntimeError(
                "ChromaDB not initialized. Run chromadb_init.py first."
            ) from e

        try:
            # Fetch all documents from collection
            # ChromaDB's get() without where clause returns all documents
            result = collection.get()

            if not result or not result.get("documents"):
                logger.warning("policy_notices collection is empty")
                self.bm25 = None
                return

            documents = result["documents"]
            ids = result["ids"]
            metadatas = result.get("metadatas", [])

            # Build corpus
            self.corpus = []
            self.doc_ids = []
            self.doc_metadata = []

            for i, (doc_id, text) in enumerate(zip(ids, documents)):
                if not text or not text.strip():
                    continue
                self.corpus.append(text)
                self.doc_ids.append(doc_id)
                self.doc_metadata.append(metadatas[i] if i < len(metadatas) else {})

            # Build BM25 index
            if self.corpus:
                tokenized_corpus = [tokenize(doc) for doc in self.corpus]
                self.bm25 = BM25Okapi(tokenized_corpus)
                logger.info(
                    "BM25 index built",
                    corpus_size=len(self.corpus),
                    collection="policy_notices"
                )
            else:
                logger.warning("No valid documents in policy_notices collection")
                self.bm25 = None

        except Exception as e:
            logger.error("Error loading BM25 corpus: %s", e)
            self.bm25 = None

    def search_policy(
        self,
        query: str,
        hts_chapter: Optional[str] = None,
        source: Optional[str] = None,
        top_k: int = 5
    ) -> List[Dict]:
        """
        Hybrid search for policy notices using dense + sparse + RRF.

        Args:
            query: Search query string
            hts_chapter: Optional filter by HTS chapter (e.g., "85")
            source: Optional filter by source ("USTR", "CBP", "USITC")
            top_k: Number of results to return

        Returns:
            List of result dicts with chunk_id, document_number, chunk_text, etc.
        """
        logger.info(
            "search_policy called",
            query=query,
            hts_chapter=hts_chapter,
            source=source,
            top_k=top_k
        )

        # Step 1: Dense search
        dense_results = self._dense_search_policy(
            query, hts_chapter, source, top_k * 3
        )
        dense_dict = {r["chunk_id"]: (r, idx) for idx, r in enumerate(dense_results)}

        # Step 2: Sparse search (BM25)
        sparse_results = self._sparse_search_policy(
            query, hts_chapter, source, top_k * 3
        )
        sparse_dict = {r["chunk_id"]: (r, idx) for idx, r in enumerate(sparse_results)}

        # Step 3: RRF Fusion
        rrf_scores = defaultdict(float)

        for chunk_id, (result, rank) in dense_dict.items():
            score = self.dense_weight / (rank + self.rrf_k + 1)
            rrf_scores[chunk_id] += score

        for chunk_id, (result, rank) in sparse_dict.items():
            score = self.sparse_weight / (rank + self.rrf_k + 1)
            rrf_scores[chunk_id] += score

        # Merge results
        all_chunks = {**dense_dict, **sparse_dict}
        ranked = sorted(
            rrf_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        results = []
        for chunk_id, rrf_score in ranked[:top_k]:
            result, _ = all_chunks[chunk_id]
            result["score"] = float(rrf_score)
            result["retrieval_method"] = "hybrid"
            results.append(result)

        logger.info("search_policy complete", result_count=len(results))
        return results

    def _dense_search_policy(
        self,
        query: str,
        hts_chapter: Optional[str] = None,
        source: Optional[str] = None,
        top_k: int = 15
    ) -> List[Dict]:
        """Dense search using ChromaDB vector similarity."""
        try:
            collection = self.chroma.get_collection("policy_notices")
        except Exception as e:
            logger.error("Failed to get policy_notices collection: %s", e)
            return []

        # Build where filter
        where_filter = None
        if hts_chapter and source:
            where_filter = {
                "$and": [
                    {"hts_chapter": {"$eq": hts_chapter}},
                    {"source": {"$eq": source}}
                ]
            }
        elif hts_chapter:
            where_filter = {"hts_chapter": {"$eq": hts_chapter}}
        elif source:
            where_filter = {"source": {"$eq": source}}

        try:
            results = collection.query(
                query_texts=[query],
                n_results=top_k,
                where=where_filter,
                include=["documents", "metadatas", "distances"]
            )

            if not results["documents"] or not results["documents"][0]:
                return []

            formatted = []
            for i, doc in enumerate(results["documents"][0]):
                metadata = results["metadatas"][0][i]
                distance = results["distances"][0][i]

                formatted.append({
                    "chunk_id": metadata.get("chunk_id", ""),
                    "document_number": metadata.get("document_number", ""),
                    "chunk_text": doc,
                    "hts_chapter": metadata.get("hts_chapter", ""),
                    "hts_code": metadata.get("hts_code", ""),
                    "source": metadata.get("source", ""),
                    "section": metadata.get("section", ""),
                    "dense_distance": distance
                })

            return formatted

        except Exception as e:
            logger.error("Dense search error: %s", e)
            return []

    def _sparse_search_policy(
        self,
        query: str,
        hts_chapter: Optional[str] = None,
        source: Optional[str] = None,
        top_k: int = 15
    ) -> List[Dict]:
        """Sparse search using BM25."""
        if not self.bm25 or not self.corpus:
            logger.warning("BM25 not available, skipping sparse search")
            return []

        try:
            # Tokenize query
            query_tokens = tokenize(query)

            # Get BM25 scores
            scores = self.bm25.get_scores(query_tokens)

            # Get top candidates
            scored_docs = [
                (idx, score)
                for idx, score in enumerate(scores)
                if score > 0
            ]
            scored_docs.sort(key=lambda x: x[1], reverse=True)

            # Filter by hts_chapter or source if provided
            filtered = []
            for idx, score in scored_docs:
                if idx >= len(self.doc_metadata):
                    continue

                metadata = self.doc_metadata[idx]

                # Apply chapter filter
                if hts_chapter:
                    if metadata.get("hts_chapter") != hts_chapter:
                        continue

                # Apply source filter
                if source:
                    if metadata.get("source") != source:
                        continue

                filtered.append((idx, score))

            # Format results
            formatted = []
            for idx, score in filtered[:top_k]:
                metadata = self.doc_metadata[idx]
                formatted.append({
                    "chunk_id": metadata.get("chunk_id", "") or self.doc_ids[idx],
                    "document_number": metadata.get("document_number", ""),
                    "chunk_text": self.corpus[idx],
                    "hts_chapter": metadata.get("hts_chapter", ""),
                    "hts_code": metadata.get("hts_code", ""),
                    "source": metadata.get("source", ""),
                    "section": metadata.get("section", ""),
                    "bm25_score": score
                })

            return formatted

        except Exception as e:
            logger.error("Sparse search error: %s", e)
            return []

    def search_hts(
        self,
        query: str,
        chapter: Optional[str] = None,
        top_k: int = 5
    ) -> List[Dict]:
        """
        Search HTS descriptions using dense search only.

        Args:
            query: Search query string
            chapter: Optional filter by HTS chapter (e.g., "84")
            top_k: Number of results to return

        Returns:
            List of HTS code result dicts
        """
        logger.info("search_hts called", query=query, chapter=chapter, top_k=top_k)

        try:
            collection = self.chroma.get_collection("hts_descriptions")
        except Exception as e:
            raise RuntimeError(
                "ChromaDB not initialized. Run chromadb_init.py first."
            ) from e

        # Build where filter
        where_filter = None
        if chapter:
            where_filter = {"chapter": {"$eq": chapter}}

        try:
            results = collection.query(
                query_texts=[query],
                n_results=top_k,
                where=where_filter,
                include=["documents", "metadatas", "distances"]
            )

            if not results["documents"] or not results["documents"][0]:
                return []

            formatted = []
            for i, doc in enumerate(results["documents"][0]):
                metadata = results["metadatas"][0][i]
                distance = results["distances"][0][i]

                formatted.append({
                    "hts_code": metadata.get("hts_code", ""),
                    "description": doc,
                    "general_rate": metadata.get("general_rate", ""),
                    "chapter": metadata.get("chapter", ""),
                    "is_chapter99": metadata.get("is_chapter99", "False"),
                    "score": distance
                })

            logger.info("search_hts complete", result_count=len(formatted))
            return formatted

        except Exception as e:
            logger.error("HTS search error: %s", e)
            return []


if __name__ == "__main__":
    print("Initializing HybridRetriever...")
    retriever = HybridRetriever()

    print("\nTesting search_policy...")
    results = retriever.search_policy(
        query="additional duties China electronics",
        hts_chapter="85",
        top_k=3
    )
    if results:
        for r in results:
            print(f"  [{r['source']}] {r['document_number']}: {r['chunk_text'][:100]}...")
    else:
        print("  No results")

    print("\nTesting search_hts...")
    results = retriever.search_hts(query="laptop computers", top_k=3)
    if results:
        for r in results:
            print(f"  {r['hts_code']}: {r['description'][:80]}")
    else:
        print("  No results")

    print("\nDone!")
