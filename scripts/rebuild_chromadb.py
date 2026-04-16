"""
Force rebuild ChromaDB collections.

Deletes existing collections and re-runs full initialization.
Useful when new documents are added to Snowflake.

Usage:
    python scripts/rebuild_chromadb.py
"""

import sys
import os
import time
import logging

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from services.chromadb_init import (
    get_chroma_client,
    build_policy_notices_collection,
    build_hts_descriptions_collection,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def rebuild_chromadb():
    """Delete and rebuild all ChromaDB collections."""
    logger.info("Starting ChromaDB rebuild...")
    start_time = time.time()

    chroma = get_chroma_client()

    # Count existing documents before deletion
    before_policy = 0
    before_hts = 0

    try:
        policy_col = chroma.get_collection("policy_notices")
        before_policy = policy_col.count()
        logger.info(f"Deleting existing policy_notices collection ({before_policy} docs)...")
        chroma.delete_collection("policy_notices")
    except Exception:
        logger.info("policy_notices collection does not exist, skipping deletion")

    try:
        hts_col = chroma.get_collection("hts_descriptions")
        before_hts = hts_col.count()
        logger.info(f"Deleting existing hts_descriptions collection ({before_hts} docs)...")
        chroma.delete_collection("hts_descriptions")
    except Exception:
        logger.info("hts_descriptions collection does not exist, skipping deletion")

    logger.info("Building collections from scratch...")

    # Rebuild
    policy_count = build_policy_notices_collection(chroma)
    hts_count = build_hts_descriptions_collection(chroma)

    elapsed = time.time() - start_time

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("ChromaDB Rebuild Complete")
    logger.info("=" * 60)
    logger.info(f"policy_notices:    {before_policy} → {policy_count} documents")
    logger.info(f"hts_descriptions:  {before_hts} → {hts_count} documents")
    logger.info(f"Total time:        {elapsed:.1f} seconds")
    logger.info("=" * 60 + "\n")


if __name__ == "__main__":
    rebuild_chromadb()
