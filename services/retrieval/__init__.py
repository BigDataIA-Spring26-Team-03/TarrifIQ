"""
TariffIQ retrieval package.

Provides:
- HybridRetriever: Dense + Sparse + RRF fusion search
- HyDEQueryEnhancer: Hypothetical document generation for better retrieval
"""

from .hybrid import HybridRetriever
from .hyde import HyDEQueryEnhancer, get_enhancer

__all__ = [
    "HybridRetriever",
    "HyDEQueryEnhancer",
    "get_enhancer",
]
