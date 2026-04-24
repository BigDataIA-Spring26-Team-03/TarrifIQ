"""
HyDE (Hypothetical Document Embeddings) Query Enhancer for TariffIQ

Improves ChromaDB retrieval by converting user questions into realistic
Federal Register excerpt text before searching.

Why this works:
  User question: "what extra duties on Chinese phones?"
  Question embedding: question-shaped vector

  HyDE generates: "Products of China classified under subheading 8517.13
  are subject to an additional 20% ad valorem rate of duty pursuant to
  Section 301 of the Trade Act of 1974..."

  Answer embedding: answer-shaped vector → matches real FR chunks better

Result: ChromaDB returns much more relevant policy documents.
"""

from __future__ import annotations

import asyncio
import structlog
from typing import Optional

logger = structlog.get_logger()


# ---------------------------------------------------------------------------
# Prompt Templates
# ---------------------------------------------------------------------------

HYDE_PROMPT = """You are a US trade policy expert writing Federal Register notice excerpts.

A procurement professional is researching tariffs on {product} imported from {country}.
{chapter_hint}

Write a SHORT paragraph (3-4 sentences) that looks exactly like a real Federal Register notice about this topic.

Your paragraph MUST use language like:
- "ad valorem rate of duty"
- "entered for consumption, or withdrawn from warehouse"
- "subchapter III of chapter 99 of the HTSUS"
- "products of [country]"
- "effective with respect to goods entered"
- HTS chapter or subheading numbers if relevant

The type of tariff action should be realistic:
- China products: mention Section 301 or IEEPA
- Steel/aluminum: mention Section 232
- Canada/Mexico: mention IEEPA or USMCA
- Other countries: mention MFN rates or reciprocal tariffs

Do NOT:
- Invent specific Federal Register document numbers
- Make up exact percentage rates unless very well known
- Write more than 4 sentences
- Add any introduction or explanation

Write ONLY the paragraph itself."""


HYDE_PROMPT_WITH_CHAPTER = """You are a US trade policy expert writing Federal Register notice excerpts.

A procurement professional is researching tariffs on {product} imported from {country}.
The relevant HTS subheading is {hts_code} (chapter {hts_chapter}).

Write a SHORT paragraph (3-4 sentences) that looks exactly like a real Federal Register notice about this topic.

Your paragraph MUST:
- Reference HTS subheading {hts_code} or other subheadings within chapter {hts_chapter} by number
- Use legal language like "ad valorem rate of duty", "entered for consumption", "subchapter III of chapter 99"
- Mention the likely tariff action type for this product/country combination
- If the product is steel (chapter 72-73), aluminum (chapter 76), or copper (chapter 74),
  mention Section 232 and that tariff rates have been increased or modified
- If the product is from China, mention Section 301 or IEEPA duties
- For Canada/Mexico, mention USMCA rules of origin and any IEEPA executive orders
- For other countries with steel/aluminum, mention Section 232 duty modifications
- Sound like it was copied from an actual Federal Register notice

Do NOT:
- Invent specific Federal Register document numbers
- Write more than 4 sentences
- Add any introduction or explanation

Write ONLY the paragraph itself."""


# ---------------------------------------------------------------------------
# HyDEQueryEnhancer
# ---------------------------------------------------------------------------

class HyDEQueryEnhancer:
    """
    Hypothetical Document Embeddings query enhancer for TariffIQ.

    Improves ChromaDB retrieval by converting user questions into
    realistic Federal Register excerpt text before searching.

    Usage:
        enhancer = HyDEQueryEnhancer()

        better_query = await enhancer.enhance(
            query="what are the tariffs on smartphones from China?",
            product="smartphones",
            country="China",
            hts_chapter="85",
            hts_code="8517.13.00"
        )

        results = retriever.search_policy(
            query=better_query,
            hts_chapter="85"
        )
    """

    def __init__(self, router=None) -> None:
        if router is not None:
            self.router = router
        else:
            from services.llm.router import get_router
            self.router = get_router()

    async def enhance(
        self,
        query: str,
        product: str,
        country: str,
        hts_chapter: Optional[str] = None,
        hts_code: Optional[str] = None,
    ) -> str:
        """
        Enhance a search query using HyDE.

        Generates a hypothetical Federal Register excerpt and returns
        it as the new search query for better semantic matching.

        Args:
            query: Original user question
            product: Extracted product name (e.g., "smartphones")
            country: Extracted country name (e.g., "China")
            hts_chapter: Optional HTS chapter (e.g., "85")
            hts_code: Optional full HTS code (e.g., "8517.13.00")

        Returns:
            Hypothetical FR text string for use as search query.
            Falls back to original query if LLM call fails.
        """
        try:
            from services.llm.router import TaskType

            if hts_chapter:
                # Use full HTS code if available, fall back to chapter-only
                effective_hts_code = hts_code or f"{hts_chapter}XX.XX.XX"
                prompt = HYDE_PROMPT_WITH_CHAPTER.format(
                    product=product,
                    country=country,
                    hts_chapter=hts_chapter,
                    hts_code=effective_hts_code,
                )
            else:
                chapter_hint = ""
                prompt = HYDE_PROMPT.format(
                    product=product,
                    country=country,
                    chapter_hint=chapter_hint,
                )

            response = await self.router.complete(
                task=TaskType.HYDE_GENERATION,
                messages=[{
                    "role": "user",
                    "content": prompt
                }],
            )

            hypothetical = response.choices[0].message.content.strip()

            if not hypothetical or len(hypothetical) < 20:
                logger.warning(
                    "hyde_output_too_short",
                    product=product,
                    country=country,
                    fallback="original_query"
                )
                return query

            logger.info(
                "hyde_enhanced",
                product=product,
                country=country,
                hts_chapter=hts_chapter,
                hts_code=hts_code,
                original_query_len=len(query),
                hypothetical_len=len(hypothetical)
            )
            return hypothetical

        except Exception as e:
            logger.warning(
                "hyde_enhance_failed",
                error=str(e),
                product=product,
                country=country,
                fallback="original_query"
            )
            return query

    def enhance_sync(
        self,
        query: str,
        product: str,
        country: str,
        hts_chapter: Optional[str] = None,
        hts_code: Optional[str] = None,
    ) -> str:
        """
        Synchronous version of enhance().

        Use this in non-async FastAPI endpoints or sync contexts.
        Falls back to original query on any error.
        """
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(
                    self.enhance(
                        query=query,
                        product=product,
                        country=country,
                        hts_chapter=hts_chapter,
                        hts_code=hts_code,
                    )
                )
                return result
            finally:
                loop.close()
        except Exception as e:
            logger.warning(
                "hyde_enhance_sync_failed",
                error=str(e),
                product=product,
                country=country,
                fallback="original_query"
            )
            return query

    async def enhance_batch(
        self,
        queries: list[tuple[str, str, str, Optional[str]]],
    ) -> list[str]:
        """
        Enhance multiple queries concurrently.

        Args:
            queries: List of (query, product, country, hts_chapter) tuples.

        Returns:
            List of enhanced queries in same order.
        """
        tasks = [
            self.enhance(
                query=q,
                product=p,
                country=c,
                hts_chapter=ch,
            )
            for q, p, c, ch in queries
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        enhanced = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.warning("hyde_batch_item_failed", index=i, error=str(result))
                enhanced.append(queries[i][0])
            else:
                enhanced.append(result)

        return enhanced


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_enhancer_instance: Optional[HyDEQueryEnhancer] = None


def get_enhancer() -> HyDEQueryEnhancer:
    """Get or create singleton HyDEQueryEnhancer instance."""
    global _enhancer_instance
    if _enhancer_instance is None:
        _enhancer_instance = HyDEQueryEnhancer()
    return _enhancer_instance