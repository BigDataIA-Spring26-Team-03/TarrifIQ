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
import logging
from typing import Optional

logger = logging.getLogger(__name__)


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
The relevant HTS chapter is {hts_chapter}.

Write a SHORT paragraph (3-4 sentences) that looks exactly like a real Federal Register notice about this topic.

Your paragraph MUST:
- Reference HTS chapter {hts_chapter} or specific subheadings within that chapter
- Use legal language like "ad valorem rate of duty", "entered for consumption", "subchapter III of chapter 99"
- Mention the likely tariff action type for this product/country
- Sound like it was copied from an actual Federal Register notice

Do NOT:
- Invent specific Federal Register document numbers
- Make up exact percentage rates unless very well known
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
            hts_chapter="85"
        )

        results = retriever.search_policy(
            query=better_query,
            hts_chapter="85"
        )
    """

    def __init__(self, router=None) -> None:
        """
        Initialize HyDE enhancer with optional custom router.

        Args:
            router: ModelRouter instance (uses singleton if None)
        """
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

        Returns:
            Hypothetical FR text string for use as search query.
            Falls back to original query if LLM call fails.
        """
        try:
            from services.llm.router import TaskType

            if hts_chapter:
                prompt = HYDE_PROMPT_WITH_CHAPTER.format(
                    product=product,
                    country=country,
                    hts_chapter=hts_chapter,
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
    ) -> str:
        """
        Synchronous version of enhance().

        Use this in non-async FastAPI endpoints or sync contexts.
        Falls back to original query on any error.

        Args:
            query: Original user question
            product: Extracted product name
            country: Extracted country name
            hts_chapter: Optional HTS chapter

        Returns:
            Enhanced query (hypothetical FR text) or original query
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
                     Pass None for hts_chapter if unknown.

        Returns:
            List of enhanced queries in same order.
            Failed enhancements fall back to original query.
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

        results = await asyncio.gather(
            *tasks,
            return_exceptions=True
        )

        enhanced = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.warning(
                    "hyde_batch_item_failed",
                    index=i,
                    error=str(result)
                )
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


# ---------------------------------------------------------------------------
# Test Block
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import asyncio

    async def test_hyde():
        """Test HyDE query enhancement."""
        enhancer = HyDEQueryEnhancer()

        test_cases = [
            {
                "query": "what are the tariffs on smartphones from China?",
                "product": "smartphones",
                "country": "China",
                "hts_chapter": "85"
            },
            {
                "query": "duty on steel pipes from Canada",
                "product": "steel pipes",
                "country": "Canada",
                "hts_chapter": "73"
            },
            {
                "query": "tariff on laptops from Mexico",
                "product": "laptops",
                "country": "Mexico",
                "hts_chapter": "84"
            },
            {
                "query": "import duty on solar panels from China",
                "product": "solar panels",
                "country": "China",
                "hts_chapter": "85"
            },
            {
                "query": "what is the rate on tuna from Thailand",
                "product": "tuna",
                "country": "Thailand",
                "hts_chapter": "16"
            },
        ]

        print("=" * 70)
        print("HyDE Query Enhancement Tests")
        print("=" * 70)

        for test in test_cases:
            print(f"\nOriginal query:")
            print(f"  {test['query']}")

            enhanced = await enhancer.enhance(
                query=test["query"],
                product=test["product"],
                country=test["country"],
                hts_chapter=test["hts_chapter"]
            )

            print(f"\nEnhanced query (HyDE):")
            print(f"  {enhanced[:300]}...")
            print(f"  Length: {len(enhanced)} chars")
            print("-" * 70)

        print("\nTesting batch enhancement...")
        batch_input = [
            (t["query"], t["product"], t["country"], t["hts_chapter"])
            for t in test_cases[:3]
        ]
        batch_results = await enhancer.enhance_batch(batch_input)
        print(f"Batch returned {len(batch_results)} results")
        for i, result in enumerate(batch_results):
            print(f"  [{i}]: {result[:100]}...")

        print("\nAll tests passed!")

    asyncio.run(test_hyde())
