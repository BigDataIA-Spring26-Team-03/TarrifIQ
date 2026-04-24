"""
TariffIQ LiteLLM Router — Cheapest Model Selection with Budget Enforcement

Routes LLM calls across task types with:
- Automatic provider fallback on failure
- Daily budget enforcement ($2/day default)
- Pre-baked system prompts per task type
- Structured cost + token logging

Model Strategy (cheapest first):
  - gemini-1.5-flash: $0.000075/1k tokens  (query parsing, hyde, chat)
  - gpt-4o-mini:      $0.000150/1k tokens  (classification)
  - claude-haiku:     $0.000250/1k tokens  (policy, synthesis — reasoning tasks)
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from enum import Enum
from typing import Any, AsyncIterator, Dict, List, Optional

import structlog
from litellm import acompletion

logger = structlog.get_logger()


def _configure_litellm() -> None:
    """Propagate env vars from app config into os.environ for LiteLLM."""
    try:
        from app.config import settings
        if settings.OPENAI_API_KEY:
            os.environ.setdefault("OPENAI_API_KEY", settings.OPENAI_API_KEY)
        if settings.ANTHROPIC_API_KEY:
            os.environ.setdefault("ANTHROPIC_API_KEY", settings.ANTHROPIC_API_KEY)
    except Exception:
        pass  # config unavailable in test; rely on env vars directly


_configure_litellm()


# ---------------------------------------------------------------------------
# Task Types
# ---------------------------------------------------------------------------

class TaskType(str, Enum):
    """Task types with specialized models and prompts."""
    QUERY_PARSING = "query_parsing"
    HTS_CLASSIFICATION = "hts_classification"
    POLICY_ANALYSIS = "policy_analysis"
    HYDE_GENERATION = "hyde_generation"
    ANSWER_SYNTHESIS = "answer_synthesis"
    CHAT_RESPONSE = "chat_response"


# ---------------------------------------------------------------------------
# ModelConfig
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    """LLM configuration for a task type."""
    primary: str
    fallbacks: List[str]
    temperature: float
    max_tokens: int
    cost_per_1k_tokens: float
    system_prompt: str = ""


# ---------------------------------------------------------------------------
# Routing Table — Cost-Optimized Models
# ---------------------------------------------------------------------------

MODEL_ROUTING: Dict[TaskType, ModelConfig] = {
    TaskType.QUERY_PARSING: ModelConfig(
        primary="gpt-4o-mini",
        fallbacks=["claude-haiku-4-5-20251001"],
        temperature=0.0,
        max_tokens=150,
        cost_per_1k_tokens=0.000150,
        system_prompt=(
            "You are a trade intelligence query parser for a US import tariff platform. "
            "Your job is to extract the importable PRODUCT and COUNTRY OF ORIGIN from the user's query. "
            "Think carefully — the product is what is being imported into the US, the country is where it comes from. "
            "\n\nRules:"
            "\n- Correct spelling errors in product and country names."
            "\n- For abbreviated or colloquial product names, return the canonical trade name "
            "(e.g. 'EVs' → 'electric vehicles', 'chips' → 'semiconductors', 'panels' → 'solar panels')."
            "\n- For vague queries like 'steel tariff history' extract the product even without a country."
            "\n- For policy questions like 'what is Section 301' extract product=null, country=null."
            "\n- For comparison queries like 'China vs Vietnam for laptops', return the first country mentioned."
            "\n- Country aliases: 'PRC' → 'China', 'EU' → country='European Union', 'Korea' → 'South Korea'."
            "\n- If no country is mentioned or implied, use country='all'."
            "\n- If no product is mentioned or identifiable, use product='unknown'."
            "\n- Never invent a product or country not present or implied in the query."
            "\nReturn ONLY valid JSON with exactly two keys: "
            "{\"product\": \"...\", \"country\": \"...\"}"
        ),
    ),

    TaskType.HTS_CLASSIFICATION: ModelConfig(
        primary="gpt-4o-mini",
        fallbacks=["claude-haiku-4-5-20251001"],
        temperature=0.1,
        max_tokens=300,
        cost_per_1k_tokens=0.000150,
        system_prompt=(
            "You are a US Harmonized Tariff Schedule (HTS) classification expert. "
            "Given a product description and candidate HTS codes from the database, "
            "select the most appropriate HTS code. "
            "Return ONLY valid JSON: "
            "{\"hts_code\": \"...\", \"confidence\": 0.0-1.0, \"reasoning\": \"...\"} "
            "Base your decision only on the HTS descriptions provided to you. "
            "Never invent HTS codes not in the list provided."
        ),
    ),

    TaskType.POLICY_ANALYSIS: ModelConfig(
        primary="claude-haiku-4-5-20251001",
        fallbacks=["gpt-4o-mini"],
        temperature=0.2,
        max_tokens=1000,
        cost_per_1k_tokens=0.000800,
        system_prompt=(
            "You are a US trade policy analyst for an import tariff platform. "
            "Answer questions using ONLY the Federal Register excerpts provided. "
            "For every factual claim cite the exact document number like: "
            "(FR: 2025-07325). "
            "If context is insufficient say explicitly: "
            "The provided Federal Register documents do not contain enough "
            "information to answer this question. "
            "Never use knowledge of tariff rates outside the documents provided."
        ),
    ),

    TaskType.HYDE_GENERATION: ModelConfig(
        primary="gpt-4o-mini",
        fallbacks=["claude-haiku-4-5-20251001"],
        temperature=0.4,
        max_tokens=200,
        cost_per_1k_tokens=0.000150,
        system_prompt=(
            "You are a US trade policy expert. "
            "Write realistic Federal Register excerpt text about tariff topics. "
            "Use authentic legal language from Federal Register notices. "
            "Include HTS subheading references when relevant. "
            "Do not invent specific document numbers or percentage rates."
        ),
    ),

    TaskType.ANSWER_SYNTHESIS: ModelConfig(
        primary="claude-haiku-4-5-20251001",
        fallbacks=["gpt-4o-mini"],
        temperature=0.2,
        max_tokens=1500,
        cost_per_1k_tokens=0.000800,
        system_prompt=(
            "You are a trade sourcing analyst for TariffIQ. "
            "Given verified tariff data, write a structured Markdown answer "
            "for a procurement professional using exactly the ## section headings provided. "
            "Rules: "
            "1. Use the ## 1 through ## 7 section headings from the format template — no prose blob, no JSON. "
            "2. Cite every Federal Register document inline as (FR: document_number) — e.g. (FR: 2025-07325). "
            "3. Never invent rates, HTS codes, or document numbers not present in the context. "
            "4. If a section has no supporting data, write one honest sentence — do not skip the heading. "
            "5. If an FTA preferential rate applies, say so explicitly in section 2. "
            "6. Do NOT include bare URLs in the text — the citations panel handles links. "
            "7. If data is missing or unverified, say so explicitly rather than guessing."
        ),
    ),

    TaskType.CHAT_RESPONSE: ModelConfig(
        primary="gpt-4o-mini",
        fallbacks=["claude-haiku-4-5-20251001"],
        temperature=0.7,
        max_tokens=600,
        cost_per_1k_tokens=0.000150,
        system_prompt=(
            "You are a helpful assistant for TariffIQ, a US import tariff "
            "intelligence platform. Help procurement professionals understand "
            "US tariff policy. Be conversational but accurate. "
            "If asked for specific rates always clarify that rates should be "
            "verified against official government sources."
        ),
    ),
}


# ---------------------------------------------------------------------------
# DailyBudget
# ---------------------------------------------------------------------------

@dataclass
class DailyBudget:
    """Thread-safe daily spend tracking with automatic reset."""
    date: date = field(default_factory=date.today)
    spent_usd: Decimal = Decimal("0")
    limit_usd: Decimal = Decimal("2.00")
    _lock: asyncio.Lock = field(
        default_factory=asyncio.Lock,
        repr=False,
        compare=False
    )

    def _reset_if_new_day(self) -> None:
        """Reset counter if calendar day changed."""
        if self.date != date.today():
            self.date = date.today()
            self.spent_usd = Decimal("0")

    def can_spend(self, amount: Decimal) -> bool:
        """Check if amount fits within daily limit."""
        self._reset_if_new_day()
        return self.spent_usd + amount <= self.limit_usd

    async def record_spend(self, amount: Decimal) -> None:
        """Thread-safe spend recording via asyncio.Lock."""
        async with self._lock:
            self._reset_if_new_day()
            self.spent_usd += amount

    @property
    def budget_remaining(self) -> Decimal:
        """Get remaining budget for today."""
        self._reset_if_new_day()
        return self.limit_usd - self.spent_usd


# ---------------------------------------------------------------------------
# ModelRouter
# ---------------------------------------------------------------------------

class ModelRouter:
    """
    Route LLM completion requests with automatic fallback, budget enforcement,
    and comprehensive cost + token logging.
    """

    def __init__(self, daily_limit_usd: Optional[float] = None) -> None:
        """
        Initialize router with optional daily budget limit.

        Args:
            daily_limit_usd: Daily budget in USD (default $2.00)
        """
        if daily_limit_usd is None:
            try:
                from app.config import settings
                daily_limit_usd = settings.LITELLM_BUDGET_USD_PER_DAY
            except Exception:
                daily_limit_usd = 2.0
        self.daily_budget = DailyBudget(limit_usd=Decimal(str(daily_limit_usd)))

    async def complete(
        self,
        task: TaskType,
        messages: List[Dict[str, str]],
        stream: bool = False,
        **kwargs: Any,
    ) -> Any:
        """
        Route a completion request with automatic fallback.

        Args:
            task: TaskType enum value
            messages: List of message dicts with role and content
            stream: Whether to stream the response
            **kwargs: Additional args passed to acompletion

        Returns:
            LiteLLM response object (or async generator if stream=True)

        Raises:
            RuntimeError if all models fail
        """
        config = MODEL_ROUTING[task]
        prepared = self._inject_system_prompt(messages, config)

        for model in [config.primary] + config.fallbacks:
            # Pre-call budget gate
            estimated = self._estimate_cost(prepared, config)
            if not self.daily_budget.can_spend(estimated):
                logger.warning(
                    "llm_budget_exceeded",
                    task=task.value,
                    model=model,
                    estimated_usd=float(estimated),
                    budget_remaining=float(self.daily_budget.budget_remaining),
                )
                continue

            try:
                if stream:
                    return self._stream_complete(
                        model, prepared, config, **kwargs
                    )

                response = await acompletion(
                    model=model,
                    messages=prepared,
                    temperature=config.temperature,
                    max_tokens=config.max_tokens,
                    **kwargs,
                )

                # Record actual spend from response token usage
                await self._record_response_cost(
                    response, config, task, model
                )
                return response

            except Exception as exc:
                logger.warning(
                    "llm_fallback",
                    task=task.value,
                    failed_model=model,
                    reason=str(exc),
                )

        raise RuntimeError(
            f"All models failed for task '{task.value}'. "
            f"Tried: {[config.primary] + config.fallbacks}"
        )

    async def _stream_complete(
        self,
        model: str,
        messages: List[Dict[str, str]],
        config: ModelConfig,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Async generator that yields content chunks from streaming response."""
        estimated = self._estimate_cost(messages, config)
        if not self.daily_budget.can_spend(estimated):
            raise RuntimeError(
                f"Daily budget exhausted. "
                f"Remaining: ${self.daily_budget.budget_remaining:.4f}"
            )

        response = await acompletion(
            model=model,
            messages=messages,
            stream=True,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            **kwargs,
        )

        total_chars = 0
        async for chunk in response:
            content = chunk.choices[0].delta.content
            if content:
                total_chars += len(content)
                yield content

        # Approximate spend recording for streamed responses
        # (token count not available during streaming)
        approx_tokens = Decimal(str(total_chars // 4))
        approx_cost = (
            approx_tokens
            * Decimal(str(config.cost_per_1k_tokens))
            / Decimal("1000")
        )
        await self.daily_budget.record_spend(approx_cost)

        logger.info(
            "llm_stream_complete",
            model=model,
            approx_tokens=int(approx_tokens),
            cost_usd=float(approx_cost),
            budget_remaining=float(self.daily_budget.budget_remaining),
        )

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    @staticmethod
    def _inject_system_prompt(
        messages: List[Dict[str, str]],
        config: ModelConfig,
    ) -> List[Dict[str, str]]:
        """Prepend task system prompt if caller hasn't already provided one."""
        if not config.system_prompt:
            return messages
        if messages and messages[0].get("role") == "system":
            return messages  # caller-supplied system message takes precedence
        return [{"role": "system", "content": config.system_prompt}] + messages

    @staticmethod
    def _estimate_cost(
        messages: List[Dict[str, str]],
        config: ModelConfig,
    ) -> Decimal:
        """
        Rough pre-call cost estimate using ~4 characters per token.
        Adds max_tokens as upper bound for completion tokens.
        """
        prompt_chars = sum(len(m.get("content", "")) for m in messages)
        prompt_tokens = Decimal(str(prompt_chars // 4))
        completion_tokens = Decimal(str(config.max_tokens))
        total_tokens = prompt_tokens + completion_tokens
        return (
            total_tokens
            * Decimal(str(config.cost_per_1k_tokens))
            / Decimal("1000")
        )

    async def _record_response_cost(
        self,
        response: Any,
        config: ModelConfig,
        task: TaskType,
        model: str,
    ) -> None:
        """Extract token usage from response and record actual spend."""
        try:
            usage = response.usage
            total_tokens = Decimal(str(usage.total_tokens))
            cost = (
                total_tokens
                * Decimal(str(config.cost_per_1k_tokens))
                / Decimal("1000")
            )
            await self.daily_budget.record_spend(cost)

            logger.info(
                "llm_complete",
                task=task.value,
                model=model,
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
                cost_usd=float(cost),
                budget_remaining=float(self.daily_budget.budget_remaining),
            )
        except Exception:
            # Usage extraction is best-effort; don't fail the caller
            logger.debug("llm_usage_unavailable", model=model)


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_router_instance: Optional[ModelRouter] = None


def get_router() -> ModelRouter:
    """Get or create singleton ModelRouter instance."""
    global _router_instance
    if _router_instance is None:
        _router_instance = ModelRouter()
    return _router_instance


# ---------------------------------------------------------------------------
# Test Block
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import asyncio

    async def test_router():
        """Basic router functionality test."""
        router = ModelRouter()
        print(f"Daily budget: ${router.daily_budget.limit_usd}")
        print(f"\nModels in use:")
        for task, config in MODEL_ROUTING.items():
            print(
                f"  {task.value:25} → {config.primary:30} "
                f"(${config.cost_per_1k_tokens:.6f}/1k)"
            )

        print("\n" + "=" * 70)
        print("Test 1: QUERY_PARSING (gemini-flash)")
        print("=" * 70)
        try:
            response = await router.complete(
                task=TaskType.QUERY_PARSING,
                messages=[{
                    "role": "user",
                    "content": "What is the tariff on laptops from China?"
                }]
            )
            print(f"Result: {response.choices[0].message.content}")
            print(f"Budget remaining: ${router.daily_budget.budget_remaining:.6f}")
        except Exception as e:
            print(f"Skipped (API key unavailable): {e}")

        print("\n" + "=" * 70)
        print("Test 2: HYDE_GENERATION (gemini-flash)")
        print("=" * 70)
        try:
            response = await router.complete(
                task=TaskType.HYDE_GENERATION,
                messages=[{
                    "role": "user",
                    "content": (
                        "Write a Federal Register excerpt about additional duties "
                        "on smartphones from China under HTS chapter 85."
                    )
                }]
            )
            content = response.choices[0].message.content
            print(f"Result: {content[:250]}...")
            print(f"Budget remaining: ${router.daily_budget.budget_remaining:.6f}")
        except Exception as e:
            print(f"Skipped (API key unavailable): {e}")

        print("\n" + "=" * 70)
        print("Test 3: CHAT_RESPONSE (gemini-flash)")
        print("=" * 70)
        try:
            response = await router.complete(
                task=TaskType.CHAT_RESPONSE,
                messages=[{
                    "role": "user",
                    "content": "What is Section 301?"
                }]
            )
            content = response.choices[0].message.content
            print(f"Result: {content[:250]}...")
            print(f"Budget remaining: ${router.daily_budget.budget_remaining:.6f}")
        except Exception as e:
            print(f"Skipped (API key unavailable): {e}")

        print("\n" + "=" * 70)
        print("Summary")
        print("=" * 70)
        total_spent = (
            router.daily_budget.limit_usd - router.daily_budget.budget_remaining
        )
        print(f"Total spent: ${total_spent:.6f}")
        print(f"Remaining budget: ${router.daily_budget.budget_remaining:.6f}")
        print("\nAll tests completed!")

    asyncio.run(test_router())