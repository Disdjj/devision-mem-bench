"""DeepSeek 客户端、基于 LLM 的对照判断器、以及 memory 内容生成器。"""

from __future__ import annotations

import json
import time

from openai import AsyncOpenAI

from . import config
from .schema import CallStats, Memory, RankResult, RecallDecision, Turn, WriteDecision
from .taxonomy import (
    MEMORY_TYPES,
    PRIORITY_LEVELS,
    PRIORITY_NAMES,
    RECALL_CRITERIA,
    RECALL_INSTRUCTION,
    STORE_CRITERIA,
    STORE_INSTRUCTION,
    TAGS,
)


class DeepSeek:
    """对 OpenAI 兼容接口的薄封装：JSON 输出 + 思考强度 + 计时。"""

    def __init__(self, client: AsyncOpenAI | None = None):
        self.client = client or AsyncOpenAI(
            api_key=config.keys().deepseek, base_url=config.DEEPSEEK_BASE_URL, max_retries=3, timeout=600
        )

    async def json(
        self,
        system: str,
        user: str,
        *,
        model: str = config.DEEPSEEK_FLASH,
        effort: str | None = "low",
        retries: int = 2,
    ) -> tuple[dict, CallStats]:
        thinking = {"type": "enabled"} if effort else {"type": "disabled"}
        extra = {"thinking": thinking} | ({"reasoning_effort": effort} if effort else {})
        stats = CallStats()
        last_err: Exception | None = None
        for _ in range(retries + 1):
            start = time.perf_counter()
            resp = await self.client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                response_format={"type": "json_object"},
                extra_body=extra,
            )
            stats.add(
                CallStats(
                    latency_s=time.perf_counter() - start,
                    input_tokens=resp.usage.prompt_tokens if resp.usage else 0,
                    output_tokens=resp.usage.completion_tokens if resp.usage else 0,
                    calls=1,
                )
            )
            try:
                return json.loads(resp.choices[0].message.content or ""), stats
            except json.JSONDecodeError as err:
                last_err = err
        raise RuntimeError(f"DeepSeek 返回的 JSON 无法解析: {last_err}")

    async def aclose(self) -> None:
        await self.client.close()


def _bullets(items: dict[str, str]) -> str:
    return "\n".join(f"- {k}: {v}" for k, v in items.items())


TAXONOMY_TEXT = f"""Memory types:
{_bullets(MEMORY_TYPES)}

Tags (multi-label):
{_bullets(TAGS)}

Priority levels:
{_bullets(dict(zip(PRIORITY_NAMES, PRIORITY_LEVELS)))}"""

WRITE_SYSTEM = f"""You are the write gate of an assistant's long-term memory system.
Given the conversation so far and the user's latest message, decide whether the latest message contains
information worth saving to long-term memory, and if so classify it.

Question: {STORE_INSTRUCTION}
- true: {STORE_CRITERIA["true"]}
- false: {STORE_CRITERIA["false"]}

{TAXONOMY_TEXT}

Always fill every field, even when should_store is false (classify the message as best you can).
Reply with JSON only:
{{"should_store": true|false, "memory_type": "<one type>", "tags": ["<tag>", ...], "priority": "low|medium|high"}}"""

RECALL_SYSTEM = f"""You are the read gate of an assistant's long-term memory system.
Given the conversation so far and the user's latest message, decide whether stored memories about this user
should be recalled, and which memory types and tags would be helpful.

Question: {RECALL_INSTRUCTION}
- true: {RECALL_CRITERIA["true"]}
- false: {RECALL_CRITERIA["false"]}

{TAXONOMY_TEXT}

Reply with JSON only:
{{"needs_recall": true|false, "types": ["<type>", ...], "tags": ["<tag>", ...]}}"""

RANK_SYSTEM = """You are the relevance filter of an assistant's long-term memory system.
Given the user's latest message and a list of candidate memories, return the ids of every memory that is
relevant and useful for responding to the latest message. Return an empty list if none are.
Reply with JSON only: {"relevant": ["<id>", ...]}"""

GENERATE_SYSTEM = """You write entries for an assistant's long-term memory about a user.
Given the conversation and the classification of the new memory, write:
- description: one short sentence (max ~25 words) summarizing the memory, used for retrieval.
- content: a self-contained, detailed memory with every concrete specific (names, dates, numbers,
  reasons, constraints) the user gave, written in third person ("The user ...").
Write both fields in the same language the user used.
Reply with JSON only: {"description": "...", "content": "..."}"""


def _turn_text(turn: Turn) -> str:
    return json.dumps(turn.state(), ensure_ascii=False, indent=1)


def _clean(values, allowed) -> list[str]:
    return [v for v in values or [] if v in allowed]


class DeepSeekDecider:
    """用 DeepSeek Flash 完成和 Jev 相同的判断，作为对照组。"""

    def __init__(self, llm: DeepSeek, effort: str | None = "low", model: str = config.DEEPSEEK_FLASH):
        self.llm, self.effort, self.model = llm, effort, model
        self.name = f"flash-{effort or 'nothink'}"

    async def _json(self, system: str, user: str) -> tuple[dict, CallStats]:
        return await self.llm.json(system, user, model=self.model, effort=self.effort)

    async def decide_write(self, turn: Turn) -> WriteDecision:
        out, stats = await self._json(WRITE_SYSTEM, _turn_text(turn))
        memory_type = out.get("memory_type")
        priority = out.get("priority")
        return WriteDecision(
            store_prob=1.0 if out.get("should_store") else 0.0,
            memory_type=memory_type if memory_type in MEMORY_TYPES else next(iter(MEMORY_TYPES)),
            tags=_clean(out.get("tags"), TAGS),
            priority=priority if priority in PRIORITY_NAMES else "medium",
            stats=stats,
        )

    async def decide_recall(self, turn: Turn) -> RecallDecision:
        out, stats = await self._json(RECALL_SYSTEM, _turn_text(turn))
        return RecallDecision(
            recall_prob=1.0 if out.get("needs_recall") else 0.0,
            types=_clean(out.get("types"), MEMORY_TYPES),
            tags=_clean(out.get("tags"), TAGS),
            stats=stats,
        )

    async def rank(self, turn: Turn, candidates: list[Memory]) -> RankResult:
        if not candidates:
            return RankResult(scores={})
        payload = {
            "latest_user_message": turn.message,
            "conversation_so_far": turn.context,
            "candidates": [{"id": m.id, "description": m.description, "details": m.content} for m in candidates],
        }
        out, stats = await self._json(RANK_SYSTEM, json.dumps(payload, ensure_ascii=False, indent=1))
        relevant = set(out.get("relevant") or [])
        return RankResult(scores={m.id: 1.0 if m.id in relevant else 0.0 for m in candidates}, stats=stats)


class MemoryGenerator:
    """判断要存之后，由 DeepSeek Flash 生成 description 与详细 content。"""

    def __init__(self, llm: DeepSeek, effort: str | None = "low"):
        self.llm, self.effort = llm, effort

    async def generate(self, turn: Turn, decision: WriteDecision) -> tuple[str, str, CallStats]:
        payload = turn.state() | {
            "classification": {
                "memory_type": decision.memory_type,
                "tags": decision.tags,
                "priority": decision.priority,
            }
        }
        out, stats = await self.llm.json(
            GENERATE_SYSTEM, json.dumps(payload, ensure_ascii=False, indent=1), effort=self.effort
        )
        return str(out.get("description", "")), str(out.get("content", "")), stats
