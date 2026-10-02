"""基于 Jev (TypeSafe System One) 的判断器。

每个判断都是一次 `system_one` 调用：所有问题在同一个 state 上并行求值，
所以多加几个 tag / 候选 memory 几乎不增加延迟。
"""

from __future__ import annotations

import time

from typesafe_sdk import AsyncTypeSafeClient, Choice, Noul, Score

from . import config
from .schema import CallStats, Memory, RankResult, RecallDecision, Turn, WriteDecision
from .taxonomy import (
    MEMORY_TYPES,
    PRIORITY_INSTRUCTION,
    PRIORITY_LEVELS,
    PRIORITY_NAMES,
    RECALL_CRITERIA,
    RECALL_INSTRUCTION,
    RECALL_TAG_INSTRUCTION,
    RECALL_TYPE_INSTRUCTION,
    RELEVANCE_INSTRUCTION,
    STORE_CRITERIA,
    STORE_INSTRUCTION,
    TAG_INSTRUCTION,
    TAGS,
    TYPE_INSTRUCTION,
)

THRESHOLD = 0.5


def _priority_name(score: float) -> str:
    return PRIORITY_NAMES[min(len(PRIORITY_NAMES) - 1, max(0, round(score)))]


def _pick_tags(probs: dict[str, float]) -> list[str]:
    tags = [t for t, p in probs.items() if p >= THRESHOLD]
    # 至少保留一个 tag，便于召回时按 tag 过滤
    return tags or [max(probs, key=probs.get)]


class JevDecider:
    name = "jev"

    def __init__(self, client: AsyncTypeSafeClient | None = None, model: str = config.JEV_MODEL):
        self.client = client or AsyncTypeSafeClient(api_key=config.keys().jev, model=model)

    async def aclose(self) -> None:
        await self.client.aclose()

    async def _ask(self, state, questions) -> tuple:
        start = time.perf_counter()
        resp = await self.client.system_one(state, questions)
        stats = CallStats(
            latency_s=time.perf_counter() - start,
            input_tokens=resp.usage.input_tokens or 0,
            output_tokens=resp.usage.output_tokens or 0,
            calls=1,
        )
        return resp, stats

    async def decide_write(self, turn: Turn) -> WriteDecision:
        questions = {
            "store": Noul(instructions=STORE_INSTRUCTION, criteria=STORE_CRITERIA),
            "type": Choice(instructions=TYPE_INSTRUCTION, criteria=MEMORY_TYPES),
            "priority": Score(instructions=PRIORITY_INSTRUCTION, criteria=PRIORITY_LEVELS),
        }
        for tag, desc in TAGS.items():
            questions[f"tag_{tag}"] = Noul(instructions=TAG_INSTRUCTION.format(desc=desc))

        resp, stats = await self._ask(turn.state(), questions)
        tag_probs = {tag: resp.nouls[f"tag_{tag}"].noul for tag in TAGS}
        type_answer = resp.choices["type"]
        priority = resp.scores["priority"]
        return WriteDecision(
            store_prob=resp.nouls["store"].noul,
            memory_type=type_answer.choice,
            tags=_pick_tags(tag_probs),
            priority=_priority_name(priority.score),
            type_confidence=type_answer.confidence,
            tag_probs=tag_probs,
            priority_score=priority.score,
            stats=stats,
        )

    async def decide_recall(self, turn: Turn) -> RecallDecision:
        questions = {"recall": Noul(instructions=RECALL_INSTRUCTION, criteria=RECALL_CRITERIA)}
        for t, desc in MEMORY_TYPES.items():
            questions[f"type_{t}"] = Noul(instructions=RECALL_TYPE_INSTRUCTION.format(desc=desc))
        for tag, desc in TAGS.items():
            questions[f"tag_{tag}"] = Noul(instructions=RECALL_TAG_INSTRUCTION.format(desc=desc))

        resp, stats = await self._ask(turn.state(), questions)
        return RecallDecision(
            recall_prob=resp.nouls["recall"].noul,
            types=[t for t in MEMORY_TYPES if resp.nouls[f"type_{t}"].noul >= THRESHOLD],
            tags=[t for t in TAGS if resp.nouls[f"tag_{t}"].noul >= THRESHOLD],
            stats=stats,
        )

    async def rank(self, turn: Turn, candidates: list[Memory]) -> RankResult:
        if not candidates:
            return RankResult(scores={})
        # 每条候选一个 Noul：绝对相关性（而不是 Choice 的相对排序），可以对全部候选都说"不相关"
        questions = {
            f"m{i}": Noul(
                instructions={
                    "memory": {"description": m.description, "details": m.content},
                    "question": RELEVANCE_INSTRUCTION,
                }
            )
            for i, m in enumerate(candidates)
        }
        resp, stats = await self._ask(turn.state(), questions)
        scores = {m.id: resp.nouls[f"m{i}"].noul for i, m in enumerate(candidates)}
        return RankResult(scores=scores, stats=stats)
