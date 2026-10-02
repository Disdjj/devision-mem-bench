"""MemorySystem：把判断器、生成器和存储组装成写入 / 召回两条路径。"""

from __future__ import annotations

from dataclasses import dataclass, field

from .deepseek import MemoryGenerator
from .schema import CallStats, Decider, Memory, RecallDecision, Turn, WriteDecision
from .store import MemoryStore
from .taxonomy import PRIORITY_NAMES


@dataclass
class ObserveResult:
    decision: WriteDecision
    memory: Memory | None
    generate_stats: CallStats = field(default_factory=CallStats)

    @property
    def total_latency_s(self) -> float:
        return self.decision.stats.latency_s + self.generate_stats.latency_s


@dataclass
class RecallResult:
    decision: RecallDecision
    memories: list[Memory]
    scores: dict[str, float]
    n_candidates: int
    rank_stats: CallStats = field(default_factory=CallStats)

    @property
    def total_latency_s(self) -> float:
        return self.decision.stats.latency_s + self.rank_stats.latency_s


class MemorySystem:
    def __init__(
        self,
        decider: Decider,
        store: MemoryStore,
        generator: MemoryGenerator | None = None,
        top_k: int = 5,
        threshold: float = 0.5,
        prefilter: bool = True,
    ):
        self.decider, self.store, self.generator = decider, store, generator
        self.top_k, self.threshold = top_k, threshold
        # prefilter=False 时跳过类型/tag 过滤，对全部 memory 做相关性判断（Jev 的 speculative fan-out）
        self.prefilter = prefilter

    async def observe(self, turn: Turn) -> ObserveResult:
        """写入路径：判断要不要存 → 生成 description/content → 落库。"""
        decision = await self.decider.decide_write(turn)
        if not decision.should_store:
            return ObserveResult(decision, None)
        if self.generator is None:
            raise RuntimeError("写入 memory 需要配置 MemoryGenerator")
        description, content, gen_stats = await self.generator.generate(turn, decision)
        memory = self.store.add(
            Memory(
                id="",
                memory_type=decision.memory_type,
                tags=decision.tags,
                priority=decision.priority,
                description=description,
                content=content,
            )
        )
        return ObserveResult(decision, memory, gen_stats)

    async def recall(self, turn: Turn) -> RecallResult:
        """召回路径：判断要不要召回及类型/tag → 代码过滤候选（可关闭）→ 逐条判断相关性。"""
        decision = await self.decider.decide_recall(turn)
        if not decision.needs_recall:
            return RecallResult(decision, [], {}, 0)
        candidates = self.store.candidates(decision.types, decision.tags) if self.prefilter else self.store.all()
        ranked = await self.decider.rank(turn, candidates)
        by_id = {m.id: m for m in candidates}
        hits = [mid for mid, s in ranked.scores.items() if s >= self.threshold]
        hits.sort(key=lambda mid: (ranked.scores[mid], PRIORITY_NAMES.index(by_id[mid].priority)), reverse=True)
        return RecallResult(
            decision,
            [by_id[mid] for mid in hits[: self.top_k]],
            ranked.scores,
            len(candidates),
            ranked.stats,
        )
