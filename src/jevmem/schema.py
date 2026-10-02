"""核心数据结构。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Protocol


@dataclass
class Turn:
    """一次需要判断的对话：此前的上下文 + 用户最新一条消息。"""

    message: str
    context: list[str] = field(default_factory=list)

    def state(self) -> dict:
        return {"conversation_so_far": self.context, "latest_user_message": self.message}


@dataclass
class CallStats:
    latency_s: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0

    def add(self, other: CallStats) -> None:
        self.latency_s += other.latency_s
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.calls += other.calls


@dataclass
class WriteDecision:
    store_prob: float
    memory_type: str
    tags: list[str]
    priority: str
    # Jev 才有的概率与置信度信息，方便调阈值；LLM 判断器留空
    type_confidence: float | None = None
    tag_probs: dict[str, float] = field(default_factory=dict)
    priority_score: float | None = None
    stats: CallStats = field(default_factory=CallStats)

    @property
    def should_store(self) -> bool:
        return self.store_prob >= 0.5


@dataclass
class RecallDecision:
    recall_prob: float
    types: list[str]
    tags: list[str]
    stats: CallStats = field(default_factory=CallStats)

    @property
    def needs_recall(self) -> bool:
        return self.recall_prob >= 0.5


@dataclass
class Memory:
    id: str
    memory_type: str
    tags: list[str]
    priority: str
    description: str
    content: str
    created_at: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RankResult:
    scores: dict[str, float]  # memory id -> 相关性
    stats: CallStats = field(default_factory=CallStats)


class Decider(Protocol):
    """负责"要不要存 / 要不要召回 / 召回哪些"的判断器，Jev 和 DeepSeek 各实现一份。"""

    name: str

    async def decide_write(self, turn: Turn) -> WriteDecision: ...

    async def decide_recall(self, turn: Turn) -> RecallDecision: ...

    async def rank(self, turn: Turn, candidates: list[Memory]) -> RankResult: ...
