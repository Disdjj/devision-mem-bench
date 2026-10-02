"""命令行入口：jevmem observe / recall / list / delete。"""

from __future__ import annotations

import argparse
import asyncio
import contextlib

from . import config
from .clef import CLEF_MAX_QUESTIONS, CloudflareSystemOne
from .deepseek import DeepSeek, DeepSeekDecider, MemoryGenerator
from .jev import JevDecider
from .memory import MemorySystem
from .schema import Turn
from .store import MemoryStore

DECIDERS = [
    "jev", "jev-fanout", "clef", "clef-fanout", "clef-flash", "clef-flash-fanout",
    "flash-nothink", "flash-low", "flash-high",
]


def make_decider(name: str, llm: DeepSeek):
    """System One 系（jev / clef / clef-flash，可带 -fanout 后缀和 @<召回门阈值>）共用 JevDecider；
    flash-<effort> 是 DeepSeek Flash，其中 nothink 表示关闭 thinking。"""
    spec, _, threshold = name.partition("@")
    base = spec.removesuffix("-fanout")
    opts = {"name": name} | ({"recall_threshold": float(threshold)} if threshold else {})
    if base == "jev":
        return JevDecider(**opts)
    if base in ("clef", "clef-flash"):
        return JevDecider(CloudflareSystemOne(base), max_questions=CLEF_MAX_QUESTIONS, **opts)
    effort = spec.removeprefix("flash-")
    return DeepSeekDecider(llm, effort=None if effort == "nothink" else effort)


def decider_spec(name: str) -> str:
    """argparse 校验：基础名必须在 DECIDERS 中，可带 @<阈值>。"""
    spec, _, threshold = name.partition("@")
    if spec not in DECIDERS:
        raise argparse.ArgumentTypeError(f"未知配置 {spec}，可选: {', '.join(DECIDERS)}")
    if threshold:
        float(threshold)
    return name


def uses_prefilter(name: str) -> bool:
    return not name.partition("@")[0].endswith("-fanout")


@contextlib.asynccontextmanager
async def build_system(decider_name: str, store: MemoryStore):
    llm = DeepSeek()
    decider = make_decider(decider_name, llm)
    try:
        yield MemorySystem(decider, store, MemoryGenerator(llm), prefilter=uses_prefilter(decider_name))
    finally:
        if isinstance(decider, JevDecider):
            await decider.aclose()
        await llm.aclose()


def _ms(seconds: float) -> str:
    return f"{seconds * 1000:.0f}ms"


async def _observe(args, store: MemoryStore) -> None:
    async with build_system(args.decider, store) as system:
        result = await system.observe(Turn(message=args.message, context=args.context or []))
    d = result.decision
    print(f"[{args.decider}] store_prob={d.store_prob:.2f} type={d.memory_type} tags={d.tags} "
          f"priority={d.priority}  判断 {_ms(d.stats.latency_s)}")
    if result.memory:
        m = result.memory
        print(f"已写入 {m.id}  生成 {_ms(result.generate_stats.latency_s)}\n  description: {m.description}\n  content: {m.content}")
    else:
        print("不需要写入")


async def _recall(args, store: MemoryStore) -> None:
    async with build_system(args.decider, store) as system:
        result = await system.recall(Turn(message=args.message, context=args.context or []))
    d = result.decision
    print(f"[{args.decider}] recall_prob={d.recall_prob:.2f} types={d.types} tags={d.tags}  "
          f"候选 {result.n_candidates} 条  总耗时 {_ms(result.total_latency_s)}")
    for m in result.memories:
        print(f"  {result.scores[m.id]:.2f} [{m.id}] ({m.memory_type}/{m.priority}) {m.description}")
    if d.needs_recall and not result.memories:
        print("  没有相关的 memory")


def main() -> None:
    parser = argparse.ArgumentParser(prog="jevmem", description="基于 Jev 的简易 memory 系统")
    parser.add_argument("--db", default=str(config.db_path()))
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("observe", "recall"):
        p = sub.add_parser(name)
        p.add_argument("message")
        p.add_argument("--context", nargs="*", help="此前的对话，按顺序给出")
        p.add_argument("--decider", type=decider_spec, default="jev", help=f"可选: {', '.join(DECIDERS)}，可带 @<召回门阈值>")
    sub.add_parser("list")
    p = sub.add_parser("delete")
    p.add_argument("id")
    args = parser.parse_args()

    store = MemoryStore(args.db)
    if args.cmd == "observe":
        asyncio.run(_observe(args, store))
    elif args.cmd == "recall":
        asyncio.run(_recall(args, store))
    elif args.cmd == "list":
        for m in store.all():
            print(f"[{m.id}] {m.memory_type}/{m.priority} {m.tags} {m.description}")
    elif args.cmd == "delete":
        print("已删除" if store.delete(args.id) else "未找到")
