"""命令行入口：jevmem observe / recall / list / delete。"""

from __future__ import annotations

import argparse
import asyncio
import contextlib

from . import config
from .deepseek import DeepSeek, DeepSeekDecider, MemoryGenerator
from .jev import JevDecider
from .memory import MemorySystem
from .schema import Turn
from .store import MemoryStore

DECIDERS = ["jev", "jev-fanout", "flash-nothink", "flash-low", "flash-high"]


def make_decider(name: str, llm: DeepSeek):
    """jev / jev-fanout 共用 JevDecider；flash-<effort> 中 nothink 表示关闭 thinking。"""
    if name.startswith("jev"):
        return JevDecider()
    effort = name.removeprefix("flash-")
    return DeepSeekDecider(llm, effort=None if effort == "nothink" else effort)


@contextlib.asynccontextmanager
async def build_system(decider_name: str, store: MemoryStore):
    llm = DeepSeek()
    decider = make_decider(decider_name, llm)
    try:
        yield MemorySystem(decider, store, MemoryGenerator(llm), prefilter=decider_name != "jev-fanout")
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
        p.add_argument("--decider", choices=DECIDERS, default="jev")
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
