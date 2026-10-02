"""在数据集上运行各配置，记录每条用例的预测、延迟与 token。

配置：
- jev        : Jev 做全部判断；写入时命中则由 Flash(low) 生成内容 → 即"Jev 判断 + Flash 生成"的混合系统
- flash-low  : DeepSeek Flash，thinking 开启，reasoning_effort=low
- flash-high : DeepSeek Flash，thinking 开启，reasoning_effort=high
- flash-nothink : DeepSeek Flash，关闭 thinking（速度最快的 LLM 对照）
- jev-fanout : 同 jev，但召回时跳过类型/tag 过滤，对全部 memory 一次性做相关性 Noul
- clef / clef-flash（及 -fanout）: Cloudflare 的 System One 兼容模型，判断逻辑与 jev 完全相同；
  单次最多 64 道题，rank 超出时分批并行
所有配置的生成器都是同一个 Flash(low)，端到端延迟的差异只来自判断器。

用法：uv run python -m bench.run_bench [--configs jev flash-low] [--langs en zh] [--limit 10]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
import traceback
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from jevmem.cli import decider_spec, make_decider, uses_prefilter
from jevmem.deepseek import DeepSeek, MemoryGenerator
from jevmem.jev import JevDecider
from jevmem.memory import MemorySystem
from jevmem.schema import Memory, Turn
from jevmem.store import MemoryStore

from .gen_dataset import OUT as DATASET

RESULTS = Path(__file__).parent / "results"
MEMORY_FIELDS = ("id", "memory_type", "tags", "priority", "description", "content")


def _stats(s) -> dict:
    return asdict(s)


async def run_write(system: MemorySystem, case: dict, generate: bool) -> dict:
    turn = Turn(message=case["message"], context=case["context"])
    if generate:
        result = await system.observe(turn)
        d, gen = result.decision, result.generate_stats
        memory = result.memory.to_dict() if result.memory else None
    else:
        d, gen, memory = await system.decider.decide_write(turn), None, None
    return {
        "pred": {
            "should_store": d.should_store,
            "store_prob": d.store_prob,
            "memory_type": d.memory_type,
            "type_confidence": d.type_confidence,
            "tags": d.tags,
            "tag_probs": d.tag_probs,
            "priority": d.priority,
            "priority_score": d.priority_score,
        },
        "decide": _stats(d.stats),
        "generate": _stats(gen) if gen else None,
        "memory": memory,
    }


async def run_recall(system: MemorySystem, case: dict) -> dict:
    result = await system.recall(Turn(message=case["message"], context=case["context"]))
    d = result.decision
    return {
        "pred": {
            "needs_recall": d.needs_recall,
            "recall_prob": d.recall_prob,
            "types": d.types,
            "tags": d.tags,
            "retrieved_ids": [m.id for m in result.memories],
            "scores": result.scores,
            "n_candidates": result.n_candidates,
        },
        "decide": _stats(d.stats),
        "rank": _stats(result.rank_stats),
    }


async def run_config(name: str, dataset: dict, args, out_file) -> None:
    llm = DeepSeek()
    decider = make_decider(name, llm)
    generator = MemoryGenerator(llm)
    stores = {}
    for lang, bank in dataset["memory_bank"].items():
        stores[lang] = MemoryStore()
        for m in bank:
            stores[lang].add(Memory(**{k: m[k] for k in MEMORY_FIELDS}))
    sem = asyncio.Semaphore(args.concurrency)

    async def one(kind: str, case: dict) -> None:
        # 写入用例用一个空的临时库，避免污染召回用的记忆库
        store = MemoryStore() if kind == "write" else stores[case["lang"]]
        system = MemorySystem(decider, store, generator, prefilter=uses_prefilter(name))
        async with sem:
            start = time.perf_counter()
            try:
                row = await (run_write(system, case, args.generate) if kind == "write" else run_recall(system, case))
                row["error"] = None
            except Exception as err:  # 记录失败而不是中断整轮 benchmark
                row = {"error": f"{type(err).__name__}: {err}", "trace": traceback.format_exc(limit=3)}
            row["wall_s"] = time.perf_counter() - start
        row |= {"config": name, "kind": kind, "id": case["id"], "lang": case["lang"], "gold": case["gold"]}
        out_file.write(json.dumps(row, ensure_ascii=False) + "\n")
        out_file.flush()
        status = "ERR" if row["error"] else "ok"
        print(f"  [{name}] {kind} {case['id']}/{case['lang']} {status} {row['wall_s']*1000:.0f}ms", flush=True)

    # 预热连接，避免第一条用例把 TLS 握手算进延迟
    await decider.decide_write(Turn(message="hello"))

    cases = [("write", c) for c in dataset["write_cases"]] + [("recall", c) for c in dataset["recall_cases"]]
    cases = [(k, c) for k, c in cases if c["lang"] in args.langs and k in args.kinds]
    if args.limit:
        cases = [(k, c) for k, c in cases if int(c["id"][1:]) <= args.limit]
    await asyncio.gather(*(one(k, c) for k, c in cases))

    if isinstance(decider, JevDecider):
        await decider.aclose()
    await llm.aclose()


async def main(args) -> Path:
    dataset = json.loads(Path(args.dataset).read_text())
    tag = f"-{args.tag}" if args.tag else ""
    run_dir = RESULTS / f"{datetime.now():%Y%m%d-%H%M%S}-{Path(args.dataset).stem}{tag}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "args.json").write_text(json.dumps(vars(args), indent=1))
    with open(run_dir / "raw.jsonl", "w") as f:
        # 各配置串行执行，避免互相争抢网络带宽而影响延迟
        for name in args.configs:
            print(f"== {name}")
            await run_config(name, dataset, args, f)
    print(f"原始结果: {run_dir / 'raw.jsonl'}")
    return run_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=str(DATASET))
    parser.add_argument("--tag", default="", help="结果目录名后缀，并行启动多个进程时用于区分")
    parser.add_argument("--configs", nargs="+", type=decider_spec, default=["jev", "flash-low", "flash-high"])
    parser.add_argument("--kinds", nargs="+", choices=["write", "recall"], default=["write", "recall"])
    parser.add_argument("--langs", nargs="+", default=["en", "zh"])
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0, help="只跑编号 <= limit 的用例，用于快速试跑")
    parser.add_argument("--no-generate", dest="generate", action="store_false", help="写入路径只测判断，不生成内容")
    args = parser.parse_args()
    run_dir = asyncio.run(main(args))

    from .report import build_report

    print(build_report(run_dir))
