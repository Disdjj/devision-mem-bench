"""从 raw.jsonl 计算指标并生成 Markdown 报告。

用法：uv run python -m bench.report bench/results/<run>
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

from jevmem import config
from jevmem.taxonomy import PRIORITY_NAMES


def pct(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    values = sorted(values)
    k = (len(values) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (k - lo)


def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def auc(scores: list[float], labels: list[bool]) -> float:
    """ROC-AUC（Mann–Whitney U）。二值输出的 LLM 也能算，只是等价于平衡准确率。"""
    pos = [s for s, y in zip(scores, labels) if y]
    neg = [s for s, y in zip(scores, labels) if not y]
    if not pos or not neg:
        return float("nan")
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def set_counts(pred: set, gold: set) -> tuple[int, int, int]:
    return len(pred & gold), len(pred - gold), len(gold - pred)


def cost_usd(cfg: str, stage: str, s: dict | None) -> float:
    if not s:
        return 0.0
    base = cfg.partition("@")[0].removesuffix("-fanout")
    system_one_price = {"jev": config.JEV_PRICE_IN, "clef": config.CLEF_PRICE_IN, "clef-flash": config.CLEF_FLASH_PRICE_IN}
    if base in system_one_price and stage != "generate":
        return s["input_tokens"] * system_one_price[base] / 1e6
    return (
        s["input_tokens"] * config.DEEPSEEK_FLASH_PRICE_IN + s["output_tokens"] * config.DEEPSEEK_FLASH_PRICE_OUT
    ) / 1e6


def ms(x: float) -> str:
    return f"{x * 1000:.0f}"


def write_metrics(rows: list[dict]) -> dict:
    ok = [r for r in rows if not r["error"]]
    y = [r["gold"]["should_store"] for r in ok]
    yhat = [r["pred"]["should_store"] for r in ok]
    tp = sum(a and b for a, b in zip(y, yhat))
    fp = sum(b and not a for a, b in zip(y, yhat))
    fn = sum(a and not b for a, b in zip(y, yhat))
    p, rc, f1 = prf(tp, fp, fn)

    pos = [r for r in ok if r["gold"]["should_store"]]
    type_rows = [r for r in pos if r["gold"]["memory_type"]]
    type_acc = sum(r["pred"]["memory_type"] == r["gold"]["memory_type"] for r in type_rows) / max(1, len(type_rows))
    ttp = tfp = tfn = 0
    for r in pos:
        a, b, c = set_counts(set(r["pred"]["tags"]), set(r["gold"]["tags"]))
        ttp, tfp, tfn = ttp + a, tfp + b, tfn + c
    prio_idx = lambda v: PRIORITY_NAMES.index(v)  # noqa: E731
    prio_acc = sum(r["pred"]["priority"] == r["gold"]["priority"] for r in pos) / max(1, len(pos))
    prio_mae = sum(abs(prio_idx(r["pred"]["priority"]) - prio_idx(r["gold"]["priority"])) for r in pos) / max(1, len(pos))

    decide = [r["decide"]["latency_s"] for r in ok]
    e2e = [r["decide"]["latency_s"] + (r["generate"] or {}).get("latency_s", 0.0) for r in ok]
    cfg = ok[0]["config"] if ok else ""
    return {
        "n": len(rows),
        "err": len(rows) - len(ok),
        "acc": sum(a == b for a, b in zip(y, yhat)) / max(1, len(ok)),
        "p": p, "r": rc, "f1": f1,
        "auc": auc([r["pred"]["store_prob"] for r in ok], y),
        "type_acc": type_acc,
        "tag_f1": prf(ttp, tfp, tfn)[2],
        "prio_acc": prio_acc,
        "prio_mae": prio_mae,
        "decide_p50": pct(decide, 0.5), "decide_p95": pct(decide, 0.95),
        "e2e_p50": pct(e2e, 0.5), "e2e_p95": pct(e2e, 0.95),
        "cost_decide_1k": 1000 * sum(cost_usd(cfg, "decide", r["decide"]) for r in ok) / max(1, len(ok)),
        "cost_e2e_1k": 1000 * sum(cost_usd(cfg, "decide", r["decide"]) + cost_usd(cfg, "generate", r["generate"]) for r in ok) / max(1, len(ok)),
        "tok_in": sum(r["decide"]["input_tokens"] for r in ok) / max(1, len(ok)),
    }


def recall_metrics(rows: list[dict]) -> dict:
    ok = [r for r in rows if not r["error"]]
    y = [r["gold"]["needs_recall"] for r in ok]
    yhat = [r["pred"]["needs_recall"] for r in ok]
    tp = sum(a and b for a, b in zip(y, yhat))
    fp = sum(b and not a for a, b in zip(y, yhat))
    fn = sum(a and not b for a, b in zip(y, yhat))
    gate_f1 = prf(tp, fp, fn)[2]

    with_rel = [r for r in ok if r["gold"]["relevant_ids"]]
    ttp = tfp = tfn = 0
    for r in with_rel:
        a, b, c = set_counts(set(r["pred"]["types"]), set(r["gold"]["types"]))
        ttp, tfp, tfn = ttp + a, tfp + b, tfn + c
    mtp = mfp = mfn = 0
    for r in ok:
        a, b, c = set_counts(set(r["pred"]["retrieved_ids"]), set(r["gold"]["relevant_ids"]))
        mtp, mfp, mfn = mtp + a, mfp + b, mfn + c
    mp, mr, mf1 = prf(mtp, mfp, mfn)
    hit = sum(bool(set(r["pred"]["retrieved_ids"]) & set(r["gold"]["relevant_ids"])) for r in with_rel)
    no_rel = [r for r in ok if not r["gold"]["relevant_ids"]]
    clean = sum(not r["pred"]["retrieved_ids"] for r in no_rel)
    # 候选覆盖率：类型/tag 过滤后，gold 相关 memory 还留在候选集里的比例
    cover_n = sum(len(r["gold"]["relevant_ids"]) for r in with_rel)
    cover = sum(len(set(r["pred"]["scores"]) & set(r["gold"]["relevant_ids"])) for r in with_rel)

    total = [r["decide"]["latency_s"] + r["rank"]["latency_s"] for r in ok]
    cfg = ok[0]["config"] if ok else ""
    return {
        "n": len(rows),
        "err": len(rows) - len(ok),
        "gate_acc": sum(a == b for a, b in zip(y, yhat)) / max(1, len(ok)),
        "gate_f1": gate_f1,
        "gate_auc": auc([r["pred"]["recall_prob"] for r in ok], y),
        "type_f1": prf(ttp, tfp, tfn)[2],
        "cover": cover / max(1, cover_n),
        "mem_p": mp, "mem_r": mr, "mem_f1": mf1,
        "hit": hit / len(with_rel) if with_rel else float("nan"),
        "clean": clean / len(no_rel) if no_rel else float("nan"),
        "cands": sum(r["pred"]["n_candidates"] for r in ok) / max(1, len(ok)),
        "p50": pct(total, 0.5), "p95": pct(total, 0.95),
        "calls": sum(r["decide"]["calls"] + r["rank"]["calls"] for r in ok) / max(1, len(ok)),
        "cost_1k": 1000 * sum(cost_usd(cfg, "decide", r["decide"]) + cost_usd(cfg, "rank", r["rank"]) for r in ok) / max(1, len(ok)),
    }


def _group(rows: list[dict], kind: str) -> dict[tuple[str, str], list[dict]]:
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        if r["kind"] == kind:
            groups[(r["config"], r["lang"])].append(r)
            groups[(r["config"], "all")].append(r)
    return dict(sorted(groups.items()))


def _f(x: float) -> str:
    return f"{x:.2f}"


def build_report(run_dir: str | Path) -> str:
    run_dir = Path(run_dir)
    rows = [json.loads(line) for line in (run_dir / "raw.jsonl").read_text().splitlines() if line.strip()]
    lines = [f"# Memory benchmark — {run_dir.name}", ""]

    lines += [
        "## 写入路径（要不要存 / 类型 / tag / 优先级）",
        "",
        "| 配置 | 语言 | n(err) | 存储 Acc | P | R | F1 | AUC | 类型 Acc | Tag F1 | 优先级 Acc | 优先级 MAE "
        "| 判断 p50/p95 ms | 端到端 p50/p95 ms | 判断 $/1k | 端到端 $/1k |",
        "|" + "---|" * 16,
    ]
    write_summary = {}
    for (cfg, lang), group in _group(rows, "write").items():
        m = write_metrics(group)
        write_summary[f"{cfg}/{lang}"] = m
        lines.append(
            f"| {cfg} | {lang} | {m['n']}({m['err']}) | {_f(m['acc'])} | {_f(m['p'])} | {_f(m['r'])} | {_f(m['f1'])} "
            f"| {_f(m['auc'])} | {_f(m['type_acc'])} | {_f(m['tag_f1'])} | {_f(m['prio_acc'])} | {_f(m['prio_mae'])} "
            f"| {ms(m['decide_p50'])}/{ms(m['decide_p95'])} | {ms(m['e2e_p50'])}/{ms(m['e2e_p95'])} "
            f"| {m['cost_decide_1k']:.4f} | {m['cost_e2e_1k']:.4f} |"
        )

    lines += [
        "",
        "## 召回路径（要不要召回 / 召回哪些类型 / 召回哪些 memory）",
        "",
        "| 配置 | 语言 | n(err) | 召回门 Acc | 门 F1 | 门 AUC | 类型 F1 | 候选覆盖 | Memory P | R | F1 | 命中率 "
        "| 无关时干净率 | 平均候选 | 调用次数 | p50/p95 ms | $/1k |",
        "|" + "---|" * 17,
    ]
    recall_summary = {}
    for (cfg, lang), group in _group(rows, "recall").items():
        m = recall_metrics(group)
        recall_summary[f"{cfg}/{lang}"] = m
        lines.append(
            f"| {cfg} | {lang} | {m['n']}({m['err']}) | {_f(m['gate_acc'])} | {_f(m['gate_f1'])} | {_f(m['gate_auc'])} "
            f"| {_f(m['type_f1'])} | {_f(m['cover'])} | {_f(m['mem_p'])} | {_f(m['mem_r'])} | {_f(m['mem_f1'])} "
            f"| {_f(m['hit'])} | {_f(m['clean'])} | {m['cands']:.1f} | {m['calls']:.1f} "
            f"| {ms(m['p50'])}/{ms(m['p95'])} | {m['cost_1k']:.4f} |"
        )

    lines += [
        "",
        "说明：",
        "- 端到端 = 判断 + （判定要存时）Flash(low) 生成 description/content；jev 行即“Jev 判断 + Flash 生成”的混合系统。",
        "- AUC 用概率输出计算；LLM 只给二值结果，其 AUC 等价于平衡准确率，Jev 的概率可以另调阈值。",
        "- 类型 / Tag / 优先级只在 gold 判定需要存储的用例上统计；召回类型 F1 只在 gold 有相关 memory 的用例上统计。",
        "- 命中率：gold 有相关 memory 时至少召回一条正确的比例；干净率：gold 无相关 memory 时什么都没召回的比例。",
        f"- 成本：Jev ${config.JEV_PRICE_IN}/M、Clef ${config.CLEF_PRICE_IN}/M、Clef-flash ${config.CLEF_FLASH_PRICE_IN}/M 输入；Flash ${config.DEEPSEEK_FLASH_PRICE_IN}/M 输入、"
        f"${config.DEEPSEEK_FLASH_PRICE_OUT}/M 输出（高峰、未命中缓存）。",
        "- gold label 由 DeepSeek Pro 生成，对 DeepSeek 系模型可能存在同源偏好。",
    ]
    report = "\n".join(lines)
    (run_dir / "report.md").write_text(report)
    (run_dir / "summary.json").write_text(json.dumps({"write": write_summary, "recall": recall_summary}, indent=1))
    return report


if __name__ == "__main__":
    print(build_report(sys.argv[1]))
