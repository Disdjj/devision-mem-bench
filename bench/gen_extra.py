"""补充数据集：边界样本集（hard）与大记忆库（large）。

hard  : 专门构造的边界写入 / 召回用例，记忆库沿用原始 36 条。
        gold 由 Pro 独立盲标注两次，二元判断一致才保留——只剔除真正有歧义的样本，
        不再像主数据集那样按"与出题意图一致"过滤（那会把难题一起删掉，造成天花板效应）。
large : 在原记忆库上为同一人设补充干扰记忆到约 300 条，原 48 条召回用例对着大库重新盲标注。

用法：uv run python -m bench.gen_extra hard|large
"""

from __future__ import annotations

import argparse
import asyncio
import json

from jevmem.deepseek import TAXONOMY_TEXT, DeepSeek
from jevmem.taxonomy import MEMORY_TYPES, PRIORITY_NAMES, TAGS

from .gen_dataset import OUT as BASE, PRO, _dump, gather_limited, label_recall, label_write, translate

HARD_OUT = BASE.with_name("dataset_hard.json")
LARGE_OUT = BASE.with_name("dataset_large.json")
SYSTEM = "You create realistic benchmark data. Reply with JSON only."

HARD_WRITE_PROMPT = """Here is a persona: {persona}

Write {n} HARD, borderline test cases for the WRITE gate of a memory system (should the user's LAST message be
saved to long-term memory?). Every case should be tricky but still have a defensible correct answer.

{taxonomy}

Cover these patterns, a few cases each, about half worth storing:
- facts about OTHER people that do or do not matter to the user ("my colleague is vegan" vs gossip)
- hypotheticals and maybes ("if I ever move to Berlin...", "maybe I'll pick up guitar someday")
- temporary vs durable states ("I'm off coffee this week" vs "I quit coffee for good")
- corrections and retractions of earlier info ("scratch that, the trip moved to December")
- durable facts buried inside an unrelated task request or a long rant
- quoted or forwarded text that is not about the user
- sarcasm, jokes, or venting that sound like preferences but are not
- instructions that apply only to the current task vs standing instructions for all future chats
- info the assistant said, merely confirmed by the user ("yes, that's right")

context: 0-3 earlier turns as strings prefixed with "user: " or "assistant: "; ids are "h01", "h02", ...
Reply with JSON: {{"cases": [{{"id", "context", "message"}}]}}"""

HARD_RECALL_PROMPT = """Here is a persona and the long-term memories stored about them:
{bank}

Write {n} HARD, borderline test cases for the READ gate of the memory system (should stored memories be
recalled for the user's LAST message, and which ones?). Every case should be tricky but defensible.

Cover these patterns, a few cases each:
- relevance that needs an inference hop ("book a table for Friday" -> dietary restriction + child's schedule)
- near-miss distractors: same topic as a memory but the memory does not actually help
- generic-sounding questions that a stored memory would actually change ("best stretches after a run?")
- personal-sounding questions where no stored memory helps
- follow-ups whose meaning depends on the earlier context turns
- requests where a standing instruction memory should apply even though the topic is unrelated
- questions needing 2-3 memories combined

context: 0-2 earlier turns as strings prefixed with "user: " or "assistant: "; ids are "q01", "q02", ...
Reply with JSON: {{"cases": [{{"id", "context", "message"}}]}}"""

EXPAND_PROMPT = """Here is a persona and memories already stored about them:
{bank}

Write {n} ADDITIONAL, distinct long-term memories about the same person, focused on: {focus}.
They must not duplicate or contradict existing memories. Make them realistic and specific, and make some of
them topically close to existing ones (same tag, different details) so retrieval has to discriminate.

{taxonomy}

description: one short sentence; content: 1-3 sentences with concrete specifics, third person ("The user ...").
ids are "{prefix}01", "{prefix}02", ...
Reply with JSON: {{"memories": [{{"id", "memory_type", "tags", "priority", "description", "content"}}]}}"""

EXPAND_FOCUS = [
    ("xa", "work, colleagues, career history, tools used at work"),
    ("xb", "food, cooking, restaurants, drinks, shopping habits"),
    ("xc", "health, fitness, running, sleep, medical appointments"),
    ("xd", "family, daughter Beatriz's school and activities, husband Miguel, the dog Sushi, friends"),
    ("xe", "travel history, upcoming trips, transport preferences, the Japan trip logistics"),
    ("xf", "home renovation, finance, budgets, subscriptions, purchases"),
    ("xg", "learning Japanese, books, hobbies, music, games, weekend routines"),
    ("xh", "how the user likes the assistant to communicate, plus miscellaneous dated events in 2025-2026"),
]


def _bilingual(cases_en: list[dict], cases_zh: list[dict]) -> list[dict]:
    out = []
    for en, zh in zip(cases_en, cases_zh):
        out.append(en | {"lang": "en"})
        out.append(en | {"lang": "zh", "context": zh["context"], "message": zh["message"]})
    return out


def _text(c: dict) -> dict:
    return {"id": c["id"], "context": c["context"], "message": c["message"]}


def _clean_memory(m: dict) -> dict | None:
    if m.get("memory_type") not in MEMORY_TYPES:
        return None
    return {
        "id": m["id"],
        "memory_type": m["memory_type"],
        "tags": [t for t in m.get("tags") or [] if t in TAGS],
        "priority": m.get("priority") if m.get("priority") in PRIORITY_NAMES else "medium",
        "description": m["description"],
        "content": m["content"],
    }


async def build_hard(llm: DeepSeek, base: dict, n: int) -> None:
    persona, bank = base["meta"]["persona"], base["memory_bank"]["en"]
    print("1/3 生成边界用例 ...")
    (w, _), (r, _) = await asyncio.gather(
        llm.json(SYSTEM, HARD_WRITE_PROMPT.format(persona=persona, n=n, taxonomy=TAXONOMY_TEXT), model=PRO, effort="high"),
        llm.json(SYSTEM, HARD_RECALL_PROMPT.format(bank=_dump({"persona": persona, "memories": bank}), n=n), model=PRO, effort="high"),
    )
    write_cases, recall_cases = w["cases"], r["cases"]

    print(f"2/3 Pro 双盲标注 {len(write_cases)} + {len(recall_cases)} 条 ...")
    wa, wb, ra, rb = await asyncio.gather(
        gather_limited([label_write(llm, c) for c in write_cases]),
        gather_limited([label_write(llm, c) for c in write_cases]),
        gather_limited([label_recall(llm, bank, c) for c in recall_cases]),
        gather_limited([label_recall(llm, bank, c) for c in recall_cases]),
    )
    kept_w = [c | {"gold": a} for c, a, b in zip(write_cases, wa, wb) if a["should_store"] == b["should_store"]]
    kept_r, jaccard = [], []
    for c, a, b in zip(recall_cases, ra, rb):
        if a["needs_recall"] != b["needs_recall"]:
            continue
        sa, sb = set(a["relevant_ids"]), set(b["relevant_ids"])
        jaccard.append(len(sa & sb) / len(sa | sb) if sa | sb else 1.0)
        kept_r.append(c | {"gold": a})

    print("3/3 翻译为中文 ...")
    wz, rz = await asyncio.gather(translate(llm, [_text(c) for c in kept_w]), translate(llm, [_text(c) for c in kept_r]))
    dataset = {
        "meta": base["meta"] | {
            "split": "hard",
            "dropped_ambiguous": {"write": len(write_cases) - len(kept_w), "recall": len(recall_cases) - len(kept_r)},
            "recall_relevant_ids_jaccard_between_labelers": sum(jaccard) / max(1, len(jaccard)),
            "note": "gold 为 Pro 第一次盲标注；两次盲标注二元判断不一致的样本已剔除",
        },
        "memory_bank": base["memory_bank"],
        "write_cases": _bilingual(kept_w, wz),
        "recall_cases": _bilingual(kept_r, rz),
    }
    HARD_OUT.write_text(_dump(dataset))
    print(f"完成：写入 {len(kept_w)}×2，召回 {len(kept_r)}×2，剔除 {dataset['meta']['dropped_ambiguous']}，"
          f"相关集合标注一致度(Jaccard) {dataset['meta']['recall_relevant_ids_jaccard_between_labelers']:.2f} → {HARD_OUT}")


async def build_large(llm: DeepSeek, base: dict, per_focus: int) -> None:
    persona, bank = base["meta"]["persona"], base["memory_bank"]["en"]
    print(f"1/3 扩充记忆库（{len(EXPAND_FOCUS)} 批 × {per_focus}）...")
    seed = _dump({"persona": persona, "memories": [{"id": m["id"], "description": m["description"]} for m in bank]})
    outs = await gather_limited([
        llm.json(SYSTEM, EXPAND_PROMPT.format(bank=seed, n=per_focus, focus=focus, prefix=prefix, taxonomy=TAXONOMY_TEXT), model=PRO, effort="high")
        for prefix, focus in EXPAND_FOCUS
    ])
    extra = [m for out, _ in outs for m in map(_clean_memory, out["memories"]) if m]
    big_bank = bank + extra

    print(f"2/3 记忆库共 {len(big_bank)} 条；召回用例对着大库重新盲标注 + 翻译新增记忆 ...")
    recall_en = [c for c in base["recall_cases"] if c["lang"] == "en"]
    golds, extra_zh = await asyncio.gather(
        gather_limited([label_recall(llm, big_bank, c) for c in recall_en]),
        translate(llm, extra),
    )
    for zh, en in zip(extra_zh, extra):
        zh.update({k: en[k] for k in ("id", "memory_type", "tags", "priority")})

    recall_cases = []
    for c in base["recall_cases"]:
        gold = golds[[x["id"] for x in recall_en].index(c["id"])]
        recall_cases.append(c | {"gold": gold, "gold_small_bank": c["gold"]})
    dataset = {
        "meta": base["meta"] | {"split": "large", "bank_size": len(big_bank),
                                "note": "召回 gold 由 Pro 对着完整大库重新盲标注"},
        "memory_bank": {"en": big_bank, "zh": base["memory_bank"]["zh"] + extra_zh},
        "write_cases": [],
        "recall_cases": recall_cases,
    }
    LARGE_OUT.write_text(_dump(dataset))
    changed = sum(set(c["gold"]["relevant_ids"]) != set(c["gold_small_bank"]["relevant_ids"]) for c in recall_cases if c["lang"] == "en")
    print(f"完成：记忆库 {len(big_bank)} 条，召回 {len(recall_en)}×2，其中 {changed} 条的相关集合因新增记忆而变化 → {LARGE_OUT}")


async def main(split: str) -> None:
    base = json.loads(BASE.read_text())
    llm = DeepSeek()
    try:
        await (build_hard(llm, base, 40) if split == "hard" else build_large(llm, base, 33))
    finally:
        await llm.aclose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("split", choices=["hard", "large"])
    asyncio.run(main(parser.parse_args().split))
