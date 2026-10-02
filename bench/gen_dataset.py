"""用 DeepSeek Pro 生成双语 benchmark 数据集。

流程：
1. 生成一个人设 + 记忆库（英文）
2. 生成写入用例、召回用例（英文，带出题意图）
3. Pro 盲标注：不看出题意图，独立给出 gold label；二元判断与意图冲突的样本视为歧义样本剔除
4. 翻译成中文，id 与 gold label 保持不变，保证中英文对比只差语言

用法：uv run python -m bench.gen_dataset [--n-bank 36 --n-write 48 --n-recall 48]
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from jevmem import config
from jevmem.deepseek import TAXONOMY_TEXT, WRITE_SYSTEM, DeepSeek
from jevmem.taxonomy import MEMORY_TYPES, PRIORITY_NAMES, RECALL_CRITERIA, RECALL_INSTRUCTION, TAGS

OUT = Path(__file__).parent / "data" / "dataset.json"
PRO = config.DEEPSEEK_PRO

BANK_PROMPT = f"""Invent one realistic, specific person (job, family, city, hobbies, health, ongoing projects, plans)
and write {{n}} long-term memories an AI assistant has stored about them from past conversations.

{TAXONOMY_TEXT}

Requirements:
- Cover every memory type at least 4 times and every tag at least twice; mix priorities (some high).
- Make several memories topically close to each other so retrieval must discriminate (e.g. two food memories,
  two travel memories with different details).
- description: one short sentence; content: 1-3 sentences with concrete specifics, third person ("The user ...").
- ids are "m01", "m02", ...

Reply with JSON: {{{{"persona": "<2 sentences>", "memories": [{{{{"id", "memory_type", "tags", "priority", "description", "content"}}}}]}}}}"""

WRITE_CASES_PROMPT = """Here is a persona: {persona}

Write {n} test cases for the WRITE gate of a memory system. Each case is a short conversation where the
persona is the user. The gate must decide whether the user's LAST message contains durable information worth
saving to long-term memory.

{taxonomy}

Requirements:
- About 55% should be worth storing and 45% not. Cover every memory type and all three priorities among the positives.
- Include hard negatives: transient states ("I'm hungry right now"), general knowledge questions,
  one-off task requests, small talk, opinions about the weather, and messages that only repeat what the
  assistant said.
- Include hard positives: durable facts mentioned in passing inside a task request, corrections of an earlier
  preference, standing instructions phrased casually.
- context: 0-3 earlier turns as strings prefixed with "user: " or "assistant: "; message: the user's last message.
- ids are "w01", "w02", ...

Reply with JSON: {{"cases": [{{"id", "context", "message", "intent_store": true|false}}]}}"""

RECALL_CASES_PROMPT = """Here is a persona and the long-term memories stored about them:
{bank}

Write {n} test cases for the READ gate of the memory system. Each case is a short conversation where the
persona is the user. The gate must decide whether stored memories should be recalled for the user's LAST
message, and which memories are relevant.

Requirements:
- About 65% should need recall, with 1-3 relevant memories each. Include indirect cases where relevance needs a
  small inference (e.g. "plan a dinner party" -> a food allergy memory; "draft my out-of-office" -> a trip).
- About 35% should not need recall: general knowledge, small talk, or self-contained tasks.
- Include a few personal requests where no stored memory is relevant (needs recall but nothing matches).
- context: 0-2 earlier turns as strings prefixed with "user: " or "assistant: "; message: the user's last message.
- ids are "r01", "r02", ...

Reply with JSON: {{"cases": [{{"id", "context", "message", "intent_relevant_ids": ["m.."]}}]}}"""

LABEL_RECALL_SYSTEM = f"""You are an expert annotator for an assistant's long-term memory system.
Given the stored memories and a conversation, label the READ gate for the user's latest message.

needs_recall — {RECALL_INSTRUCTION}
- true: {RECALL_CRITERIA["true"]}
- false: {RECALL_CRITERIA["false"]}

relevant_ids — every stored memory that is relevant and useful for responding to the latest message
(may be empty, even when needs_recall is true).

Think carefully, then reply with JSON only: {{"needs_recall": true|false, "relevant_ids": ["m.."]}}"""

LABEL_WRITE_SYSTEM = WRITE_SYSTEM.replace(
    "You are the write gate of", "You are an expert annotator labeling the write gate of"
)

TRANSLATE_SYSTEM = """Translate the JSON values from English into natural, colloquial Simplified Chinese as a
native speaker would write in a chat. Keep every key, id, enum value (memory_type, tags, priority) and the
"user: " / "assistant: " prefixes unchanged. Keep the exact same structure and item order.
Reply with the translated JSON only."""


def _dump(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=1)


async def gather_limited(coros, limit: int = 8):
    sem = asyncio.Semaphore(limit)

    async def run(c):
        async with sem:
            return await c

    return await asyncio.gather(*(run(c) for c in coros))


async def label_write(llm: DeepSeek, case: dict) -> dict:
    state = {"conversation_so_far": case["context"], "latest_user_message": case["message"]}
    out, _ = await llm.json(LABEL_WRITE_SYSTEM, _dump(state), model=PRO, effort="max")
    return {
        "should_store": bool(out.get("should_store")),
        "memory_type": out.get("memory_type") if out.get("memory_type") in MEMORY_TYPES else None,
        "tags": [t for t in out.get("tags") or [] if t in TAGS],
        "priority": out.get("priority") if out.get("priority") in PRIORITY_NAMES else "medium",
    }


async def label_recall(llm: DeepSeek, bank: list[dict], case: dict) -> dict:
    payload = {
        "stored_memories": [{k: m[k] for k in ("id", "description", "content")} for m in bank],
        "conversation_so_far": case["context"],
        "latest_user_message": case["message"],
    }
    out, _ = await llm.json(LABEL_RECALL_SYSTEM, _dump(payload), model=PRO, effort="max")
    ids = {m["id"] for m in bank}
    relevant = sorted(i for i in out.get("relevant_ids") or [] if i in ids)
    by_id = {m["id"]: m for m in bank}
    return {
        "needs_recall": bool(out.get("needs_recall")),
        "relevant_ids": relevant,
        # 类型与 tag 的 gold 由相关 memory 推导，避免再引入一层主观标注
        "types": sorted({by_id[i]["memory_type"] for i in relevant}),
        "tags": sorted({t for i in relevant for t in by_id[i]["tags"]}),
    }


async def translate(llm: DeepSeek, items: list[dict], chunk: int = 12) -> list[dict]:
    chunks = [items[i : i + chunk] for i in range(0, len(items), chunk)]

    async def one(part: list[dict]) -> list[dict]:
        out, _ = await llm.json(TRANSLATE_SYSTEM, _dump({"items": part}), model=PRO, effort="low")
        result = out.get("items") or []
        if [x.get("id") for x in result] != [x["id"] for x in part]:
            raise RuntimeError("翻译结果的 id 顺序与原文不一致")
        return result

    return [x for part in await gather_limited([one(c) for c in chunks]) for x in part]


async def main(n_bank: int, n_write: int, n_recall: int) -> None:
    llm = DeepSeek()
    try:
        print("1/4 生成人设与记忆库 ...")
        bank_out, _ = await llm.json(
            "You create realistic benchmark data. Reply with JSON only.",
            BANK_PROMPT.format(n=n_bank),
            model=PRO,
            effort="high",
        )
        persona, bank = bank_out["persona"], bank_out["memories"]
        for m in bank:
            m["tags"] = [t for t in m["tags"] if t in TAGS]

        print("2/4 生成写入 / 召回用例 ...")
        (write_out, _), (recall_out, _) = await asyncio.gather(
            llm.json(
                "You create realistic benchmark data. Reply with JSON only.",
                WRITE_CASES_PROMPT.format(persona=persona, n=n_write, taxonomy=TAXONOMY_TEXT),
                model=PRO,
                effort="high",
            ),
            llm.json(
                "You create realistic benchmark data. Reply with JSON only.",
                RECALL_CASES_PROMPT.format(bank=_dump({"persona": persona, "memories": bank}), n=n_recall),
                model=PRO,
                effort="high",
            ),
        )
        write_cases, recall_cases = write_out["cases"], recall_out["cases"]

        print(f"3/4 Pro 盲标注 {len(write_cases)} + {len(recall_cases)} 条 ...")
        write_gold = await gather_limited([label_write(llm, c) for c in write_cases])
        recall_gold = await gather_limited([label_recall(llm, bank, c) for c in recall_cases])

        kept_write, kept_recall, dropped = [], [], {"write": [], "recall": []}
        for case, gold in zip(write_cases, write_gold):
            if gold["should_store"] != bool(case.pop("intent_store")):
                dropped["write"].append(case["id"])
                continue
            kept_write.append(case | {"gold": gold})
        for case, gold in zip(recall_cases, recall_gold):
            intent = case.pop("intent_relevant_ids") or []
            # 意图有相关 memory 但标注认为无需召回（或反之且标注也无相关项），视为歧义
            if bool(intent) and not gold["needs_recall"]:
                dropped["recall"].append(case["id"])
                continue
            kept_recall.append(case | {"gold": gold, "intent_relevant_ids": intent})

        print("4/4 翻译为中文 ...")
        text_fields = lambda c: {"id": c["id"], "context": c["context"], "message": c["message"]}  # noqa: E731
        bank_zh, write_zh, recall_zh = await asyncio.gather(
            translate(llm, bank),
            translate(llm, [text_fields(c) for c in kept_write]),
            translate(llm, [text_fields(c) for c in kept_recall]),
        )
        for zh, en in zip(bank_zh, bank):
            zh.update({k: en[k] for k in ("id", "memory_type", "tags", "priority")})

        def bilingual(cases_en, cases_zh):
            out = []
            for en, zh in zip(cases_en, cases_zh):
                out.append(en | {"lang": "en"})
                out.append(en | {"lang": "zh", "context": zh["context"], "message": zh["message"]})
            return out

        dataset = {
            "meta": {
                "generator": PRO,
                "persona": persona,
                "dropped_ambiguous": dropped,
                "note": "gold label 由 DeepSeek Pro 盲标注，中文版由 Pro 翻译并沿用英文 gold",
            },
            "memory_bank": {"en": bank, "zh": bank_zh},
            "write_cases": bilingual(kept_write, write_zh),
            "recall_cases": bilingual(kept_recall, recall_zh),
        }
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(_dump(dataset))
        print(
            f"完成：记忆库 {len(bank)} 条，写入用例 {len(kept_write)}×2，召回用例 {len(kept_recall)}×2；"
            f"剔除歧义 写入 {len(dropped['write'])} / 召回 {len(dropped['recall'])} → {OUT}"
        )
    finally:
        await llm.aclose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-bank", type=int, default=36)
    parser.add_argument("--n-write", type=int, default=48)
    parser.add_argument("--n-recall", type=int, default=48)
    args = parser.parse_args()
    asyncio.run(main(args.n_bank, args.n_write, args.n_recall))
