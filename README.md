# devision-mem-bench

一个简易 memory 系统，用 System One 决策模型做“要不要存 / 要不要召回 / 召回哪些”的判断，用 DeepSeek Flash 生成记忆内容（Python 包名 `jevmem`），并附带 benchmark。判断模型可选：
- [Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev)（TypeSafe）
- [Clef / Clef-flash](https://blog.cloudflare.com/clef-decision-models/)（Cloudflare Workers AI，与 Jev API 兼容）
- 作为对照的纯 LLM 方案（DeepSeek Flash）

## 结论

> 测试时间 2026-10-02，对比 `jev-1.13.0`、`@cf/cloudflare/clef`、`@cf/cloudflare/clef-flash` 与 `deepseek-flash`（DeepSeek-V4.1-Flash）。完整数据见 [bench/RESULTS.md](bench/RESULTS.md)。

**一句话：在这个 memory 场景里，Jev 是判断门的首选，最快、最便宜，召回质量最好或持平。它的短板是需要理解上下文的写入判断，这部分需要 LLM 兜底。**

| | Jev | Clef | Clef-flash | Flash 关闭 thinking | Flash low |
|---|---|---|---|---|---|
| 写入判断 p50 / p95 | **0.4s / 0.5s** | 1.0s / 1.6s | 0.8s / 1.2s | 1.1s / 1.4s | 2.1–2.4s / 3.9–5.5s |
| 召回 p50 / p95（36 条记忆） | **0.8s / 0.9s** | 2.5s / 3.0s | 1.6s / 3.3s | 2.1s / 2.6s | 5.3s / 18.8s |
| 召回 p50 / p95（286 条记忆） | **0.9s / 1.2s** | 3.4s / 4.6s | 2.2s / 4.6s | 2.6s / 3.4s | 11.0s / 40.2s |
| 写入判断成本 $/1k 次 | **$0.044** | $0.41 | $0.15 | $0.23 | $0.50–0.65 |
| 边界写入：存储判断 Acc | 0.83 | 0.88 | 0.87 | 0.88 | **0.96** |
| 召回记忆 F1：小库 / 边界集 / 大库 | 0.80 / 0.75 / **0.59** | 0.76 / 0.72 / 0.58 | 0.74 / 0.69 / 0.57 | 0.65 / 0.62 / 0.45 | **0.82 / 0.77** / 0.53 |

System One 系的召回取 fan-out 版本；Clef-flash 的召回门阈值为 0.2，原因见第 5 条。

1. **速度和成本**：Jev 比最快的 LLM 方案（Flash 关闭 thinking）快约 2.5–3 倍、便宜 5 倍；比开 thinking 的 Flash 快 6–12 倍，p95 快 20–30 倍。Jev 的延迟约 0.4s，几乎都是固定开销：单次请求放 1 / 15 / 64 道题都约 425ms，放 300 道题也只要 875ms。
2. **没有“又快又准”的 LLM 方案**：Flash 关闭 thinking 后速度接近，但召回准确率在所有数据集上都是最差的；开 thinking 质量够好，但慢一个数量级，尾延迟不可控。
3. **Jev 的短板是依赖上下文的写入判断**：边界样本上 Jev 精确率 0.98，但覆盖率只有 0.79–0.80，偏向漏存，调阈值也救不回来。漏判集中在两类：要结合前文才能理解的更正（“Scratch that, the trip moved to December”），以及短期但重要的状态（“明天下午 3 点看牙”）。
4. **召回是 Jev 的主场**：Jev 输出连续的相关性概率，可以排序；LLM 只能给出“相关 / 不相关”。在 286 条的大库上返回 Top-5 时，Jev 的 F1 最高；小库上与 Flash low 持平，但快 7 倍。
5. **Clef / Clef-flash（Cloudflare）**：
   - 质量接近 Jev，召回 F1 略低 0.01–0.06；在边界写入上比 Jev 更少漏存（Clef 覆盖率 0.91 vs Jev 0.79），类型判断则更弱。
   - 从本机访问时比 Jev 慢 2–4 倍，延迟随题数近似线性增长（1 / 15 / 64 道题：Clef 709 / 879 / 1791ms），与 Cloudflare 宣称的 209ms / 39ms 差距很大。单次最多 64 道题，大库召回要分批。成本是 Jev 的 3.5–9 倍。
   - Clef-flash 的概率整体偏低。默认 0.5 阈值下，召回门准确率只有 0.47–0.76；按原数据集把阈值调到 0.2 后，在三个数据集上回到 0.89–0.94。**换 System One 模型时，阈值必须重新校准。**
   - Clef 的优势在 benchmark 之外：开源权重（Apache 2.0）、支持图片输入、可在 Cloudflare 边缘就近部署。服务本身跑在 Workers 上时值得重测。
6. **fan-out**：跳过类型和 tag 过滤，对全部记忆判断相关性。小库上覆盖率从约 0.90 升到 0.97，F1 +0.01–0.08，适合作为默认；大库上基本没有收益。
7. **中文**：召回质量中英文基本没有差别；写入时的类型准确率，中文比英文低约 0.07。

**推荐架构**
- **召回**：交给 Jev。小库用 fan-out，大库先按类型 / tag 过滤。
- **写入**：Jev 做实时第一道门，概率达到 0.5 就写入。Jev 判为不存、但概率处在中间区间的消息，交给异步运行的 Flash 生成环节复核，同时修正类型和优先级。用户只需等约 0.4s。

**局限**：只有一个人设；gold label 由 DeepSeek Pro 标注，可能偏向 DeepSeek；大库的 gold 偏宽；所有 API 都通过本机代理访问，延迟的绝对值不能直接外推，尤其是依赖边缘就近部署的 Cloudflare。

## 设计

Jev 不生成文本，只回答三类有类型的问题（Noul 是非概率 / Choice 单选 / Score 有序打分），且同一次请求里的所有问题并行求值。因此：

- **分类体系是封闭集合**（`src/jevmem/taxonomy.py`）：5 种 memory 类型、11 个 tag、3 档优先级。Jev 与 DeepSeek 判断器共用同一套题目文案，保证对比公平。
- **写入路径** `MemorySystem.observe`：
  1. Jev 一次调用：`store`（Noul）+ `type`（Choice）+ `priority`（Score）+ 每个 tag 一道 Noul（多标签，阈值 0.5）
  2. 判定要存 → DeepSeek Flash 生成 `description`（检索用的一句话）与 `content`（细节完整的记忆）→ SQLite
- **召回路径** `MemorySystem.recall`：
  1. Jev 一次调用：`recall`（Noul）+ 每种类型、每个 tag 各一道 Noul
  2. 代码按“类型命中或任一 tag 命中”过滤候选
  3. Jev 第二次调用：每条候选一道相关性 Noul（绝对判断，可以全部判为不相关），取 ≥0.5 的 Top-K
- **对照组**：`DeepSeekDecider` 用 Flash 以 JSON 输出完成同样的三个判断，接口与 `JevDecider` 一致（`schema.Decider`）。

```
src/jevmem/
  taxonomy.py   类型 / tag / 优先级定义与题目文案
  schema.py     Turn / Memory / WriteDecision / RecallDecision / Decider 协议
  jev.py        JevDecider（System One 判断器，Jev / Clef 共用）
  clef.py       Cloudflare Workers AI 客户端（Clef / Clef-flash）
  deepseek.py   DeepSeek 客户端、DeepSeekDecider、MemoryGenerator
  store.py      SQLite 存储与候选过滤
  memory.py     MemorySystem（observe / recall）
  cli.py        命令行
bench/
  gen_dataset.py  DeepSeek Pro 生成双语数据集与 gold label
  run_bench.py    运行各配置并记录原始结果
  report.py       计算指标、生成 Markdown 报告
```

## 使用

`.env`：

```
jev_api_key=...
deepseek_api_key=...
CLOUDFLARE_ACCOUNT_ID=...      # 使用 clef / clef-flash 时需要
# CLOUDFLARE_API_TOKEN=...     # 可选；不填则调用 `wrangler auth token` 借用 wrangler 的登录态
```

```bash
uv sync
uv run jevmem observe "我对花生严重过敏，以后推荐菜谱时一定要避开。"
uv run jevmem observe "I prefer concise answers with code examples." --decider flash-low
uv run jevmem recall "推荐一道今晚可以做的菜"
uv run jevmem list
uv run jevmem delete <id>
```

`--decider` 可选 `jev`（默认）/ `clef` / `clef-flash` / `flash-nothink` / `flash-low` / `flash-high`；System One 系可加 `-fanout`（召回时跳过类型/tag 过滤）和 `@<阈值>`（召回门阈值，如 `clef-flash-fanout@0.2`）；`--context` 传入此前的对话；`--db` 或 `JEVMEM_DB` 指定数据库（默认 `data/memory.db`）。

## Benchmark

```bash
uv run python -m bench.gen_dataset          # 生成 bench/data/dataset.json（已生成可跳过）
uv run python -m bench.gen_extra hard       # 边界样本集 bench/data/dataset_hard.json（双盲标注）
uv run python -m bench.gen_extra large      # 约 300 条的大记忆库 bench/data/dataset_large.json
uv run python -m bench.run_bench            # 跑默认配置，结果在 bench/results/<时间戳>-<数据集>/
uv run python -m bench.run_bench --no-generate --dataset bench/data/dataset_hard.json \
    --configs jev jev-fanout flash-nothink flash-low
uv run python -m bench.run_bench --limit 3 --configs jev   # 快速试跑
uv run python -m bench.report bench/results/<时间戳>      # 重新出报告
```

数据集：DeepSeek Pro 生成一个人设、36 条记忆库、写入与召回用例，再**盲标注**得到 gold label（与出题意图在二元判断上冲突的样本作为歧义剔除），最后翻译为中文并沿用英文 gold，使中英文对比只差语言。

配置：`jev`（Jev 判断 + Flash(low) 生成，即混合系统）、`clef`、`clef-flash`（及各自的 `-fanout` / `@阈值` 变体）、`flash-nothink`（关闭 thinking）、`flash-low`、`flash-high`。`--kinds recall` 只跑召回，`--tag` 区分并行启动的多个进程。所有配置的生成器相同，端到端差异只来自判断器。同一进程内各配置串行、配置内并发 4；不同服务的配置可以用 `--tag` 分成多个进程并行跑（不同服务之间不会互相影响延迟）。

注意事项：

- gold label 来自 DeepSeek Pro，对 DeepSeek 系模型可能有同源偏好。
- Jev 官方说明英文效果最好，CJK 支持但较弱；双语数据集正是为了量化这一点。
- Jev 服务部署在美国西海岸，延迟数据受本机网络位置影响。

结果与结论见 [bench/RESULTS.md](bench/RESULTS.md)。
