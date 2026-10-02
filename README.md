# devision-mem-bench

用 [Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev)（TypeSafe 的 System One 模型）做判断、用 DeepSeek Flash 生成内容的简易 memory 系统（Python 包名 `jevmem`），并附带与纯 LLM 方案的 benchmark。

## 结论

> 测试时间 2026-10-02，对比 `jev-1.13.0` 与 `deepseek-flash`（DeepSeek-V4.1-Flash）。完整数据见 [bench/RESULTS.md](bench/RESULTS.md)。

**一句话：Jev 适合做"快、稳、带概率的排序和过滤"，不适合做"需要理解上下文的判断"。**

| | Jev | Flash 关闭 thinking | Flash low |
|---|---|---|---|
| 写入判断 p50 / p95 | **0.4s / 0.5s** | 1.1s / 1.4s | 2.1–2.4s / 3.9–5.5s |
| 召回 p50 / p95（36 条记忆） | **0.8s / 0.9s** | 2.1s / 2.6s | 5.3s / 18.8s |
| 召回 p50 / p95（286 条记忆） | **0.9s / 1.3s** | 2.6s / 3.4s | 11.0s / 40.2s |
| 写入判断成本 $/1k 次 | **$0.044** | $0.23 | $0.50–0.65 |
| 边界写入：存储判断 Acc | 0.84 | 0.88 | **0.96** |
| 召回记忆 F1：小库 / 边界集 / 大库 | 0.80 / 0.75 / **0.60** | 0.65 / 0.62 / 0.45 | **0.82 / 0.77** / 0.53 |

Jev 的召回 F1 取 `jev-fanout`（小库、边界集）和 `jev`（大库）中各自更好的一种。

1. **速度和成本**：比最快的 LLM 方案（Flash 关闭 thinking）快约 2.5–3 倍、便宜 5 倍；比开 thinking 的 Flash 快 6–12 倍，p95 快 20–30 倍。Jev 的延迟约 0.4s，几乎都是固定开销：单次请求放 64 / 150 / 300 道题，分别耗时 629 / 710 / 875ms。
2. **没有"又快又准"的 LLM 方案**：Flash 关闭 thinking 后速度接近，但召回准确率在所有数据集上都是最差的；开 thinking 质量够好，但慢一个数量级，尾延迟不可控。
3. **Jev 的短板是依赖上下文的写入判断**：边界样本上，Jev 精确率 0.98，但覆盖率只有 0.80，偏向漏存，调阈值也救不回来。漏判集中在两类：要结合前文才能理解的更正（"Scratch that, the trip moved to December"），以及短期但重要的状态（"明天下午 3 点看牙"）。类型和优先级的准确度也落后于 Flash low。
4. **召回是 Jev 的主场**：记忆库越大优势越明显。在 286 条的库上返回 Top-5 时，Jev 的 F1 最高，因为它输出连续的相关性概率，可以排序；LLM 只能给出"相关 / 不相关"。小库上两者持平。
5. **fan-out**：跳过类型和 tag 过滤，对全部记忆判断相关性。小库上覆盖率从约 0.90 升到 0.97，F1 +0.01–0.04，只多 20–30ms，适合作为默认；大库上没有收益。
6. **中文**：召回质量中英文基本没有差别；写入时的类型准确率，中文比英文低约 0.07。

**推荐架构**
- **召回**：全部交给 Jev。小库用 fan-out，大库先按类型 / tag 过滤。
- **写入**：Jev 做实时第一道门，概率达到 0.5 就写入。Jev 判为不存、但概率处在中间区间的消息，交给异步运行的 Flash 生成环节复核，同时修正类型和优先级。用户只需等约 0.4s。

**局限**：只有一个人设；gold label 由 DeepSeek Pro 标注，可能偏向 DeepSeek；大库的 gold 偏宽；两家 API 都通过本机代理访问，延迟的绝对值不能直接外推。

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
  jev.py        JevDecider
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
```

```bash
uv sync
uv run jevmem observe "我对花生严重过敏，以后推荐菜谱时一定要避开。"
uv run jevmem observe "I prefer concise answers with code examples." --decider flash-low
uv run jevmem recall "推荐一道今晚可以做的菜"
uv run jevmem list
uv run jevmem delete <id>
```

`--decider` 可选 `jev`（默认）/ `jev-fanout`（召回时跳过类型/tag 过滤）/ `flash-nothink` / `flash-low` / `flash-high`；`--context` 传入此前的对话；`--db` 或 `JEVMEM_DB` 指定数据库（默认 `data/memory.db`）。

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

配置：`jev`（Jev 判断 + Flash(low) 生成，即混合系统）、`jev-fanout`、`flash-nothink`（关闭 thinking）、`flash-low`、`flash-high`。所有配置的生成器相同，端到端差异只来自判断器。各配置串行执行，配置内并发 4。

注意事项：

- gold label 来自 DeepSeek Pro，对 DeepSeek 系模型可能有同源偏好。
- Jev 官方说明英文效果最好，CJK 支持但较弱；双语数据集正是为了量化这一点。
- Jev 服务部署在美国西海岸，延迟数据受本机网络位置影响。

结果与结论见 [bench/RESULTS.md](bench/RESULTS.md)。
