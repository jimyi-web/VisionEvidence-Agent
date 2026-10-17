# T04-4 固定样本与 qrels 检索评测报告

2026-10-17，Windows 本地 CPU。正式运行代码提交：`9917055831a5d861c48bbe2718d20748d363ab5b`，运行时 Git 工作区干净。
固定集 `t04-synthetic-observation-retrieval / 1.0.0 / synthetic_test`；10 个样本、12 篇证据，9 个样本有 qrels，S09 缺证排除。
预先设计审查见 [协议](t04_4_protocol.md)。本轮没有调整既有查询构造策略、BM25 实现或 Qwen3-VL 后端，没有 AutoDL、GPU、模型下载或网络调用。

## 数据来源、标注与有效性边界

所有场景文字规格、原始描述、结构化字段、虚构证据和 qrels 均由 Codex 合成；不是从新闻、论文或真实图像采集。来源使用 synthetic:// 标识，不冒充可验证外部网页。
`visual_observation` 是文字场景规格，原始描述是同一场景的完整文字，结构字段为其可见对象/文字片段；其中保留预先构造的遗漏与失败。

| 类型 | 本轮状态 |
| --- | --- |
| 人工标注 | 未完成；human_reviewed=false，qrels 是待人工复核的合成判定 |
| 合成数据 | 10 个文字场景、12 篇虚构证据及全部结构化字段/qrels |
| Mock | 离线 fixture 直接评测，不调用 MockVLM 或 StructuredFixtureVLM；不是模型生成 |
| 真实 VLM 输出 | 0；没有真实图像，也没有像素分析 |

每条 qrels 均有判定理由与 evidence_status。全闭集其余文档按不相关处理；此规则只是合成任务标注，不证明证据的真实性。
构造者同时知道场景与证据，词汇重合和选择偏差不能被代码隔离消除。代码隔离仅保证 qrels 不参与查询/排名；字段提取约束测试保证没有新增原始描述之外的关键词。
S01 与 S08 共用场景主题，本集并非 10 个独立随机抽样；不做显著性检验、泛化结论或真实多模态能力宣称。

## 实测结果

同一个 BM25Index，k1=1.5、b=0.75、ASCII tokenizer、top_k=5；Baseline 使用完整原始描述，Structured 使用已有 objects+visible_text 策略。
两组都检索全部 10 个样本，以下端到端宏平均分母均为 9；有 qrels 的查询失败记零。

| 方法 | Recall@1 | Recall@3 | Recall@5 | MRR@5 |
| --- | ---: | ---: | ---: | ---: |
| Baseline | 0.555556 | 0.777778 | 0.777778 | 0.722222 |
| Structured | 0.611111 | 0.666667 | 0.666667 | 0.666667 |

这里只能说明该固定合成集的检索行为：Structured 的 Recall@1 较高，但 Recall@3/5 和 MRR@5 较低，不存在一致优势。没有测量视觉准确率或最终核验准确率，不能宣称准确率提升。

成功构造查询且有 qrels 的子集：Baseline n=8，Recall@1/3/5=0.625/0.875/0.875、MRR@5=0.8125；Structured n=7，分别为 0.785714/0.857143/0.857143、0.857143。两组子集不同，不能用这些数替代端到端比较。

## 逐例排名与失败分析

`—` 表示无候选；列出实际 Top-5，不补零分候选。完整查询、来源字段、证据全文、分数、耗时、逐例四项指标和诊断见 [results.jsonl](../experiments/t04_v1/run-20261017-01/results.jsonl)。

| 样本 | qrels | Baseline | Structured | 分析 |
| --- | --- | --- | --- | --- |
| S01 | D01,D02 | D02,D01,D12,D03,D04 | D01,D02,D12 | 两篇相关文档，Recall@1 都为 0.5 |
| S02 | D03 | D03,D04,D06,D09,D10 | D03,D04,D06 | 关闭桥梁；两组首位命中 |
| S03 | D05 | D05,D06,D03,D04,D02 | D05,D06,D04 | 周日开放图书馆；两组首位命中 |
| S04 | D07 | D07,D08,D03,D04,D02 | D07,D08 | Structured 遗漏 Platform 7；D07/D08 同分，ID 排序使 D07 首位。高分掩盖了语义遗漏，不能解释成正确辨认站台 |
| S05 | D09 | D09,D10,D03,D01,D07 | D09,D10 | 饮水喷泉；两组首位命中 |
| S06 | D11 | — | — | 检索无命中：automobile 与 car 无词面交集；两组指标均零 |
| S07 | D05 | — | — | 查询构造错误：中文没有 ASCII 词项；有 qrels，均计零 |
| S08 | D01,D02 | D02,D01,D12,D03,D04 | — | Structured 空提取导致查询失败计零；不回退到原始描述 |
| S09 | 空 | D10,D03,D07,D08,D09 | — | 证据不足：语料没有天文记录。Baseline 通用词引来无关候选；Structured 同时无命中。指标为 null 并排除，不把缺证当作检索零分 |
| S10 | D04 | D03,D04,D05,D02,D10 | D04,D03,D05 | Baseline 原描述通用词影响首位排序；Structured 在这一例首位命中，不能外推 |

诊断计数允许重叠，分母是全部 10 个样本：Baseline 无候选 1、查询错误 1、证据不足 1；Structured 分别为 2、2、1。
有候选但 Top-5 未找到任何 qrels 的样本本次为 0；运行器另支持此诊断。S04 是需要人工语义审查的查询遗漏，不能仅靠命中指标自动判断。

## 记录与复现

正式运行：

```powershell
.\.venv\Scripts\python.exe -m visionevidence.evaluation.observation --dataset data/evaluation/t04_v1 --config configs/t04_evaluation.toml --output outputs/t04-4-v1-20261017-01
```

原始输出已逐字节归档到 `experiments/t04_v1/run-20261017-01/`，包含输入快照、配置、逐例 JSONL、汇总 JSON 和 manifest。
manifest 保存实际消费字节哈希、源码哈希、Git Commit/dirty、Python/依赖版本、CPU 平台、运行时间、固定 seed=42。流程无随机采样，模型与峰值内存为 null（未使用模型，未测量内存）。耗时含查询构造、检索、索引和总时间；不包含 VLM 推理。
复跑请更换新输出目录；可以把归档目录作为 --dataset，并使用归档 config.toml。排名、分数和指标应一致，耗时与环境字段可不同。

验证命令（项目 Python 3.12.7）：

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_observation_evaluation.py --basetemp outputs/pytest-t044-focused-01
.\.venv\Scripts\python.exe -m pytest --basetemp outputs/pytest-t044-full-01
git diff --check
```

新增测试 11 passed（1.38s）；完整测试 319 passed（22.30s），包含原 T01–T04 的 308 项测试。diff 检查通过。
验证覆盖手算指标、标签更改不影响排名、查询字段来源、失败计零、空 qrels 排除、异常 qrels 拒绝、CLI 实际运行、快照哈希、复现一致及禁止覆盖。
正式运行后未按分数调整样本、qrels、查询策略或参数。报告与归档另作提交，避免把运行时代码提交与包含结果的后续提交混为一谈。

## 下一步

先由人逐例复核该小集的 qrels 和失败解释。后续另建有来源和授权记录的真实图片/dev 集，独立人工标注并冻结测试集；真实 VLM 输出必须单独采集，保留失败，不以本合成集指导调参。GPU 验收另行批准。
