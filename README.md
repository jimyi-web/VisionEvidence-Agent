# VisionEvidence-Agent · V1.0

一个轻量、可复现、可审计的多模态证据检索研究基线，面向科研实习申请展示。短期目标：**2026-10-20 前完成 V1.0**；长期目标：**10,000+ 查询的大规模可复现评测**。

研究问题：将图像的可见对象和文字组织成结构化观察，能否比完整自然语言描述构造更好的证据检索查询？如何保留失败、来源、标注边界和计算成本，使比较可审核？

当前覆盖视觉观察 → 查询构造 → BM25 候选证据 → 可恢复批量评测。**最终事实核验决策、Agent 规划、Dense/Hybrid Retrieval 尚未实现。** 候选证据与模型描述均不是已确认事实，模型输出始终 verified=false。

## 流程与状态

```mermaid
flowchart LR
  A[JSONL 与本地图片] --> B[SQLite 断点与调度]
  B --> C[Qwen3-VL 或明确标注的 Mock]
  C --> D[严格结构化观察]
  D --> E[objects + visible_text 查询]
  E --> F[BM25 候选证据]
  F --> G[结果与运行记录]
  H[独立相关性 qrels] --> I[Recall / MRR]
  G --> I
```

- 真实 Qwen3-VL-4B：单 GPU BF16、本地权重、首次加载复用、耗时/显存记录；已有单图实测。
- 严格观察解析：拒绝截断/散文，保留失败原响应；不自动补造字段。
- BM25：保留文档 ID、来源、原文、分数；独立 qrels 驱动端到端 Recall@K/MRR@K。
- T05-A：流式 JSONL、稳定 ID、分批调度、SQLite 进度、断点、重试、成功缓存、输入/源码/模型哈希、Commit/环境追踪。
- CPU/Mock 可本地运行；新多图结构化链路尚待真实 GPU 验收。
- VERITE 适配器已用官方 CSV 小子集验证；真实图片受机构邮箱申请及条款限制。[数据选择](docs/t05_datasets.md)。
- Flickr8k 真实图片 proxy：已取得 test 前 20 张 JPEG，保留 6000/1000/1000 原划分与每图五条描述；两路共享 5,000 条 test 候选，CPU/Mock 端到端和恢复已通过，真实 GPU 由开发者手动验收。[协议](docs/flickr8k_protocol.md)、[下载/离线/GPU 命令](docs/flickr8k_runbook.md)、[验收报告](docs/flickr8k_report.md)。

## 已有实验与边界

三种规模始终分开：**证据库文档数、评测查询数、真实 VLM 图片推理调用数**。独立图片另计，重试不是新图片。

| 实验 | 证据文档 | 查询/样本 | 真实图片推理调用 | 结论 |
| --- | ---: | ---: | ---: | --- |
| T03-B-5 GPU smoke | 0 | 0 个检索查询，1 张输入图 | 6 次，独立图 1 张 | RTX 4090D 加载复用成功；6/6 达 token 上限，观察质量未通过验收 |
| T04-4 合成对照 | 12 | 10，9 条进入指标分母 | 0 | Structured 无一致优势 |
| T04-5 人工辅助审核 | 12 | 同一 10 个场景 | 0 | 120 个样本-文档关系，非 120 个查询或图片 |
| T05 CPU/Mock 分阶段 | 5 | 100 → 1,000 → 10,000 | 0 | 验证调度/缓存/恢复；复用 1 个合成白像素 fixture，不代表万张真实图片 |
| VERITE 元数据子集 | 0 | 12 条配对记录 | 0 | 8 图片路径、本地图片 0，非检索效果实验 |
| Flickr8k 真实 JPEG + Mock | 5,000 条候选描述 | 20 图片查询/路，共 40 查询槽位 | 0 | 20 张真实图已解码；Mock 不分析像素，尚无真实 VLM 质量指标 |

T04 固定合成集主结果（宏平均 n=9，非空 qrels 的查询失败记零）：

| 方法 | Recall@1 | Recall@3 | Recall@5 | MRR@5 |
| --- | ---: | ---: | ---: | ---: |
| 完整描述 Baseline | 0.555556 | 0.777778 | 0.777778 | 0.722222 |
| Structured | 0.611111 | 0.666667 | 0.666667 | 0.666667 |

场景、描述、结构字段和证据由 AI 合成；人工在 AI 辅助阅读后确认 120 个关系：相关 11、不相关 109，与冻结标签一致。这不是盲审或标注者间一致性。全部指标差异来自 S08 空提取和 S10 首位改善；ASCII 分词、同义/否定处理不足及闭集合成偏差仍存在。[协议](docs/t04_4_protocol.md)、[报告](docs/t04_4_report.md)、[失败审计](docs/t04_5_report.md)、[后续人工审核](docs/t04_5_human_archive.md)。

历史 GPU 首次加载约 4.78 秒；256-token 复用 wall 均值约 6.44 秒，512-token 约 12.97 秒；PyTorch allocated 峰值约 8,751 MiB，整卡采样最大 9,241 MiB。口径不同，不能保证新图片/并发容量；没有视觉或最终核验准确率。[单图 GPU 报告](docs/t03_b5_report.md)。

## Windows 本地运行

要求已有项目 `.venv`、Python 3.12；无需激活，不使用全局 Python。没有环境时先由开发者确认创建方案。运行基础设施无第三方依赖；pytest/Pillow 是开发依赖，Qwen 使用独立 Linux GPU 环境。

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m visionevidence.cli --version
.\.venv\Scripts\python.exe -m pytest
```

若 Git 报 dubious ownership，可仅对当前 PowerShell 进程配置项目例外后测试，不改全局设置：

```powershell
$env:GIT_CONFIG_COUNT='1'
$env:GIT_CONFIG_KEY_0='safe.directory'
$env:GIT_CONFIG_VALUE_0='D:/Research/VisionEvidence-Agent'
.\.venv\Scripts\python.exe -m pytest
```

## CPU/Mock：100 → 1000 → 10000

以下是合成调度验收，不分析真实图片。先准备完整输入，保持文件不变；新实验使用新目录，--resume 扩大前缀并跳过成功样本。

```powershell
.\.venv\Scripts\python.exe scripts/prepare_t05_mock.py --output outputs/t05-input-new.jsonl --count 10000
.\.venv\Scripts\python.exe -m visionevidence.batch --mode mock --samples outputs/t05-input-new.jsonl --corpus examples/evidence_demo.jsonl --output outputs/t05-run-new --limit 100 --batch-size 32
.\.venv\Scripts\python.exe -m visionevidence.batch --mode mock --samples outputs/t05-input-new.jsonl --corpus examples/evidence_demo.jsonl --output outputs/t05-run-new --limit 1000 --batch-size 128 --resume
.\.venv\Scripts\python.exe -m visionevidence.batch --mode mock --samples outputs/t05-input-new.jsonl --corpus examples/evidence_demo.jsonl --output outputs/t05-run-new --limit 10000 --batch-size 512 --resume
```

输出含固定 manifest、输入快照、state.sqlite3、每阶段独立 summary/results。错误和原响应查 attempts；--max-attempts 默认 2，是跨 session 累计预算，修复缺图后可显式提高，成功结果不覆盖。退出码 0=所选样本完成，1=有未完成样本，2=配置/输入/目录错误。

prompt、源码、证据库、配置或模型改变要新目录；成功图片哈希变化拒绝缓存。batch-size 是调度大小，GPU 逐图推理。[输入/qrels/恢复协议与 GPU 小批命令](docs/t05_pipeline.md)、[本轮验收](docs/t05_report.md)。

## 真实数据与 GPU 下一步

VERITE 图片申请 PENDING 时，当前优先用 Flickr8k 做 **image-to-caption retrieval proxy**。20 张真实图片已获取，候选库固定为 test 的 5,000 条人工描述；正确描述只作为检索目标，不进入 VLM prompt。配对 qrels 显式标记 caption_pair_proxy，不用于外部事实核验。当前完整测试 **371 passed**。

```powershell
.\.venv\Scripts\python.exe -m visionevidence.caption_experiment --dataset data/processed/flickr8k-proxy-v1/test --output outputs/flickr8k-mock-new --mode mock --limit 20
```

运行前按 [Flickr8k runbook](docs/flickr8k_runbook.md) 下载/离线适配。后续 20→100→1000 仅扩大同一个 test 查询前缀，候选库不变；8,000 图按 train/dev/test 分开运行。指标是同图描述关联命中率/召回率，不能称为谣言识别准确率；相似图片的描述可能也是合理答案，五条同图描述也不是独立图片样本。

按 [VERITE 元数据步骤](docs/t05_datasets.md) 接入，再经 [官方 HF](https://huggingface.co/datasets/stefpapad/VERITE) 申请图片；保留图片哈希与条款。当前官方 CSV 和论文总数差 1 条，适配器按实际字节原样保留。

图文配对标签**不是检索 qrels**；模型生成不是人工核验事实。没有独立相关性标注时指标 null。真实语料需独立准备，不能把正确 caption 塞入证据库制造高分。

开发者在已有批准的 GPU 环境中手动运行 [12 条命令](docs/t05_pipeline.md)，先检查 JSON 成功率和成本再扩大规模。本轮不启动 AutoDL、不下载模型权重。万级 GPU、真实检索标注和最终 verifier 留待后续。

## 历史复现与导航

```powershell
.\.venv\Scripts\python.exe -m visionevidence.evaluation.observation --dataset data/evaluation/t04_v1 --config configs/t04_evaluation.toml --output outputs/t04-replay-new
.\.venv\Scripts\python.exe scripts/validate_t04_human_review.py
.\.venv\Scripts\python.exe -m visionevidence.cli retrieve --query "flood road closures" --corpus examples/evidence_demo.jsonl --top-k 3 --json
```

| 路径 | 用途 |
| --- | --- |
| src/visionevidence/batch.py | 批量 CLI、SQLite、来源、评测 |
| src/visionevidence/datasets/verite.py | 官方 CSV 适配与标签隔离 |
| src/visionevidence/datasets/flickr8k.py | 固定版本小图下载、离线导入、原始划分和五描述 |
| src/visionevidence/caption_experiment.py | Baseline/Structured 同候选对照与 proxy 指标 |
| src/visionevidence/description.py | 原始散文 VLM 描述直接检索基线 |
| src/visionevidence/vlm/ | Mock/Qwen、本地加载与契约 |
| src/visionevidence/observation.py | 严格解析、查询来源、观察到检索 |
| src/visionevidence/retrieval/ | BM25 与语料校验 |
| configs/ | 模型/检索配置、观察 Schema |
| experiments/ | 小型审计归档，不放图片/大数据 |
| docs/qwen3_vl_deployment.md | 人工部署与 GPU 环境要求 |

实验输出独占新目录，不覆盖旧结果。Commit 经开发者授权，push 由开发者手动执行。
