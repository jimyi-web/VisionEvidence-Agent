# T05-A 批量流水线协议

短期：2026-10-20 前交付 V1.0；长期：10,000+ 查询可复现评测。V1.0 覆盖观察、检索、记录、评测基础设施，最终事实核验决策仍未实现。

## 缺口与本次最小实现

已有 Qwen 单次加载复用、严格观察解析器、BM25、固定合成检索对照和人工辅助审核。旧 CLI 面向单请求，固定评测器面向已准备文本，缺少多图输入、持久化状态、缓存、重试和扩容。自然语言 visual_observation_v1 无法直接进入严格 JSON 解析器。

新增 `python -m visionevidence.batch`：JSONL → SQLite 样本表 → 分批调度 → 一个复用的 Qwen/StructuredMock → 现有 observe_and_retrieve → 严格解析 → BM25 → 逐样本结果与 attempt journal。没有新增运行依赖。

batch-size 控制调度内存，不是 GPU tensor batch；GPU 逐图推理。SQLite 每次状态变化提交，OS 文件锁禁止同目录多个 worker。证据库仍使用原内存 BM25，大库需单独测内存；本轮 10,000 查询只用了 5 文档，未验证百万文档容量。

## 规模字段

| 字段 | 定义 |
| --- | --- |
| evidence_documents | 实际索引的文档 ID 数 |
| evaluation_queries | 当前前缀选中的样本数，含失败；successful_queries 另计成功观察检索 |
| real_vlm_image_calls | qwen3_vl 模式后端调用数，重试重复计，Mock 恒为 0 |
| real_vlm_responses | 已返回的真实后端响应数；JSON 不合格也可能已有推理 |
| unique_real_vlm_images | 按 SHA256 去重的真实响应图片数，不把重复请求算新图 |

调用意图在进入后端前提交。强制终止落在提交和调用之间时，该次实际推理是否完成未知，标记 interrupted，不能当成已确认成功推理。来源矛盾的响应原文保留但不接入成功结果。

## 输入与标签

每行必需 sample_id、image_path、claim、dataset、dataset_version、split、source，均为非空字符串；可选 metadata 对象、pair_label。相对图片路径基于样本文件目录。重复 ID、未知字段、非法 JSON 在推理前拒绝；缺图/格式错误进入逐样本失败记录。

JSONL 流式入 SQLite，调度只读 batch-size 行。VLM 只接受图片与固定观察 prompt，不接受 claim、pair_label、qrels、source article；claim 仅保留给后续研究。

可选 --qrels 每行：

```json
{"sample_id":"sample-1","relevant_ids":["document-1"],"annotation_source":"documented review or synthetic fixture","kind":"human_relevance"}
```

kind 只接受 human_relevance/synthetic_relevance；声明 kind 也不能证明可信，仍需核查 annotation_source。未知文档/样本、重复标注拒绝。非空 qrels 的失败样本按零进入端到端 Recall@K/MRR@K；空标注与无标注分别计数并排除，没有可评测标签时指标 null。配对标签不用作 qrels；未实现分类 Accuracy/Macro-F1。

## 自然语言与严格 JSON

批量入口覆盖旧配置的 prompt，固定现有 structured_visual_v1，记录全文与 SHA256。[Schema](../configs/observation.schema.json) 对应 parse_observation：只允许 objects/visible_text/uncertainties，数组最多 8 项，字符串 1–200 字符且非空白；解析器还限制响应总长 8,000，拒绝重复键、NaN、截断。

保留 description 字符串接口。prompt 请求 JSON **不等于启用了约束解码**。散文、Markdown、缺字段或达到 token 上限均显式失败，保留原响应，不猜字段、不修复、不退回 claim。默认累计最多 2 次尝试；确定性错误重试可能仍失败。更改 token 上限/prompt 须新 run，不污染旧缓存。真实结构化成功率尚待 GPU 验收。

StructuredMock 固定输出 road sign 和 [MOCK]，不看像素/文件名/caption；is_mock=true、inference_performed=false、verified=false。真实生成也保持 verified=false，不冒充人工事实。

## 恢复、缓存与记录

新运行独占创建目录，已有目录须 --resume。固定完整输入文件，limit 取前 N 行（超出时取全部），100→1000→10000 只做未成功且有预算的样本。max-attempts 是跨 session 累计预算，非每次重置。修复缺图后可显式从 1 增至 2；成功结果不替换。

身份包含完整 samples/corpus/qrels SHA256、规范化配置、prompt、全部 Python 源码哈希、模型目录每文件 SHA256、BM25 参数。成功图片变化拒绝缓存。配置/模型/代码变化要新目录；limit/batch-size/重试预算可变。Git Commit/dirty 每 session 记录，未提交代码也可用源码哈希核对。模型逐文件哈希有磁盘开销，不联网下载权重。

- manifest.json：身份、配置、Git、环境/依赖、seed=0、设备；input-*.jsonl 为原字节快照。
- state.sqlite3：samples 状态/缓存、完整 attempts（时间、图片 hash、错误、原响应）、独立 qrels、sessions；synchronous=FULL。
- summary-NNNN.json/results-NNNN.jsonl：每 session 独占导出，不覆盖旧文件。results 是累计成功前缀，完整失败详情查 attempts。
- runtime-NNNN.json：真实模式的 CUDA/GPU 名称/总显存；响应保留现有 Qwen 时间和 torch allocated 峰值口径。Mock 不采 GPU。

按需创建一个模型并复用；greedy+seed=0 不保证跨 CUDA 逐位相同。token 上限不是硬超时；驱动卡死由外层 timeout/人工中断。中断后 OS 释放锁，续跑保留 interrupted。GPU 返回和事务提交之间被杀可能重做最后一次推理，语义为 at-least-once。

## 真实 GPU 小批验收（本轮未执行）

仅在开发者已批准的 Linux 实例、已有独立 GPU 环境/模型目录上执行。先合法取得图片，按适配器路径摆放；另准备有来源的 corpus，不能用合成演示证据库声称真实检索效果。

```bash
export VEA_ENV=/absolute/path/to/existing/gpu-venv
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
# 人工核对 configs/t05_qwen3_vl.toml 的已有 model_path，并安装当前代码。
timeout --signal=INT --kill-after=30s 20m "$VEA_ENV/bin/python" -m visionevidence.batch \
  --mode qwen3_vl --config configs/t05_qwen3_vl.toml \
  --samples /absolute/path/to/verite-subset/samples.jsonl \
  --corpus /absolute/path/to/independent/corpus.jsonl \
  --output outputs/t05-real-small-01 --limit 12 --batch-size 4 --max-attempts 1
```

核对真实来源、原始 JSON 与错误率、模型首次加载复用、时延/显存、三种规模与中断记录。没有 qrels 时只报告操作性成功率/耗时，不报检索质量。成功后再考虑独立新 run 的 100/1000/10000；不在小 benchmark 上反复挑有利样本后声称泛化。脚本不启动 AutoDL。
