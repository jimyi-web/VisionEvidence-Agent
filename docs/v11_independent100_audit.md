# V1.1 独立 test 100 图正式离线审计

审计日期：2026-10-27（Asia/Shanghai）。冻结 Commit：`8428c626a7ba55bb723a0e1506889a04595c1307`。

**结论：本次独立测试未提供 format_details 优于 Baseline 的证据。预先指定的主要指标 Hit@10 从 0.78 降至 0.72，配对差 −0.06，95% CI [−0.17, 0.05]。format_details 有六次附加格式合约失败，全部计零并保留。** Recall@10 小幅增加，MRR@10 基本相同；五项指标差值区间均跨零，不能据此宣称提升，也不能宣称等效或确定下降。

任务始终为 **image-to-caption retrieval proxy**，不是真实事实核验、谣言检测或真实性分类。审计使用真实保存的原始文件；新增 VLM/GPU 调用为 **0**，不修复后替换失败、不重新选策略、不修改冻结文件或测试协议。

## 原件、完整性与来源边界

输入原件：

- `outputs/v11-independent100-audit-transfer.gFF0L3/`：SCP 同步包及 SHA256SUMS、Git 信息。
- `outputs/v11-independent100-source-01/`：安全解包的普通文件；未展开指向 AutoDL 的绝对软链接。
- `outputs/flickr8k-v11-test-100.tar.gz`：本地已有的 100 图及元数据备份，用于补足结果包中未随软链接打包的 raw 数据。

| 原件 | SHA256 |
| --- | --- |
| raw-artifacts.tar.gz | `04606e894bc82d0f538469275d13d638171c2137c2bdb6bf9546ecb19e9a740f` |
| flickr8k-v11-test-100.tar.gz | `9870fc2f64a3dd1b2aa5fcea35b5c405d9a94a14e62014568b7b046748ca5f70` |

结果包 SHA256SUMS 中全部五项匹配，47 个解包普通文件逐字节匹配包内成员。所有实验原件在审计前后哈希不变。Git HEAD 收据、两路 manifest 及四条 session 的 Git 字段均为冻结 Commit、dirty=false；记录的源码哈希匹配冻结 Commit。源码/配置采用冻结的 CRLF→LF 规范化规则，输入/图片使用精确字节 SHA256。正式冻结目录也与该 Commit 对照。

已核对正式冻结 Manifest、Prompt、策略、生成参数、检索参数、输入快照、qrels、模型文件哈希记录及环境。两路模型哈希记录均与正式冻结的 16 个模型文件哈希一致；**未传输或在本地重新计算模型权重哈希**。

备份存在两项来源缺口，不隐去：

1. 结果包的 `data/raw/flickr8k-v11-test-01` 仅为指向 `/root/autodl-tmp/flickr8k-v11-test-01` 的软链接。使用独立图片包中的 100 张 JPEG，逐张对照两路 attempts/samples 的哈希及图片 inventory，全部一致；原始镜像元数据也匹配冻结哈希。不存在为审计重新下载、替换或重选图片。
2. `included-logs.txt` 为空，包内没有终端日志。SQLite sessions、原始响应及编号快照完整，但无法独立证明准确 shell 调用、wrapper preflight 输出或不存在目录外实验。

“200 次真实调用”的审计依据是保存的 backend=qwen3_vl、is_mock=false、inference_performed=true、request_id、时间、模型/图片哈希、代码及状态记录的一致性；这不是独立硬件认证，不能据此排除未提交记录中的其他运行。

## 协议与会话核对

数据为 `intro/flickr8k@c7a02cb487da0eca729e133194b0f842cb578c08`。原 test 文件名 ASCII 升序第 21–120 张，offset=20、limit=100；此前诊断前 20 张排除。sample_id 与顺序完全匹配正式冻结 Manifest。全部 100 个图片哈希唯一，候选 caption 文档 5,000，每图五个同图 caption 为冻结 qrels。

两路均使用 Qwen3-VL-4B-Instruct、BF16、greedy、seed=0、max_new_tokens=384、BM25(k1=1.5,b=0.75)、top_k=10、lifetime max_attempts=1。环境记录为 RTX 4090 D、Python 3.12.3、torch 2.8.0+cu128、transformers 4.57.1、Pillow 11.3.0，与正式冻结的 dev 环境一致。

| 路径 | 会话 1 新增调用 | 会话 2 新增调用 | 累计 attempts/响应 | completed | failed |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 2 | 98 | 100/100 | 100 | 0 |
| structured（format_details） | 2 | 98 | 100/100 | 94 | 6 |

会话发生于 2026-10-27 北京时间 13:32–13:38。Smoke 和完整续跑复用同一队列与目录，不把 `0001`、`0002` 累计快照相加。首个 structured 样本在 Smoke 已失败，完整会话仍保留该条失败，无再次尝试。

两路各恰好 100 个唯一查询槽位，合计 200 个分支槽位、100 张独立图片。每个 sample_id 每路恰好一个 attempt、一次记录的 backend 调用和一个返回响应，无 missing/pending/running 槽位。SQLite 完整性检查通过。全部 attempts/results/summary/runtime 编号快照、sessions、qrels 及两份 comparison 均已核对；无意外额外编号快照。

Baseline `results-0002.jsonl` 有 100 条，structured 有 94 条，因为 results 只导出成功记录；六条失败在 attempts 和 SQLite 中完整保留。`status=incomplete` 表示流程有失败，不表示六张图片未执行。不得把状态重写为 completed。

194 条成功查询从保存的原始响应按冻结规则重建，CPU BM25 重放的排名、分数、证据文本与来源全部匹配。失败没有执行替代检索。可核验范围内未发现协议偏差、重复调用、丢失响应、样本/qrels 修改或哈希冲突。

## 指标和配对置信区间

分母始终为 100，六条失败全部计零。设一张图片对应的相关 caption 集合大小为 5：

- Hit@K：前 K 项至少命中一个相关 caption 时为 1，否则为 0。
- Recall@10：前十项命中的相关 caption 数 ÷ 5；本报告沿用 summary 的命名，不将它与 Hit@10 混淆。
- MRR@10：首个相关结果位于第 r≤10 名时为 1/r，否则为 0。
- 所有指标对 100 张图片取算术平均；未按成功数量重新归一化。

| 指标 | Baseline | format_details | 配对差（后者−前者） | 95% paired bootstrap CI | 胜/平/负 |
| --- | ---: | ---: | ---: | --- | --- |
| Hit@1 | 0.400000 | 0.420000 | +0.020000 | [−0.090000, 0.140000] | 18/66/16 |
| Hit@5 | 0.650000 | 0.680000 | +0.030000 | [−0.090000, 0.150000] | 19/65/16 |
| **Hit@10（主指标）** | **0.780000** | **0.720000** | **−0.060000** | **[−0.170000, 0.050000]** | **13/68/19** |
| Recall@10 | 0.308000 | 0.322000 | +0.014000 | [−0.048000, 0.076000] | 40/23/37 |
| MRR@10 | 0.5165793651 | 0.5166785714 | +0.0000992063 | [−0.0981114087, 0.0978214286] | 34/36/30 |

统计方法沿用冻结方案：以**图片对**为单位，固定 sample_id 顺序，Python `random.Random(0).choices` 每轮有放回抽取 100 对，重复 20,000 次。每轮计算平均配对差，取差值分布的 2.5%/97.5% 分位点，线性插值。所有指标使用相同抽样索引；失败从始至终保留为零。Hit@10 是预注册主区间，其余四项是本次审计按用户要求补充的描述性区间，未做多重比较调整，不用其中较有利的一项替代主指标。

胜/平/负按每图该指标差值的正/零/负计数。六个 structured 失败图的 Baseline 全部 Hit@10=1；其余 94 图两路各命中 72 图，因此净 −6 次命中在数值上由失败图贡献。这只是观测差值分解，不能推断修复这些响应就会得到六次正确检索，更不能用成功子集替代主结果。

## 六条 ValueError 的根因

100 个 structured 响应均为完整、符合原 Schema 的严格 JSON，原 Schema 成功率 **100%**；附加格式合约通过率及端到端成功率均为 **94%**。六条失败不是 JSON 语法错误，也不是 384-token 截断。它们都是附加字段项数上限违规。

| sample_id（均带 flickr8k: 前缀） | 原错误类型 | 首个错误字段 | 全部项数违规 | generated_tokens |
| --- | --- | --- | --- | ---: |
| 1258913059_07c613f7ff.jpg | ValueError | visible_text | visible_text 3>2；uncertainties 3>2 | 63 |
| 1329832826_432538d331.jpg | ValueError | visible_text | visible_text 4>2 | 88 |
| 1417031097_ab656bc4bd.jpg | ValueError | visible_text | visible_text 3>2 | 67 |
| 1461667284_041c8a2475.jpg | ValueError | objects | objects 7>6 | 43 |
| 1472882567_33dc14c8b6.jpg | ValueError | objects | objects 8>6 | 59 |
| 2041867793_552819a40b.jpg | ValueError | visible_text | visible_text 3>2 | 80 |

精确首个错误分别为 `v11 format: visible_text must contain at most 2 items`（4 条）和 `v11 format: objects must contain at most 6 items`（2 条）。第一张图的 uncertainties 超限被 fail-fast 顺序遮住，审计额外列出，但不能重复计为第七个失败样本。其余字符串长度、去重、总词数约束未发现额外违规。

完整原始描述在私用 [report.md](../outputs/v11-independent100-audit-02/report.md) 逐条展示，原始响应及模型元数据在 [failures.json](../outputs/v11-independent100-audit-02/failures.json)。描述可能包含模型视觉错误，本审计未进行逐图视觉正确性人工标注；合约通过不等于内容真实。

## 成本与限制

| 指标 | Baseline | format_details |
| --- | ---: | ---: |
| generated_tokens 总数（各100响应，含失败） | 5,378 | 3,386 |
| 响应平均 latency_ms（不含加载） | 1,376.73 | 888.80 |
| attempt 平均 wall ms（各100次） | 1,542.83 | 897.37 |
| 成功检索平均 ms（覆盖100/94条） | 12.18 | 4.81 |
| 记录的最大 torch allocated MiB | 8,565.07 | 8,644.00 |

Baseline 两次会话各首条记录冷加载，分别约 6,296.79 ms 和 4,767.40 ms；structured 复用同进程模型。执行顺序固定，未作随机交叉，不能把 wall time 或这次平均延迟比例当成一般加速结论。显存是 torch allocated MiB，排除模型加载峰值，并非完整设备占用。

固定 ASCII 排序窗口不等于随机总体样本；图片的五个 caption 高度相关；其他图 caption 是假定负例；模型预训练可能见过 Flickr8k；镜像原 split 未重新对照作者原始划分认证。Bootstrap 衡量这一固定队列经验分布下的差值不确定性，不能消除这些限制或证明总体泛化。

## 交付、复现与验证

- [完整私用审计报告及原始响应](../outputs/v11-independent100-audit-02/report.md)
- [机器可读审计、逐文件哈希及全部排名](../outputs/v11-independent100-audit-02/audit.json)
- [100 对逐样本指标及差值 CSV](../outputs/v11-independent100-audit-02/paired.csv)
- [100 对逐样本指标及差值 JSON](../outputs/v11-independent100-audit-02/paired.json)
- [六条完整失败响应](../outputs/v11-independent100-audit-02/failures.json)
- [离线审计脚本](../scripts/audit_v11_independent.py)、[计算测试](../tests/test_v11_independent_audit.py)

输出 01 为首次审计保留产物；输出 02 增加了逐项 trace/快照验证及完整报告，二者数值一致。原实验和此前审计产物均未覆盖。

已实际执行：

```powershell
.\.venv\Scripts\python.exe scripts/audit_v11_independent.py --source outputs/v11-independent100-source-01 --transfer outputs/v11-independent100-audit-transfer.gFF0L3 --images outputs/flickr8k-v11-test-100.tar.gz --output outputs/v11-independent100-audit-02
.\.venv\Scripts\python.exe -m pytest tests/test_v11_independent_audit.py -q -p no:cacheprovider --basetemp outputs/pytest-v11-independent-audit-01 --junitxml outputs/pytest-v11-independent-audit-01.xml
```

审计 exit=0，`status=audited_with_provenance_limits`；**6 tests passed**。测试覆盖失败计零、排名10/11边界、重复排名/qrels拒绝、与冻结 Hit@10 bootstrap 完全一致、非二元退化区间、首个错误之外的全部合约违规。这些是计算验证，不是模拟实验结果。真实结果始终来自原件。

复现时使用新输出目录（例如 `outputs/v11-independent100-audit-recheck-01`），不可重用已存在目录；脚本拒绝覆盖原件，按只读模式打开 SQLite，运行前后重新核验来源哈希。首次审计未运行 GPU runner、未提交或推送 Git；下面记录此次正式封存。

## 正式封存复核（2026-10-27）

本轮在新目录 `outputs/v11-independent100-seal-recheck-01` 重新执行同一离线审计，`audit.json`、`paired.json`、`paired.csv`、`failures.json`、`report.md` 五个文件与审计02逐字节相同，报告五行指标、配对差与区间已程序化比对通过。无新增 VLM/GPU 调用，无冻结文件或原始产物修改。

```powershell
.\.venv\Scripts\python.exe scripts/audit_v11_independent.py --source outputs/v11-independent100-source-01 --transfer outputs/v11-independent100-audit-transfer.gFF0L3 --images outputs/flickr8k-v11-test-100.tar.gz --output outputs/v11-independent100-seal-recheck-01
$env:GIT_CONFIG_COUNT='1'
$env:GIT_CONFIG_KEY_0='safe.directory'
$env:GIT_CONFIG_VALUE_0='D:/Research/VisionEvidence-Agent'
.\.venv\Scripts\python.exe -m pytest tests/test_v11_independent_audit.py tests/test_caption_pair.py tests/test_v11_audit.py tests/test_flickr8k.py tests/test_flickr8k_cohort.py tests/test_retrieval.py tests/test_retrieval_evaluation.py -q -p no:cacheprovider --basetemp outputs/pytest-v11-seal-01 --junitxml outputs/pytest-v11-seal-01.xml
```

**89 passed，0 failed**。覆盖新增审计计算、冻结配对报告/来源拒绝、Flickr8k 适配与队列、BM25 和检索评价；并非声称重跑整个历史测试集。进程级 Git safe.directory 设置未修改全局配置。

[封存索引](../experiments/v11/independent100-sealed-20261027/seal-index.json) 版本化保存主次指标、全部区间、六条失败ID/错误/响应哈希、47个原文件哈希、两个私用压缩包哈希及测试/审计输出指纹。索引不包含原始响应文本、图片或数据库。原始 GPU 产物与审计02保持在本地忽略目录；普通 Git clone 只含报告和索引，完整原件仍需从私用备份取得，不能把索引当成原件。

本次封存只创建本地 Commit，不 push。封存 Commit 用 `git log -1 --format=%H -- experiments/v11/independent100-sealed-20261027/seal-index.json` 解析，避免把提交自己的 SHA 写入提交内容。实验冻结 Commit 仍为 `8428c626a7ba55bb723a0e1506889a04595c1307`。

下一步：保留该结果作为 V1.1 独立测试结论，由人工复核报告；如仍有终端日志，可另行补齐审计来源链。不得因为此结果不利而修改 format_details、重跑失败后替换结果或更换主指标。未来方法变更须另立版本和未用评测集。
