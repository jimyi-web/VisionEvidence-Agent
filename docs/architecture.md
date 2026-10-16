# Architecture — T03-B-1

## 当前执行路径

CLI 参数 → `ClaimInput` 校验 → JSON 输入校验回执 → 退出。

T02 新增两条独立 CPU 路径：

- `retrieve` → 严格 JSONL 读取 → `Evidence` → 英文分词/内存 BM25 索引 → Top-K 证据输出。
- `evaluate` → 同一索引 + 独立 qrels → 逐查询检索 → Recall@K/MRR@K 宏平均 → 可审计 JSON。

旧输入校验命令及 `--help`、`--version` 保持兼容；禁止将 `--text/--image`
与子命令混用；VLM 图片参数放在 `vlm` 后面。旧路径不调用真实模型、网络搜索、Agent 或验证决策。

T03-A 增加独立路径：`vlm` → TOML 配置/后端选择 → `VLMRequest` 校验
→ `BaseVLM.generate` → `VLMResponse` → 明确后端标识的文本或 JSON。
该路径尚未与检索、Agent 或验证逻辑串联。

T03-B-1 增加显式 `qwen3_vl` 分支：本地目录预检 → Pillow 完整解码/EXIF/RGB/尺寸约束
→ 首次加载并复用 Qwen3VLForConditionalGeneration 与 AutoProcessor → 单 GPU BF16 生成
→ 裁剪输入 token → 未验证描述与运行元数据。GPU 框架延迟导入，Mock 和检索不依赖它们。
真实 CUDA 路径尚待云端验收，CPU fake 测试不代表真实推理已成功。

CLI 使用 argparse；数据契约使用 dataclass、pathlib 等标准库组件。
输入文本去除首尾空白后必须非空。图片可省略；提供时必须指向现有文件，
相对路径基于进程工作目录。此处指 T01 的 `ClaimInput` 路径，不检查格式、解码或提取图像特征。
回执包含 `status: input_validated`、`inference_performed: false`、输入和包版本。
回执不是 `VerificationResult`，不包含预测标签或检索证据。

## 数据契约

| 数据结构 | 字段 | 约束与含义 |
| --- | --- | --- |
| `ClaimInput` | `text`, `image_path`, `metadata` | 非空声明，可选现有文件路径，元数据字典 |
| `Evidence` | `document_id`, `source`, `text`, `score`, `metadata` | 前三项非空；分数可缺省，否则须为有限数值；保留来源与原文 |
| `VerificationResult` | `predicted_label`, `evidence_ids`, `metadata` | 标签必须显式提供且非空；引用对应 `Evidence.document_id`；元数据可记录运行来源 |
| `VLMRequest` | `image_path`, `prompt`, `request_id`, `max_image_size_mb` | 非空提示词、本地普通文件、PNG/JPEG/WEBP 扩展名与文件头一致、默认 ≤10 MiB |
| `VLMResponse` | `request_id`, `description`, `model_name`, `backend`, `is_mock`, `latency_ms`, `metadata` | 字符串描述、显式模拟标记、有限非负耗时、严格 JSON 元数据；文本不是已验证事实 |

标签词表留给后续具体数据集和协议确定，不在基础工程阶段预设标签类别。
证据引用允许为空（具体任务是否接受由未来验证层决定）；当前仅检查引用字符串，
尚未进行跨对象引用完整性校验。分数允许负值，不被解释为概率或可信度。
T01 三种结构的 `metadata` 须为字符串键的字典，值暂不限制；其序列化层需负责 JSON 兼容性。
默认容器独立创建，构造时执行校验；这些是可变数据对象，后续赋值不会自动重新校验。
T02 外部 JSONL 的 `evidence_id` 映射到 `Evidence.document_id`，输出再映射回来，
不修改 T01 的公共数据结构。索引构建和返回结果时复制对象，防止调用者修改对象导致索引失效。
JSONL 读取层限制元数据为合法 JSON，拒绝损坏记录、非有限数字、重复 JSON 键和未知顶层字段。
所有测试数据及结果标签都是合成测试夹具，不代表模型输出或研究结论。

VLM 数据结构独立放在 `vlm/schemas.py`，不修改 T01 契约；VLM dataclass 冻结字段赋值。
响应 metadata 构造时校验并复制，只接受 JSON 类型、字符串键和有限数值，拒绝循环引用；
嵌套容器仍可修改，`to_dict()`/`to_json()` 会重新校验。图片在构造请求及 Mock 调用时检查，
只读取最多 33 字节文件头，不做完整解码。文件头检查不是完整图像可解码性或恶意文件检测。

## 模块职责与实现状态

| 模块 | 计划职责 |
| --- | --- |
| `vlm` | 已实现同步 Protocol、Mock、提示词、工厂和本地 Qwen3-VL 加载/推理/统计逻辑；真实 GPU 执行尚未验证 |
| `retrieval` | 已实现 JSONL 校验、英文分词、BM25 和稳定排序；稠密/混合检索尚未实现 |
| `agent` | 受约束工具选择；显式接口、调用上限、超时、停止条件和中间记录 |
| `verification` | 基于输入及证据产生结果，检查引用，按数据集协议解释标签 |
| `evaluation` | 已实现显式 qrels、Recall@K、MRR@K、宏平均和逐查询记录；模型验证评估尚未实现 |

`retrieval/corpus.py` 负责文件格式及 schema 适配；`retrieval/bm25.py` 负责分词与索引；
`evaluation/retrieval.py` 负责 qrels 和指标。CLI 负责参数、配置、输入哈希、环境记录和结果保存。
BM25 索引只接收证据，`search` 只接收查询和 K，不接受相关性标注。
同分按证据 ID 升序排列，无词项匹配时不填充零分文档。空标注查询排除并记录数量；
全部无标注则报错，qrels 引用库外 ID 也报错。具体协议见 [检索基线](retrieval_baseline.md)。

未来数据流：文本声明及可选图片 → 多模态理解 → 证据检索 → 受约束工具编排
→ 有证据支撑的验证 → 结构化结果及评估。
此流程是规划，不是当前已连通的实现。检索得到的内容不自动视为真实证据。

## 配置、工程与实验边界

- `pyproject.toml` 管理包元数据、安装入口、开发依赖及 pytest；采用 `src` 布局。
- `configs/base.yaml` 是版本控制中的规划模板。T01 不加载 YAML，因此无需 YAML 依赖。
- `configs/retrieval_bm25.toml` 是 T02 可执行配置，通过标准库 `tomllib` 加载；
  未提供 `--config` 时使用相同默认值 k1=1.5、b=0.75，实际参数写入结果。
- `configs/vlm_mock.toml` 通过 `tomllib` 严格读取；未提供路径时使用相同内置默认值。
  `--backend` 覆盖后端配置，`--prompt` 覆盖提示词并将实际提示词版本标记为 custom。
- `configs/vlm_qwen3_vl.toml` 配置本地权重、单 GPU、BF16、生成长度及像素/尺寸上限。
  GPU 依赖独立放在 `requirements/`；模型和处理器均 local_files_only，不自动下载或降级 Mock。
- `scripts/` 为后续可复现脚本预留，`.gitkeep` 保留空目录；当前复现命令见 README。
- `outputs/` 用于本地产物，只有 `.gitkeep` 进入 Git；实验需各自独立目录以避免覆盖。
- T02 只执行本地文本检索与检索评测，不进行模型调用、网络检索或 GPU 操作。
- 评测 JSON 记录输入路径/内容哈希、配置、数据集声明、Python/包版本、CPU 平台、
  索引和检索耗时、样本数、指标及逐查询排序。无随机操作，种子和模型字段为 null；
  峰值内存尚未测量，显式记录为 null。结果文件使用独占创建，禁止静默覆盖。
- 未来评估记录数据集名称/版本/划分、模型/revision、环境、种子、配置、硬件、
  时间和内存、指标与样本数；不得混用训练、开发和测试数据。

## 验证范围

pytest 覆盖输入校验、可选文件路径、证据字段与分数、结果契约、独立默认容器，
以及通过当前 Python 解释器启动的 CLI 帮助、版本、JSON 回执和错误退出码。
T02 增加严格 JSONL 校验、BM25 公式手算、排序和边界条件、同分稳定性、
指标手算/宏平均/空标注处理，以及检索、评测 CLI 和输出文件不覆盖检查。
不以这些基础测试推断检索质量、模型准确率或 GPU 可用性。

T03-A 增加图片校验、JSON 安全性、Mock 内容确定性与接口注入、工厂/配置错误和 CLI 测试。
独立 Python 子进程在导入 VLM 前拦截网络、GPU 框架导入与外部进程调用，验证 Mock 路径。
69 字节 PNG demo fixture 另验证完整 chunk CRC 与解压后的像素，避免把伪造文件当示例图片。
更多细节见 [VLM 接口说明](vlm_interface.md)。

T03-B-1 使用真实 Pillow 和注入的 torch/processor/model doubles 验证图片、加载复用、
CUDA/BF16 预检、token 裁剪、计时口径、失败传播和真实后端 CLI 输出契约。
部署与尚未验证事项见 [Qwen3-VL 部署说明](qwen3_vl_deployment.md)。
