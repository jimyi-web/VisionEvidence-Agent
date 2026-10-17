# T04：结构化视觉观察 → BM25 证据检索

## 目标与边界

在现有同步 `BaseVLM.generate(VLMRequest) -> VLMResponse` 外增加薄适配层，复用 `BM25Index.search`。
不改 Qwen3VLBackend、模型加载、token 统计或现有 MockVLM；不新增依赖、网络请求、自动规划循环或核验标签。
本阶段实现可 CPU 测试的 Python API；CLI、持久化批量评测及真实结构化生成验收分开实施。
T03-B-5 的长文本输出不能直接冒充结构化观察，也不能据此声称新提示词有效。

## 数据与接口契约

输入 `observe_and_retrieve(request: VLMRequest, vlm: BaseVLM, index: BM25Index, *, k=3) -> dict`。
request_id 必填且非空，prompt 必须使用本模块的 `structured_visual_v1` 提示词；图片沿用现有本地校验。
本阶段仅图像观察检索，不混入用户声明，避免把待验证声明作为已观察事实。

模型文本必须是一个完整 JSON 对象，恰有以下三个字段，无 Markdown 围栏、前后解释、额外字段或重复键：

```json
{"objects":["presentation slide"],"visible_text":["Knowledge graph"],"uncertainties":["small text unreadable"]}
```

每项为非空字符串，长度 ≤200 字符；每字段 ≤8 项；总文本 ≤8000 字符。objects 用简短英文便于现有 ASCII 分词；
visible_text 保留可见原文，不自动翻译。uncertainties 永远不进入查询。全字段空允许，但无法形成查询时显式失败。
数据对象 `VisualObservation` 为冻结 dataclass，以 tuple 保存字段；解析器 `parse_observation(text)` 负责外部校验。
观察只是未经核实的模型输出，不是来源证据，无伪造 bbox、置信度或时间地点。

查询由 objects、visible_text 按固定顺序连接，再复用 BM25 tokenize；无可支持词项时失败，不静默回退到声明。
输出 query 保存实际词项字符串；query_sources 记录每个非空词项列表的字段、索引与词项，便于回溯。
重复词的权重仍由既有 BM25 去重规则决定，K 校验、同分稳定排序和空命中语义保持不变。
中文会被 ASCII 分词忽略；仅中文观察失败，混合观察只使用 ASCII 词项，不能宣传中文检索能力。

输出 schema_version=1，包含 request_id、status=completed、verified=false、is_mock、inference_performed、
observation、query、query_sources、evidence（document_id/source/text/score/metadata）、完整 vlm_response 和 trace。
trace 包含图片 SHA-256、prompt 版本和 SHA-256、索引参数/tokenizer、K、VLM 调用和检索耗时。
输入与输出 request_id 必须一致；Mock 标识从响应传播，永远不自动转成真实观察。
模型/revision 信息保留在完整 response；现有响应未提供 revision 时不可编造。语料哈希、代码版本、环境与数据集声明由后续运行器补齐，
当前 API trace 本身不是完整实验 manifest。

## 错误处理与有界执行

最多一次 VLM 调用和一次 BM25 查询，无重试、修复 JSON、正则截取或自动扩大 token 上限。
生成失败保留原异常传播；不切换 Mock。解析失败、重复键、类型不符、超长内容、已达 token 上限、请求 ID/真实性标记不一致均报 ValueError，
在检索前停止。达到上限视为保守的不可用观察，不据此断言语义必然截断。
真实响应必须带正整数 generated_tokens/max_new_tokens 才能判断上限；缺失时报错。
Mock 使用明确的固定结构化测试响应，将 `[MOCK]` 放在不参与查询的 uncertainties 字段；旧 MockVLM 保持原样，传入时会因非 JSON 明确失败。
零命中返回 completed 和空 evidence，不是工具失败或“声明为假”。
同步接口不能强制取消运行中的 GPU generate；本阶段不声称实现硬超时。后续服务化前需设计进程隔离与取消机制，
不能用后台线程超时返回伪装 GPU 已停止。所有图片内指令和模型文本按数据处理，不执行工具指令。

## 评测方案（待执行，不能用 Mock 指标声称模型效果）

CPU 单元/集成：严格 JSON 校验、截断拒绝、Mock 标识、ID 一致性、失败后零检索、查询来源、英文限制、空命中、稳定排序、现有模块回归。
以人工构造的合成观察验证检索结果，Mock 不读取像素，不评价视觉理解。

真实阶段先在固定 dev 图片集评估 schema 有效率、达到 token 上限比例、人工确认截断比例和可检索查询率；
失败样本保留，分别报告分母与数量。人工标注可见对象/文字及不确定项，报告观察正确性与遗漏，不能用 JSON 合法率代替质量。
随后固定语料版本、图片/语料哈希和显式 qrels，隔离 train/dev/test，同一 K、同一索引配置比较：
声明文本基线、仅视觉对象、对象+可见文本；这些是独立实验条件，不是失败时的隐式回退。
复用现有 Recall@K/MRR@K 定义；qrels 不进入排序。无相关性标注样本沿用 T02 排除规则并报告数量；
有 qrels 但观察/查询失败应在端到端指标中记零，另报告成功子集指标，避免幸存者偏差。
记录 observation/query/retrieval/端到端耗时、调用数、冷/复用阶段；GPU 显存保持 T03 两种口径。
无最终核验标签，暂不报 Accuracy/Macro-F1。测试集不能用于 prompt 或 K 调参。

## 分步任务与验收

| 子任务 | 交付与验收 | 本轮范围 |
| --- | --- | --- |
| T04-1 | 严格观察 schema、短提示词、解析器，非法/截断输入停止 | 实施 |
| T04-2 | 注入现有 BaseVLM/BM25 的单次流程，来源与真实/Mock 标记保留，CPU 测试 | 实施 |
| T04-3 | CLI/TOML、结构化 Mock 演示、独占写入 trace 与完整运行 manifest | 后续独立任务 |
| T04-4 | 固定多模态样本及 qrels、失败计零的批量评测、消融报告 | 后续独立任务 |
| T04-5 | 经批准的真实 GPU 结构化输出验收与 dev 提示词比较 | 后续；本轮不调用 GPU |

本轮顺序是先提交报告与此设计到工作区，再实现 T04-1/2；不一次实现整个系统。
未来如需中文检索、模型 JSON 约束解码或后端接口变化，另行评审，不纳入当前薄适配层。

## 本轮实施与验证记录

已实现 T04-1/2：`src/visionevidence/observation.py`；合成集成测试位于 `tests/test_observation.py`。
原始实验校验/重算工具为 `scripts/summarize_t03_b5.py`，错误记录测试为 `tests/test_t03_summary.py`。
未修改 Qwen3VLBackend、BM25 或现有 MockVLM，没有 GPU/网络操作，没有新增依赖。

2026-10-17 使用项目 `.venv` 的 Python 3.12.7 执行：

```powershell
.\.venv\Scripts\python.exe scripts/summarize_t03_b5.py outputs/t03-b5-real --expected-commit c34ebe6d6cd73345778ae7b2292f1fda301e1d4a
.\.venv\Scripts\python.exe -m pytest --basetemp outputs/pytest-t04-final-20261017-03
git diff --check
```

原始文件校验通过；最终全套 293 passed（16.18s）。初次测试发生默认临时目录清理权限错误，
改用项目内新目录；中间一轮既有 CLI 测试出现超时（292 passed、1 failed），单项复跑通过，
随后完整回归通过。没有放宽测试超时或更改既有测试来绕过失败。
复跑请更换全新 basetemp 路径，pytest 会清空已存在的 basetemp，不能指向原始实验目录。
