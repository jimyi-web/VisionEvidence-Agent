# T02：BM25 文本证据检索基线

本模块仅实现本地文本检索与检索评测。所有 `examples/` 数据及其相关性标注都是
**synthetic/demo**，版本 1，划分 demo，不是真实新闻、事实证据或研究效果结论。
不使用 GPU、VLM、Agent、网络搜索或最终事实核验。

## BM25 原理与实现选择

BM25 根据查询词的文档频率、文档内词频及文档长度计算词面相关性。
本实现只使用 Python 3.12 标准库，不使用 rank-bm25 或新增运行依赖：当前规模下，
显式实现便于审计公式、固定边界行为，并保留 T01 的轻量环境。

对每个不重复的查询词 t：

- IDF(t) = ln(1 + (N − df(t) + 0.5) / (df(t) + 0.5))。
- TF 部分 = f(t,d) × (k1 + 1) / (f(t,d) + k1 × (1 − b + b × |d| / avgdl))。
- score(q,d) 是查询词的 IDF × TF 部分之和。

N 是文档数，df 是包含该词的文档数，f 是文档内词频，|d| 是分词后的长度，
avgdl 是全库平均长度。采用正值 log1p 平滑 IDF；这与某些 BM25 包的负 IDF
截断方案不同，不应直接比较不同实现的原始分数。默认 k1=1.5、b=0.75，
k1 必须为有限正数，b 为 [0,1] 内的有限数值。

分词先转小写，再提取 `[a-z0-9]+`。标点作为分隔符；例如 `Road-closures!`
得到 `road, closures`，`don't` 得到 `don, t`。不做停用词移除、词干化或同义词扩展。
查询词去重并按字典序累加，重复查询词不加权。分数不是概率、事实真假或真实性置信度。

索引在内存中保存文档、词频倒排表及文档长度，只遍历查询词的倒排表。
同一 `BM25Index` 可复用多次；CLI 每次运行重建一次，评测中所有查询共用一个索引。
候选按 score 降序、`evidence_id` 字符串升序排序，无磁盘索引或增量更新。

## 证据与结果格式

UTF-8 JSONL，每行一个对象，必填字段 `evidence_id`、`text`、`source` 都为非空字符串；
可选 `metadata` 为 JSON 对象。其他信息放进 metadata，未知顶层字段报错。

```json
{"evidence_id":"E001","text":"Heavy rainfall caused several road closures.","source":"synthetic/demo","metadata":{"synthetic":true}}
```

ID、正文和来源按现有 `Evidence` 契约去除首尾空白；保留正文内部空白和元数据。
去除空白后重复的 ID 会报错。`evidence_id` 在 Python 对象中对应 `document_id`，
无需改变现有 schemas.py。返回 `Evidence` 列表；score 存放 BM25 分数。

文件读写错误、无效 UTF-8、损坏 JSON、空白行、非对象行、缺字段、字段类型错误、
空文本、重复 ID、重复 JSON 键、NaN/Infinity 等均报错，不静默跳过。
记录级错误含文件路径和行号；UTF-8 解码错误含文件路径。
允许末尾正常换行，空文件报错。可读取中文正文和元数据，但中文字符不会生成检索词；
非空但没有支持词项的文档留在库中，词项长度为零。如果全库均无支持词项，英文查询返回空列表。

边界行为：

| 输入 | 行为 |
| --- | --- |
| k 为非整数、布尔值或 ≤ 0 | 报错；CLI 退出码 2 |
| k 大于匹配文档数 | 返回全部匹配文档，不填充零分结果 |
| 空白、纯标点或纯中文查询 | 无支持词项，报错 |
| 有支持词项但无文档匹配 | 返回空列表，CLI 退出码 0 |
| 完全相同的分数 | 按 ID 字符串升序，例如 E10 排在 E2 前 |

## 检索命令与 Python API

在项目根目录、已有 `.venv` 下运行，无需激活环境：

```powershell
.\.venv\Scripts\python.exe -m visionevidence.cli retrieve --query "flood road closures" --corpus examples/evidence_demo.jsonl --top-k 3 --config configs/retrieval_bm25.toml
.\.venv\Scripts\visionevidence.exe retrieve --query "flood road closures" --corpus examples/evidence_demo.jsonl --top-k 3 --json
```

默认每条结果输出一行 JSON，包含 evidence_id、text、source、score、metadata。
`--json` 输出一个包含 query、config、results 的 JSON 对象，空结果为 `results: []`。
终端 JSON 使用 Unicode 转义以兼容 Windows 输出编码，解码后恢复原始字符。
无匹配的默认输出为 `No matching evidence.`；错误走 stderr，退出码 2。
配置文件可省略，默认值与仓库 TOML 相同；提供文件时必须恰含 `[bm25]` 的 k1、b。

```python
from visionevidence.retrieval import BM25Index, load_corpus

index = BM25Index(load_corpus("examples/evidence_demo.jsonl"), k1=1.5, b=0.75)
results = index.search("flood road closures", k=3)
```

## Qrels 与评测协议

评测 JSONL 每行必须恰含 query_id、query、relevant_ids：

```json
{"query_id":"Q001","query":"flood road closures","relevant_ids":["E001","E002"]}
```

query_id 唯一且非空，query 必须有支持词项。relevant_ids 是非空字符串的列表，
按集合解释，重复标注只计一次；引用证据库之外的 ID 报错以防止数据/标注配错。
空 qrels 文件报错。空 relevant_ids 表示无相关性标注：排除该查询，记录其 ID 和数量；
若全部查询均被排除则报错，不制造 0 分结果。无词面命中但有标注的查询正常参与宏平均。

- Recall@K = Top-K 命中的不同相关证据数量 / 该查询全部相关证据数量。
- 单查询 reciprocal rank@K = Top-K 第一个相关证据排名的倒数，未命中为 0。
- 多查询 Recall@K 和 MRR@K 分别对参与评测查询的上述值取宏平均。

人工可计算例子：查询 Q1 的相关集合为 {B,D}，Top-3 为 [A,B,C]，
因此 Recall@3=1/2、reciprocal rank@3=1/2。Q2 的相关集合为 {A}，排名相同，
两项均为 1；Q3 无标注而排除。两项宏平均均为 (1/2+1)/2=0.75，参与数量为 2。
此例在测试中用同分 BM25 文档构造，标注更改不会改变检索排序。

```powershell
.\.venv\Scripts\python.exe -m visionevidence.cli evaluate --corpus examples/evidence_demo.jsonl --qrels examples/qrels_demo.jsonl --config configs/retrieval_bm25.toml --top-k 3 --dataset-name synthetic/demo --dataset-version 1 --split demo --output outputs/retrieval-demo.json
```

评测始终在 stdout 输出 JSON；`--output` 可选，只允许创建新文件，父目录须存在。
再次运行须选新文件名，或省略 `--output`。JSON 包括配置、输入路径和 SHA-256、
数据集声明、包/Python 版本、CPU 平台信息、索引耗时、逐查询检索耗时、总耗时、
文档数和总/参与/排除查询数、指标、逐查询证据 ID 和分数。没有随机抽样或模型，
种子与模型字段为 null；峰值内存未测量，明确记录 null。总耗时不包括结果写盘。
未提供数据集参数时记录 unspecified，正式实验应显式填写。

示例在 K=3 时：Q001、Q002、Q003 的两项指标均为 1；Q004 的查询词与
相关证据 E005 无词面重合，两项均为 0；Q005 被排除。合计 5 个文档，
5 个查询，4 个参与、1 个排除，Recall@3=0.75、MRR@3=0.75。
这些值由实际排序与 qrels 计算，程序不包含指标常量。
输入哈希、排序和指标可复现；时间戳、运行耗时与硬件字段随运行而变化。

## 验证与当前局限

```powershell
.\.venv\Scripts\python.exe --version
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m visionevidence.cli --help
git diff --check
git status --short
```

如果系统 pytest 临时目录的清理权限受限，可指定新的项目内目录，例如
`-m pytest --basetemp outputs/t02-pytest-run-01`。每次选择未使用的目录名，
因为 pytest 会清空已存在的 basetemp；不要指向任何实验结果目录。

仅覆盖英文 ASCII 词面检索；不具备中文分词、语义理解、拼写纠错、图像理解或来源可信度判断。
全库驻内存，返回前对全部匹配候选排序，尚未面向大规模语料优化。
合成数据不用于泛化结论；没有真实数据集、跨平台验证、模型性能或峰值内存基准。

## 后续 Dense Retrieval 对比计划（尚未实现）

先确定真实语料的授权、版本、固定 train/dev/test 划分和显式 qrels，冻结测试集，
BM25 调参只使用开发集。保持语料、查询、相关性定义、K 和空标注排除规则一致，
再引入稠密检索。记录编码器标识/revision、分词/截断、向量归一化、相似度与索引参数，
比较 Recall@K、MRR@K、索引时间、检索延迟及实际内存需求，分析词面匹配与语义匹配差异。
不将测试标注用于排序或调参。任何大型模型下载、GPU 环境或花费另行批准。
