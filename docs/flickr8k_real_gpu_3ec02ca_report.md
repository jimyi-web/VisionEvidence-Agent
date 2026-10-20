# Flickr8k 真实 GPU 实验审计：3ec02ca，20 图

日期：2026-10-20（北京时间）。实验原目录名：`flickr8k-real-3ec02ca-01`。

**结论：本次 Structured 工作流不如 Baseline，且 Structured 整体 incomplete。** 失败查询全部保留并计零。两路共同成功的 15 图上，Structured 的 Hit@10 和 MRR@10 仍较低，不能把差距完全归因于 JSON/Schema 失败，也不能声称已经提高检索性能。值得开展独立、小规模、预先登记的 V1.1 修复实验；当前不建议直接扩大到 100/1000 图 GPU 运行。

任务是 **image-to-caption retrieval proxy**：查询图片，检索其原配对的五条人工描述。本文指标不是事实核验准确率、外部证据可信度、真实性分类或谣言识别准确率。

## 1. 数据、来源与冻结

实际读取了压缩包中的 `comparison-0002.json`、两路 `summary-0002.json`、`results-0002.jsonl`、`state.sqlite3` 的 samples/attempts/qrels/sessions，以及输入快照、manifest 和 runtime。另核对了第一次 2 图验收的 session 记录。不是根据用户转述重建报告。

原压缩包：`outputs/flickr8k-real-3ec02ca-01.tar.gz`，711,336 字节。

SHA256：`4708eb995f12cc0986a3f274eaf9fb58f499dea41e913ff77cbf72a2cfd15b06`。

解压至新目录 `outputs/flickr8k-real-3ec02ca-01-audit-20261020-01/source/outputs/flickr8k-real-3ec02ca-01`，未使用原结果目录作为分析输出。所有 28 个归档文件均与压缩包成员逐字节哈希一致；SQLite 用 `mode=ro&immutable=1` 打开，两库 integrity_check 均为 ok。审计完成后原包与所有提取文件的 SHA256 均不变。此处“冻结”指保留原包及可核验快照、只读分析，不是声称存储介质具有硬件级不可变性。

没有修改 qrels、删除失败样本、补全截断 JSON、放宽原 Schema、覆盖已有输出、修改算法/提示词，或重新调用 GPU。原始失败仍是原始失败。本轮没有提交或推送 Git。

可审阅的精简归档：

- [全部原件大小与哈希](../experiments/flickr8k_v1/gpu-3ec02ca-20261020/freeze-inventory.json)
- [指标、格式统计及验证结果](../experiments/flickr8k_v1/gpu-3ec02ca-20261020/audit-summary.json)
- [五条精确错误与分类](../experiments/flickr8k_v1/gpu-3ec02ca-20261020/failure-taxonomy.json)
- [全部 20 图配对指标](../experiments/flickr8k_v1/gpu-3ec02ca-20261020/paired-metrics.jsonl)
- [最终验证记录](../experiments/flickr8k_v1/gpu-3ec02ca-20261020/validation.json)

本地完整证据包含模型原响应、每图五条目标描述、两路查询/前十结果、词项差异与 BM25 分数分解，保留于被 Git 忽略的 [paired-samples.jsonl](../outputs/flickr8k-real-3ec02ca-01-audit-20261020-01/analysis/paired-samples.jsonl)；五条完整失败原响应另见 [failures.json](../outputs/flickr8k-real-3ec02ca-01-audit-20261020-01/analysis/failures.json)。不把原始图片、完整人工描述库或 SQLite 复制进可版本控制的精简归档。

## 2. 实验配置与实际规模

| 项目 | 归档核实值 |
| --- | --- |
| 数据 | Flickr8k，intro/flickr8k @ c7a02cb487da0eca729e133194b0f842cb578c08 |
| 划分/抽样 | test，文件名升序前 20 张；本地 20 图内容哈希均与 attempts 相同 |
| 候选库 | 固定 test 1,000 图 × 5 描述 = 5,000 文档 |
| qrels | 每图五个原配对描述 ID，caption_pair_proxy |
| 图片 / 查询槽位 / 真实调用 | 20 张独立图片 / 40 个槽位 / 40 次 backend 调用，40 个真实响应 |
| 状态 | Baseline completed；Structured incomplete |
| 源码 | 3ec02ca331e54b06595f32953d4d921f5c50bf7c，云端 dirty=false；全部记录源码哈希与本地审计所用源码一致 |
| 模型 | Qwen/Qwen3-VL-4B-Instruct，本地权重；16 个模型目录文件的内容哈希锁定在 manifest |
| 模型 revision 边界 | 记录为 local content hashes；包中没有上游模型 revision 文件正文，不能据此补写一个 HF 模型 Commit |
| GPU | NVIDIA GeForce RTX 4090 D，cuda:0 |
| 系统/软件 | Linux 5.15.0-101-generic-x86_64，glibc 2.35；Python 3.12.3；torch 2.8.0+cu128；transformers 4.57.1；Pillow 11.3.0 |
| 生成配置 | BF16，do_sample=false，seed=0，max_new_tokens=384，max_image_side=1280 |
| 检索 | BM25 k1=1.5、b=0.75；ASCII 小写字母数字分词；无词干化/语义同义映射；K=1/5/10 |
| Baseline | concise_visual_description_v1，英文散文整段作为查询 |
| Structured | structured_visual_v1，严格 objects/visible_text/uncertainties；仅前两字段构造查询 |

两路 samples/corpus/qrels/模型配置/模型文件/源码/seed/检索设置的控制字段全部相同。Linux 样本快照含 Linux 相对路径，其指纹按云端快照核对，不拿 Windows 适配文件的路径差异误判为样本变化。

两次 session 分别是先 2 图（batch_size=2），再断点扩至 20 图（batch_size=4），累计每路 20 次 attempts，每个图片/分支只有一次 attempt，max_attempts=1。`comparison-0002` 是累计快照，不能把 0001 与 0002 的累计数字相加成 44 次。相同进程内两路共享模型；两次命令进程各在 Baseline 首个新增图片加载一次模型，共两次加载。batch_size 是调度批大小，不是 GPU tensor batch。

首次 Baseline session 开始于北京时间 09:42:54；最终 Structured session 结束于 09:49:03，中间含人工间隔，不能把这段时间作为连续端到端运行耗时。

## 3. 主结果：全部 20 图，失败计零

Hit@K 是前 K 条是否至少命中一个同图描述，再按图片平均；CaptionRecall@K 是命中描述数除以五；MRR@K 是首个目标描述排名的倒数，超出 K 或失败为零。

| 指标 | Baseline | Structured | Structured − Baseline |
| --- | ---: | ---: | ---: |
| 流程成功数 | 20/20 | 15/20 | −5 |
| Hit@1 | 0.3500（7/20） | 0.1500（3/20） | −0.2000 |
| Hit@5 | 0.4500（9/20） | 0.2000（4/20） | −0.2500 |
| Hit@10 | 0.6500（13/20） | 0.3500（7/20） | −0.3000 |
| MRR@10 | 0.411012 | 0.178611 | −0.232401 |
| CaptionRecall@1 | 0.0700 | 0.0300 | −0.0400 |
| CaptionRecall@5 | 0.1400 | 0.1000 | −0.0400 |
| CaptionRecall@10 | 0.1900（19/100） | 0.1600（16/100） | −0.0300 |

结果与 comparison、summary、SQLite 重算完全一致。summary 中 `recall_at_k=0.19/0.16` 是五描述召回比例，不是 Hit@10=0.65/0.35。

### 成功样本与同一队列的诊断结果

| 队列（仅诊断） | 方法 | n | Hit@1 | Hit@5 | Hit@10 | MRR@10 | CaptionRecall@10 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 各自成功样本 | Baseline | 20 | 0.3500 | 0.4500 | 0.6500 | 0.411012 | 0.1900 |
| 各自成功样本 | Structured | 15 | 0.2000 | 0.2667 | 0.4667 | 0.238148 | 0.2133 |
| **两路共同成功图片** | **Baseline** | **15** | **0.3333** | **0.4667** | **0.6667** | **0.405159** | **0.1867** |
| **两路共同成功图片** | **Structured** | **15** | **0.2000** | **0.2667** | **0.4667** | **0.238148** | **0.2133** |
| Structured 失败的图片 | Baseline | 5 | 0.4000 | 0.4000 | 0.6000 | 0.428571 | 0.2000 |
| Structured 失败的图片 | Structured | 5 | 0 | 0 | 0 | 0 | 0 |

不同成功子集不能当作公平主比较。共同成功的 15 图才控制了图片集合，但这是按运行结果选择的诊断队列，仍不能替代全部 20 图的主结果。

共同成功队列中，Structured 的 CaptionRecall@10 较高（16 个目标描述，对比 Baseline 14 个），因为个别成功图片能集中取回多个同图描述；其命中图片覆盖、首位命中与 MRR 仍较差。应如实报告这个局部指标，不能把它概括为整体检索提升。

全部 20 图按 MRR@10 配对，Baseline 较好 9 图、Structured 较好 4 图、相同 7 图；共同成功 15 图中分别为 6、4、5 图。共同成功队列中 Hit@10 新增命中 2 图、丢失命中 5 图，净少 3 图。

MRR@10 差额可作记账分解：Structured 失败的五图贡献差额 0.107143，共同成功十五图贡献 0.125258，合计 0.232401。它说明“并非只有格式失败”；不是修好五条后一定可收回 0.107143 的因果或反事实估计。

## 4. 五个 ValueError：精确错误、原响应与分类

| 序号 / 图片 ID | SQLite 精确 error | token | 原响应检查 | 根因分类 |
| --- | --- | ---: | --- | --- |
| 3 / 1082379191_ec1e53f996.jpg | `token limit reached; structured observation rejected` | 384/384 | 对象列表从人物/湖边场景扩展到大量身体部位、形状、方位等泛化词；末尾停在未闭合的 `"next to` | 达上限、JSON 未闭合；枚举失控 |
| 13 / 115684808_cb01227802.jpg | `objects must contain at most 8 items as a tuple` | 38/384 | 完整合法 JSON；objects=9，visible_text=0，uncertainties=0 | Schema 数量上限失败 |
| 16 / 1174629344_a2e1a2bdbf.jpg | `objects must contain at most 8 items as a tuple` | 110/384 | 完整合法 JSON；objects=17，visible_text=1，uncertainties=3 | Schema 数量上限失败 |
| 18 / 1224851143_33bcdd299c.jpg | `token limit reached; structured observation rejected` | 384/384 | 多轮重复 appliance / stone wall / stacked items / green and blue chair / wooden bar，末尾在逗号后结束 | 达上限、JSON 未闭合；循环重复 |
| 19 / 1237985362_dbafc59280.jpg | `token limit reached; structured observation rejected` | 384/384 | 初始人物/气泡后大量重复 blue shirt，最后一个字符串未闭合 | 达上限、JSON 未闭合；循环重复 |

三条未闭合响应的独立 JSON 解析诊断分别为：

- 3：`Unterminated string starting at: line 1 column 1108 (char 1107)`
- 18：`Expecting value: line 1 column 1303 (char 1302)`
- 19：`Unterminated string starting at: line 1 column 1322 (char 1321)`

这些 JSONDecodeError 是本轮离线检查的诊断，不是 SQLite 中原始捕获的 error。原流程先检查生成 token 上限，因此数据库存的是 token-limit ValueError；不能把同一条再计作一个独立 JSON 失败。归档没有保存逐 token ID 或显式 finish_reason，但 generated_tokens=384/max_new_tokens=384 与明显未闭合结尾共同支持“达到上限并输出不完整”。

两条 `as a tuple` 消息来自内部 dataclass 校验：JSON 数组成功解析后才转为 tuple。模型并未输出 Python tuple，也不是把数组误写成 tuple 的语法问题。两条的直接问题均为对象项数超过 8。

### 格式合法率及异常分母

| 检查层级 | 数量 / 分母 | 比例 |
| --- | ---: | ---: |
| 真实响应到达 | 20/20 | 100% |
| 严格 JSON 语法合法（含重复键/非有限数检查） | 17/20 | 85% |
| 严格原 Schema 合格 | 15/20 | 75% |
| JSON 合法响应中的 Schema 合格 | 15/17 | 88.24% |
| 完整可用查询并成功检索 | 15/20 | 75% |
| token 上限触发 | 3/20 | 15% |
| 非截断的 JSON 语法错误 | 0/20 | 0% |
| 完整 JSON 的 Schema 失败 | 2/20 | 10% |
| 其他失败（图片/后端/OOM/请求来源/空查询等） | 0/20 | 本归档未观测到 |

三条截断响应无法作为完整对象校验 Schema，不能将其形式化计为另外三条已验证的 Schema 错误。未出现 Markdown 包裹、缺键、额外键、字段类型错误、非法常量或重复键导致的失败。视觉内容是否正确是另一层问题，合法 JSON 不代表观察正确。

`results-0002.jsonl` 的 Baseline 有 20 行、Structured 只有 15 行。导出逻辑只写 completed 行；五条失败保存在 SQLite attempts，不在这个 JSONL 中。仅分析 JSONL 会误得出“15/15 全成功”，甚至漏掉三条最长响应与失败成本。原始记录是完整保留的，缺少的是单独 JSONL 对失败的可见性。

## 5. 全部 20 图逐样本对照

下面排名是用**保存的原查询**在原 5,000 条候选上 CPU 重放得到的首个同图描述排名，包含超出前十的诊断排名。主指标依然只用 K≤10，超出前十记零；未改查询、qrels 或评分窗口。“无匹配”表示五个目标在该查询的全部非零 BM25 结果中均不存在；“失败”表示原流程根本没有产出可用查询。

| # | 图片 ID | Baseline 排名 | Structured 排名 | 逐样本观察 |
| --- | --- | ---: | ---: | --- |
| 1 | 1056338697_4f7d7ce270 | 1 | 244 | woman car road hand 丢失 blonde/blue/top/street 等区分特征；hand 带来其他图高分 |
| 2 | 106490881_5a2dd9b7bd | 3 | 9 | boy/swim/trunks 保留，但 beach/blue/smiles 等消失；近似水边男孩描述居前 |
| 3 | 1082379191_ec1e53f996 | 11 | 失败 | 对象枚举失控到 384 token；Baseline 也因 bench 等表述未进入前十 |
| 4 | 1084040636_97d9633581 | 1 | 25 | white dog 丢失 rests/head/its 等休息姿态，退化为大类匹配 |
| 5 | 1096395242_fc69f0ae5a | 64 | 10 | Structured 去掉散文噪声后改善，但仍把手持物描述为 magnifying glass；glass 也吸引门窗描述 |
| 6 | 1107246521_d16a476380 | 8 | 4 | dog toy 减少 chain-link fence 等背景竞争，改善；也丢失 black/rope/动作 |
| 7 | 1119015538_e8e796281e | 1 | 无匹配 | 仅 puppy；五条目标都写 dog/dogs；两者无同义映射，丢失 grass/ears/runs 等补偿词 |
| 8 | 1122944218_8eb3607403 | 3 | 1 | baby flag 比散文中 sideways 等细节更契合目标；原语言 OCR 被 ASCII 分词过滤 |
| 9 | 1131800850_89c7ffd477 | 7 | 290 | 仅 dog，丢失 brown/white/water 等信息，目标区分性几乎消失 |
| 10 | 1131932671_c8d17751b3 | 1 | 1 | child bed/shorts 足以命中；Structured 未保留跳跃动作，但前十取回更多同图描述 |
| 11 | 113678030_87a6a6e42e | 19 | 126 | 未必缺动作，但 snowboard/snowboarding/skiing 与目标 snowboarder/skiers 有词形差异；两路前十均未命中 |
| 12 | 114051287_dd85625a04 | 7 | 30 | boy car shoe sock seat 等丢失 red/soccer；shoe/shoes、sock/socks 无词干映射 |
| 13 | 115684808_cb01227802 | 1 | 失败 | JSON 合法但九项 objects，Schema 拒绝；不假设放宽后会得到什么排名 |
| 14 | 1167669558_87a8a467d6 | 14 | 29 | Structured 仅 man，丢失黄衣/表情；Baseline 又受 lights/eyes/background 等背景词吸引 |
| 15 | 1174525839_7c1e6cfa86 | 74 | 78 | person ocean sand clouds 丢失跳跃、衣色与海滩细节，clouds 吸引“扬尘”描述；两路均失败 |
| 16 | 1174629344_a2e1a2bdbf | 7 | 失败 | 十七项 objects；两路街牌解读也不同，Structured 的不确定 OCR 未实际进入检索，不能归为已发生的检索降级 |
| 17 | 1220401002_3f44b1f3f7 | 37 | 9 | 保留 two children/playing/grass 后改善；wooden/bench 与 SCHOOL 仍是偏离目标的词，未保留 laughing |
| 18 | 1224851143_33bcdd299c | 1 | 失败 | 循环列举环境词到 384 token；Baseline 的 spoon/chair/mouth 能直接命中 |
| 19 | 1237985362_dbafc59280 | 77 | 失败 | blue shirt 循环到 384 token；Baseline 也被 no/sits/shirt 等吸引，原本未命中前十 |
| 20 | 1248940539_46d33ed487 | 1 | 1 | 三个孩子/沙滩/沙子等场景核心保留，两路均首位命中 |

对全部 20 图完成文本/查询/目标描述/排名审计；另目视检查 1、4、5、7、8、12、15、17 共八张本地图片。相应图片字节哈希与 GPU attempts 一致。目视支持白狗伏地、幼犬奔跑、海边跳跃、车中红色球服等特征确实存在；未将八图抽查冒充全量视觉正确性标注。第 5 图手持物外观也说明两路同写 magnifying glass 并不等于它正确，更不能为迎合 toy gun 标签而改 qrels。

### 信息丢失发生在哪里

Structured 成功十五图的查询中位数只有 6 个不同词，均值 5.2；Baseline 全部二十图分别为 35、34.35。该长度统计的队列不同，只作描述，不作为处理效应估计。Structured 中 puppy、dog、man 三条成功查询各仅一个不同词。原 JSON 已经如此简略，build_query 只是把 objects 和 visible_text 分词拼接：多数属性/动作的消失发生在模型受提示生成观察时，而不是解析器把已有动作字段错误删除。

原 Schema 没有禁止 objects 中写动作短语：第 11 图确实输出 person sitting on snowboard，第 17 图有 two children playing outdoors。因此不能笼统说“JSON 无法表达动作”。更准确的假设是，objects 导向的提示使本次许多响应只枚举名词，省略了有区分力的属性、姿态和关系。当前两路的提示和长度目标同时不同，本实验未隔离“结构化格式本身”的单一作用。

### 可核验的检索词竞争

BM25 对查询不同词做集合化，按每个匹配词的 IDF、文档词频及长度计算贡献；没有理解语义、短语或否定，也没有词形/同义映射。重复 person skiing 不会增加查询权重，重复生成却仍浪费 token。以下是**原查询原排名的分数分解**，不是删除词后重新调参的反事实实验：

- **第 1 图，信息压缩并引入竞争词。** Baseline 同图首位得分 25.0626，blonde/top/hair/street 合计贡献约 14.0271。Structured 的第一条他图结果得分 9.4922，其中 hand 贡献 6.0966；其最佳同图描述仅由 woman 匹配，得分 2.6664，排名 244。不能将 hand 本身判成视觉幻觉；问题是它对原配对描述的区分力不足。
- **第 4 图，姿态丢失。** Baseline 同图首位的 rests/head/its 合计贡献约 17.2290。Structured 只有 white dog，他图中“白狗与棕狗”的描述因 dog 文档词频和长度获得 5.5429，最佳同图描述得分 4.5796、排名 25。
- **第 7 图，语义正确也可能词面失配。** puppy 在整个候选库出现于 15 个文档，但同图五条目标均不含 puppy；查询既没有 dog，也没有奔跑、草地、耳朵等补偿词，因此没有任何同图文档得到正分。Baseline 自身也使用 puppy，但其 ears/large/grass/through 等词仍足以首位命中。
- **第 15 图，背景词/词义竞争。** 原查询 person ocean sand clouds 的首位他图结果是全地形车扬尘场景，clouds 贡献 6.0915、person 贡献 3.0820。最佳同图描述只匹配 ocean，得分 4.3223，排名 78。这支持多义背景词可能主导排名，不证明 clouds 是图中不存在的物体。
- **第 17 图，不确定文本未被自动隔离。** 原响应 visible_text=[SCHOOL]，同时 uncertainties 指出衣服文字不清楚，但 school 仍进入 query。它在五个候选文档中出现，却不在同图目标描述中。当前第一条他图结果实际主要由 wooden（5.0213）、children/playing/two 驱动，所以只能把 OCR 视为潜在污染通道，不能说 SCHOOL 已被证实导致本例名次下降。此图 Structured 本来就由 37 改善至 9。
- **第 5 图，Baseline 也有明显噪声。** Baseline 第一条他图描述中的 text/visible 合计贡献约 12.1300；这些词来自否定性套话和场景描述。Structured 去掉这类散文噪声后进入前十，但其 glass 又以 7.7422 吸引一条玻璃门描述；magnifying 在候选库词频为零。
- **第 6 图，精简可能有益。** Baseline 他图第一条由 chain/link/fence 约 19.0650 分主导；Structured dog toy 从第 8 升到第 4。这是保留反例的必要性：修复不应简单恢复所有散文或无差别增加词量。

第 8 图的原语言 visible_text 被 ASCII tokenizer 全部过滤，并非神秘“多模态 OCR 增益”；此图改善可由 baby/flag 等词解释。第 16 图中 Elephant St 与 Baseline 的另一街牌读法存在分歧，但原 Structured 已在 Schema 层失败，没有可供分析的实际检索排名。

这些是工作流记录及当前评分函数支持的机制解释。没有做删词、同义替换、补动作或放宽 Schema 后的另一次检索实验，因此不把上述解释写成已验证的改进效果。

## 6. 实际时延、token 与显存

| 全部真实响应统计 | Baseline（20） | Structured（20，含失败） |
| --- | ---: | ---: |
| latency_ms 均值 | 1,380.17 ms | 2,134.51 ms |
| latency_ms 中位数 | 1,331.25 ms | 786.85 ms |
| latency_ms 合计 | 27.603 s | 42.690 s |
| generated_tokens 合计 | 1,050 | 1,680 |
| 最大单响应 torch allocated 峰值 | 8,551.81 MiB | 8,582.61 MiB |

latency_ms 的记录范围是解码、processor、传输、生成与输出解码，排除模型加载和锁等待，不是全流程 wall time。Structured 成功十五条的平均 latency_ms 约 655.97 ms，但这是排除了失败后的条件统计，不能据此宣称当前 Structured 整体加速。五条失败消耗 32.851 s，占该路响应 latency_ms 合计约 77.0%；三条截断响应占 Structured 生成 token 的 68.6%。

Baseline 第 1、第 3 图分别记录一次模型加载，耗时约 6.394 s、4.492 s，对应两个进程的首个新增调用。其余调用复用。两路顺序和冷加载不对称，也不支持简单拿分支 wall time 比作推理加速比。summary-0002 的 elapsed_seconds 只对应第二个 session；其 18 条新增调用耗时不能与累计 20 条响应时延混用。

显存数值是包含驻留模型、排除加载峰值的 torch allocated MiB，不是 nvidia-smi 总显存、峰值 reserved 或通用 GPU 容量保证。本报告不把这 20 图配置下的测量扩展为所有图片/配置都能容纳模型。

## 7. 最小修复建议与 V1.1 决策

下面仅是建议；本轮没有实施生成、查询或检索行为变更。

| 优先级 | 最小动作 | 依据与验收边界 |
| --- | --- | --- |
| P0：记录可见性 | 今后额外导出逐 attempt 状态、精确错误、raw_response 引用及独立 syntax/schema/token 类别；保留成功导出与原始 SQLite | 本轮离线审计已补足可见性；生产导出改动另行实施，不重写 V1.0 记录 |
| P1：生成合约 | 独立提示版本仍沿用三个字段和严格 Schema；约束至少于八项、例如最多六个有信息量的短语，禁止穷举/重复；先保持 384 token | 两条超项与三条枚举/循环失败有直接证据。不先用更大 token 上限延长循环；仅在独立实验验证，不能宣称已修复 |
| P1：保留视觉核心 | 在同一 schema 的 objects 字符串中表达显著对象＋可见属性/动作/关系，避免只写 puppy/dog/man；不强迫填满对象槽位 | 原格式本来允许短语，改动可保持接口。动作/属性应该来自图片，绝不从本次 qrels 或正确 caption 自动抽词填入 |
| P2：查询噪声 | 在独立版本审查不确定 OCR、否定套话、背景细节及词形/同义失配；一次只评估一种规则 | 先修结构化表示；不要同轮又换检索器、扩词、改变候选池，造成无法归因。OCR 分歧需和图片核对，不盲信模型 |
| P3：受限恢复 | 如确有必要，再评估有界重试或显式回退，并记录额外调用/成本 | 回退到 Baseline 会改变方法定义；必须单列“带回退策略”，不能把回退成功当成 Structured 自身成功，不能改写本轮五条失败 |

**值得做独立 V1.1，但定位应是可靠性与表示修复，不是扩大规模以寻找更好分数。** 建议顺序：

1. 先用保存响应和合成 fixture 做 CPU 合约检查；原 token 截断仍拒绝，不用字符串补括号伪造合格观察。本轮新增审计用例覆盖了 JSON 合法但 Schema 失败、完整 JSON 达上限仍被拒绝、非截断语法错、有效 Schema 却无支持词、重复键/NaN，以及失败计零与两种 Recall/Hit 的区分。
2. 本次 test 前二十图已经被用于诊断，应标为开发诊断样本；不能在其上反复调 prompt 再把同组成绩当作独立测试提升。后续先在原 dev split 的预先固定图片子集上开发，使用原 dev 全部 5,000 描述候选与原 qrels；与此次 test 结果分开报告。新增图片获取与真实 GPU 运行均需开发者另行授权。
3. 先只评估生成合约改动，再单独加入属性/动作保留，必要时再评估 OCR/查询规则。对每个版本冻结 prompt/config/源码/模型哈希、图片 ID、候选/qrels、预算和停止条件，创建全新输出目录。
4. 主指标继续用全部预定图片计分、失败计零；同时报告严格 JSON 合法率、Schema/流程成功率、共同成功配对结果、全部调用和失败成本。可靠性验收可预设严格 Schema/流程 20/20；未达到先停止扩容。任何检索改善都必须由实际运行产生。
5. 配置锁定后，再预先指定未用于开发的 test 图片进行独立比较。若重复使用已有 20 图，仅称回归检查。小规模结果无论高低均完整保存，达到事先约定的可靠性/成本要求后才考虑 100→1000。

本报告不写“V1.1 将提升多少”。小样本、词面代理标签、跨图相似描述和可能的预训练污染都限制泛化。更多候选文档不等于更多真实图片推理；5,000 描述不能写成 5,000 张 GPU 测试图片，40 次调用也不能写成 40 张独立图片。

## 8. 实施、验证与边界

**Implemented：** 新增只读 CPU 审计脚本、审计分类测试、精简证据归档与本报告；算法、提示词和实验原件保持原样。

**Files Changed：** [审计脚本](../scripts/audit_flickr8k_gpu.py)、[审计测试](../tests/test_flickr8k_gpu_audit.py)、本报告，以及 `experiments/flickr8k_v1/gpu-3ec02ca-20261020/` 下五份精简归档（四份分析/冻结记录和一份最终验证记录）。完整提取原件、原始响应及分析数据留在 outputs，不进入 Git 精简归档。

**Validation：**

```powershell
.\.venv\Scripts\python.exe scripts\audit_flickr8k_gpu.py --run outputs\flickr8k-real-3ec02ca-01-audit-20261020-01\source\outputs\flickr8k-real-3ec02ca-01 --archive outputs\flickr8k-real-3ec02ca-01.tar.gz --output outputs\flickr8k-real-3ec02ca-01-audit-20261020-01\analysis
.\.venv\Scripts\python.exe -m pytest tests\test_flickr8k_gpu_audit.py tests\test_flickr8k.py -q --basetemp outputs\pytest-flickr8k-gpu-audit-20261020-01
```

审计首次执行成功：28 文件匹配压缩包；两库完整性正常；全部源码/输入快照/控制字段哈希匹配；comparison、summary、SQLite、成功 JSONL 一致；35 条原查询的前十 ID 与分数复现；主指标及逐样本指标一致；原件前后哈希不变；GPU 新调用为零。最初针对性测试 **17 passed in 1.24s**。以上输出路径已存在，再次运行应指定新的输出目录，脚本拒绝覆盖。

最终以新目录 `analysis-validated` 再执行一次审计，结果一致；再次执行针对性测试为 **17 passed in 0.79s**，JUnit 保存在 outputs。另实际验证已有输出目录被拒绝覆盖、20 张本地图片哈希与 GPU 记录一致、原件最终哈希不变、文档本地链接可解析及 Git diff --check。命令与结果见同目录 `validation.json`；不把此前 CPU/Mock 验收的历史通过数当作本轮执行结果。

**Limitations：** 20 图仅作小规模验收；八图目视抽查不构成完整视觉质量标注。两种提示和输出长度不同，不是纯格式单变量对照。五描述相关，未标注跨图的语义相关性，不能做事实核验或准确率结论。没有开展生成/检索行为修复或新 GPU 实验，任何建议的收益均未验证。

**Next Step：** 开发者审阅本负结果及修复优先级，再决定是否批准独立 V1.1；本轮不触发 GPU 运行。
