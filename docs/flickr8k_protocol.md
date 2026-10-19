# Flickr8k V1.0：image-to-caption retrieval proxy 协议

冻结日期：2026-10-19。优先交付 20 张真实图片的可运行输入、两路实验和 GPU 命令；随后扩展至 100/1000 图。8,000 图保持三个原划分分别运行。本协议不定义谣言识别、真实性分类或外部事实核验。

## 来源、版本、许可与实际下载

用户指定来源：[intro/flickr8k](https://huggingface.co/datasets/intro/flickr8k)，是 Flickr8k 的第三方 imagefolder 重打包，并非原始作者新发布的事实核验数据。[数据卡](https://huggingface.co/datasets/intro/flickr8k/blob/c7a02cb487da0eca729e133194b0f842cb578c08/README.md) 声明 CC0-1.0；[原作者项目页](https://hockenmaier.cs.illinois.edu/Framing_Image_Description/KCCA.html) 则说明图片版权属于原权利人，使用应尊重 Flickr 条款，面向非商业研究/教育。因此记录 provider_license_claim=cc0-1.0、original_photo_rights_verified=false，不宣称所有照片均逐图核实为公有领域。本地科研使用，不在 Git/公开演示包中再分发图片或整套描述。

数据卡声明文件结构为 train/、dev/、test/ 各有 JPEG 和 metadata.csv；字段严格为 file_name、split、caption_0…caption_4，image 由 imagefolder 根据 file_name 推断。划分声称来自 Flickr_8k.trainImages.txt/devImages.txt/testImages.txt，描述来自 Flickr8k.token.txt 的 #0…#4。本次按镜像字段原样保留，并核对 6000/1000/1000 图、全局文件名 ID 无交叉；未另外取得原始划分列表做逐文件独立认证。其原始 8,092 组中无明确划分的 92 组不加入这 8,000 图。

固定 HF Commit：`c7a02cb487da0eca729e133194b0f842cb578c08`。直接访问 `/resolve/<commit>/<split>/<filename>`，不调用会获取全部图片的 load_dataset。下载器获取数据卡和三个小体积 CSV，图片只选 **test 文件名 ASCII 升序前 20 个 ID**，独立于 caption、模型响应和分数。本次已实际获取 20 张，JPEG 解码均通过，图片总字节 2,421,770；未下载仓库页面所列约 1.11 GB 全集。每文件 URL、SHA256、字节、时间及异常都有记录。

## 输入与防泄漏边界

离线适配一次生成所有原划分，即使磁盘只有 20 张图片。每个 split 各有：

| 文件 | 内容与用途 |
| --- | --- |
| samples.jsonl | 一个图片 ID 一条查询样本，保留原 split、相对图片路径、源版本；claim 只含通用任务文本，无人工描述 |
| captions.jsonl | 图片 ID、原 split、原样保存的五条人工描述（包括原空格），仅用于审计 |
| corpus.jsonl | 每条描述一个文档，ID=flickr8k:原文件名#0…#4，保留 image_id/caption_index/split/source |
| qrels.jsonl | 每图恰好五个同图描述 ID；kind=caption_pair_proxy，明确是原配对关联代理标签 |

VLMRequest 只含本地图片、固定 prompt、request_id 和大小限制。原 filename 和 request_id 不拼入 prompt；Qwen 处理器只接收解码后的像素与 prompt，标签/人工描述/候选列表均不进入模型输入。BM25 只索引 corpus.text，文档 ID/metadata 不参与分词。qrels 只用于输出后评分。测试覆盖更换所有参考描述后两路生成查询完全不变。

人工描述属于检索目标；它出现在候选库中是任务定义，不是模型输入。严禁把五条描述抽出的关键词、正确 caption、测试答案、图片文件名语义或 qrels 带入 prompt/query。Mock 固定模板不看像素，不用参考描述生成模拟观察。

## 候选、qrels 与负例

test 候选库从开始就固定为 **1,000 图 × 5 描述 = 5,000 文档**。20/100/1000 阶段只扩大查询前缀，不缩小/扩大候选库，不依据检索结果选择图片或负例。dev 也为 5,000 文档，train 为 30,000 文档；每路比较严格在同一原 split 内进行。

五个同图描述 ID 全为 proxy positives；同 split 其他图片的描述全是 assumed negatives。无额外 hard-negative 挖掘、随机负例重采样、同义扩展、答案 caption 合并、候选重排或基于分数的样本剔除。本轮不导入原论文可能另行提供的跨图人工相关性判断，也不声称配对标签穷尽语义相关性。

同图五描述高度相关，不能算五个独立样本。跨图可能出现相同/近似描述或相似场景，导致代理 false negatives；相同文本保留为不同原始文档 ID，不为提高指标去重/合并 qrels。BM25 同分按文档 ID 排，可能产生任意的配对命中差异。以图片为统计单位；20 图只做验收，无显著性/泛化结论。模型可能见过 Flickr8k，预训练污染无法排除。

## 两路受控比较

- Baseline：现有英文观察约束提示追加“单段英文散文、最多 60 词”，版本 concise_visual_description_v1；原始 response.description 整体作为 BM25 查询，不做参考描述引导的重写。
- Structured：现有 structured_visual_v1，严格 JSON objects/visible_text/uncertainties；只用 objects+visible_text 构造查询，uncertainties 排除。

相同图片前缀、候选库 SHA256、qrels、BM25(k1=1.5,b=0.75,ASCII tokenizer)、K=1/5/10、模型文件、384 token 上限、图片处理配置、greedy/seed=0。完整 prompt/配置/源码/模型哈希均保留。两种 prompt 的长度/格式目标不同，这是“受约束表示工作流”的比较，不是隔离 JSON 语法这一单变量的实验。

同一进程共享一个 Qwen 实例；先 Baseline 后 Structured，首个真实调用含冷加载，不能把两路总 wall time 直接解释为表示方案的推理加速比。每请求保留首次加载/复用与时间/显存元数据。真实散文达到 token 上限也失败，与 Structured 使用相同截断拒绝政策；Structured 额外要求严格 Schema 合格。没有自动 JSON 修复、caption 回退或只统计成功子集的主结果。

## 指标和规模口径

主分母是所选图片数，两路相同；生成、解析、查询或图片失败记零，保留错误/原响应。报告：

- HitRate@K（常规 image-to-text R@K）：前 K 中是否至少有一个同图描述；每图 0 或 1 后宏平均。
- CaptionRecall@K：前 K 中同图描述数÷5；不是上一项。K=1 时最大 0.2。
- MRR@K：第一个同图描述的倒数排名，前 K 没有则为零。

| 阶段 | 候选描述文档 | 图片查询数/每路 | 两路查询槽位 | 无重试时计划真实 VLM 调用 | 独立图片 |
| --- | ---: | ---: | ---: | ---: | ---: |
| test smoke | 5,000 | 20 | 40 | 40 | 20 |
| test 扩容 | 5,000 | 100 | 200 | 200 | 100 |
| test 全划分 | 5,000 | 1,000 | 2,000 | 2,000 | 1,000 |
| 全部原划分 | train 30,000；dev/test 各 5,000 | 总计 8,000 | 16,000 | 16,000 | 8,000 |

本轮真实调用 **0**，本地为真实 JPEG + Mock 生成。实际真实调用重试会增加，独立图片按响应 SHA256 去重。8,000 阶段分别跑 train/dev/test，分别报告；如另报加权汇总须明确权重 6000/1000/1000，不能称为 8,000 条独立 held-out test 或单个 40,000 文档候选池。train 用作描述性规模验收，不用于在 test 上调参。

## 断点与扩容

复用 T05 SQLite attempts/samples/sessions、OS 锁、图片哈希、累计重试预算、成功缓存和独占 session 导出。两路各自缓存，Structured 失败不会使成功 Baseline 重做。默认每图每路累计 1 次，允许显式提高 --max-attempts 重试；模型/源码/配置/候选变化要求新 run。之前 T05 归档原样保留。

**扩大图片下载时保留同一个适配包，不重新 adapt**：samples 已包含原 split 所有 ID，未来文件出现在既定路径即可。adapter manifest 的 available_images 是准备时快照，实际请求重新校验文件。重新生成 manifest 会改变比较身份并要求新目录。Windows 路径写为 POSIX 相对形式；按相同目录树复制可在 Linux 使用，或在 Linux 重新离线 adapt 后使用新 run。

GPU 驱动挂起需要外层 timeout，生成 token 上限不是硬时限；中断边界可能重做最后一个尚未提交结果，保持 T05 at-least-once 语义。
