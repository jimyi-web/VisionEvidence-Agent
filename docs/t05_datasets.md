# T05 公开数据选择与接入结论

核查日期：2026-10-19。**V1.0 首先接入 VERITE 小子集元数据；长期万级查询优先考虑 NewsCLIPpings 的指定 split；VisualNews 留作候选语料来源。** 图片获取和人工检索相关性标注是独立工作。

## 比较

| 数据集 | 规模、模态 | 获取方式与下载量 | 许可、限制 | 检索相关性与兼容性 |
| --- | --- | --- | --- | --- |
| VisualNews | 官方介绍超过百万图文条目；图、caption、文章、元数据 | [官方仓库](https://github.com/FuxiaoLiu/VisualNews-Repository) 指向 [Rice 目录](https://www.cs.rice.edu/~vo9/visualnews/)；当前列 origin.tar 91G、articles.tar.gz 4.2G，非本地实测字节。本轮不下载 | 上述仓库/目录未找到足以明确覆盖第三方图片/文章再分发的统一数据许可；不能由公开可下载推定可再分发 | 图文关联不是独立核验事实或检索 qrels。适合候选语料，须控制同文/同图泄漏 |
| NewsCLIPpings | [论文 Table 1](https://arxiv.org/html/2104.05893v2)：Total/Unique train/val/test 为 816,922 / 85,609 / 85,752；Merged/Balanced 为 71,072 / 7,024 / 7,264 配对 | [官方索引与脚本](https://github.com/g-luo/news_clippings)；id/image_id 连接 VisualNews data.json 和图片。完整下载脚本还含大 embedding，不能直接执行来获取小子集 | 官方根目录/README 未核实适用于全部数据的统一再分发授权；还继承新闻来源权利，扩容前补充确认 | falsified 和构造 similarity_score 都不是相关性真值。大型 split 支持万级查询；Merged/Balanced test 仅 7,264，不能单独称万级测试 |
| VERITE | [论文](https://arxiv.org/html/2304.14133v3)/README 描述 1,000 对：338 true、338 miscaptioned、324 out-of-context。**本次官方 CSV 实际 1,001 对：338/338/325；338 article groups、663 图片路径** | [官方 CSV](https://github.com/stevejpapad/image-text-verification/tree/master/VERITE)。[2026-07 更新](https://raw.githubusercontent.com/stevejpapad/image-text-verification/master/README.md) 称旧 URL 大量失效，改提供 HF 图片。CSV+articles+LICENSE+README 共下载 367,784 字节；[HF 页面](https://huggingface.co/datasets/stefpapad/VERITE) 列总大小 65.3 MB | 仓库 Apache-2.0；README 限科研。HF 需机构/大学邮箱审核、非商业科研、禁止再分发，并保留第三方权利责任 | 作者分类标签有事实核查文章背景，但无本项目检索 qrels；不把正确 caption 当现成证据。适合小批接入，不能单独支撑万级独立查询 |

选择 VERITE 是为了快速验证真实字段与来源，不表示已有图片闭环或有效检索基准。官方当前 CSV 与论文计数不一致，按实际字节记录，不静默删掉额外一行；论文版本复现应另取明确历史版本。

## 本次实际接入

只下载官方 GitHub 的两个 CSV、README、LICENSE，没有图片、embedding、大数据包或权重。API 403 限流和 GitHub 连接失败后，显式采用 master 的内容快照，**上游 Commit 未知**：

`snapshot-sha256:453fa6f76617a22edd8af7d568d8595a15f9a9fe700fb72dba800a6fe6c78254`

定义为 SHA256(VERITE.csv 原始字节 + VERITE_articles.csv 原始字节)。[来源记录](../experiments/t05_v1/acceptance-20261019/verite-source.json) 保存 URL 和各文件哈希；不声称不同 branch 请求属于同一上游 Commit。

前 12 行生成 12 个 sample_id、8 个图片路径、4 个文章组，三类各 4 条。可用图片 0、证据库 0、真实 VLM 调用 0、qrels=null。[适配 manifest](../experiments/t05_v1/acceptance-20261019/verite-subset.json)。缺图不替换成白像素，也不按可下载性挑更容易的样本。

pair_label、文章 URL 只是元数据。VLM 只看图片和固定 prompt，不接收 caption、标签、true_caption 或文章。当前做观察驱动检索，尚不做图文分类；引入 claim-assisted 查询需要另立实验条件。

## 复现

```powershell
.\.venv\Scripts\python.exe scripts/fetch_verite_metadata.py --output data/raw/verite-metadata-new --ref master
$veriteSource = Get-Content data/raw/verite-metadata-new/source.json -Raw | ConvertFrom-Json
.\.venv\Scripts\python.exe -m visionevidence.datasets.verite --prepared data/raw/verite-metadata-new/VERITE.csv --articles data/raw/verite-metadata-new/VERITE_articles.csv --image-root data/raw/verite-metadata-new --output outputs/verite-subset-new --revision $veriteSource.revision --limit 12
```

默认不加 --ref 时先解析官方 Commit，再按 SHA 下载；--revision 可指定已知的 40 位 Commit。--ref master 的新结果可能变化，必须对照归档哈希。每个响应上限 2 MiB，HTTPS 校验开启。脚本不下载图片。

取得合法访问权后，人工将图片放到 image-root 下的 images/true_ID.jpg、images/false_ID.jpg。不要公开再分发 HF 数据。图片哈希由批量运行记录。保留 group_id，未来切分按文章组与图片去重，不能把同组真假 caption 随机分入 train/test。本子集作为 benchmark，不用于调参。

## GPU 预算（假设，非实测）

历史仅 1 图 6 次调用，不能代表新闻图片吞吐。暂按 **10 秒/调用** 规划：100 次约 16.7 分钟、1,000 次约 2.8 小时、10,000 次约 27.8 小时；另假设 20% 重试则约 20 分钟、3.3 小时、33.3 小时。不含下载、模型哈希、加载、排队，也不保证显存容量。

费用=人工确认的小时单价×实际 GPU 时长，本轮未承诺 AutoDL 价格。先人工验收 8–12 条，测量 JSON 合格率、重试、时延与显存，再估算。当前缓存按 sample，不跨不同 caption 的同图复用，因此独立图片数可能少于推理调用数。万级 GPU 全量运行留待后续，本轮不启动 AutoDL。
