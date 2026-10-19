# Flickr8k 真实图片接入与本地验收报告

日期：2026-10-19。结论：真实 20 图可获取，数据/两路 proxy/记录链路在 CPU/Mock 下运行成功；真实 Qwen GPU 结果仍待开发者执行，不报告真实检索提升或谣言识别准确率。

## 已实现

复用 T05 的 SQLite 调度、断点/缓存/累计重试、来源校验、模型复用、结构化解析和 BM25，仅增加 Baseline 散文分支与显式 caption_pair_proxy 任务，未重构核心检索算法。批量入口默认仍为原 evidence_retrieval/structured。proxy 配对标签在普通事实证据相关性任务中拒绝使用。

Flickr8k 下载器按固定 HF Commit 获取三份 CSV 和少量指定图片，支持失败记录与 --resume；离线适配保持全局图片 ID、五条原描述与三个原 split。样本不含人工描述，候选文本与 qrels 分文件保存，Linux 可读取 POSIX 相对路径。比较入口在一个进程复用模型，固定候选/qrels/模型/配置，记录两路原响应、失败和一致分母指标。

## 真实获取与三类规模

版本：intro/flickr8k @ c7a02cb487da0eca729e133194b0f842cb578c08。按 test 文件名升序选前 20 张，未按 caption/分数选样。本轮元数据共 2,515,706 字节（含 README），20 张真实 JPEG 共 2,421,770 字节；全部实际解码通过，并目视检查了一张下载图片。不下载完整约 1.11 GB 镜像、额外 92 组图片或模型权重。[来源收据](../experiments/flickr8k_v1/acceptance-20261019/source.json)、[下载记录](../experiments/flickr8k_v1/acceptance-20261019/download-0001.json)、[逐图 SHA256](../experiments/flickr8k_v1/acceptance-20261019/image-inventory.json)。

| 项目 | 候选描述文档数 | 查询图片/每路 | 两路查询槽位 | 真实 VLM 图片推理调用 |
| --- | ---: | ---: | ---: | ---: |
| 本轮 CPU/Mock | 5,000 | 20 | 40 | 0 |
| 后续 GPU 计划（未执行） | 5,000 | 20 | 40 | 无重试预计 40 次，独立图片 20 |

适配器保留 train 6,000 图/30,000 描述、dev 1,000 图/5,000 描述、test 1,000 图/5,000 描述。本地只存在 test 前 20 张图片；元数据覆盖 8,000 不能冒充已下载/推理 8,000 图。[适配 manifest](../experiments/flickr8k_v1/acceptance-20261019/adapter-manifest.json)。

许可核查区分镜像 CC0 声明与原图片权利；本轮不宣称逐图版权验证，不公开提交图片/全部人工描述。原 split 来自镜像声称的原始列表映射，计数/ID/字段已验证，未独立获取原始划分文件逐项认证。详见 [来源与协议](flickr8k_protocol.md)。

## CPU/Mock 实际运行

```powershell
.\.venv\Scripts\python.exe -m visionevidence.datasets.flickr8k download --output data/raw/flickr8k-20261019 --revision c7a02cb487da0eca729e133194b0f842cb578c08 --split test --limit 20
.\.venv\Scripts\python.exe -m visionevidence.datasets.flickr8k adapt --input-root data/raw/flickr8k-20261019 --output data/processed/flickr8k-proxy-v1
.\.venv\Scripts\python.exe -m visionevidence.caption_experiment --dataset data/processed/flickr8k-proxy-v1/test --output outputs/flickr8k-mock-20-01 --mode mock --limit 20 --batch-size 4
# 同命令增加 --resume，再次成功；两路 attempts 仍各为 20。
```

两路 20/20 完成，无错误，首次累计 40 个 Mock 调用，续跑新增尝试 0。调度循环耗时 Baseline 0.220 秒、Structured 0.148 秒；该字段不含启动/建索引/结果导出，不能当作端到端或真实 GPU 吞吐，更不能声称 Structured 加速。Mock 固定模板不看像素，任何 proxy 分数仅是程序输出，不能解释为真实视觉检索质量。

全量逐图/逐描述结果在被忽略的 outputs 和 data/processed；Git 仅归档来源、hash、逐查询 proxy 指标与运行/测试摘要，不提交 JPEG、原始描述库、SQLite 或模型。[首次比较](../experiments/flickr8k_v1/acceptance-20261019/mock-comparison-0001.json)、[缓存续跑](../experiments/flickr8k_v1/acceptance-20261019/mock-comparison-0002.json)、[本地产物哈希](../experiments/flickr8k_v1/acceptance-20261019/local-artifacts.json)。

运行时父 Commit 为 e097148，工作区 dirty=true，具体实现由 manifest 源码哈希确认；不把父 Commit 冒充包含本轮改动的最终 Commit。全部文件冻结后创建本地 Commit，不推送。旧 T03/T04/T05 实验与标注原件未修改。

## 测试与验证

- 新增 19 项测试：原 ID/五描述/划分/空格保留，异常字段与路径/跨 split ID 拒绝，离线版本/源哈希，小图下载故障后只重取失败文件，两路相同候选/配置，模型实例复用、逐路缓存、缺图恢复、Schema 错误计零、代理标签隔离、截断拒绝、HitRate 与五目标 Recall 区分、CLI，以及改变参考描述不改变生成查询。
- 完整命令：`.\.venv\Scripts\python.exe -m pytest --basetemp outputs/pytest-flickr8k-full-01 --junitxml=outputs/pytest-flickr8k-full-01.xml`，**371 passed，48.13 秒**；已有 352 项和新增 19 项全部通过。进程级 GIT_CONFIG_* 指定 safe.directory，未修改全局 Git。
- 重点测试先运行 35 项全部通过。新增测试均用合成图片/CPU doubles，不冒充真实 GPU 故障测量。真实 JPEG 下载/解码和 CPU/Mock 20 图 smoke 为另行实际运行。
- `.\.venv\Scripts\python.exe -m pip check`：No broken requirements found。没有新增依赖或改变模型后端/权重。
- git diff --check、Markdown 本地链接和归档完整性检查通过。[JUnit 报告](../experiments/flickr8k_v1/acceptance-20261019/pytest.xml)、[机器验证摘要](../experiments/flickr8k_v1/acceptance-20261019/validation.json)。

## 局限与下一步

这是 image-to-caption retrieval proxy：同图五描述是配对目标，其他图描述是未经穷尽人工判断的代理负例；相似图/重复描述导致 false negatives 与同分排名偏差。不能等同外部证据相关性、事实核验或谣言识别。20 图不支持泛化/显著性结论；两种 prompt 与输出格式不同，不单独归因于 JSON 语法。预训练污染未排除。

GPU 仍逐图生成，batch-size 是调度批大小；只有真实 GPU 执行后才能得到 JSON 合格率、观察质量、时延/显存与两路真实 proxy 结果。资源暂按两次调用/图、10 秒/调用估算：20 图约 6.7 分钟，100 图约 33 分钟，1000 图约 5.6 小时，8000 图约 44.4 小时，均非本轮实测。

下一步：开发者手动传输 20 图和元数据到已有 GPU 环境，执行 [runbook 中的 20 图命令](flickr8k_runbook.md)，完整保留成功和失败；再按实测成本决定 100→1000，8,000 图按原 train/dev/test 分开报告。没有自行启动 AutoDL，也没有执行 git push。
