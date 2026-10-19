# T05-A / V1.0 本地验收报告

日期：2026-10-19，Windows / 项目 Python 3.12.7。结论：可恢复批量基础设施、CPU/Mock 万级查询、真实公开元数据小子集已验收；真实多图结构化 GPU 验收与真实检索质量仍未完成。没有启动 AutoDL、下载权重或执行新的 GPU 推理。

## 已实现

新增批量入口，复用原 Qwen3VLBackend、observe_and_retrieve、parse_observation、BM25Index，不改冻结 T04 协议/结果或原算法。SQLite 保存样本、attempts、qrels、sessions；支持分阶段前缀、累计重试、成功缓存、OS 锁和中断恢复。自然语言/严格 JSON 冲突通过固定结构化 prompt、严格拒绝与错误归档处理，未声称模型必定输出有效 JSON。

独立 qrels 才计算 Recall/MRR；配对分类标签隔离。成功、失败均保留明确来源，Mock 不标成真实推理，所有观察仍未核验。输入/模型/源码 SHA256、规范化配置、Git Commit/dirty、环境与运行时间均可追踪。详细契约见 [流水线协议](t05_pipeline.md)。

## 实测规模与结果

同一个 outputs/t05-mock-scale-20261019-01 目录，完整输入 10,000 行，共用 1 个白像素 synthetic fixture；固定证据库 5 文档，所有结果均 Mock。三阶段没有重跑已成功样本：

| session | 证据文档数 | 累计评测查询 | 新处理查询 | 累计 attempts | 真实 VLM 图片调用 | 调度处理耗时 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 5 | 100 | 100 | 100 | 0 | 0.711 秒 |
| 2 | 5 | 1,000 | 900 | 1,000 | 0 | 5.977 秒 |
| 3 | 5 | 10,000 | 9,000 | 10,000 | 0 | 66.042 秒 |

10,000/10,000 成功，无错误，未提供 qrels，retrieval_metrics=null。耗时只含该 session 调度/缓存图片校验/推理检索记录循环，不含启动、索引/入库和末尾导出，不能当完整端到端吞吐或 GPU 性能。不能据此声称万张图片视觉理解、万文档检索容量、真实检索质量或分类效果。

[三阶段归档](../experiments/t05_v1/acceptance-20261019/mock-summary-0003.json) 保留各 summary 与原始 manifest。运行基于 a4429a338e81cb671738750a99147c56c3be821d 的未提交修改（dirty=true），精确实现以 manifest 的源码哈希识别，不将父 Commit 冒充最终代码 Commit。初始 manifest 记录的已安装包元数据仍为 0.1.0；运行期间完成 editable 包升级至 1.0.0，未改本次流水线源码；保留原记录，不回填版本。本轮最终 CLI 实测为 visionevidence 1.0.0。

## 失败恢复

另建 outputs/t05-recovery-20261019-01：两个样本、一张有效 fixture、一张初始缺失，max_attempts=1。首次结果 completed=1、failed=1，错误 ValueError 保留。补齐缺图，显式 --resume 并提高累计预算至 2：completed=2，累计 attempts=3；原成功样本仍 1 次，修复样本累计 2 次。旧 summary 不覆盖，历史错误仍计 1。[恢复前](../experiments/t05_v1/acceptance-20261019/recovery-summary-0001.json)、[恢复后](../experiments/t05_v1/acceptance-20261019/recovery-summary-0002.json)。

pytest 另覆盖 KeyboardInterrupt 留下 running attempt 后恢复为 interrupted、原始散文响应拒绝/重试、缓存变化拒绝、同目录并发锁、CLI 与来源矛盾拒绝。上述都是 CPU 合成测试，不冒充 CUDA 故障实测。

## 真实数据适配

成功获取 367,784 字节的官方 VERITE 元数据/许可文件，内容快照固定，未取得上游 Commit。实际 CSV 为 1,001 行，与论文 1,000 有差异，原样保留。适配前 12 行、三类各 4、8 图片路径，本地图片 0、证据文档 0、真实 VLM 调用 0。下载/标签/许可边界和选择结论见 [数据比较](t05_datasets.md)。

官方 2026-07 更新和 HF 访问门槛已核对；图片需机构邮箱申请，非商业科研、禁止再分发。本轮不自动申请或接受协议。不存在真实 VERITE 图像推理结果，也没有把分类标签转成检索 qrels。

## 验证与文件

- 完整测试：`.\.venv\Scripts\python.exe -m pytest --basetemp outputs/pytest-t05-full-03`，**352 passed，49.91 秒**。沿用原 332 项，新增批量/适配 20 项。测试进程临时 GIT_CONFIG_* 指定本项目 safe.directory，未改全局配置。
- 初轮全测 4 失败/345 通过，均因 Git dubious ownership；进程级配置后 349 项全部通过，再补充 CLI/锁/快照测试后得到最终 352。首轮新测试发现 Mock 输出缺 [MOCK] 标志，已修正合成响应，未放宽契约。
- `.\.venv\Scripts\python.exe -m pip check`：No broken requirements found。editable 安装 1.0.0 完成，运行依赖未新增。构建依赖在隔离环境按 pyproject 已固定版本安装。
- `.\.venv\Scripts\python.exe scripts/validate_t04_human_review.py`：120 对完整，差异 0。
- `.\.venv\Scripts\python.exe scripts/summarize_t03_b5.py outputs/t03-b5-real --expected-commit c34ebe6d6cd73345778ae7b2292f1fda301e1d4a`：原 GPU 记录哈希/指标校验通过。仅重算历史文件，未执行 GPU。
- CPU 三阶段与缺图修复 smoke 实际运行并断言通过；git diff --check 通过。

关键改动：batch.py、datasets/verite.py、两个数据准备脚本、观察 JSON Schema、T05 GPU 配置、20 项测试、README/目标更新、本报告与协议/数据报告。小型机器归档在 experiments/t05_v1/acceptance-20261019；大 SQLite/逐条结果和原 CSV 留在被忽略的 outputs/data/raw，见 [本地产物哈希清单](../experiments/t05_v1/acceptance-20261019/local-artifacts.json)。这些大产物和官方元数据原件应独立备份，未随 Git 提交。

## 限制与下一步

这是可展示的 V1.0 研究流水线，不是完整自动事实核验系统。没有真实公开数据检索 qrels、最终 verifier、稠密检索或多 GPU；GPU 仍逐图，缓存按样本，不跨 caption 的同图共享；证据库在内存。CPU 验收未量测峰值 RAM。驱动卡死靠外层 timeout，强制终止边界可能重复最后一次推理。

下一步由开发者取得 VERITE 图片访问权，准备独立有来源的证据库，手动执行 [12 条 GPU 验收命令](t05_pipeline.md)。先验收 JSON 合格率、失败/重试与成本，再决定人工检索标注和万级 NewsCLIPpings 指定 split；不在本轮启动或承诺万级 GPU 结果。
