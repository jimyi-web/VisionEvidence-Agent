# T04-5 人工审核归档交付记录

2026-10-18，用户在 AI 辅助阅读后亲自确认全部标签。本次完成状态与历史未填写模板分离，详见 [审核报告](../experiments/t04_v1/human-review-20261018-01/report.md)。

新增独立目录 `experiments/t04_v1/human-review-20261018-01/`，包含确认来源、120 条标签及证据全文、统计、审核报告、完整性 manifest。10 样本 × 12 文档；相关 11、不相关 109、不确定 0；与冻结 qrels 一致 120/120。这不是独立盲审或多标注者一致性结果。

新增 [只读校验器](../scripts/validate_t04_human_review.py)、数据完整性测试与 [T04-6 候选设计](t04_6_design.md)。原始复核模板、T04-5 审计报告、冻结 qrels、历史实验、查询/评测实现和 Qwen 后端均保留不变。

本地提交将同时收录此前尚未提交的 T04-5 审计准备材料与本次人工审核归档/方案，使历史审计链接和测试可完整复现。不推送 GitHub；推送须另行取得用户批准。

## 验证

- `.\.venv\Scripts\python.exe scripts/validate_t04_human_review.py`：通过；120 对覆盖完整、标签/原文/来源/哈希/统计一致，差异 0。
- `.\.venv\Scripts\python.exe -m pytest --basetemp outputs/pytest-t045-human-full-01`：**332 passed（32.16s）**，包括前轮 322 项和本轮新增 10 项。使用已有项目 Python 3.12.7；通过临时 GIT_CONFIG_* 环境项设置该项目 safe.directory，未修改全局 Git 配置。
- 原冻结输入副本逐字节一致；原 manifest 中全部 21 个源码文件哈希不变；`git diff --check` 通过。
- 本轮没有运行 GPU、修改 Qwen 后端或调整测试数据/查询算法/评测规则。审核方法的局限及候选改进尚未验证的收益已明确记录。
- 暂存检查发现 Windows 归档的 CRLF 被识别为行尾空白；保留归档字节，在 `.gitattributes` 明确 CRLF 检查规则（仍检查真实尾随空格），并对两份被哈希的历史文档禁止行尾转换，防止后续 checkout 破坏哈希。历史文件内容未改动。
