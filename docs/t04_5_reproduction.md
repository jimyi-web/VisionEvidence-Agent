# T04-5 复现说明

在项目根目录使用已有 `.venv`，无需安装依赖、网络或 GPU。

```powershell
.\.venv\Scripts\python.exe scripts/audit_t04_5.py --archive experiments/t04_v1/run-20261017-01 --output outputs/t04-5-audit-replay-01
# 当前环境的仓库所有权与执行用户不一致；供测试子进程读取 Git 信息。
# 仅在未设置其他 GIT_CONFIG_* 覆盖项的新 PowerShell 进程中使用。
$env:GIT_CONFIG_COUNT = '1'
$env:GIT_CONFIG_KEY_0 = 'safe.directory'
$env:GIT_CONFIG_VALUE_0 = 'D:/Research/VisionEvidence-Agent'
.\.venv\Scripts\python.exe -m pytest tests/test_t04_audit.py tests/test_observation_evaluation.py --basetemp outputs/pytest-t045-focused-01
.\.venv\Scripts\python.exe -m pytest --basetemp outputs/pytest-t045-full-01
git -c safe.directory=D:/Research/VisionEvidence-Agent diff --check
```

每次使用新的 output 和 basetemp 路径。审计输出目录必须不存在，已有目录报错以保护人工填写内容。pytest 会清空指定 basetemp，不能指向实验或人工复核目录。

脚本读取冻结归档中的样本、证据、qrels、配置、dataset、results、summary 和 manifest。先校验 manifest 所列 7 个文件 SHA-256，再以原 BM25/查询函数重放全部 20 条方法结果，比较查询、来源、错误、候选原文/来源/元数据/顺序、分数、标签、诊断和指标；汇总完全一致。浮点分数容差 1e-12，不比较耗时。随后使用同一索引的单词查询拆解可加分数，不改变原查询或排名。

输出 `human_review.md` 和 `audit.json`；前者为人工表，后者包含哈希、检查结果及逐词贡献。输入哈希与 manifest 之间的一致性不是独立真实性证明。协议先后关系需结合 Git 历史，见审计报告；脚本不自动证明预注册。

归档审计材料位于 `experiments/t04_v1/audit-20261018-01/`。人工填写请保留该模板，复制一份到新文件并记录审核人/日期；不要将重导出的空模板覆盖已填版本。建议先独立判断原场景与证据，再看方法排名；可填写“不确定”，不强制改变标签。

本轮执行记录（2026-10-18）：

- `python scripts/audit_t04_5.py --output experiments/t04_v1/audit-20261018-01`：PASS，实际使用项目 `.venv/Scripts/python.exe`。
- 首轮定向测试：12 passed、2 failed。新增测试的样本标题计数误把 Structured 小标题计入，已改为行首精确匹配；既有 CLI 测试因 Git ownership 检查导致 commit=null，使用上述进程环境配置解决，未修改既有源码或测试断言。
- 完整回归：`.\.venv\Scripts\python.exe -m pytest --basetemp outputs/pytest-t045-full-01`，322 passed（37.31s），包含原有 319 项和新增 3 项；使用上述临时 Git 环境配置。
- 固定数据与归档逐字节一致，归档 manifest 中全部 21 个原始源码文件哈希一致；`git diff --check` 通过。

仓库初次 `git status` 遇到 ownership 检查；后续 Git 命令以 `git -c safe.directory=D:/Research/VisionEvidence-Agent ...` 指定可信项目路径。测试使用上述进程环境变量，使子进程同样能读取提交号；没有修改全局 Git 配置。
