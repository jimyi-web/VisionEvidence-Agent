# VisionEvidence-Agent

一个轻量、可复现的多模态证据验证研究项目。研究目标是结合视觉语言模型、
证据检索和受约束的工具调用，对文本声明及可选图片进行有证据支撑的验证，
并通过可审计的评估与消融实验分析效果。目标里程碑为 2026 年 11 月。

## 当前状态：T01 基础工程

已实现：

- `src` 布局、Python 包元数据和 CLI 入口。
- 标准库 dataclass：输入 `ClaimInput`、证据 `Evidence`、结果契约 `VerificationResult`。
- 非空文本、图片文件路径、证据字段、有限数值分数及基本元数据校验。
- CLI 的 `--help`、`--version` 和输入校验；JSON 输出明确包含 `inference_performed: false`。
- CPU 单元测试与 CLI 子进程测试。

尚未实现：真实 VLM 推理、图片解码、BM25/稠密/混合检索、Agent 决策、
验证预测、数据集接入、评估指标及演示界面。当前没有模型效果或 GPU 性能实验结果。
`configs/base.yaml` 仅为后续配置模板，当前 CLI 不加载它。

## 本地安装（Windows / PowerShell）

在项目根目录操作，要求已有 Python 3.12 的 `.venv`。
若 `.venv` 不存在，先由开发者确认环境创建方案；不要使用全局 Python 代替。
无需激活环境或修改 PowerShell 执行策略。

```powershell
.\.venv\Scripts\python.exe --version
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m pip check
```

运行时无第三方依赖。开发依赖为 pytest，构建使用 setuptools；直接依赖版本在
`pyproject.toml` 中固定。安装需要能够访问相应 Python 包索引。
当前验证目标是 Python 3.12，其他版本尚未验证。

## 使用与测试

```powershell
.\.venv\Scripts\python.exe -m visionevidence.cli --help
.\.venv\Scripts\python.exe -m visionevidence.cli --version
.\.venv\Scripts\python.exe -m visionevidence.cli --text "这是一条待验证的声明。"
# 将路径替换为本地已有图片；相对路径基于当前工作目录。
.\.venv\Scripts\python.exe -m visionevidence.cli --text "图片中的声明。" --image ".\example.png"
.\.venv\Scripts\python.exe -m pytest
```

安装也会生成 `.\.venv\Scripts\visionevidence.exe` 命令入口。
CLI 成功校验返回退出码 0；缺失参数、空声明、图片路径不存在或指向目录返回 2。
图片目前只检查文件路径，不读取内容或验证图片编码。校验成功不表示声明为真，
CLI 不输出预测标签或检索证据。结果数据结构供未来模块使用，不包含默认预测。

## 工程结构

```text
configs/base.yaml           后续配置模板（T01 不加载）
docs/architecture.md        模块边界、数据契约与数据流
src/visionevidence/         Python 包、数据结构与 CLI
  vlm/                     模型加载和推理预留模块
  retrieval/               证据检索预留模块
  agent/                   工具编排预留模块
  verification/            验证决策预留模块
  evaluation/              评估预留模块
scripts/                   后续可复现脚本目录
tests/                     数据结构与 CLI 测试
outputs/                   本地输出目录（只跟踪 .gitkeep）
```

`scripts/.gitkeep` 用于让 Git 保留空目录；模块目录用带职责说明的 `__init__.py` 保留。
测试针对安装后的包运行，不通过修改 `sys.path` 绕过 `src` 布局。

## 可复现性与后续 GPU 计划

后续实验必须记录数据集版本和划分、模型标识与 revision、随机种子、配置、
软件环境、硬件、运行耗时、峰值显存、指标和样本数。配置模板中的种子尚未用于实验。
可用以下命令检查环境清单，运行正式实验时应将其保存至该次独立实验目录：

```powershell
.\.venv\Scripts\python.exe -m pip freeze
```

当前固定直接开发依赖和构建依赖，尚未提供跨平台传递依赖锁文件。
不要覆盖已有实验输出。密钥、原始数据、模型权重、环境和临时输出不得提交。

未来将先在独立 Ubuntu GPU 环境验证 Qwen3-VL-4B 或其他合适的开放权重模型兼容性，
支持可配置模型路径、适用时的 BF16、复用模型实例，并实测延迟和峰值显存；
仅必要时评估 4-bit 量化。不预先承诺任何模型能装入特定显卡。
云实例操作、大型权重下载和 GPU 花费需要开发者明确批准。
Windows `.venv` 不复制到 Linux。T01 不安装 PyTorch、Transformers 或 CUDA，
也不连接 GPU 服务器。
