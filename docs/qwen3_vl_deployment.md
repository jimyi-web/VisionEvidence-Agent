# Qwen3-VL 部署与人工验收 — T03-B-1

本阶段实现本地权重后端，并用 Windows CPU 测试验证控制流和 Pillow 解码。
**没有执行真实 GPU 推理、云端安装或模型下载，也没有测得真实显存或延迟。**
以下为逐步人工执行的 Linux Bash 命令，不是一键部署脚本；每一步成功后再继续。
出现安装冲突、导入失败或 CUDA 错误时停止，保存错误和环境清单，不反复升级试装。
启动/修改/停止云实例、下载权重及产生费用仍由开发者批准和执行。

## 1. 环境与已核实依赖

用户提供的基线：AutoDL Ubuntu 22.04、RTX 4090D 约 24GB、Python 3.12.3、
PyTorch 2.8.0+cu128、CUDA runtime 12.8、Driver 580.76.05、Pillow 11.3.0。
数据盘 `/root/autodl-tmp` 50GB，系统盘 30GB。这是环境描述，不是本次实机检查结果。

2026-10-16 核对的固定安装目标如下；“源码/依赖约束支持”不等于“GPU 实测通过”。

| 组件 | 固定版本 | 依据与用途 |
| --- | --- | --- |
| PyTorch | 2.8.0+cu128 | 保持用户给定版本；不升级镜像基础 PyTorch |
| torchvision | 0.23.0+cu128 | PyTorch 官方 2.8.0 配对版本；Qwen3-VL 的 AutoProcessor 包含视频处理器组件，需 torchvision，即使本项目只传图片 |
| Transformers | 4.57.1 | 已核实该 tag 提供 Qwen3VLForConditionalGeneration、Qwen3VLProcessor 和 AutoProcessor 映射 |
| Accelerate | 1.10.1 | 用于显式单 GPU device_map；其要求 torch≥2.0.0、hub≥0.21.0 |
| huggingface-hub | 0.35.3 | 满足 Transformers 4.57.1 的 ≥0.34.0、<1.0 约束；用于人工下载 |
| Pillow | 11.3.0 | 图片完整解码、EXIF 方向与 RGB；满足该 Transformers 的 Pillow 范围 |

官方来源：

- [模型卡](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct)确认模型类与 AutoProcessor。卡中“4.57.0 尚未发布”的文字是历史说明；本项目不按其建议安装浮动主分支。
- [Transformers 4.57.1 模型源码](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py)、[处理器源码](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/qwen3_vl/processing_qwen3_vl.py)确认该版本 API；不是推测旧版 Qwen2-VL 模型接口兼容。
- [AutoVideoProcessor 源码](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/auto/video_processing_auto.py)声明 vision/torchvision 依赖。
- [Transformers 版本约束](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/dependency_versions_table.py)包括 torch≥2.2、accelerate≥0.26.0；[Accelerate 1.10.1 安装声明](https://github.com/huggingface/accelerate/blob/v1.10.1/setup.py)用于交叉核对。
- [PyTorch 官方历史版本安装表](https://pytorch.org/get-started/previous-versions/#v280)列出 2.8.0 / 0.23.0 与 cu128 配对；[Hub 0.35.3 发布元数据](https://pypi.org/pypi/huggingface-hub/0.35.3/json)确认该版本存在。

这组直接依赖在声明范围内相容；Linux 二进制导入、CUDA/BF16 和完整模型运行仍待验证。
不额外安装 FlashAttention、qwen-vl-utils、torchaudio 或量化工具。本后端用 SDPA 和原生处理器。
`requirements/gpu-qwen3-vl.txt` 固定直接依赖，`gpu-constraints.txt` 锁住 torch/torchvision；
它们不是所有传递依赖的跨平台锁文件，成功安装后必须保存 `pip freeze`。

## 2. 数据盘与代码

建议布局：

```text
/root/autodl-tmp/
  VisionEvidence-Agent/                  Git 仓库
    models/qwen3-vl-4b-instruct/          权重、tokenizer、processor、chat template
    outputs/                            每次独立运行目录
  vea-envs/qwen3-vl/                     独立 Linux venv
  vea-cache/huggingface/                 HF_HOME，含 hub/xet 缓存
  vea-cache/pip/                         pip 缓存
  vea-cache/tmp/                         安装临时文件
```

先在 AutoDL 终端检查，不修改驱动或系统 CUDA：

```bash
df -h / /root/autodl-tmp
nvidia-smi
python3.12 --version
python3.12 -c "import torch; print(torch.__version__, torch.version.cuda)"
```

只有经人工验收、提交并推送后，云端才能从 GitHub 获得本次代码。本任务不自动 commit/push。
新仓库使用下面命令；若目标已存在，不重复 clone，不覆盖其内容：

```bash
git clone https://github.com/jimyi-web/VisionEvidence-Agent.git /root/autodl-tmp/VisionEvidence-Agent
cd /root/autodl-tmp/VisionEvidence-Agent
git rev-parse HEAD
git status --short
```

已有仓库则先进入目录、检查 `git status --short`，工作区干净且确认远端版本后执行
`git pull --ff-only`。不得用 reset/强制同步覆盖实验文件或本地修改。
核对云端 HEAD 与本地人工验收的 commit SHA 一致。

```bash
export VEA_ROOT=/root/autodl-tmp/VisionEvidence-Agent
export VEA_ENV=/root/autodl-tmp/vea-envs/qwen3-vl
export HF_HOME=/root/autodl-tmp/vea-cache/huggingface
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_XET_CACHE="$HF_HOME/xet"
export PIP_CACHE_DIR=/root/autodl-tmp/vea-cache/pip
export TMPDIR=/root/autodl-tmp/vea-cache/tmp
mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$HF_XET_CACHE" "$PIP_CACHE_DIR" "$TMPDIR" "$VEA_ROOT/models"
```

新终端需重新设置这些变量。安装包、venv、权重和缓存都放数据盘；安装/下载前后检查剩余空间。
50GB 还要容纳 CUDA wheels、安装展开文件和模型，不能将标称容量全部当作权重可用空间。

## 3. 独立虚拟环境与分步安装

不继承 system-site-packages，不修改基础镜像的 PyTorch，不复制 Windows `.venv`。
新 venv 路径必须尚不存在；已有路径先人工核查，不能直接重建覆盖。

```bash
test ! -e "$VEA_ENV"
# 仅上一条退出码为 0 时执行：
python3.12 -m venv "$VEA_ENV"
"$VEA_ENV/bin/python" --version
"$VEA_ENV/bin/python" -m pip --version
```

独立 venv 需要安装与基线相同的 CUDA wheels；这是在数据盘新增副本，基础环境保持原样。
若 venv 已经装有其他 torch 版本，停止并报告；不要让下面命令替换它。

```bash
"$VEA_ENV/bin/python" -m pip install --no-cache-dir "torch==2.8.0+cu128" "torchvision==0.23.0+cu128" --index-url https://download.pytorch.org/whl/cu128
"$VEA_ENV/bin/python" -c "import torch, torchvision; assert torch.__version__ == '2.8.0+cu128'; assert torchvision.__version__ == '0.23.0+cu128'; print(torch.__version__, torchvision.__version__, torch.version.cuda)"
```

在安装其余依赖前先审阅解析计划；若它计划改动已固定 torch/torchvision，停止。

```bash
cd "$VEA_ROOT"
"$VEA_ENV/bin/python" -m pip install --dry-run -r requirements/gpu-qwen3-vl.txt
"$VEA_ENV/bin/python" -m pip install -r requirements/gpu-qwen3-vl.txt
"$VEA_ENV/bin/python" -m pip install -e .
"$VEA_ENV/bin/python" -m pip check
"$VEA_ENV/bin/python" -c "import torch, torchvision, transformers, accelerate, huggingface_hub, PIL; from transformers import Qwen3VLForConditionalGeneration, AutoProcessor; print(torch.__version__, torchvision.__version__, transformers.__version__, accelerate.__version__, huggingface_hub.__version__, PIL.__version__)"
```

以上命令本次未在云端执行。任何一步失败都停止；不使用 `pip install -U` 或更换 CUDA wheel 来碰运气。
项目基础安装没有 GPU 依赖；Windows 开发额外依赖只有 pytest 与用于解码测试的 Pillow。

## 4. GPU/BF16 检查

```bash
"$VEA_ENV/bin/python" - <<'PY'
import torch
assert torch.__version__ == "2.8.0+cu128", torch.__version__
assert torch.cuda.is_available(), "CUDA unavailable"
with torch.cuda.device(0):
    assert torch.cuda.is_bf16_supported(including_emulation=False), "Native BF16 unavailable"
    info = torch.cuda.get_device_properties(0)
    print(info.name, info.total_memory / 1024**3, "GiB", torch.version.cuda)
    value = torch.ones((16, 16), device="cuda:0", dtype=torch.bfloat16)
    print((value @ value).dtype)
    torch.cuda.synchronize(0)
PY
```

`nvidia-smi` 的 CUDA 字段表示驱动能力，`torch.version.cuda` 表示 wheel runtime；两者不是同一指标。
此检查不证明完整模型能装入 24GB 显存。

## 5. 显式模型下载（人工批准后）

后端只读本地目录，不执行这一步。下载包括模型运行所需的配置、tokenizer、processor、
chat template 和 safetensors；不要只复制权重分片。首次下载前确认空间和网络访问条件。

先查看当前模型 commit，再将其完整 SHA 填入下载命令；不要把 main 当作实验 revision：

```bash
"$VEA_ENV/bin/python" -c "from huggingface_hub import HfApi; print(HfApi().model_info('Qwen/Qwen3-VL-4B-Instruct').sha)"
export VEA_MODEL_REVISION=REPLACE_WITH_REVIEWED_40_CHARACTER_COMMIT_SHA
```

```bash
"$VEA_ENV/bin/python" - <<'PY'
import os, re
from pathlib import Path
from huggingface_hub import snapshot_download
revision = os.environ["VEA_MODEL_REVISION"]
assert re.fullmatch(r"[0-9a-f]{40}", revision), "Replace the revision placeholder first"
destination = Path(os.environ["VEA_ROOT"]) / "models/qwen3-vl-4b-instruct"
assert not destination.exists(), "Existing model directory: inspect it instead of overwriting"
snapshot_download("Qwen/Qwen3-VL-4B-Instruct", revision=revision, local_dir=destination)
(destination / "download_revision.txt").write_text(revision + "\n", encoding="utf-8")
print(destination)
PY
```

中断后保留目录并人工检查，再使用相同 revision 续传；不要删除/覆盖其他 revision 的权重。
缓存由 HF_HOME/HF_HUB_CACHE/HF_XET_CACHE 放在数据盘；不得将 token 写入配置或提交 Git。

## 6. 单张图片推理与输出保存

确认 `configs/vlm_qwen3_vl.toml` 中的 model_path 是实际目录；默认 `cuda:0`、BF16、
`max_new_tokens=256`、`do_sample=false`、原图最多 2000 万像素、处理副本长边最多 1280。
原文件不变，超像素原图直接拒绝。可复制配置到本次运行目录后调整。

设置离线模式作为部署层的额外约束：

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
"$VEA_ENV/bin/visionevidence" vlm --image examples/vlm_demo_fixture.png --backend qwen3_vl --config configs/vlm_qwen3_vl.toml --json
```

1×1 fixture 只能测试接口接通，不能衡量描述质量。真实图像 smoke test 用现有本地图像：

```bash
export VEA_IMAGE=/root/autodl-tmp/your-test-image.jpg
export VEA_RUN="$(mktemp -d "$VEA_ROOT/outputs/qwen-smoke-XXXXXXXX")"
cp configs/vlm_qwen3_vl.toml "$VEA_RUN/config.toml"
"$VEA_ENV/bin/python" -m pip freeze > "$VEA_RUN/pip-freeze.txt"
git rev-parse HEAD > "$VEA_RUN/code-revision.txt"
cp "$VEA_ROOT/models/qwen3-vl-4b-instruct/download_revision.txt" "$VEA_RUN/model-revision.txt"
sha256sum "$VEA_IMAGE" > "$VEA_RUN/image-sha256.txt"
nvidia-smi > "$VEA_RUN/nvidia-smi.txt"
"$VEA_ENV/bin/visionevidence" vlm --image "$VEA_IMAGE" --backend qwen3_vl --config "$VEA_RUN/config.toml" --request-id smoke-001 --json > "$VEA_RUN/response.pending" 2> "$VEA_RUN/stderr.txt"
```

检查最后一条命令退出码为 0、JSON 可读且 metadata.inference_performed=true、
backend=qwen3_vl、is_mock=false 后，才把 `response.pending` 重命名为 `response.json`。
失败时保留 stderr，空输出不是成功结果。每次使用新 VEA_RUN；不要覆盖历史实验文件。
成功文本仍是未验证的模型生成内容，`verified=false`，不能当作事实核验结论。

## 7. 耗时和显存口径

- `model_load_ms`：首次调用的运行库导入、processor/model 加载、eval 与 CUDA 同步；实例复用后保留该值，`model_loaded_this_call=false`。
- `latency_ms`：本次图片解码/处理、输入编码和传输、生成及输出解码；排除加载和实例锁等待。
- `generation_ms`：仅 model.generate，计时边界执行 CUDA synchronize，避免只计入异步提交时间。
- `generated_tokens`：扣除完整输入 token 长度后的序列长度，包含生成的 EOS 等特殊 token；显示文本跳过特殊 token。
- `peak_gpu_memory_mb`：生成前重置峰值后，torch.cuda.max_memory_allocated / 1024²（实际单位 MiB），包括常驻模型和本次输入/生成张量，**不包括加载阶段峰值**，也不等于 nvidia-smi 的整个进程占用。

每次新起 CLI 都是冷进程。需要测 warm 调用时，在同一 Python 进程中复用后端实例：

```bash
"$VEA_ENV/bin/python" - <<'PY'
import os
from pathlib import Path
from visionevidence.vlm import VLMRequest, create_vlm, load_vlm_config
from visionevidence.vlm.prompts import get_prompt
cfg = load_vlm_config("configs/vlm_qwen3_vl.toml")
backend = create_vlm(cfg.backend, config=cfg)
for index in range(3):
    response = backend.generate(VLMRequest(os.environ["VEA_IMAGE"], get_prompt(cfg.prompt_version), f"repeat-{index}", cfg.max_image_size_mb))
    with (Path(os.environ["VEA_RUN"]) / f"repeat-{index}.json").open("x", encoding="utf-8") as stream:
        stream.write(response.to_json() + "\n")
PY
```

第一条冷调用与后两条分开分析。此示例只做 smoke test，不是正式性能基准。
测量时同一进程/GPU 不并行运行别的推理任务，否则进程级峰值可能受干扰。
默认 greedy + eval 减少随机性，但不保证跨硬件/库版本逐位一致；开启采样时当前不固定 seed。
正式实验还需登记数据集、split、样本数、模型 revision、prompt/图像处理配置、完整环境及硬件。

## 8. 常见错误

| 错误 | 下一步 |
| --- | --- |
| model_path 不存在 | 检查数据盘挂载和 TOML；后端不会自动下载 |
| 缺少配置、tokenizer、processor 或分片 | 对照下载 revision 检查目录完整性；不启用在线 fallback |
| Qwen3VLForConditionalGeneration 无法导入 | 检查独立 venv 与固定版本，保存 pip check/导入错误，停止 |
| torchvision 算子缺失 | 检查 torch 2.8.0+cu128 / torchvision 0.23.0+cu128 配对，不覆盖基础环境 |
| CUDA/BF16 不可用 | 检查 nvidia-smi、设备索引、驱动/runtime；不自动降级 CPU/FP16/Mock |
| 解码失败/像素超限 | 使用有效静态 PNG/JPEG/WEBP；调整限制或另存缩小副本，保留原图 |
| CUDA OOM | 记录发生在 loading 还是 inference；检查其他进程，降低 max_image_side（如 896）和 max_new_tokens（如 128），另存配置后做单图测试 |

若加载阶段就 OOM，降低生成长度通常无助于常驻权重占用。停止该实验并讨论显存或量化方案；
本阶段不自动安装 4-bit 依赖、不启用 CPU offload、不重试循环。失败异常保留原始原因，
不生成成功 JSON。进程退出后检查显存释放；不要终止其他人的进程。

## 9. 备份、关机与释放实例

先备份每个独立 outputs 目录、配置、代码 SHA、模型 revision 和输入标识到本地或获准存储，
核对文件数量与校验和。不要只把重要结果留在临时实例盘上；权重可按固定 revision 重下载。
本地下载可使用 AutoDL 文件管理或其提供的 SSH/SCP 地址、端口，不能在仓库保存访问凭据。

由开发者在 AutoDL 控制台停止实例并确认计费状态。释放/删除实例前，现场核对该实例的
系统盘/数据盘保留规则与费用，并完成可读备份；不假定关机等于释放，也不假定释放后数据仍保留。
本任务没有连接、启动、关闭或释放任何云实例。
