# T03-B-5：单进程模型复用与生成上限实验

更新：真实六次实验已完成并同步原始文件，见 [正式实验报告](t03_b5_report.md)。
下文保留原始执行协议；其中“待执行”描述是制定协议时的历史状态。

## 依据与边界

用户报告 T03-B-4 已在 RTX 4090D 24GB、PyTorch 2.8.0+cu128、
Transformers 4.57.1 上完成 Qwen3-VL-4B-Instruct 真实 smoke test：
加载 12983 ms，请求 7828.6 ms，生成 7683.9 ms，已分配显存峰值 8751.8 MiB。
中文 PPT 处理后 1280×717，生成 256 token 达到上限，描述较准确但截断。
这些是用户提供的历史观察，本仓库此次没有独立复核原始产物，也没有运行 T03-B-5 GPU 实验。

## 协议

复用现有 Qwen3VLBackend 的单实例和加载缓存，不修改依赖、不下载模型。
默认顺序 256、512、256、512、256、512；每配置三次，相同图片、提示词、BF16、
greedy、seed=0。只有第 0 次包含模型加载，标记 cold；其余标记 reuse。
512 首次调用可能还有形状/kernel 预热，不能把两组所有数据直接求平均比较。
分析时保留原始顺序，排除 cold，并单独观察每种配置首次出现；这是单图 smoke，
不是准确率评测，也不足以宣称统计显著的性能提升。可在新的输出目录用
`--tokens 512 256` 反转顺序复查冷启动与顺序效应；每个进程仍只加载一次。

`--prompt` 可传入 T03-B-4 原提示词，否则使用配置指定的版本化提示词。
必须使用同一原图；处理后尺寸保存在每条 response.metadata 中。
`token_limit_reached=true` 只表示 token 数达到上限，并不能单独证明语义截断；
人工检查两种上限的输出完整性、幻觉和延迟。512 也可能被截断。

## 云端执行（使用已存在的 Linux 环境）

同步本次提交后，在仓库根目录执行。VEA_ENV、VEA_IMAGE 应替换为现有路径；
沿用 T03-B-4 的环境，不执行 pip install。配置中的 model_path 必须指向现有完整模型目录。

```bash
export VEA_ENV=/absolute/path/to/existing/gpu-venv
export VEA_IMAGE=/absolute/path/to/original-slide.png
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
"$VEA_ENV/bin/python" -m visionevidence.vlm.experiment \
  --config configs/vlm_qwen3_vl.toml \
  --image "$VEA_IMAGE" \
  --output "outputs/t03-b5-$(date -u +%Y%m%dT%H%M%S)-$$" \
  --tokens 256 512 --repeats 3 --seed 0 --interval 0.2
```

如已知下载的精确 revision，追加 `--model-revision <revision>`；未知时明确记录 unspecified，
仍通过文件 SHA256 确认实际模型内容。哈希所有模型目录文件（排除 .cache 下载元数据），
包括各权重分片、tokenizer、processor、config，逐文件流式读取，可能需要数十秒。
哈希阶段在推理计时之外，会预热文件系统缓存，因此 cold 指模型首次加载，非磁盘冷缓存。
不要在实验期间改图片/模型文件或并发运行其他 GPU 任务。

## 输出和计量口径

- `manifest.json`：推理前快照；配置、输入 SHA256、模型逐文件 SHA256、代码 commit、
  dirty 状态、依赖清单、Python/平台、种子、提示词、顺序、设备/驱动。正式记录应 git_dirty=false。
- `runs.jsonl`：每请求写一行并 flush，包含完整 response 和采样。Mock 的 token/显存/生成耗时为 null，
  phase=mock、is_mock=true、inference_performed=false，不能用作真实 GPU 结果。
- `result.json`：最终状态 completed/failed、已完成次数及错误；失败不回退 Mock，命令返回非零。
  强制终止/断电可能来不及生成最终文件，应视为未完成。
- `wall_ms` 包含 generate 整个调用（首次含加载），不含哈希、seed 初始化、采样器启动/关闭。
  `latency_ms` 与 `generation_ms` 沿用后端口径，排除模型加载。
- `torch_peak_allocated_mib` 为 PyTorch 已分配显存峰值，包含常驻模型，不含模型加载峰值，
  不是 allocator reserved 或整卡占用。每次请求重置峰值统计。
- `gpu_samples` 通过 nvidia-smi 约每 0.2 秒加查询耗时采集整卡 utilization%、memory.used/total MiB，
  含 UUID、驱动和名称；记录所有物理卡以避免 CUDA_VISIBLE_DEVICES 重映射混淆。
  以 UUID 对应实际使用卡分析；这些读数包含其他进程，不是本进程占用，不能视为精确瞬时峰值。
  采样首条在请求之前，后续涵盖加载和推理；可能漏掉短峰值，并有测量开销。
  不支持 nvidia-smi 或数值不可读会失败；运行中采样失败保留错误并停止实验。

输出目录必须不存在，拒绝覆盖。仅 status=completed、completed_runs=6 且全部真实标记正确、
首条 model_loaded_this_call=true、其余 false、采样无错误时，视为本协议执行完成。

## 本地 CPU 验证

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m visionevidence.vlm.experiment --config configs/vlm_mock.toml --image examples/vlm_demo_fixture.png --output outputs/t03-b5-mock
```

Mock 只验证调度、记录与失败处理；注入的 CPU runtime double 验证六次调用只加载一次及 token 参数传递。
均不能验证真实 CUDA 性能、显存或 512 token 的内容完整性。下一步由开发者执行上述云端实验并保存原始产物。
