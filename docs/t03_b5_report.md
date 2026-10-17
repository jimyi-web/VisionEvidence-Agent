# T03-B-5 正式实验报告：单图 GPU Smoke Test

## 结论与证据范围

2026-10-17 在单张用户提供的图片上完成六次真实推理，单进程首次加载一次、后续五次复用。
这是部署与复用机制的 Smoke Test，不是通用性能基准，不是 OCR、幻觉率或事实核验准确率评测。
本报告依据本地同步的三个原始文件，未以聊天摘要替代原始记录；本地仅解析数据，没有重新运行 GPU。

原始目录：`outputs/t03-b5-real/`（被 Git 忽略，需独立备份）。文件 SHA-256：

| 文件 | SHA-256 |
| --- | --- |
| manifest.json | `0a24fe843e14dfafdd9f8f2ce4416d1af90ae412a48778603992d4bb606dfea7` |
| runs.jsonl | `1d8113cb88888035ba9fd16adede461b116f711ae9202c21bcb7c4bc77c51fd8` |
| result.json | `ea7b7b1de07bd47cd515bfff82f6fc7172e8cb38b2562f0d4923d86a8c566bc7` |

三份文件严格按 UTF-8 JSON/JSONL 解析，拒绝重复键和非有限 JSON 常量。
manifest 的 `status=preparing` 是推理前快照，最终 result 为 `completed`、`completed_runs=6`，两者不冲突。
除 status 外 manifest 字段与 result 一致；两者 `is_mock=false`、`git_dirty=false`。
六条记录 run_index=0…5，response 均 `is_mock=false`、`inference_performed=true`，采样错误列表均为空。
模型、后端、版本、配置、token、耗时和显存字段通过交叉核对。run 本身不另存 commit，归属由同目录 manifest/result 确认。

## 环境与输入

| 项目 | 原始记录 |
| --- | --- |
| 代码版本 | `c34ebe6d6cd73345778ae7b2292f1fda301e1d4a` |
| GPU | NVIDIA GeForce RTX 4090 D；整卡报告容量 24564 MiB |
| GPU UUID | `GPU-19f6fd71-6df7-605c-aa61-3e87bde81dec` |
| 驱动 / CUDA runtime | 580.76.05 / 12.8 |
| 平台 | Linux 5.15.0-101-generic x86_64，glibc 2.35 |
| Python / PyTorch / Transformers | 3.12.3 / 2.8.0+cu128 / 4.57.1 |
| 模型 | Qwen/Qwen3-VL-4B-Instruct，BF16，cuda:0 |
| revision | unspecified；manifest 记录 16 个模型目录文件的大小和 SHA-256 |
| 输入 | user_supplied_single_image，split=smoke_only，独立样本数 1 |
| 原图 SHA-256 | `1b5edc5106c7c1de9a7b3bf80ca9aad60dd6c52f7e3d7e40b51bd4df13903422` |
| 处理后尺寸 | 六次均为 1280×717（送入 processor 前，不等于最终 patch 网格） |
| 时间 | UTC 06:34:51.119512 至 06:36:03.670693；北京时间 14:34:51 至 14:36:03 |

完整依赖清单、提示词全文、配置和模型文件哈希见 manifest。未取得云端模型文件及原图进行独立哈希复验；
这里只确认同步记录内部一致。未从平台字符串推断 Ubuntu 发行版。模型下载来源的精确 revision 仍未知。

## 方法与计量口径

同一后端实例、同一图片、同一 `visual_observation_v1` 提示词；seed=0、do_sample=false、greedy。
按 256、512、256、512、256、512 的生成上限交替运行，每种上限三次。第 0 次 cold，其余 reuse。
哈希在推理计时外，可能预热文件系统缓存，cold 不代表磁盘冷缓存。

- wall_ms：整个 generate 调用，首次含模型加载；不含哈希、种子初始化及采样器启动/停止。
- latency_ms：图片解码、processor、传输、生成及解码；不含加载和锁等待。
- generation_ms：CUDA 同步边界内的生成耗时。
- token/s：generated_tokens × 1000 / generation_ms，是生成阶段平均速率；不是端到端吞吐或纯 decode 速率，也没有 TTFT 数据。

## 原始指标（显示四舍五入，计算使用原始精度）

| Run | 上限/实际 token | phase | wall ms | latency ms | generation ms | 生成 token/s | torch allocated 峰值 MiB | 整卡采样最大 MiB | 采样数 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 256/256 | cold | 11905.66 | 7123.29 | 7005.35 | 36.54 | 8751.79 | 9241 | 41 |
| 1 | 512/512 | reuse | 12764.10 | 12763.67 | 12689.08 | 40.35 | 8750.69 | 9241 | 49 |
| 2 | 256/256 | reuse | 6404.69 | 6404.29 | 6333.93 | 40.42 | 8750.69 | 9241 | 24 |
| 3 | 512/512 | reuse | 13061.09 | 13060.67 | 12985.45 | 39.43 | 8750.69 | 9241 | 49 |
| 4 | 256/256 | reuse | 6466.67 | 6466.20 | 6394.66 | 40.03 | 8750.69 | 9241 | 24 |
| 5 | 512/512 | reuse | 13084.65 | 13084.24 | 13020.14 | 39.32 | 8750.69 | 9241 | 49 |

256 reuse（runs 2、4，n=2）的 wall 均值为 6435.68 ms；512 reuse（runs 1、3、5，n=3）为 12969.95 ms。
单独排除首次 512 调用后，runs 3、5 的均值为 13072.87 ms。512 首次调用并没有在本次记录中表现为更慢；
仍保留首次形状/内核预热和顺序效应的可能性，不用混合 cold/reuse 的总体平均声称加速比或统计显著差异。

## 模型复用

首次加载 4782.008443 ms，model_loaded_this_call 序列为 true、false、false、false、false、false。
后续记录中的 model_load_ms 仍保留首次加载耗时，它是缓存的历史值，不能累计成六次加载。
记录支持“同一进程六次请求中仅首次加载”的结论，不证明跨进程缓存、长期服务稳定性或并发能力。

## 显存：两种指标不可混用

PyTorch 的 `torch_peak_allocated_mib` / response 的 `peak_gpu_memory_mb` 实际单位均为 MiB，
是每请求重置统计后的已分配显存峰值，包含常驻模型但排除加载阶段峰值。
它不包含 allocator reserved 的全部空间，也不表示整卡占用。

nvidia-smi 以同一 UUID 汇总的六次采样最大 memory.used 均为 9241 MiB，整卡容量为 24564 MiB。
计划采样间隔 0.2 秒，实际还叠加查询耗时；共 236 条采样。首条在请求前，首轮涵盖加载与推理。
整卡读数包含 CUDA 上下文、缓存和可能的其他进程，不能归因于本进程，不能称为精确瞬时峰值。
没有 reserved 峰值、独立加载峰值、逐进程整卡占用或其他进程清单，不用两种口径的差值推导这些指标。
只说明本次配置在该卡上成功完成，不能外推其他图片尺寸、并发数或上下文长度的容量。

## 输出截断与质量边界

六次 token_limit_reached=true（6/6）。该标志单独只能证明达到上限；本次还检查了原始 description：

- 三次 256 输出均在 `- Center column, with a diagram` 结束，段落描述未完成。
- 三次 512 输出均在“节点间存在两种力：”结束，尚未展开列表且引号未闭合。
- 相同 token 上限的三份文本完全相同；这只是本次 greedy 重复观察，不保证跨硬件逐位一致。

因此 512 扩大了输出范围但没有解决本图的完整性问题。没有人工真值和原图对照审核，不能声称 OCR 正确率、
描述准确率或幻觉率。下一阶段应尝试短小、受约束的结构化观察，显式拒绝截断 JSON；不能默认加大上限就解决问题。

## 局限性与复现

只有一张文本密集的图片，六次请求不是六个独立样本；没有多图、随机顺序、多进程、多精度或其他 GPU 对照。
不存在性能置信区间、泛化效果结论、检索指标和事实核验指标。原始记录由用户同步，哈希提供后续完整性核对，
并非第三方运行真实性认证。原始文件不随 Git 提交，应独立备份。

本地重算（不导入 GPU 框架）：

```powershell
.\.venv\Scripts\python.exe scripts/summarize_t03_b5.py outputs/t03-b5-real --expected-commit c34ebe6d6cd73345778ae7b2292f1fda301e1d4a
```

GPU 重跑协议见 [T03-B-5 实验协议](t03_b5_experiment.md)。以下是复现模板，非声称原 shell 命令已被记录；
需使用上述 commit 的独立 checkout、既有 Linux 环境和相同模型/图片哈希，先获得 GPU 执行批准。本轮未执行：

```bash
export VEA_ENV=/absolute/path/to/existing/gpu-venv
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
"$VEA_ENV/bin/python" -m visionevidence.vlm.experiment \
  --config configs/vlm_qwen3_vl.toml \
  --image /root/autodl-tmp/test-image.jpg \
  --output "outputs/t03-b5-repeat-$(date -u +%Y%m%dT%H%M%S)-$$" \
  --tokens 256 512 --repeats 3 --seed 0 --interval 0.2
```

配置 model_path 需对应原始记录中的现有模型目录，复现时对照配置哈希和完整 prompt。
输出目录必须全新，不覆盖当前产物；不以 Windows 环境替代 Linux GPU 环境。
