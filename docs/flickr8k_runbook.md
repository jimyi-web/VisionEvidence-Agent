# Flickr8k：Windows 下载、离线导入与 GPU 验收

本任务已准备真实 20 图；以下执行命令使用项目解释器，不启动云实例，不下载模型。方法与指标见 [冻结协议](flickr8k_protocol.md)。

## Windows 小样本获取与 CPU 验收

在 D:\Research\VisionEvidence-Agent，使用现有 .venv。首次使用新目录：

```powershell
.\.venv\Scripts\python.exe -m visionevidence.datasets.flickr8k download --output data/raw/flickr8k-20261019 --revision c7a02cb487da0eca729e133194b0f842cb578c08 --split test --limit 20
.\.venv\Scripts\python.exe -m visionevidence.datasets.flickr8k adapt --input-root data/raw/flickr8k-20261019 --output data/processed/flickr8k-proxy-v1
.\.venv\Scripts\python.exe -m visionevidence.caption_experiment --dataset data/processed/flickr8k-proxy-v1/test --output outputs/flickr8k-mock-20-new --mode mock --limit 20 --batch-size 4
```

前两步本轮已执行，已有目录勿重复覆盖。下载器每个请求 30 秒 timeout、图片响应最多 10 MiB，校验 JPEG 解码；source.json 和 image-inventory.json 保存固定版本、URL、字节/哈希。失败图片记录到 download-NNNN.json，退出 1；修复网络后同一命令增加 --resume，仅获取未成功图片。坏缓存显式报错，不悄悄替换。

```powershell
.\.venv\Scripts\python.exe -m visionevidence.datasets.flickr8k download --output data/raw/flickr8k-20261019 --split test --limit 20 --resume
```

下载成功退出 0，配置/输入错误退出 2。不要使用无约束的 load_dataset 或 snapshot_download 下载整个数据集。

## 无网络时的 Windows 下载/离线导入

可在另一台有网络的 Windows 电脑用同一项目和上述下载命令准备目录，再把 data/raw/flickr8k-20261019 按原字节复制到目标项目。只需本轮 20 JPEG、三个 metadata.csv、README、source.json 和 image-inventory.json，不含模型。

也可从 HF 的固定 Commit 手动保存文件，目录必须如下；三个 CSV 都保留完整内容，图片仅保存 test 前 20 个文件名，禁止改名或把缺图换 fixture：

```text
data/raw/flickr8k-local/
  README.md
  train/metadata.csv
  dev/metadata.csv
  test/metadata.csv
  test/1056338697_4f7d7ce270.jpg
  ...（按 test CSV 文件名升序选定的前 20 张）
```

固定下载 URL 模板：`https://huggingface.co/datasets/intro/flickr8k/resolve/c7a02cb487da0eca729e133194b0f842cb578c08/<split>/<filename>`。列文件名的命令：

```powershell
Import-Csv data/raw/flickr8k-local/test/metadata.csv | Sort-Object file_name | Select-Object -First 20 -ExpandProperty file_name
.\.venv\Scripts\python.exe -m visionevidence.datasets.flickr8k adapt --input-root data/raw/flickr8k-local --output data/processed/flickr8k-local-proxy
```

没有 source.json 的手工导入自动用三个 CSV 的内容哈希作 offline-sha256 版本，source_verified=false，不冒充已认证上游 Commit。若从完整原始 Flickr8k 包导入，需要先整理成相同 imagefolder/CSV 格式；不要把没有原划分的额外 92 组随机塞进 test。离线 adapt 全程无网络。

## 20 图真实 GPU（由开发者手动执行）

只在开发者已批准的 Linux 实例、现有 GPU 虚拟环境和本地模型权重目录中运行。先手动同步代码和小数据目录；Windows .venv 不复制。核对 configs/t05_qwen3_vl.toml 中已有 model_path，当前两路共用 BF16、384 token、max_image_side=1280。若 Linux 没有 processed 包，离线生成：

可在 Windows 把已经取得的目录压缩后人工传到 Linux（本轮目录只有 20 张图片，不含完整数据）：

```powershell
Compress-Archive -LiteralPath data/raw/flickr8k-20261019 -DestinationPath outputs/flickr8k-20-transfer.zip
```

在 Linux 仓库的 data/raw 下解压，确保最终路径是 data/raw/flickr8k-20261019/test/原文件名.jpg。下载目录和图片不随 git push 同步；压缩包仅供自己的研究环境传输，不公开发布。

```bash
export VEA_ENV=/absolute/path/to/existing/gpu-venv
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
# 在仓库根目录；已有 GPU 环境需已安装当前代码，或指定 PYTHONPATH。
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
"$VEA_ENV/bin/python" -m visionevidence.datasets.flickr8k adapt \
  --input-root data/raw/flickr8k-20261019 --output data/processed/flickr8k-proxy-linux
timeout --signal=INT --kill-after=30s 20m "$VEA_ENV/bin/python" -m visionevidence.caption_experiment \
  --dataset data/processed/flickr8k-proxy-linux/test \
  --output outputs/flickr8k-real-20-01 --mode qwen3_vl \
  --config configs/t05_qwen3_vl.toml --limit 20 --batch-size 4 --max-attempts 1
```

输出根目录 experiment.json、comparison-NNNN.json；baseline/ 和 structured/ 各自含 manifest、state.sqlite3、输入快照、session summary/results。比较 JSON 含逐样本指标，原响应和错误详情在各分支记录；failed 查询仍在分母，不能仅挑成功样本报告。确认 is_mock=false、每请求 inference_performed=true、GPU 名称/时延/显存、首调用加载与后续复用；40 个计划调用对应 20 张图片。至少人工抽查输出与图片，区分 JSON 合格率与视觉内容正确性。

失败后同目录用 --resume --max-attempts 2 重试；不通过提高预算悄悄改 prompt/上限。如果需要改配置，选择新 run，记录与预设的偏离。

## 100 / 1000 / 8000

下载扩容明确由开发者后续执行，本轮未执行。Windows 增加 --resume，不重新适配：

```powershell
.\.venv\Scripts\python.exe -m visionevidence.datasets.flickr8k download --output data/raw/flickr8k-20261019 --split test --limit 100 --resume
# 后续按预算再将 limit 改为 1000。
```

同步新增图片到 Linux 原位置后，将同一个 GPU 实验命令的 limit 改为 100，增加 --resume；后续再改 1000，同时人工调整外层 timeout。候选库始终 5,000，不重做前 20 个成功样本。若此前失败预算耗尽，明确提高 max-attempts。

8,000 图分别下载 test 1000、dev 1000、train 6000；使用同一适配包的三个 split，分别建输出目录，保持候选池与原划分。不要对 --dataset .../test 使用 --limit 8000，程序会拒绝。完整下载/运行需要后续预算决定，不是本轮默认动作。

## 资源估算与停止条件

规划假设为每次 VLM 调用 10 秒，**每图两次**，不是本轮实测：20/100/1000/8000 图约 6.7 分钟/33 分钟/5.6 小时/44.4 小时；加 20% 重试约 8 分钟/40 分钟/6.7 小时/53.3 小时。另计加载、两次模型文件哈希、索引与磁盘时间。先测 20 图后按实际调用分位数重新估算；小时价格由开发者确认。

历史 RTX 4090D 单图 BF16 allocated 峰值约 8,751 MiB，但不是本 Flickr8k 配置的容量保证；建议沿用已验证的 24 GB GPU 环境，实际显存以此次测量为准。逐图推理、单模型复用，batch-size 不代表 GPU tensor batch。当前数据小样本图片约 2.42 MB；全库镜像页面约 1.11 GB，后续预留几 GB 数据/结果空间，模型权重空间另算。CPU 峰值 RAM 尚未量测。

若大量 token 截断/Schema 失败、模型来源不符、图片哈希变化或 OOM，保留失败并停止扩大规模，分析后另建配置实验。不要因低分删样本、改 qrels、替换负例，或把代理检索命中率命名为谣言识别准确率。
