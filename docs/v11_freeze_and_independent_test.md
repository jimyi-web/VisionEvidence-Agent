# V1.1 正式冻结与独立 test 部署

日期：2026-10-20。**已审计真实 dev 原件，正式冻结 format_details；独立 test 尚未运行。** 任务始终是 **image-to-caption retrieval proxy**，不是谣言检测、真实性分类或事实核验。冻结后不得根据 Smoke 或完整 test 反馈修改策略。

## 五路 dev 结果与选择依据

原方法 Commit：`53e41f88cd057dde55a4a41f41db1b9b86516a61`。五路均为 clean Git，源码逐文件与该 Commit 的 blob 哈希匹配；同用 dev 文件名升序前 20 图、5,000 描述、相同 qrels、BM25(k1=1.5,b=0.75)、greedy、seed=0、384-token 和 BF16。

| 方法 | 成功/20 | 失败率 | Hit@1 | Hit@5 | Hit@10 | MRR@10 | 原 Schema/20 | 附加合约/20 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | 20 | 0% | 0.35 | 0.75 | 0.80 | 0.532143 | 不适用 | 不适用 |
| V1.0 Control | 17 | 15% | 0.35 | 0.50 | 0.55 | 0.425000 | 17 | 不启用 |
| Format Only | 14 | 30% | 0.35 | 0.50 | 0.55 | 0.415476 | 20 | 14 |
| Details Only | 17 | 15% | 0.45 | 0.60 | 0.65 | 0.531250 | 17 | 不启用 |
| Format + Details | 20 | 0% | 0.50 | 0.80 | 0.90 | 0.665476 | 20 | 20 |

全部指标分母为 20，失败计零，未删除失败图。原 Schema 与附加格式合约分开统计。

| 方法 | 失败根因 | generated_tokens 总量 | 响应延迟均值 ms，含失败响应 |
| --- | --- | ---: | ---: |
| Baseline | 无 | 1114 | 1474.26 |
| V1.0 Control | 2 次 token-limit；1 次 objects 超过原 Schema 的 8 项 | 1379 | 1756.69 |
| Format Only | 3 次 objects>6；2 次 uncertainties>2；1 次 objects 重复 | 800 | 1064.59 |
| Details Only | 2 次 token-limit；1 次 objects 超过原 Schema 的 8 项 | 1391 | 1769.35 |
| Format + Details | 无 | 731 | 961.65 |

token-limit 为 generated_tokens≥384；截断可能同时导致 JSON 不完整，根因与症状不重复相加。精确错误、失败 ID、原始响应与全部 attempt/快照均保留。延迟不含冷加载，不将此小样本差异解释为普遍加速结论。

依据预先版本化的 `configs/v11/evaluation_plan.toml`，Structured 候选先须 20/20 流程成功，再按 Hit@10、MRR@10、较少 generated_tokens 排序，完全平局再优先开关少、再 format_only。**只有 format_details 通过可靠性门槛，因此正式选定它。** Baseline 不参与 Structured 候选选择。这是 dev 选择依据，不是独立 test 提升证明。

## 真实来源、原件与正式 manifest

五路 100 条响应均为 backend=qwen3_vl、is_mock=false、inference_performed=true；核对 request_id、JPEG 哈希、同一组 16 个模型文件哈希、生成参数、SQLite 完整性、input/attempts/results/summary 一致性。CPU 重放 88 条成功原始查询，排名与 BM25 分数一致；12 条失败计零。本轮新增真实调用 0。

保存的 runtime 为 NVIDIA GeForce RTX 4090 D，显存总量 25,250,627,584 bytes；Python 3.12.3、torch 2.8.0+cu128、transformers 4.57.1、Pillow 11.3.0。安装包元数据为 visionevidence-agent 0.1.0；保留原环境，用 PYTHONPATH 指向同步源码，不重新安装或升级。

传输包还含早期 dev 两图 Baseline+V1.0 Control Smoke：4 次真实调用、均成功，另 CPU 重放 4 条查询并核对 comparison。单独归档，不并入五路选择。合计 **104 次真实调用记录、20 张独立图片**。来源结论是保存记录与代码的一致性，不是独立硬件证明。

正式目录：`experiments/v11/frozen-format-details-20261020-01/`，包含 `freeze-manifest.json`、`dev-audit.json`、四份策略 TOML、两路完整 prompt、CPU 验证记录。manifest 的 status=frozen、selected_strategy=format_details，保存模型/提示/Schema/代码/配置哈希、384-token/生成与图像参数、BM25、版本/划分/队列/qrels/失败规则、全部 dev 指标、运行环境、区间/预算和备份约束。

method_git_commit 指原 dev 实现；包含正式 manifest 的封存 Commit 用 `git log -1 --format=%H -- experiments/v11/frozen-format-details-20261020-01/freeze-manifest.json` 获取，避免提交 SHA 自引用。源码/脚本/配置哈希统一 CRLF→LF，原实验 manifest 另存 Linux 原字节哈希；冻结文件在 Git 中保留精确字节。

原始传输包 `outputs/v11b-dev20-gpu-results.tar.gz` 的 SHA256：`acaa5fa12e083ea0b2248c373741ff4dae138d1eb8b747a353b6a3f475b45fad`。安全解包在 `outputs/v11-dev-real-source-01/`，全部 81 文件逐字节匹配原包。最终审计在 `outputs/v11-dev-real-audit-03/`：`all-dev-originals.tar.gz` 是含早期 Smoke 的原包副本；`dev-originals.tar.gz` 是五路归档，SHA256 为 `1f0e9cbcea2769e5cb303e8a370738cfd45baa8030f62f99ced9b69e0a4dffdd`。所有失败、早期/后期编号快照、原始输出均未覆盖或清理。TOML/完整提示与源码/Schema 身份一同冻结；权重仅记录哈希，不下载、不入 Git。

`experiments/v11/independent-test-20261020-01/` 保留原件到达前的准备快照；其 blocked 状态已由正式目录取代，不是当前运行入口。旧 `flickr8k-v11b-dev20-53e41f8-01.tar.gz` 只有图片/元数据，历史 `v11a-five-arm-cli` 为 Mock，均未冒充真实结果。

## 固定独立 test 与准确下载

数据版本：`intro/flickr8k@c7a02cb487da0eca729e133194b0f842cb578c08`。原 test 文件名 ASCII 升序位置 21–120，offset=20、limit=100；前 20 图仅作诊断，明确排除。沿用镜像原 split 字段，原作者列表未另行逐文件独立认证。

首图 `1258913059_07c613f7ff.jpg`，末图 `2078311270_f01c9eaf4c.jpg`，完整顺序在 manifest。两路仅 Baseline+format_details，共用 100 图、完整 5,000 描述与相同 qrels。每图五个同图描述为 proxy positives，其他图描述为 assumed negatives；不挖掘负例、不去重、不用正确 caption 生成查询。

- corpus SHA256：`2b6007f1ffd51d40f31bc123b624e3d7aa9344ca3e0404a5691e264ae8cbd599`
- cohort qrels SHA256：`24bf1beb2e4be453558812a5f24da19fc63d71f4b4e6d7d7f9bb8d1efcea334d`
- 私用输入传输包：`outputs/flickr8k-v11-test-inputs-01.tar.gz`，SHA256 `0e3c74db4fbd9f362323c7312089843bfe8ced4a77254b20815676286ff479b5`。含 raw 元数据种子、原 adapter 与 100 图队列，没有图片或检索结果，不入 Git。

旧下载器无 offset。本次实际新增并测试 `--offset`，默认 0 保持旧前缀行为，严格选择排序后 `[offset:offset+limit]`。**需要获取 100 张新图片，本轮下载 0 张**，无需重新获取诊断前 20 张。

以下是后续人工操作，本轮未执行。手动推送并同步此 Commit，复制输入包到原项目 outputs/ 后，使用原环境。解包目标三棵数据目录必须不存在，不覆盖历史数据。

```bash
export VEA_ENV=/root/autodl-tmp/vea-envs/qwen3-vl
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
echo '0e3c74db4fbd9f362323c7312089843bfe8ced4a77254b20815676286ff479b5  outputs/flickr8k-v11-test-inputs-01.tar.gz' | sha256sum --check
tar -xzf outputs/flickr8k-v11-test-inputs-01.tar.gz
"$VEA_ENV/bin/python" -m visionevidence.datasets.flickr8k download \
  --output data/raw/flickr8k-v11-test-01 --split test --offset 20 --limit 100 \
  --revision c7a02cb487da0eca729e133194b0f842cb578c08 --resume
```

首次也使用 --resume，因为包内已有 source.json 和已验证 CSV。下载失败重跑同一命令，缓存须通过 SHA256/JPEG 解码，新增收据、不换图。网络恢复不增加 VLM attempts。100 图齐备才 Smoke，不重新 adapt 或改写冻结 manifest；available_images=0 是准备时快照，入口另查最新 inventory 的全部 100 图。

## 后续 GPU 授权后的 Smoke 与完整命令

**本轮不启动 GPU 推理，下列命令只在后续明确授权后执行。** 入口验证已提交且未变的正式 manifest、代码/脚本/配置/prompt、候选/qrels、全部图片、模型文件哈希、原 Python 和关键依赖；执行时检查 RTX 4090D。不匹配即停止，不安装升级或下载模型。

```bash
export FREEZE=experiments/v11/frozen-format-details-20261020-01/freeze-manifest.json
# CPU preflight，不加载模型；模型哈希需要磁盘时间。
"$VEA_ENV/bin/python" scripts/run_v11_test.py --manifest "$FREEZE" --limit 2
# 最小 Smoke：固定前两图，两路最多 4 次调用。
timeout --signal=INT --kill-after=30s 10m \
  "$VEA_ENV/bin/python" scripts/run_v11_test.py \
  --manifest "$FREEZE" --limit 2 --execute \
  > outputs/flickr8k-v11-independent-smoke-01.log 2>&1
# 同目录续跑到 100 图；含 Smoke 累计最多 200 次，成功或失败均不重做。
timeout --signal=INT --kill-after=30s 60m \
  "$VEA_ENV/bin/python" scripts/run_v11_test.py \
  --manifest "$FREEZE" --limit 100 --resume --execute \
  > outputs/flickr8k-v11-independent-full-01.log 2>&1
```

固定输出 `outputs/flickr8k-v11-independent100-01/`；两路各保存 SQLite、manifest 和全部编号 snapshots。中断后复用完整续跑命令，仅将日志改成新编号，避免覆盖。每图每路 lifetime max_attempts=1，已登记的中断 attempt 消耗额度，失败计零。不删样本、不扩大 token/重试、不根据 test 改策略。未来若改方法，须登记新版本并另留未用 test。

规划预算：200 次×10 秒×1.2 余量约 40 分钟，另计模型哈希、冷加载和 I/O，尚无本次实测。Smoke 外层 10 分钟，完整续跑 60 分钟，不自动重启。费用为实际占用小时×用户确认单价；本轮无 GPU 开支。OOM、环境/哈希不符或超时均保留记录并停止。

## 预定指标、95% 区间与备份

完整分母 100；失败、未完成和中断计零，不能以成功子集替代主结果。

- Hit@1/5/10：前 K 至少命中一个同图描述；MRR@10：首个相关排名倒数，未命中/失败为零。CaptionRecall@K 辅助报告五个目标中命中的比例。
- Structured 原 Schema 与附加合约成功率分开，分母 100，独立于 token-limit 接受状态；Baseline JSON Schema 率不适用。两路均报流程成功率/失败率。
- 返回响应（包括被拒绝者）的 latency_ms、有起止记录的全部 attempt wall time、成功者 retrieval_ms（明确覆盖数量）和冷加载。固定先 Baseline 后 Structured，不把总 wall time 差异当推理加速比。
- 每图状态及 Hit/MRR/CaptionRecall 配对差。Hit@10 差值方向为 format_details−Baseline；95% 区间采用**图片对** percentile bootstrap，20,000 次，seed=0，2.5%/97.5% 线性插值分位点，失败保留为零。五个 caption 不能当独立样本。恒定差值可能产生退化区间，须如实报告；区间不解决预训练暴露或证明总体泛化。

仅在授权完成实验、进程停止后评分与备份；本轮没有 test 评分：

```bash
"$VEA_ENV/bin/python" -m visionevidence.evaluation.caption_pair \
  --run outputs/flickr8k-v11-independent100-01 --limit 100 \
  --output outputs/flickr8k-v11-independent100-01/paired-report-0001.json
# 备份使用新编号，确保没有进程继续写 SQLite。
tar -czf outputs/flickr8k-v11-independent100-backup-01.tar.gz \
  outputs/flickr8k-v11-independent100-01 \
  outputs/flickr8k-v11-independent-smoke-01.log \
  outputs/flickr8k-v11-independent-full-01.log \
  experiments/v11/frozen-format-details-20261020-01 \
  data/raw/flickr8k-v11-test-01
sha256sum outputs/flickr8k-v11-independent100-backup-01.tar.gz \
  > outputs/flickr8k-v11-independent100-backup-01.tar.gz.sha256
```

将全部 dev 原件、test 备份与 SHA256 下载到 Windows 校验，确认后由用户停止付费实例。数据/完整输出只作私用研究备份，不自动公开再分发。本轮未启动/停止云实例、下载模型或推送 GitHub。

## CPU 验证与限制

见正式目录 validation.json/pytest.xml。审计只重放保存的 dev 查询，不生成新查询；配对报告器也已在真实 dev 副本上运行，没有接触独立 test 排名。首次全量测试受原有 Git 所有权检查影响，使用仅对测试进程生效的 safe.directory 重跑，不修改全局 Git 配置。

相似图片描述可能被当作负例，五个 caption 高度相关，模型预训练可能见过 Flickr8k，20 图 dev 选择可能过拟合。独立 100 图结果未知，不能声称已提升独立 test 性能或事实核验能力。下一步由用户手动推送、传输输入包并下载固定 100 图；后续明确授权后才运行 Smoke 与完整实验。
