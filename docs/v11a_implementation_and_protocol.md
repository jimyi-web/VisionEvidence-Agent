# V1.1-A：生成约束、信息保留与独立评测方案

日期：2026-10-20。状态：**本地实现与 CPU/Mock 验收完成，未进行 V1.1 真实 GPU 评测，尚未选择最终策略。不能声称检索性能或真实生成合法率已经提升。**

本阶段依据 [V1.0 GPU 审计](flickr8k_real_gpu_3ec02ca_report.md) 中的枚举/循环截断、objects 超项和视觉细节丢失问题，增加可单独开关的最小约束。既有 Flickr8k test 前二十图现在仅作诊断集，不再称为独立测试集。任务仍是 **image-to-caption retrieval proxy**，不是事实核验准确率。

## Implemented：最小行为改动

原 `structured_visual_v1` 提示词文本、三个字段、旧版每字段最多八项/每项最多二百字符的 Schema、严格 JSON 解析器和查询构造器均保留。散文 Baseline 也不改。默认没有配置参数时继续使用旧策略；新增的 trace、manifest、summary 和 attempts 导出属于记录增强，因此不声称新运行文件与历史文件逐字节相同。

新增两个正交开关，均只接受 TOML 布尔值：

| 配置 | format_constraints | retain_visual_details | prompt_version |
| --- | --- | --- | --- |
| [v10_control](../configs/v11/v10_control.toml) | false | false | structured_visual_v1 |
| [format_only](../configs/v11/format_only.toml) | true | false | structured_visual_v11a_format |
| [details_only](../configs/v11/details_only.toml) | false | true | structured_visual_v11a_details |
| [format_details](../configs/v11/format_details.toml) | true | true | structured_visual_v11a_format_details |

四种 Structured 以外，另保留 `concise_visual_description_v1` 散文 Baseline；不要混淆“旧版 Structured 控制组”和“散文 Baseline”。配置协议为 observation_policy_v1，查询构造版本为 objects_visible_text_ascii_v1。

### 格式约束开关

在原提示之后追加生成合约：objects≤6、visible_text≤2、uncertainties≤2；不要求填满；每字符串≤100 字符；全部字符串合计≤60 个按空白分隔的词。先选择显著项，再输出完整 JSON 并停止；不展开“物体的部件的部件”、备选名称、同义词或泛化词表。不重复同一字段中的条目，明确可见的多个实例可合并成带数量的描述，不猜数量。

执行顺序为：检查 token 元数据与上限 → 原严格 JSON/Schema → 新格式合约 → 查询构造 → BM25。新格式合约进一步拒绝超项、超长、总词数超限及同字段重复（casefold 与空白归一后的相同字符串）。不会删除第七项、去重非法列表、截断字符串、补括号、提取 Markdown 中的 JSON、回退散文或自动扩大 token 上限。任何异常都保留 error 和原始 response。

这是**提示＋更严格接收合约**的工作流消融，不是仅更改提示的单变量试验。有效 JSON 可以仍违反原 Schema 或新合约，报告时必须分开统计。60 个空白词也不等价于 384 个模型 token；提示不能数学保证解码永不重复，真实生成是否遵守仍待 GPU 测量。更严格合约可能降低成功率，这属于应报告的实验结果，不能静默修复。

### 信息保留开关

不加新字段，在 objects 字符串中要求使用简洁的视觉短语：显著对象与其可见动作/姿态、区分属性放在一起；有区分力时保留场景/空间关系；可确定的计数或可辨认数字原样保留。优先这些信息，避免无关背景堆砌；不添加词库、同义扩展、推断意图/地点或参考描述。只把确定可辨认的文字放进 visible_text，不确定读法只放 uncertainties；不输出“No text is visible”类套话。

**查询构造器不做扩词或答案引导重写。** 它原本就会保留短语中的 ASCII 词和数字，新增测试明确覆盖对象、动作、属性、场景、数量词和数值的传递及来源追踪。例如合成观察可传递为 `two red cars crossing a bridge wet city road 27`；若模型只输出 object，程序不会自行补出动作或颜色。不确定 OCR 的隔离目前是生成提示要求，没有新增声称能判断视觉置信度的分类器。

两开关均不改检索器、候选库、qrels、token 上限、greedy、seed 或模型权重。保留旧行为使得可以单独观察格式约束、信息保留及两者结合的结果。

### 接入、记录与兼容性

`visionevidence.batch` 和 `visionevidence.caption_experiment` 新增 `--observation-config`。配对入口仅把该策略用于 Structured，散文分支保持原样。manifest/experiment/summary/comparison/trace 保存开关、配置版本、实际完整提示与提示版本；原有源码/模型/数据哈希继续生效。开关变化会使续跑身份校验失败，必须建新目录。

新生成的 `attempts-NNNN.jsonl` 是累计尝试快照，含成功、失败、精确错误与嵌套原响应；`results-NNNN.jsonl` 继续只导出成功结果。缺图等生成前失败的 raw_response 为 null，不伪造响应。旧 SQLite 和编号快照不被迁移或重写。新源码不能直接续跑旧版本实验：应保留历史 Commit 和原件，另建目录比较。

## Files Changed

- [observation_policy.py](../src/visionevidence/observation_policy.py)：配置、独立提示后缀、严格附加合约。
- [observation.py](../src/visionevidence/observation.py)、[batch.py](../src/visionevidence/batch.py)、[caption_experiment.py](../src/visionevidence/caption_experiment.py)：策略接入、来源记录、全 attempts 导出；旧默认生成/检索行为保留。
- [flickr8k_cohort.py](../src/visionevidence/datasets/flickr8k_cohort.py)：离线队列副本，原 split 和完整候选保持，源文件不改。
- [四个策略配置及实验计划](../configs/v11/evaluation_plan.toml)、[策略测试](../tests/test_observation_policy.py)、[队列测试](../tests/test_flickr8k_cohort.py)、本说明与 `experiments/v11a/acceptance-20261020/` 的验收证据。
- 前一轮尚未提交的 V1.0 GPU 审计报告/脚本/测试/精简归档也随本地提交保存；不是对 V1.0 原始实验输出的改写。

## dev/test 方案：选择前固定队列，选择后冻结方法

版本化计划见 [evaluation_plan.toml](../configs/v11/evaluation_plan.toml)。此文件是科研计划，不是 `--config` 或 `--observation-config` 的输入。当前 selected_strategy=pending_dev_selection。

1. **诊断边界。** V1.0 test 位置 1–20 已公开分析，仅可标为诊断/回归。不能通过反复查看它的标签或排名来调 prompt、采样规则或检索词。V1.1 单元测试和 Mock 验收仅使用合成图片/参考文本，没有重跑真实 test 检索来挑策略。
2. **开发队列。** 同一固定数据 Commit 下原 dev 文件名升序前 20 图（offset=0, limit=20），所有方法同一队列，候选保持原 dev 全 5,000 描述。可先跑固定前两图检查加载/记录，再在同配置/同目录续跑至二十图，累计调用数不重复相加。故障、零命中都不换图、不删样本；实际获取新图片需要后续人工执行。
3. **五路开发比较。** 散文 Baseline 一次，Structured 四种策略各一次；每路同模型、384 token、greedy、seed=0、BM25 参数、图片前缀、候选与 qrels，max_attempts=1。全图片计分，失败/待处理计零。记录严格 JSON 率、原 Schema 率、新合约率、流程成功率、Hit@1/5/10、CaptionRecall@K、MRR@10、调用/token/时延/显存、错误和原响应；共同成功子集只作辅助诊断。
4. **预定选择规则。** Structured 候选必须先达到开发队列 20/20 流程成功，再依次按 dev Hit@10、MRR@10、较少 generated_tokens 选择；完全相同优先开关更少，再优先 format_only。若没有候选通过可靠性门槛，保留负结果并停止进入独立测试，另行登记下一版方案。该规则不是性能提升承诺；不得用 test 结果打破平局。
5. **方法冻结。** 完成开发选择后，先在新文件记录选中配置全文/哈希、实际完整 prompt/哈希、源码 Commit 与源码哈希、模型文件哈希、软件/硬件、生成/图像/检索配置、开发所有候选结果与选择理由、独立 test 图片 ID/顺序、候选/qrels 哈希、指标/失败规则/预算；完成本地 Commit。此轮只冻结计划与候选实现，不伪造一个尚未完成的开发选择。
6. **独立评测。** 冻结后使用原 test 位置 21–120（offset=20, limit=100），完整 test 5,000 描述候选，原配对 qrels。这批图未被本项目用于策略开发；不能保证模型预训练未见过 Flickr8k。运行散文 Baseline 与已冻结选中策略，不在 test 上扫描四种配置后挑最好者。无论结果如何都保留并报告；若再改策略，这一批也变成已使用数据，下一版需另留未见队列。

离线队列工具只按 samples 的原文件名顺序切片，再复制原 qrels；它不按 caption、分数或结果选择。原 corpus/captions 保持完整字节，候选不会缩成二十图的 100 条描述。工具改写副本中的相对图片路径，使它仍指向同一原图；源 adapter 与标签不改。它拒绝把 test 声明为 development、把前二十图纳入 heldout、在原 adapter 内写输出或再次切分已有 cohort。`heldout_requires_prior_method_freeze` 是协议标记，**程序不校验一个外部冻结文件是否已签署**，冻结顺序由开发者执行并审阅。

## 本地验证与 Mock 使用

```powershell
.\.venv\Scripts\python.exe -m pytest -q --basetemp outputs/pytest-v11a-full-20261020-02 --junitxml=outputs/pytest-v11a-full-20261020-02.xml
.\.venv\Scripts\python.exe -m visionevidence.caption_experiment --dataset <合成适配包>/dev --output outputs/v11a-mock-new --mode mock --limit 2 --observation-config configs/v11/format_details.toml
```

本机 Git 所有权检查要求为测试子进程添加 safe.directory。实际完整测试由项目解释器启动 subprocess，并在其环境中追加 GIT_CONFIG_COUNT / GIT_CONFIG_KEY_n=safe.directory / GIT_CONFIG_VALUE_n=项目绝对路径；不改全局 Git 配置。首次不带该设置时 414 passed、4 failed，四项均在读取 Git Commit 的旧代码路径失败；带该进程设置重跑为 **418 passed in 54.74s**。

新增 39 项策略/队列测试覆盖四种消融、旧 prompt SHA256 与八项边界、未知/类型错误配置、七项/九项对象、重复、长度/总词数、OCR/uncertainties 超项、坏 JSON、完整 JSON 但已达 token 上限、查询保留与不扩词、失败原响应导出、CLI 失败状态、来源隔离、缓存/续跑策略变更拒绝、dev/test 边界和原 corpus/qrels/图片路径保持。已有解析、VLM CPU doubles、Mock、断点恢复、错误路径等均纳入完整测试；CPU doubles 不是 GPU 运行证据。

另实际运行合成 dev 两图的四种配对 Mock＋续跑，八个分支均 completed，续跑新增 attempts=0；并以五个独立 CLI 命令跑过合成两图五路流程和上述 CPU 计分方式，十次 Mock 调用，控制字段一致、分母完整。两项均真实 VLM 调用=0。固定 Mock 对所有策略返回相同合成观察，分数只用于接口验证，不能作为视觉/检索改善依据。验收材料在 [validation.json](../experiments/v11a/acceptance-20261020/validation.json)、[Mock 验收](../experiments/v11a/acceptance-20261020/mock-acceptance.json) 和 [五路 CLI 验收](../experiments/v11a/acceptance-20261020/five-arm-cli.json)，完整输出在 `outputs/v11a-cpu-acceptance-20261020-01/` 与 `outputs/v11a-five-arm-cli-20261020-01/`。

## GPU 运行命令：留待开发者另行执行

以下都未在本轮执行。只使用现有、已获准的 Linux GPU 环境和本地模型目录；不复制 Windows .venv，不自动启动实例或下载模型。同步此 Commit 后先检查 `configs/t05_qwen3_vl.toml` 的 model_path。数据下载写新目录，不覆盖 V1.0 数据/输出。

### 后续获取 dev，并建立队列

```bash
export VEA_ENV=/root/autodl-tmp/vea-envs/qwen3-vl
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
# 下载是后续人工动作；固定数据版本，只获取 dev 前 20 图。
"$VEA_ENV/bin/python" -m visionevidence.datasets.flickr8k download \
  --output data/raw/flickr8k-v11-dev --split dev --limit 20 \
  --revision c7a02cb487da0eca729e133194b0f842cb578c08
"$VEA_ENV/bin/python" -m visionevidence.datasets.flickr8k adapt \
  --input-root data/raw/flickr8k-v11-dev --output data/processed/flickr8k-v11-original
"$VEA_ENV/bin/python" -m visionevidence.datasets.flickr8k_cohort \
  --dataset data/processed/flickr8k-v11-original/dev \
  --output data/processed/flickr8k-v11-dev20 --purpose development --limit 20
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
```

### 节省调用的五路开发运行

用 batch 分别运行五路，使散文只调用一次；每个进程内部复用模型，五个命令各有加载/模型哈希成本。不要同时运行这些 GPU 进程争抢显存。下面可先将所有 limit 改成 2 检查，随后保持其余配置不变、改回 20 并对每条命令加 `--resume`；失败仍保留，max_attempts 不提高。

```bash
export DEV=data/processed/flickr8k-v11-dev20/dev
export RUN=outputs/flickr8k-v11-dev20-run01
timeout --signal=INT --kill-after=30s 10m "$VEA_ENV/bin/python" -m visionevidence.batch \
  --samples "$DEV/samples.jsonl" --corpus "$DEV/corpus.jsonl" --qrels "$DEV/qrels.jsonl" \
  --output "$RUN/baseline" --mode qwen3_vl --config configs/t05_qwen3_vl.toml \
  --task image_to_caption_retrieval_proxy --query-mode baseline \
  --limit 20 --batch-size 4 --max-attempts 1 --top-k 10
for POLICY in v10_control format_only details_only format_details; do
  timeout --signal=INT --kill-after=30s 10m "$VEA_ENV/bin/python" -m visionevidence.batch \
    --samples "$DEV/samples.jsonl" --corpus "$DEV/corpus.jsonl" --qrels "$DEV/qrels.jsonl" \
    --output "$RUN/$POLICY" --mode qwen3_vl --config configs/t05_qwen3_vl.toml \
    --task image_to_caption_retrieval_proxy --query-mode structured \
    --observation-config "configs/v11/$POLICY.toml" \
    --limit 20 --batch-size 4 --max-attempts 1 --top-k 10
  # incomplete 返回 1：保留记录并审查；不要使用 set -e 自动跳过后面的负结果记录。
done
```

批量 summary 的 recall 是五描述召回比例；Hit@K 必须另外计算。以下在 CPU 上以全部二十图计分并先核对五路控制字段，输出文件用 x 模式，已有结果不会覆盖。若进程发生硬中断，先以相同配置 resume 完成快照导出，不静默增加重试预算。

```bash
"$VEA_ENV/bin/python" - "$RUN" <<'PY'
import json, sys
from pathlib import Path
from visionevidence.caption_experiment import score_branch
root = Path(sys.argv[1])
arms = ['baseline', 'v10_control', 'format_only', 'details_only', 'format_details']
identities = [json.loads((root / arm / 'manifest.json').read_text())['identity'] for arm in arms]
for field in ('samples_sha256', 'corpus_sha256', 'qrels_sha256', 'vlm_config',
              'model_file_sha256', 'source_sha256', 'retrieval', 'seed'):
    assert all(item[field] == identities[0][field] for item in identities), field
scores = {arm: score_branch(root / arm, 20, (1, 5, 10)) for arm in arms}
assert all(value['denominator'] == 20 for value in scores.values())
with (root / 'dev-scores-0001.json').open('x') as stream:
    json.dump({'task': 'image_to_caption_retrieval_proxy', 'scores': scores,
               'selection_source': 'dev only', 'method_selected': False}, stream, indent=2)
PY
```

也可使用 caption_experiment 针对四种配置各跑一次配对比较，它会自动输出 Hit@1/5/10；但 Baseline 会重复运行，二十图合计 160 次调用，比五路方案的 100 次多 60 次。不得把重复调用当成更多独立图片或更大样本。

### 完成 dev 选择并冻结后，获取独立 test

在修改本地图片可用性后应重新 adapt 到**新目录**，不能修改已生成的 adapter manifest。现有下载器按前缀获取，因此获取 test 前 120 图，再由队列工具排除已诊断前二十图；实际评测只有一百张。若新增下载失败，先保留收据并恢复下载，再 adapt，不用换图凑数。

```bash
# 仅在方法冻结、下载与 GPU 预算另行获准后执行。
"$VEA_ENV/bin/python" -m visionevidence.datasets.flickr8k download \
  --output data/raw/flickr8k-v11-dev --split test --limit 120 --resume \
  --revision c7a02cb487da0eca729e133194b0f842cb578c08
"$VEA_ENV/bin/python" -m visionevidence.datasets.flickr8k adapt \
  --input-root data/raw/flickr8k-v11-dev --output data/processed/flickr8k-v11-original-with-test
"$VEA_ENV/bin/python" -m visionevidence.datasets.flickr8k_cohort \
  --dataset data/processed/flickr8k-v11-original-with-test/test \
  --output data/processed/flickr8k-v11-heldout100 --purpose heldout_test --offset 20 --limit 100
# 将下一行替换成 dev 选中并已冻结的真实配置；不能在 test 上扫描配置。
export SELECTED_POLICY=/absolute/path/to/frozen-observation-policy.toml
timeout --signal=INT --kill-after=30s 1h "$VEA_ENV/bin/python" -m visionevidence.caption_experiment \
  --dataset data/processed/flickr8k-v11-heldout100/test \
  --output outputs/flickr8k-v11-heldout100-frozen01 --mode qwen3_vl \
  --config configs/t05_qwen3_vl.toml --observation-config "$SELECTED_POLICY" \
  --limit 100 --batch-size 4 --max-attempts 1
```

## 资源估算与停止条件

这是计划预算，**不是 V1.1 实测**。V1.0 单响应均值约 1.38 s（散文）/2.13 s（Structured，含失败），384-token 截断约 9.7 s；新提示更长、dev 图不同，不能直接保证沿用该均值。按保守每调用 10 s，再加 20% 时间余量：

| 阶段 | 独立图片 | 候选描述 | 真实调用（无重试） | 推理规划时间＋20% |
| --- | ---: | ---: | ---: | ---: |
| dev 前二图五路 smoke | 2 | 5,000 | 10 | 约 2 min |
| dev 二十图五路累计（含 smoke） | 20 | 5,000 | 100 | 约 20 min |
| test 一百图散文＋冻结策略 | 100 | 5,000 | 200 | 约 40 min |

另计每个命令的模型文件哈希、冷加载、建索引和磁盘成本；五路开发方案有五个进程加载，不能用响应合计当作端到端 wall time。总计开发与独立评测预计 300 次调用，不把先二图再续跑的快照重复相加。GPU 价格由开发者按届时实际计费确认，费用估算用实际占用小时 × 单价，不在本报告编造价格。

沿用已有 BF16、384-token、max_image_side=1280、本地 Qwen3-VL-4B 权重和顺序单模型推理；不增加下载权重或改量化。V1.0 torch allocated 峰值约 8,583 MiB 只作历史参考，不保证新配置容量。首次 dev 两图必须记录实际显存/时延/JSON/Schema/合约结果，之后再决定是否继续。数据只计划获取 20 dev＋前 120 test 图片，体积取决于实际下载收据；元数据和两个完整 5,000 描述候选不能算作 10,000 张图推理。

发生模型来源不符、OOM、枚举/截断或合约系统性失败时，保留记录并停止扩容；不通过删除失败或提高预算粉饰主结果。多个候选都不达成功门槛就停在开发阶段，不消耗独立 test。完成后备份全部输出，由开发者检查并关闭付费实例。

## Limitations 与 Next Step

这次已验证接口、严格拒绝、缓存/记录与数据边界；没有测量 V1.1 的真实 JSON 合格率、语义保留程度、检索性能或 GPU 成本。所有改善表述均为待检验假设。相似图/重复描述的代理负例、五描述相关性和预训练污染仍存在；独立一百图也不自动构成显著性或事实核验结论。

下一步是开发者审阅本地 Commit 与上述计划，另行批准 dev 小规模真实运行；先完成策略选择和方法冻结，再执行未用于开发的 test。此次没有启动/修改 GPU 实例、下载模型、改 qrels、覆盖 V1.0 结果或推送 GitHub。
