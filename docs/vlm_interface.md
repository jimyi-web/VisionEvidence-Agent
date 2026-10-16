# VLM 接口、Mock 与 Qwen3-VL — T03-B-1

## 当前实现和用途

已实现统一同步接口、`MockVLM` 和可选 `Qwen3VLBackend`，用于独立的图像描述路径。
Qwen3-VL 后端从本地目录加载模型，在单 CUDA GPU 上执行 BF16 推理；本任务仅通过 CPU
模拟测试验证控制流，未在真实 GPU 上安装、加载或运行模型。没有在线模型 API 或隐式权重下载。
Mock 测试通过只说明软件接口符合契约，不能用于推断模型准确率、推理速度或显存需求。

## 输入和输出

`vlm/base.py` 定义 `BaseVLM`（runtime-checkable Protocol）：
`generate(request: VLMRequest) -> VLMResponse`。实现此方法的对象可以直接注入消费者，
不要求继承某个基类，也没有全局模型实例或隐式初始化。

请求为冻结的 dataclass：

| 字段 | 含义 |
| --- | --- |
| `image_path` | `Path` 或字符串，本地普通文件；相对路径按当前工作目录解释 |
| `prompt` | 非空字符串，去除首尾空白；自定义提示词不保证遵守观察提示词约束 |
| `request_id` | 可选非空字符串，缺省为 None；调用者负责唯一性，不生成随机 ID |
| `max_image_size_mb` | 有限正数，默认 10；单位为 MiB，即 1024×1024 字节 |

响应为冻结的 dataclass：

| 字段 | 含义 |
| --- | --- |
| `request_id` | 原样关联经过校验的请求标识，未提供则为 None/JSON null |
| `description` | 字符串；生成文本始终是未验证内容，不等于事实或检索证据 |
| `model_name` | 非空的后端/模型标识；Mock 默认 mock-vlm，不代表已安装模型 |
| `backend` | Mock 为 mock；与 is_mock 标记保持一致 |
| `is_mock` | 必须为 bool；Mock 始终为 True，描述必须含 [MOCK] |
| `latency_ms` | 有限非负数，Mock 实测本地校验与文本准备耗时，不含响应构造/序列化 |
| `metadata` | JSON 对象，允许嵌套对象、列表、字符串、布尔、有限数字及 null |

`to_dict()` 返回独立的 JSON 安全字典，`to_json()` 生成 JSON 字符串。
metadata 拒绝非字符串键、NaN/Infinity、Path、集合、元组、循环引用等；
构造时深拷贝，序列化时再次验证。冻结 dataclass 不会冻结嵌套 metadata 容器。
真实后端显式使用 `is_mock=False` 和 `verified=False`，这不代表其描述已经被验证。

```python
from visionevidence.vlm import BaseVLM, MockVLM, VLMRequest, VLMResponse
from visionevidence.vlm.prompts import get_prompt

def observe(request: VLMRequest, vlm: BaseVLM) -> VLMResponse:
    return vlm.generate(request)

request = VLMRequest(
    image_path="examples/vlm_demo_fixture.png",
    prompt=get_prompt(),
    request_id="demo-1",
)
result = observe(request, vlm=MockVLM())
print(result.to_json())
```

Mock 返回固定 `[MOCK]` 文本，不根据文件名、用户声明或像素编造观察。
`metadata` 明确包含 `inference_performed=False`、`verified=False` 和
`content_source="fixed_mock_template"`。相同输入的内容与元数据确定；
`latency_ms` 是实测耗时，允许每次不同，不能比较成模型性能指标。

## 提示词与配置

`prompts.py` 保留英文 `visual_observation_v1`，要求描述直接可见的对象、场景和文字，
区分直接观察与推测，不凭空推断拍摄时间/地点，不将用户声明视为事实，明确不确定性，
不输出谣言真假标签。图片内的指令也作为未验证内容处理。提示词尚未经过真实模型评测。

`configs/vlm_mock.toml`：

```toml
[vlm]
backend = "mock"
model_name = "mock-vlm"
max_image_size_mb = 10
prompt_version = "visual_observation_v1"
```

标准库 `tomllib` 读取配置；只允许 `[vlm]` 表。Mock 保留以上四项必填字段。
Qwen 配置必填 backend/model_name/prompt_version，其余使用 VLMConfig 默认值；
model_path 在创建真实后端时必须指向现有目录。允许的扩展字段见下表，未知键仍拒绝。
未知键、缺字段、类型/取值错误、未知提示词版本、无效 TOML/UTF-8 和读文件失败均报清晰错误。
不提供 `--config` 时使用相同内置默认值，不依赖仓库相对路径来寻找配置。
CLI `--backend` 覆盖已校验配置中的 backend；其他配置字段仍需有效。
`--prompt` 覆盖默认提示词，输出 metadata 中的 `prompt_version` 改为 custom。

`create_vlm("mock", model_name="mock-vlm")` 返回 `MockVLM`。
`create_vlm("qwen3_vl", config=config)` 返回惰性加载的 Qwen3VLBackend。
未提供有效 model_path 时抛出 ValueError，早于运行库导入；未知后端也抛出 ValueError。
可直接构造 `Qwen3VLBackend(config)`，公开方法仍为 `generate(request) -> VLMResponse`。
缺少可选运行库抛出 BackendUnavailableError（保留原有 NotImplementedError 继承关系），
CUDA 不可用、BF16 不支持、加载或生成失败抛出 VLMInferenceError，并保留异常原因。
不存在自动 CPU/Mock fallback 或伪造成功结果。

| 配置字段 | 默认值/作用 |
| --- | --- |
| model_path | None；真实后端必须配置现有本地目录，相对路径按当前工作目录解释 |
| device | cuda:0；支持 cuda（当前设备）或 cuda:N，不跨 GPU 自动分片 |
| dtype | bfloat16；当前拒绝其他精度，检查硬件原生 BF16 支持 |
| max_new_tokens | 256，正整数 |
| do_sample | false；另固定 num_beams=1、num_return_sequences=1 并执行 model.eval() |
| max_image_pixels | 20000000；原图宽×高上限，超限在解码像素前拒绝 |
| max_image_side | 1280，至少 32；EXIF 后 RGB 副本按比例缩小，原文件不变 |

`configs/vlm_qwen3_vl.toml` 提供完整示例，model_name 为 Qwen/Qwen3-VL-4B-Instruct。
`runtime_loader` 是后端构造器的 CPU 测试注入点，不是生产降级机制。
实例内锁串行化首次加载和生成；成功加载后每次复用，失败不会缓存半加载模型。
生成使用 torch.inference_mode()；模板生成图像占位符和文本，再由 processor 接收已解码 PIL
图片编码输入，避免处理器自行读取 URL。使用 Qwen3-VL 官方类及已核实的 4.57.1 API。
输出按输入序列长度裁剪，只解码新增 token。

真实响应 metadata 包含 inference_performed、verified、device、dtype、torch_version、
transformers_version、model_path、max_new_tokens、do_sample、generated_tokens、
peak_gpu_memory_mb、prompt_version，以及 model_load_ms、model_loaded_this_call、
generation_ms、image_preprocessing_ms、processed_image_size 和图片限制。
latency_ms 排除模型加载和锁等待；generation_ms 在 CUDA 同步边界计时。
峰值字段为 torch 分配显存 MiB，包含常驻模型但排除加载阶段峰值。
完整测量口径、依赖版本依据与云端验收命令见 [部署说明](qwen3_vl_deployment.md)。

## Windows CLI 与测试

在项目根目录使用已有 `.venv`，无需激活或修改执行策略：

```powershell
.\.venv\Scripts\python.exe -m visionevidence.cli --help
.\.venv\Scripts\python.exe -m visionevidence.cli vlm --help
.\.venv\Scripts\python.exe -m visionevidence.cli vlm --image examples/vlm_demo_fixture.png --backend mock
.\.venv\Scripts\python.exe -m visionevidence.cli vlm --image examples/vlm_demo_fixture.png --config configs/vlm_mock.toml --request-id demo-1 --json
.\.venv\Scripts\python.exe -m visionevidence.cli vlm --image missing-vlm-fixture.png --backend mock
.\.venv\Scripts\python.exe -m visionevidence.cli vlm --image examples/vlm_demo_fixture.png --backend qwen3_vl
.\.venv\Scripts\python.exe --version
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m pytest
git diff --check
git status --short
```

也支持安装后的 `.\.venv\Scripts\visionevidence.exe vlm --image ... --backend mock`。
成功返回 0，参数/图片/配置/依赖/推理错误返回 2，错误走 stderr，不输出成功 JSON。
`vlm --image` 与旧的顶层 `--text/--image` 参数分离；T01 输入回执及 T02 retrieve/evaluate 保留。
JSON 使用 Unicode 转义兼容 Windows 终端；文本显式显示 MOCK/is_mock=True 或 REAL/is_mock=False，
真实文本标题同时标记 unverified model output。

如果系统 pytest 临时目录有清理权限问题，可使用
`.\.venv\Scripts\python.exe -m pytest --basetemp outputs/t03a-pytest-acceptance-01`。
每次选新的、未使用的临时目录名，因为 pytest 会清空已有 basetemp，不可指向实验结果目录。

示例图片 `examples/vlm_demo_fixture.png` 是标准库 struct/zlib 生成的 1×1 白色 RGB PNG，
仅 69 字节，明确为 synthetic/demo fixture。测试校验 PNG chunk CRC、结构与解压像素。
VLM 测试覆盖请求、响应、图片边界、配置、Mock 确定性、依赖注入与 CLI。
隔离子进程在导入 VLM 之前阻止 GPU 框架导入、网络连接及外部进程调用，运行 Mock 全路径。
测试不下载模型，不依赖外网或 GPU，也不安装大型依赖。

## 图片校验与安全边界

- 必须是现有普通文件；扩展名不区分大小写，支持 .png、.jpg、.jpeg、.webp。
- 在读取内容前检查大小，默认 ≤10 MiB；只读取最多 33 字节文件头，不整文件读入内存。
- PNG 检查 signature/IHDR/非零尺寸；JPEG 检查 SOI、首个段标记与段长度；
  WEBP 检查 RIFF/WEBP、VP8/VP8L/VP8X 类型和容器/首块长度，并要求与扩展名一致。
- 请求构造及 Mock generate 时分别检查，防止普通调用中请求创建后文件被删除或替换。
- 拒绝 URL 和 UNC 路径，不自动下载；不使用 pickle 或其他对象反序列化。
- 这是轻量文件头校验，不能保证完整图像可解码性、全部 chunk/压缩流有效性或文件无恶意内容。
  Mock 保留该行为；真实后端另用 Pillow verify/load 完整解码，处理 EXIF、转 RGB、约束像素和尺寸。
- Qwen 后端将 Pillow DecompressionBombWarning 视为错误，也拒绝 DecompressionBombError，
  不关闭或修改 Pillow 全局像素保护；不放宽截断图片解码。动画/多帧图片拒绝。
  处理器还可能按 patch 对齐再次缩放，限制 max_pixels=max_image_side²、min_pixels=32²；
  metadata 的 processed_image_size 是交给处理器前的 RGB 副本尺寸，不是最终 patch 网格。
- 路径校验不是操作系统沙箱，不防止文件系统竞争替换、符号链接或映射盘重定向。
  当前用于可信本地开发目录，不作为不可信文件上传服务的安全边界。

## 测试范围与未实现功能

`tests/test_qwen3_vl.py` 用真实 Pillow 测试 PNG/JPEG/WEBP、EXIF、RGB、损坏图片、
像素上限和解压炸弹；注入运行库验证加载参数、模型复用、输入 token 裁剪、输出契约、
计时分离、CUDA/BF16 不可用和失败传播。CPU doubles 的输出和显存数值仅是测试夹具。
原有测试保留；仅随本阶段更新“后端未实现”错误与 CLI 帮助的旧断言。

尚未完成真实 GPU 验收、模型质量评估、批处理/异步、采样 seed 管理、Agent 编排、
证据融合与真假判断。未实现专用 OCR 模块，也未验证模型 OCR 质量。
默认 greedy 不保证跨硬件/版本逐位一致；正式实验需保存模型 revision、输入与完整环境。
模型下载是部署文档中的独立人工步骤，不属于 generate。云 GPU 操作与花费需人工批准。
