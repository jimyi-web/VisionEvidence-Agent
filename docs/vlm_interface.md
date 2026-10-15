# VLM 接口与 Mock 测试 — T03-A

## 当前实现和用途

当前仅实现统一同步接口及 `MockVLM`，用于验证数据流、CLI 和未来 Agent 的依赖注入。
没有真实模型、图片语义理解、GPU 计算、网络调用、权重下载或在线模型 API。
未来目标为 Qwen3-VL-4B-Instruct，尚未安装、加载或验证其兼容性。
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
真实后端将来必须显式使用 `is_mock=False`，这仍然不代表其描述已经被验证。

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

标准库 `tomllib` 读取配置；必须且只能包含以上四项及 `[vlm]` 表。
未知键、缺字段、类型/取值错误、未知提示词版本、无效 TOML/UTF-8 和读文件失败均报清晰错误。
不提供 `--config` 时使用相同内置默认值，不依赖仓库相对路径来寻找配置。
CLI `--backend` 覆盖已校验配置中的 backend；其他配置字段仍需有效。
`--prompt` 覆盖默认提示词，输出 metadata 中的 `prompt_version` 改为 custom。

`create_vlm("mock", model_name="mock-vlm")` 返回 `MockVLM`。
`create_vlm("qwen3_vl")` 抛出 `BackendUnavailableError`（NotImplementedError 子类），
未知后端抛出 ValueError。不存在降级伪造成功或隐式真实模型加载。

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
成功返回 0，参数/图片/配置错误及未实现后端返回 2，错误走 stderr，不输出成功 JSON。
`vlm --image` 与旧的顶层 `--text/--image` 参数分离；T01 输入回执及 T02 retrieve/evaluate 保留。
JSON 使用 Unicode 转义兼容 Windows 终端；默认文本显式显示 MOCK 和 is_mock=True。

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
  后续真实后端必须使用受维护的解码器并限制像素数/解压资源；当前没有像素数上限。
- 路径校验不是操作系统沙箱，不防止文件系统竞争替换、符号链接或映射盘重定向。
  当前用于可信本地开发目录，不作为不可信文件上传服务的安全边界。

## 后续接入位置与未实现功能

未来可新增 `vlm/qwen3_vl.py` 并实现 `BaseVLM.generate`，在 `factory.py` 中增加显式分支。
模型初始化与复用属于后端实例，保持业务调用接口不变；加载路径/revision、确定性生成参数、
BF16 支持、实测显存/延迟及异常处理需要单独实现和验证。
模型依赖、权重下载、云 GPU 操作与花费必须先获得授权。

当前未实现：真实 VLM、像素解码/OCR、GPU 内存统计、模型下载、在线 API、批处理/异步、
模型质量评估、Agent 编排、证据融合与真假判断。T03-A 不引入以上功能或额外依赖。
