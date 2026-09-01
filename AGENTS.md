# 项目概述 (Project Overview)
`transvideo` 是一个基于 Python 的实用工具，用于处理视频文件，将音频转录为文本并翻译成中文。转录默认使用 `mlx-whisper`（跑 Apple Silicon GPU），并保留 `faster-whisper`（CTranslate2）作为备选后端；媒体处理依赖 `ffmpeg`。该工具可以分多个阶段执行（音频提取、转录、翻译、生成字幕以及视频合成），并支持从失败的阶段继续执行。此外，项目还包含一个 `srt_to_transcript.py` 脚本，用于将 SRT 字幕文件转换为纯文本转录内容。

## 主要技术栈 (Main Technologies)
- **Python**: >=3.12
- **依赖管理**: `uv`（见 `pyproject.toml`, `uv.lock` 以及 `Makefile`）
- **核心依赖库**: `mlx-whisper`（仅 Apple Silicon）、`faster-whisper`, `translators`, `lxml`, `requests`
- **系统要求**: `ffmpeg`（用于视频和音频处理）

# 构建与运行 (Building and Running)

## 前置条件 (Prerequisites)
请确保您的系统中已安装 `ffmpeg`：
```bash
brew install ffmpeg
```

## 全局安装 (Installing Globally)
想在项目目录之外直接用 `transvideo` 命令，用 pipx 装打包好的 wheel：
```bash
make install     # 清 dist -> uv build -> pipx 安装 dist/*.whl
make uninstall   # pipx uninstall transvideo
```
`make install` 会按 `pyproject.toml` 里的 `requires-python` 挑一个 uv 管理的解释器传给 `pipx --python`，因为 pipx 默认的解释器版本未必满足这个约束。它先 `pipx uninstall` 再安装而不是用 `--force`：`--force` 会复用已存在的 venv 并忽略 `--python`，导致旧 venv 的解释器版本一直沿用下去。

## 运行项目 (Running the Project)
由于项目使用 `uv` 管理依赖和运行脚本，统一通过 `transvideo` 命令调用（基于 click 实现，入口在 `src/cli.py`）：

**运行视频处理主流程：**
```bash
uv run transvideo run {PATH_TO_VIDEO}
```
*可选参数：*
- `--stages {audio,transcribe,translate,srt,compile}`: 运行指定的阶段，可重复传递以选择多个；不传则跑完所有阶段。
- `--soft`: 以软字幕的形式将字幕外挂在视频上（可以在播放器中关闭），而不是直接烧录进视频。
- `--translator {bing,ollama}`、`--ollama-url`、`--ollama-model`: 选择翻译引擎及其参数。
- `--whisper-backend {mlx,faster}`: 推理后端，默认 `mlx`。
- `--whisper-model`: 所选后端的模型。`mlx` 传 MLX 仓库名或本地路径（默认 `mlx-community/whisper-large-v3-turbo`）；`faster` 传模型尺寸（如 `tiny`、`medium.en`、`large-v3`）或本地 CTranslate2 路径（默认 `large-v3`）。
- `--whisper-device`、`--whisper-compute-type`: 推理设备与 CTranslate2 计算精度，**仅对 `faster` 后端生效**。
- `--language`: 指定语种（如 `en`）以跳过自动检测，不传则自动检测。
- `--glossary`: 专有名词术语表路径，默认读 `src/glossary.txt`。该文件未入库（见 `.gitignore`）——里面是内部产品代号和客户名，而仓库是公开的。缺失时会告警并退化为不加偏置的转录，格式见下面的「专有名词转录」一节。

### 推理后端 (Inference Backends)
CTranslate2 没有 Metal 后端，`faster` 在 macOS 上只能跑 CPU，`large-v3` 大约是实时速度。`mlx` 走 Apple Silicon GPU，因此是默认值。

M3 Max 上对同一段 5 分 13 秒英文音频实测（均带 glossary）：

| 后端 | 模型 | 耗时 | 相对实时 |
|---|---|---|---|
| `faster` | `large-v3` (int8) | 272.1s | 1.15x |
| `mlx` | `whisper-large-v3-turbo` | 13.3s（首次）/ 10.3s（模型已加载） | 23.5x / 30.3x |

换算过来，一小时视频的转录阶段从约 52 分钟降到约 2 分钟。

- **`mlx`（默认）**：模型 `whisper-large-v3-turbo` 保留了 large-v3 的 encoder，但 decoder 从 32 层砍到 4 层，速度提升主要来自这里。追求精度可换 `mlx-community/whisper-large-v3-mlx`，追求更快可换 `-q4` / `-8bit` 量化版。仅 Apple Silicon 可用（`pyproject.toml` 里带 `platform_machine == 'arm64'` marker，因此 darwin x86_64 仍能解析依赖）。
- **`faster`**：非 Apple Silicon 机器上的兜底，也是唯一支持 `hotwords` 的后端。CTranslate2 默认线程数偏保守（M3 Max 上只吃满约 5 核），被迫用这个后端时可以试 `--whisper-cpu-threads`。

### 专有名词转录 (Proper Nouns)
Whisper 容易把专有名词转错（例如 `Veeva` → `Vive`/`Viva`）。`src/glossary.txt` 中的术语用于在解码时偏向这些拼写，两个后端的注入方式不同：

- `faster` 后端拼成 `hotwords` 传入。
- `mlx` 后端没有 `hotwords` 等价物，术语改为拼成 `initial_prompt`（`This transcript mentions: ...`）作为前文上下文注入。这条路径实测有效：同一段音频在 `mlx` 上带 glossary 时 `Veeva` 命中 14 次，不带时 **0 次**（全被转成 Viva/Vieva 之类）。若某个专有名词仍然转错，把它挪到 `glossary.txt` 靠前的位置，或改用 `--whisper-backend faster`。

新增术语直接编辑该文件（一行一个，`#` 为注释；该文件不入库，需自行创建）。保持精简有两个理由：两个后端都有约 223 token 的截断上限，超出部分静默丢弃；且在 `mlx` 上 glossary 是作为解码前缀注入的，当前这个长度的词表会让转录慢约一倍（10.3s → 5.5s 是去掉 glossary 后的耗时）。把最常错的词放在前面，不要无限堆砌。

**SRT 转纯文本：**
```bash
uv run transvideo srt2txt input.srt [-o output.txt]
```

## 测试 (Testing)
项目中有一个测试文件 `src/test_transvideo.py`。你可以通过以下方式运行测试：
```bash
uv run pytest src/test_transvideo.py
```
*（假设使用了 `pytest`，虽然依赖中没有明确列出，但标准的 Python `unittest` 也可能适用：`uv run python -m unittest src/test_transvideo.py`）*

# 开发约定 (Development Conventions)
- **工具使用**: 优先使用 `uv` 进行环境管理和脚本运行，而不是全局的 `pip`（尽管 README 中仍有 `pip install` 的历史遗留描述）。适用时请使用 `Makefile` 中的命令（如 `make build`, `make run`, `make install`, `make publish`, `make lint`）。
- **源码结构**: 所有的源代码和测试文件统一存放在 `src/` 目录下；CLI 入口位于 `src/cli.py`，业务逻辑分别在 `src/transvideo.py` 与 `src/srt_to_transcript.py`。
- **容错与恢复**: 主流程 (`Transvideo` 类) 会保留中间文件，以便在流程失败时能够从特定阶段恢复。在修改数据处理流水线时，请确保维持这一工作流特性。
- **Lint**: 完成任何 coding 之后必须执行 `make lint`，确保 `ruff check`、`ruff format --check` 与 `ty check` 全部通过后再交付。
