# 项目概述 (Project Overview)
`transvideo` 是一个基于 Python 的实用工具，用于处理视频文件，将音频转录为文本并翻译成中文。它使用 `openai-whisper` 进行转录，并依赖 `ffmpeg` 进行媒体处理。该工具可以分多个阶段执行（音频提取、转录、翻译、生成字幕以及视频合成），并支持从失败的阶段继续执行。此外，项目还包含一个 `srt_to_transcript.py` 脚本，用于将 SRT 字幕文件转换为纯文本转录内容。

## 主要技术栈 (Main Technologies)
- **Python**: >=3.12
- **依赖管理**: `uv`（见 `pyproject.toml`, `uv.lock` 以及 `Makefile`）
- **核心依赖库**: `openai-whisper`, `translators`, `lxml`, `requests`
- **系统要求**: `ffmpeg`（用于视频和音频处理）

# 构建与运行 (Building and Running)

## 前置条件 (Prerequisites)
请确保您的系统中已安装 `ffmpeg`：
```bash
brew install ffmpeg
```

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
- **工具使用**: 优先使用 `uv` 进行环境管理和脚本运行，而不是全局的 `pip`（尽管 README 中仍有 `pip install` 的历史遗留描述）。适用时请使用 `Makefile` 中的命令（如 `make build`, `make run`, `make publish`, `make lint`）。
- **源码结构**: 所有的源代码和测试文件统一存放在 `src/` 目录下；CLI 入口位于 `src/cli.py`，业务逻辑分别在 `src/transvideo.py` 与 `src/srt_to_transcript.py`。
- **容错与恢复**: 主流程 (`Transvideo` 类) 会保留中间文件，以便在流程失败时能够从特定阶段恢复。在修改数据处理流水线时，请确保维持这一工作流特性。
- **Lint**: 完成任何 coding 之后必须执行 `make lint`，确保 `ruff check`、`ruff format --check` 与 `ty check` 全部通过后再交付。
