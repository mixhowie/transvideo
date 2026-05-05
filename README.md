# Quick Start
1. 安装 ffmpeg
```
brew install ffmpeg
```
2. 安装依赖
```
pip install -r requirements.txt
```
3. 执行脚本
```
python .transvideo.py {PATH_TO_VIDEO}
```

# Options
```
❯ ./transvideo.py --help
usage: transvideo.py [-h] [--stages {audio,transcribe,translate,srt,compile}] [--soft] video_file

Transcribe video to text and translate to Chinese

positional arguments:
  video_file

optional arguments:
  -h, --help            show this help message and exit
  --stages {audio,transcribe,translate,srt,compile}
  --soft
```

|参数|说明|
|---|---|
|stages|受未知因素影响，整个过程有时候会失败，因为保留了中间文件，可以从失败的阶段继续执行|
|soft|字幕默认以不可取消的模式烧入视频，这个参数可以使字幕以外挂在视频上，从而在视频软件上取消字幕，锻炼听力|

# SRT to Transcript Converter

A simple Python tool to convert SRT subtitle files to plain transcript text.

## Usage

```bash
python srt_to_transcript.py input.srt [-o output.txt]
```

### Arguments:
- `input.srt`: Path to the input SRT file
- `-o, --output`: (Optional) Path to the output transcript file. If not provided, a default name will be used based on the input filename.

## Example

For an SRT file with content:

```
1
00:00:00,000 --> 00:00:07,140
Welcome everyone. We're going to go ahead and get started on time because we have so many excellent questions.
欢迎大家。我们将继续按时开始，因为我们有很多很好的问题。

2
00:00:07,680 --> 00:00:11,320
Um, it's gone pretty fast. I need to we haven't even got to do it.
嗯，它过得真快。我需要，我们甚至都不需要这样做。
```

The output transcript will be:

```
Welcome everyone. We're going to go ahead and get started on time because we have so many excellent questions.
欢迎大家。我们将继续按时开始，因为我们有很多很好的问题。
Um, it's gone pretty fast. I need to we haven't even got to do it.
嗯，它过得真快。我需要，我们甚至都不需要这样做。
```

This tool automatically removes subtitle numbers, timestamps, and preserves all text content including multilingual text.
