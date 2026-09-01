#!/usr/bin/env python3
"""Command-line entry point for transvideo."""

import logging
import os

import click

STAGE_CHOICES = ["audio", "transcribe", "translate", "srt", "compile"]
WHISPER_BACKENDS = ["mlx", "faster"]


@click.group()
def main():
    """Transvideo CLI."""


@main.command()
@click.argument("video_file", type=click.Path())
@click.option(
    "--stages",
    type=click.Choice(STAGE_CHOICES),
    multiple=True,
    default=(),
    help="Stages to run. Repeat to select multiple. Default runs all stages.",
)
@click.option("--soft", is_flag=True, default=False, help="Use soft subtitles instead of burning in.")
@click.option(
    "--translator",
    default="bing",
    show_default=True,
    help=(
        "Translation engine to use. 'ollama' uses the local Ollama service; "
        "any other value is passed through to the `translators` library "
        "(e.g. bing, google, alibaba)."
    ),
)
@click.option("--ollama-url", default="http://localhost:11434", show_default=True, help="Ollama API URL.")
@click.option("--ollama-model", default="qwen2.5:7b", show_default=True, help="Ollama model name.")
@click.option(
    "--whisper-backend",
    type=click.Choice(WHISPER_BACKENDS),
    default="mlx",
    show_default=True,
    help=(
        "Inference backend. 'mlx' runs on the Apple Silicon GPU and is several "
        "times faster; 'faster' is CTranslate2 (CPU-only on macOS) and is the "
        "only backend supporting hotwords."
    ),
)
@click.option(
    "--whisper-model",
    default=None,
    help=(
        "Model for the chosen backend: an MLX repo or path for 'mlx' "
        "(default mlx-community/whisper-large-v3-turbo), a size such as "
        "tiny/medium.en/large-v3 or a CTranslate2 path for 'faster' "
        "(default large-v3)."
    ),
)
@click.option(
    "--whisper-device",
    default="auto",
    show_default=True,
    help="Inference device: auto, cpu, or cuda. Only used by the 'faster' backend.",
)
@click.option(
    "--whisper-compute-type",
    default="int8",
    show_default=True,
    help=(
        "CTranslate2 compute type, only used by the 'faster' backend. 'int8' keeps "
        "large-v3 usable on CPU; 'float32' is slower but avoids quantization loss "
        "on rare proper nouns."
    ),
)
@click.option(
    "--whisper-cpu-threads",
    type=int,
    default=0,
    show_default=True,
    help=(
        "CTranslate2 CPU thread count, only used by the 'faster' backend. 0 uses "
        "its own default, which saturates only about 5 cores on an M3 Max -- try "
        "your performance-core count if you are stuck on this backend."
    ),
)
@click.option(
    "--language",
    default=None,
    help="Spoken language code (e.g. en). Skips auto-detection when set.",
)
@click.option(
    "--glossary",
    "glossary_file",
    type=click.Path(dir_okay=False),
    default=None,
    help=(
        "Proper-noun glossary. Fed to whisper as hotwords on the 'faster' backend "
        "and as an initial prompt on 'mlx'. Defaults to the bundled glossary.txt."
    ),
)
def run(
    video_file: str,
    stages: tuple[str, ...],
    soft: bool,
    translator: str,
    ollama_url: str,
    ollama_model: str,
    whisper_backend: str,
    whisper_model: str | None,
    whisper_device: str,
    whisper_compute_type: str,
    whisper_cpu_threads: int,
    language: str | None,
    glossary_file: str | None,
):
    """Transcribe a video and compile it with translated subtitles."""
    from transvideo import Transvideo

    logging.info(
        "args: video_file=%s stages=%s soft=%s translator=%s ollama_url=%s ollama_model=%s "
        "whisper_backend=%s whisper_model=%s whisper_device=%s whisper_compute_type=%s "
        "whisper_cpu_threads=%s language=%s glossary=%s",
        video_file,
        stages,
        soft,
        translator,
        ollama_url,
        ollama_model,
        whisper_backend,
        whisper_model,
        whisper_device,
        whisper_compute_type,
        whisper_cpu_threads,
        language,
        glossary_file,
    )

    video_file = os.path.expanduser(video_file)
    transvideo = Transvideo(
        video_file,
        translator=translator,
        ollama_url=ollama_url,
        ollama_model=ollama_model,
        whisper_backend=whisper_backend,
        whisper_model=whisper_model,
        whisper_device=whisper_device,
        whisper_compute_type=whisper_compute_type,
        whisper_cpu_threads=whisper_cpu_threads,
        language=language,
        glossary_file=os.path.expanduser(glossary_file) if glossary_file else None,
    )
    if not stages or "audio" in stages:
        transvideo.video_to_audio()
    if not stages or "transcribe" in stages:
        transvideo.save_whisper_result()
    if not stages or "translate" in stages:
        transvideo.translate_whisper_result()
    if not stages or "srt" in stages:
        transvideo.create_srt()
    if not stages or "compile" in stages:
        transvideo.compile_video_with_srt(soft=soft)


@main.command("srt2txt")
@click.argument("input_file", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "-o", "--output", "output_file", type=click.Path(dir_okay=False), default=None, help="Output transcript file path."
)
def srt2txt(input_file: str, output_file: str | None):
    """Convert an SRT subtitle file to plain transcript text."""
    from srt_to_transcript import srt_to_transcript

    if output_file is None:
        base, _ = os.path.splitext(input_file)
        output_file = f"{base}_transcript.txt"
    srt_to_transcript(input_file, output_file)


if __name__ == "__main__":
    main()
