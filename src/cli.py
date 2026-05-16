#!/usr/bin/env python3
"""Command-line entry point for transvideo."""

import logging
import os

import click

STAGE_CHOICES = ["audio", "transcribe", "translate", "srt", "compile"]


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
def run(video_file: str, stages: tuple[str, ...], soft: bool, translator: str, ollama_url: str, ollama_model: str):
    """Transcribe a video and compile it with translated subtitles."""
    from transvideo import Transvideo

    logging.info(
        "args: video_file=%s stages=%s soft=%s translator=%s ollama_url=%s ollama_model=%s",
        video_file,
        stages,
        soft,
        translator,
        ollama_url,
        ollama_model,
    )

    video_file = os.path.expanduser(video_file)
    transvideo = Transvideo(
        video_file,
        translator=translator,
        ollama_url=ollama_url,
        ollama_model=ollama_model,
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
