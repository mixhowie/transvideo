#!/usr/bin/env python3
"""
Transvideo - A tool to transcribe and translate videos with subtitles.

This module provides functionality to convert video to audio, transcribe using
whisper, translate text to Chinese, create SRT subtitles, and compile a video
with embedded or soft subtitles.
"""

import logging
import os
import shutil
import subprocess
import tempfile
import warnings

# Suppress SyntaxWarning from translators package in Python 3.12+
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=SyntaxWarning)
    import translators

import ollama
import whisper

logging.basicConfig(level=logging.INFO)


WHIPER_MODEL = "tiny"


class Transvideo:
    """
    A class to transcribe and translate videos with subtitles.

    This class provides methods to convert video to audio, transcribe using whisper,
    translate text to Chinese, create SRT subtitles, and compile a video with
    embedded or soft subtitles.

    Attributes:
        video_file (str): Path to the input video file.
        audio_file (str): Path to the extracted audio file.
        whisper_original_file (str): Path to the original whisper transcription.
        whisper_combined_file (str): Path to the combined whisper transcription.
        translate_file (str): Path to the translated text file.
        srt_file (str): Path to the generated SRT subtitle file.
        output_file (str): Path to the output video file with subtitles.
    """

    def __init__(self, video_file, translator="bing", ollama_url="http://localhost:11434", ollama_model="qwen2.5:7b"):
        """
        Initialize Transvideo with a video file.

        Args:
            video_file (str): Path to the input video file.
            translator (str): Translation engine to use ('bing' or 'ollama').
            ollama_url (str): URL of the Ollama service.
            ollama_model (str): Ollama model to use.
        """
        self.video_file = video_file
        self.translator = translator
        self.ollama_url = ollama_url
        self.ollama_model = ollama_model
        self.audio_file = os.path.splitext(video_file)[0] + ".wav"
        self.whisper_original_file = os.path.splitext(video_file)[0] + ".original.txt"
        self.whisper_combined_file = os.path.splitext(video_file)[0] + ".combined.txt"
        self.translate_file = os.path.splitext(video_file)[0] + ".trans.txt"
        self.srt_file = os.path.splitext(video_file)[0] + ".srt"
        self.output_file = os.path.splitext(video_file)[0] + ".trans" + os.path.splitext(video_file)[-1]

    def video_to_audio(self):
        """
        Convert video file to audio using ffmpeg.

        Extracts audio from the video file and saves it as a WAV file.
        """
        logging.info("Converting video to audio...")

        command = f'ffmpeg -i "{self.video_file}" "{self.audio_file}"'
        exec_command(command)

    def save_whisper_result(self):
        """
        Transcribe audio using whisper and save the results.

        This method performs the following steps:
        1. Loads the whisper model and audio file
        2. Detects the language of the audio
        3. Transcribes the audio with timestamps for each segment
        4. Saves the original transcription with timestamps
        5. Combines segments into sentences and saves as a combined file

        The method will skip transcription if the output file already exists.
        """
        logging.info("Getting whisper result...")

        if not os.path.exists(self.whisper_original_file):
            model = whisper.load_model(WHIPER_MODEL)
            audio = whisper.load_audio(self.audio_file)

            mel = whisper.log_mel_spectrogram(whisper.pad_or_trim(audio)).to(model.device)
            _, probs_any = model.detect_language(mel)
            probs: dict[str, float] = probs_any  # ty: ignore[invalid-assignment]
            detected_language = max(probs, key=probs.get)  # ty: ignore[no-matching-overload]
            logging.info(f"Detected language: {detected_language}")

            whisper_result = model.transcribe(
                verbose=True,
                audio=audio,
                language=detected_language,
                fp16=False,
                word_timestamps=True,
            )

            content = []
            for segment in whisper_result["segments"]:
                start_time = seconds_to_hms(segment["start"])
                end_time = seconds_to_hms(segment["end"])
                segment_text = segment["text"].replace("\n", "").strip()
                content.append(f"{start_time}|{end_time}|{segment_text}")
            save_text_to_file("\n".join(content), self.whisper_original_file)

        combined_result = []
        buffer_parts: list[str] = []
        buffer_start: str | None = None
        last_end: str | None = None
        with open(self.whisper_original_file) as f:
            whisper_original = f.read().strip().split("\n")
            for line in whisper_original:
                start_time, end_time, segment_text = line.split("|")
                segment_text = segment_text.strip()
                last_end = end_time
                if not segment_text:
                    continue
                if buffer_start is None:
                    buffer_start = start_time
                buffer_parts.append(segment_text)

                buffer = " ".join(buffer_parts)
                split_at = _find_last_sentence_end(buffer)
                if split_at == -1:
                    if len(buffer) < MAX_CUE_CHARS:
                        continue
                    split_at = _find_soft_break(buffer)
                    if split_at == -1:
                        continue

                completed = buffer[: split_at + 1].strip()
                leftover = buffer[split_at + 1 :].strip()
                logging.info("%s %s", buffer_start, completed)
                combined_result.append("|".join([buffer_start, end_time, completed]))

                if leftover:
                    buffer_parts = [leftover]
                    buffer_start = end_time
                else:
                    buffer_parts = []
                    buffer_start = None

        if buffer_parts and buffer_start is not None and last_end is not None:
            tail = " ".join(buffer_parts).strip()
            if tail:
                combined_result.append("|".join([buffer_start, last_end, tail]))

        save_text_to_file("\n".join(combined_result), self.whisper_combined_file)

    def translate_whisper_result(self):
        logging.info("Translating whisper result using %s...", self.translator)
        if self.translator == "ollama":
            self._translate_with_ollama()
        else:
            self._translate_with_translators()

    def _translate_with_translators(self):
        translate_result = []
        with open(self.whisper_combined_file, encoding="utf-8") as f:
            for line in f.read().strip().split("\n"):
                if not line:
                    continue
                start_time, end_time, text_original = line.split("|")
                text_translated = translators.translate_text(
                    text_original,
                    translator=self.translator,
                    from_language="en",
                    to_language="zh",
                    if_ignore_limit_of_length=True,
                )

                logging.info("%s %s %s", start_time, text_original, text_translated)
                translate_result.append("|".join([start_time, end_time, text_original, text_translated]))
        save_text_to_file("\n".join(translate_result), self.translate_file)

    def _translate_with_ollama(self):
        lines = []
        with open(self.whisper_combined_file, encoding="utf-8") as f:
            lines = [line.split("|") for line in f.read().strip().split("\n") if line]

        translate_result = []
        batch_size = 50

        # Initialize Ollama client
        client = ollama.Client(host=self.ollama_url)

        for i in range(0, len(lines), batch_size):
            batch = lines[i : i + batch_size]
            text_to_translate = "\n".join([f"{idx + 1}. {item[2]}" for idx, item in enumerate(batch)])

            prompt = (
                "You are a professional translator. Translate the following English subtitles into Chinese. "
                "Keep the original numbering and format. Only output the translated text, one per line, "
                "starting with the number. Do not add any explanations.\n\n"
                f"{text_to_translate}"
            )

            try:
                response = client.generate(
                    model=self.ollama_model,
                    prompt=prompt,
                    stream=False,
                    options={
                        "temperature": 0.3,
                    },
                )
                translated_content = response.get("response", "").strip()

                # Simple parsing of the numbered list
                translated_lines = {}
                for line in translated_content.split("\n"):
                    line = line.strip()
                    if ". " in line:
                        parts = line.split(". ", 1)
                        if parts[0].isdigit():
                            translated_lines[int(parts[0])] = parts[1].strip()
                    elif ":" in line:  # Some models might use '1:' instead of '1.'
                        parts = line.split(":", 1)
                        if parts[0].isdigit():
                            translated_lines[int(parts[0])] = parts[1].strip()

                for idx, item in enumerate(batch):
                    original_idx = idx + 1
                    text_translated = translated_lines.get(original_idx, "")
                    logging.info("%s %s %s", item[0], item[2], text_translated)
                    translate_result.append("|".join([item[0], item[1], item[2], text_translated]))

            except Exception as e:
                logging.error("Ollama translation failed for batch starting at %d: %s", i, e)
                for item in batch:
                    translate_result.append("|".join([item[0], item[1], item[2], ""]))

        save_text_to_file("\n".join(translate_result), self.translate_file)

    def create_srt(self):
        logging.info("Converting whisper result to srt...")

        transcript_result = []
        index = 1
        with open(self.translate_file, encoding="utf-8") as f:
            for line in f.read().strip().split("\n"):
                start_time, end_time, text_original, text_translated = line.split("|")
                transcript_result.append(str(index))
                transcript_result.append(f"{start_time} --> {end_time}")
                transcript_result.append(text_original)
                transcript_result.append(text_translated)
                transcript_result.append("")

                index += 1
        save_text_to_file("\n".join(transcript_result), self.srt_file)

    def compile_video_with_srt(self, soft=False):
        if soft:
            self._compile_video_with_srt_soft()
        else:
            self._compile_video_with_srt_hard()

    def _compile_video_with_srt_hard(self):
        logging.info("Compiling video with srt...")

        # ffmpeg's `subtitles` filter applies its own two-level escaping to the
        # filename, which makes paths containing quotes/colons/commas/etc.
        # extremely fragile. Stage the srt under a sanitized temp path so the
        # filter only ever sees plain ASCII.
        with tempfile.TemporaryDirectory() as tmpdir:
            safe_srt = os.path.join(tmpdir, "subs.srt")
            shutil.copyfile(self.srt_file, safe_srt)
            vf = f"subtitles={safe_srt}:force_style='FontSize=12,Fontname=PingFang SC'"
            cmd = ["ffmpeg", "-i", self.video_file, "-vf", vf, self.output_file]
            logging.info("Executing command: %s", cmd)
            subprocess.run(cmd)

    def _compile_video_with_srt_soft(self):
        logging.info("Compiling video with srt...")

        command = (
            f'ffmpeg -i "{self.video_file}" -i "{self.srt_file}" '
            f'-c copy -c:s mov_text -metadata:s:s:0 language=eng "{self.output_file}"'
        )
        exec_command(command)


SENTENCE_TERMINATORS = (".", "!", "?", "。", "！", "？")
SOFT_BREAK_CHARS = (",", "，", ";", "；", ":", "：", " ")
MAX_CUE_CHARS = 300


def _find_last_sentence_end(text: str) -> int:
    """Return the index of the rightmost sentence-terminator in text, or -1."""
    return max((text.rfind(t) for t in SENTENCE_TERMINATORS), default=-1)


def _find_soft_break(text: str) -> int:
    """Return the index of the rightmost soft break (comma/space) in text, or -1."""
    return max((text.rfind(c) for c in SOFT_BREAK_CHARS), default=-1)


def seconds_to_hms(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    hms = f"{int(h):02d}:{int(m):02d}:{s:06.3f}"
    hms = hms.replace(".", ",")
    return hms


def exec_command(command):
    logging.info("Executing command: %s", command)

    os.system(command)


def save_text_to_file(text, filepath):
    logging.info("Saving text to file...")

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(text)


def read_text_from_file(filepath):
    with open(filepath, encoding="utf-8") as f:
        return f.read()


if __name__ == "__main__":
    from cli import main

    main()
