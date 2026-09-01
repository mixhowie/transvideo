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
from concurrent.futures import ThreadPoolExecutor, as_completed

# Suppress SyntaxWarning from translators package in Python 3.12+
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=SyntaxWarning)
    import translators

import ollama

logging.basicConfig(level=logging.INFO)


# CTranslate2 has no Metal backend, so `faster` inference is CPU-only and runs
# large-v3 at roughly real time on an M-series laptop. `mlx` runs on the Apple
# Silicon GPU instead and is several times faster, at the cost of `hotwords`
# support -- the glossary degrades to an `initial_prompt` there.
WHISPER_BACKEND = "mlx"

# large-v3-turbo keeps large-v3's encoder but cuts the decoder from 32 layers to
# 4, which is where the speedup comes from. Swap in `mlx-community/whisper-large-v3-mlx`
# for maximum accuracy, or a `-q4`/`-8bit` repo to trade accuracy for more speed.
# Throughput is bounded by how many sequences the server decodes at once, which is
# OLLAMA_NUM_PARALLEL on the server side (llama-server's `-np`, default 1). Against a
# default server anything above ~4 is wasted; against `OLLAMA_NUM_PARALLEL=4` this
# measured ~2x, since batched decode amortises the weight reads that dominate on a
# unified-memory GPU.
OLLAMA_CONCURRENCY = 12
OLLAMA_CONTEXT_BEFORE = 4
OLLAMA_CONTEXT_AFTER = 2

MLX_WHISPER_MODEL = "mlx-community/whisper-large-v3-turbo"
FASTER_WHISPER_MODEL = "large-v3"
DEFAULT_WHISPER_MODELS = {"mlx": MLX_WHISPER_MODEL, "faster": FASTER_WHISPER_MODEL}

WHISPER_DEVICE = "auto"
WHISPER_COMPUTE_TYPE = "int8"

# 0 lets CTranslate2 pick. Its default is conservative -- on a 12-performance-core
# M3 Max it saturates only ~5 cores -- so raising this is worth trying if you are
# stuck on the 'faster' backend.
WHISPER_CPU_THREADS = 0

DEFAULT_GLOSSARY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "glossary.txt")

# Both backends cap glossary-style conditioning at roughly `max_length // 2`
# tokens (~223): faster-whisper truncates `hotwords`, MLX truncates
# `initial_prompt`. Warn well before that so silently dropped terms are
# noticeable; ~4 chars per token is a rough but adequate estimate.
MAX_HOTWORDS_CHARS = 800


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

    def __init__(
        self,
        video_file,
        translator="bing",
        ollama_url="http://localhost:11434",
        ollama_model="qwen2.5:7b",
        whisper_backend=WHISPER_BACKEND,
        whisper_model=None,
        whisper_device=WHISPER_DEVICE,
        whisper_compute_type=WHISPER_COMPUTE_TYPE,
        whisper_cpu_threads=WHISPER_CPU_THREADS,
        language=None,
        glossary_file=None,
    ):
        """
        Initialize Transvideo with a video file.

        Args:
            video_file (str): Path to the input video file.
            translator (str): Translation engine to use ('bing' or 'ollama').
            ollama_url (str): URL of the Ollama service.
            ollama_model (str): Ollama model to use.
            whisper_backend (str): Inference backend, 'mlx' (Apple GPU) or
                'faster' (CTranslate2, CPU-only on macOS).
            whisper_model (str | None): Model identifier for the chosen backend --
                an MLX repo/path for 'mlx', a size or CTranslate2 path for
                'faster'. Defaults to that backend's entry in
                DEFAULT_WHISPER_MODELS.
            whisper_device (str): Device for inference ('auto', 'cpu', 'cuda').
                Only used by the 'faster' backend.
            whisper_compute_type (str): CTranslate2 compute type ('int8', 'float32',
                ...). Only used by the 'faster' backend.
            whisper_cpu_threads (int): CTranslate2 CPU thread count, 0 for its
                default. Only used by the 'faster' backend.
            language (str | None): Spoken language code (e.g. 'en'). None
                auto-detects.
            glossary_file (str | None): Path to the proper-noun glossary. Defaults
                to the bundled glossary.txt.
        """
        if whisper_backend not in DEFAULT_WHISPER_MODELS:
            raise ValueError(f"Unknown whisper backend: {whisper_backend}")

        self.video_file = video_file
        self.translator = translator
        self.ollama_url = ollama_url
        self.ollama_model = ollama_model
        self.whisper_backend = whisper_backend
        self.whisper_model = whisper_model or DEFAULT_WHISPER_MODELS[whisper_backend]
        self.whisper_device = whisper_device
        self.whisper_compute_type = whisper_compute_type
        self.whisper_cpu_threads = whisper_cpu_threads
        self.language = language
        self.glossary_file = glossary_file or DEFAULT_GLOSSARY_FILE
        self.audio_file = os.path.splitext(video_file)[0] + ".wav"
        self.whisper_original_file = os.path.splitext(video_file)[0] + ".original.txt"
        self.whisper_combined_file = os.path.splitext(video_file)[0] + ".combined.txt"
        self.translate_file = os.path.splitext(video_file)[0] + ".trans.txt"
        self.srt_file = os.path.splitext(video_file)[0] + ".srt"
        self.output_file = os.path.splitext(video_file)[0] + ".trans" + os.path.splitext(video_file)[-1]

    def video_to_audio(self):
        """
        Convert video file to audio using ffmpeg.

        Extracts audio from the video file and saves it as a WAV file. Whisper
        resamples everything to 16 kHz mono anyway, so downmixing here keeps the
        intermediate file small and saves the decoder that work.
        """
        logging.info("Converting video to audio...")

        command = f'ffmpeg -i "{self.video_file}" -vn -ac 1 -ar 16000 -c:a pcm_s16le "{self.audio_file}"'
        exec_command(command)

    def save_whisper_result(self):
        """
        Transcribe audio using whisper and save the results.

        This method performs the following steps:
        1. Transcribes the audio with the configured backend, biasing the decoder
           toward the glossary terms
        2. Saves the original transcription with timestamps
        3. Combines segments into sentences and saves as a combined file

        Language is auto-detected as part of transcription unless `language` is set.

        The method will skip transcription if the output file already exists.
        """
        logging.info("Getting whisper result...")

        if not os.path.exists(self.whisper_original_file):
            if self.whisper_backend == "mlx":
                segments = self._transcribe_mlx()
            else:
                segments = self._transcribe_faster()

            content = []
            for start_time, end_time, segment_text in segments:
                logging.info("%s %s", start_time, segment_text)
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

    def _transcribe_mlx(self) -> list[tuple[str, str, str]]:
        """
        Transcribe on the Apple Silicon GPU via MLX.

        MLX has no `hotwords` equivalent, so the glossary is injected as an
        `initial_prompt` instead. That biases the decoder more weakly than
        hotwords do -- if a proper noun still comes out wrong, move it to the
        front of glossary.txt or fall back to the 'faster' backend.

        Returns:
            list[tuple[str, str, str]]: (start, end, text) per segment, timestamps
            already formatted as SRT-style hh:mm:ss,mmm.
        """
        try:
            import mlx_whisper
        except ImportError as exc:  # pragma: no cover - depends on the host arch
            raise RuntimeError(
                "The 'mlx' backend requires mlx-whisper, which only installs on "
                "Apple Silicon. Re-run with --whisper-backend faster."
            ) from exc

        initial_prompt = build_initial_prompt(load_glossary(self.glossary_file))
        if initial_prompt:
            logging.info("Biasing transcription with initial prompt: %s", initial_prompt)

        logging.info("Transcribing with MLX model %s...", self.whisper_model)
        result = mlx_whisper.transcribe(
            self.audio_file,
            path_or_hf_repo=self.whisper_model,
            initial_prompt=initial_prompt or None,
            language=self.language,
        )
        logging.info("Detected language: %s", result.get("language"))

        return [
            (
                seconds_to_hms(segment["start"]),
                seconds_to_hms(segment["end"]),
                segment["text"].replace("\n", "").strip(),
            )
            for segment in result["segments"]
        ]

    def _transcribe_faster(self) -> list[tuple[str, str, str]]:
        """
        Transcribe with faster-whisper (CTranslate2).

        CPU-only on macOS, but it is the only backend that supports `hotwords`,
        and it is the fallback on non-Apple-Silicon machines.

        Returns:
            list[tuple[str, str, str]]: (start, end, text) per segment, timestamps
            already formatted as SRT-style hh:mm:ss,mmm.
        """
        from faster_whisper import WhisperModel

        hotwords = build_hotwords(load_glossary(self.glossary_file))
        if hotwords:
            logging.info("Biasing transcription with hotwords: %s", hotwords)

        model = WhisperModel(
            self.whisper_model,
            device=self.whisper_device,
            compute_type=self.whisper_compute_type,
            cpu_threads=self.whisper_cpu_threads,
        )
        # No `word_timestamps`: nothing downstream reads word-level timings, and
        # the cross-attention DTW alignment it needs is pure overhead. `vad_filter`
        # keeps silent stretches out of the decoder entirely.
        segments, info = model.transcribe(
            self.audio_file,
            hotwords=hotwords or None,
            language=self.language,
            vad_filter=True,
        )
        logging.info("Detected language: %s (probability %.2f)", info.language, info.language_probability)

        # `segments` is a generator -- transcription happens as it is consumed.
        return [
            (
                seconds_to_hms(segment.start),
                seconds_to_hms(segment.end),
                segment.text.replace("\n", "").strip(),
            )
            for segment in segments
        ]

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

        client = ollama.Client(host=self.ollama_url)
        texts = [item[2] for item in lines]

        # Asking for a numbered batch lets the model merge, split or renumber lines,
        # which shifts every translation after that point onto the wrong subtitle
        # while still looking complete. One line per request cannot misalign, and it
        # measures the same throughput here because decoding is what costs, not the
        # round trips. Neighbours ride along as context so wording stays coherent.
        translations = [""] * len(texts)
        done = 0
        with ThreadPoolExecutor(max_workers=OLLAMA_CONCURRENCY) as pool:
            pending = {pool.submit(self._translate_line_with_ollama, client, texts, i): i for i in range(len(texts))}
            for future in as_completed(pending):
                translations[pending[future]] = future.result()
                done += 1
                # Results are only written out at the end, so without this a long
                # transcript spends an hour looking indistinguishable from a hang.
                if done % 25 == 0 or done == len(texts):
                    logging.info("Translated %d/%d lines", done, len(texts))

        untranslated = [i for i, t in enumerate(translations) if not t]
        if untranslated:
            logging.error("%d lines left untranslated: %s", len(untranslated), untranslated[:20])

        translate_result = []
        for item, text_translated in zip(lines, translations, strict=True):
            logging.info("%s %s %s", item[0], item[2], text_translated)
            translate_result.append("|".join([item[0], item[1], item[2], text_translated]))

        save_text_to_file("\n".join(translate_result), self.translate_file)

    def _translate_line_with_ollama(self, client, texts, index, attempts=3):
        before = texts[max(0, index - OLLAMA_CONTEXT_BEFORE) : index]
        after = texts[index + 1 : index + 1 + OLLAMA_CONTEXT_AFTER]

        prompt = (
            "You are translating an English talk transcript into Chinese, one subtitle line at a time.\n\n"
            f"Preceding lines (context only, do not translate):\n{chr(10).join(before) or '(none)'}\n\n"
            f"Following lines (context only, do not translate):\n{chr(10).join(after) or '(none)'}\n\n"
            f"Translate ONLY this line into Chinese:\n{texts[index]}\n\n"
            "Output only the Chinese translation of that one line, with no numbering, "
            "quotes, English, or explanation."
        )

        # The model occasionally appends a note explaining its word choice, and on a
        # source line that is itself a whisper repetition artefact it can fall into a
        # repetition loop of its own -- one such reply came back at 1.1 MB. Ask again
        # rather than burning either into the video. The bound has to scale with the
        # source: a long sentence has a legitimately long translation (measured ratio
        # stays near 1.0), so only a multiple of it separates prose from a runaway.
        limit = 3 * len(texts[index]) + 20
        overlong = ""

        for attempt in range(attempts):
            try:
                response = client.generate(
                    model=self.ollama_model,
                    prompt=prompt,
                    stream=False,
                    options={
                        "temperature": 0.3,
                    },
                )
            except Exception as e:
                logging.error("Ollama translation failed for line %d (attempt %d): %s", index, attempt + 1, e)
                continue

            translated = _flatten_translation(response.get("response", ""))
            if not translated:
                continue
            if len(translated) <= limit:
                return translated

            logging.warning("Line %d came back %d chars, over the %d limit, retrying...", index, len(translated), limit)
            overlong = translated

        return overlong[:limit]

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


def load_glossary(path: str | None = None) -> list[str]:
    """
    Load proper nouns from a glossary file.

    One term per line; blank lines and `#` comments are ignored. Duplicates are
    dropped while preserving order, so the most important terms stay in front.

    Args:
        path (str | None): Glossary path. Defaults to the bundled glossary.txt.

    Returns:
        list[str]: The glossary terms, in file order.
    """
    path = path or DEFAULT_GLOSSARY_FILE
    if not os.path.exists(path):
        logging.warning("Glossary file not found, transcribing without hotwords: %s", path)
        return []

    terms: list[str] = []
    seen: set[str] = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            term = line.split("#", 1)[0].strip()
            if term and term not in seen:
                seen.add(term)
                terms.append(term)
    return terms


def _warn_if_glossary_too_long(text: str) -> None:
    """Warn when glossary conditioning is long enough to be silently truncated."""
    if len(text) > MAX_HOTWORDS_CHARS:
        logging.warning(
            "Glossary is %d chars; whisper may silently drop trailing terms. "
            "Trim it to the terms Whisper actually gets wrong.",
            len(text),
        )


def build_hotwords(terms: list[str]) -> str:
    """
    Join glossary terms into the `hotwords` string faster-whisper expects.

    Warns when the result is long enough that faster-whisper's internal token
    cap would silently drop the trailing terms.
    """
    hotwords = ", ".join(terms)
    _warn_if_glossary_too_long(hotwords)
    return hotwords


def build_initial_prompt(terms: list[str]) -> str:
    """
    Phrase glossary terms as an `initial_prompt` for the MLX backend.

    MLX exposes no `hotwords` parameter, so the terms are fed as prior context
    instead. Whisper conditions on this as if it were previously transcribed
    text, which is why it reads as a sentence rather than a bare word list.
    """
    if not terms:
        return ""

    prompt = f"This transcript mentions: {', '.join(terms)}."
    _warn_if_glossary_too_long(prompt)
    return prompt


def _find_last_sentence_end(text: str) -> int:
    """Return the index of the rightmost sentence-terminator in text, or -1."""
    return max((text.rfind(t) for t in SENTENCE_TERMINATORS), default=-1)


def _find_soft_break(text: str) -> int:
    """Return the index of the rightmost soft break (comma/space) in text, or -1."""
    return max((text.rfind(c) for c in SOFT_BREAK_CHARS), default=-1)


def _flatten_translation(text: str) -> str:
    """Collapse a model reply into one delimiter-safe line.

    Replies sometimes span several lines -- a multi-sentence subtitle split up, or a
    trailing note. Either way a newline would split the record across two lines of
    the pipe-delimited file and break the srt stage, and a literal pipe would shift
    every field after it.
    """
    joined = " ".join(part.strip() for part in text.split("\n") if part.strip())
    return joined.replace("|", "/").strip()


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
