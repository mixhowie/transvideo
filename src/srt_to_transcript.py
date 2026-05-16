#!/usr/bin/env python3
"""
Convert SRT subtitle files to plain transcript text.
"""

import re


def parse_srt(srt_content: str) -> list[str]:
    """
    Parse SRT content and extract only the text parts.

    Args:
        srt_content: Content of the SRT file

    Returns:
        List of text lines from the SRT file
    """
    # Pattern to match SRT index and timestamp lines
    pattern = r"^\d+$|^\d{2}:\d{2}:\d{2},\d{3}\s-->\s\d{2}:\d{2}:\d{2},\d{3}$"

    lines = srt_content.strip().split("\n")
    text_lines = []

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Skip empty lines
        if not line:
            i += 1
            continue

        # Skip index and timestamp lines
        if re.match(pattern, line):
            i += 1
            continue

        # Add text content
        text_lines.append(line)
        i += 1

    return text_lines


def srt_to_transcript(input_file: str, output_file: str) -> None:
    """
    Convert SRT file to transcript by extracting only the text content.

    Args:
        input_file: Path to input SRT file
        output_file: Path to output transcript file
    """
    try:
        with open(input_file, encoding="utf-8") as f:
            srt_content = f.read()

        transcript_lines = parse_srt(srt_content)

        with open(output_file, "w", encoding="utf-8") as f:
            f.write("\n".join(transcript_lines))

        print(f"Successfully converted {input_file} to {output_file}")

    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    from cli import main

    main()
