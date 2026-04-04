"""TextSplitter — split text into overlapping chunks."""

from __future__ import annotations

from typing import Any

from misen.core.block import Block


class TextSplitter(Block):
    """Split text into chunks by character count with overlap.

    Config (constructor):
        chunk_size: Maximum characters per chunk.
        overlap: Characters to repeat between consecutive chunks.
        separator: Split boundary (default newline).
        input_key / output_key: Dict keys to read from / write to.
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        overlap: int = 200,
        separator: str = "\n",
        *,
        input_key: str = "text",
        output_key: str = "chunks",
    ) -> None:
        super().__init__(name="TextSplitter", description="Split text into overlapping chunks")
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.separator = separator
        self.input_key = input_key
        self.output_key = output_key

    async def run(self, input: dict[str, Any]) -> dict[str, Any]:
        text: str = input[self.input_key]
        return {self.output_key: self._split(text)}

    def _split(self, text: str) -> list[str]:
        if not text:
            return []

        segments = text.split(self.separator)
        chunks: list[str] = []
        current: list[str] = []
        current_len = 0

        for segment in segments:
            seg_len = len(segment) + (len(self.separator) if current else 0)

            if current_len + seg_len > self.chunk_size and current:
                chunks.append(self.separator.join(current))
                # keep trailing segments for overlap
                overlap_segments: list[str] = []
                overlap_len = 0
                for s in reversed(current):
                    candidate = len(s) + (len(self.separator) if overlap_segments else 0)
                    if overlap_len + candidate > self.overlap:
                        break
                    overlap_segments.insert(0, s)
                    overlap_len += candidate
                current = overlap_segments
                current_len = overlap_len

            current.append(segment)
            current_len += seg_len

        if current:
            chunks.append(self.separator.join(current))

        return chunks
