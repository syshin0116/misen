"""Tests for built-in tools."""

import pytest

from misen import sequential
from misen.tools import TextSplitter, Transformer


class TestTextSplitter:
    async def test_basic_split(self):
        splitter = TextSplitter(chunk_size=20, overlap=0, separator="\n")
        text = "line one\nline two\nline three\nline four"
        result = await splitter.run({"text": text})
        chunks = result["chunks"]
        assert len(chunks) >= 2
        # all original content should be present across chunks
        assert all(line in "\n".join(chunks) for line in text.split("\n"))

    async def test_empty_text(self):
        splitter = TextSplitter()
        result = await splitter.run({"text": ""})
        assert result["chunks"] == []

    async def test_small_text_single_chunk(self):
        splitter = TextSplitter(chunk_size=1000)
        result = await splitter.run({"text": "short text"})
        assert len(result["chunks"]) == 1
        assert result["chunks"][0] == "short text"

    async def test_custom_keys(self):
        splitter = TextSplitter(
            chunk_size=50,
            overlap=0,
            input_key="content",
            output_key="parts",
        )
        result = await splitter.run({"content": "a" * 100})
        assert "parts" in result

    async def test_overlap_produces_repeated_content(self):
        splitter = TextSplitter(chunk_size=20, overlap=10, separator=" ")
        text = "one two three four five six seven"
        result = await splitter.run({"text": text})
        chunks = result["chunks"]
        if len(chunks) >= 2:
            # overlap means some words appear in consecutive chunks
            words_0 = set(chunks[0].split())
            words_1 = set(chunks[1].split())
            assert words_0 & words_1  # should share at least one word


class TestTransformer:
    async def test_basic_transform(self):
        upper = Transformer(str.upper, input_key="text")
        result = await upper.run({"text": "hello"})
        assert result["text"] == "HELLO"

    async def test_different_output_key(self):
        length = Transformer(len, input_key="text", output_key="length")
        result = await length.run({"text": "hello"})
        assert result["length"] == 5

    async def test_name(self):
        t = Transformer(str.upper, input_key="text", name="Uppercase")
        assert t.name == "Uppercase"

    async def test_default_name(self):
        t = Transformer(str.upper, input_key="text")
        assert "text" in t.name


class TestToolIntegration:
    async def test_splitter_pipe_transformer(self):
        splitter = TextSplitter(chunk_size=20, overlap=0, separator="\n")
        counter = Transformer(len, input_key="chunks", output_key="chunk_count")

        pipeline = splitter | counter
        result = await pipeline.run({"text": "line one\nline two\nline three\nline four"})

        assert "chunks" in result
        assert "chunk_count" in result
        assert result["chunk_count"] == len(result["chunks"])
