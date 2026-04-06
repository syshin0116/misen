"""Tests for built-in tools."""

from misen.tools import TextSplitter, Transformer


class TestTextSplitter:
    async def test_basic_split(self):
        splitter = TextSplitter(chunk_size=20, overlap=0, separator="\n")
        text = "line one\nline two\nline three\nline four"
        result = await splitter.run({"text": text})
        chunks = result["chunks"]
        assert len(chunks) >= 2
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
            words_0 = set(chunks[0].split())
            words_1 = set(chunks[1].split())
            assert words_0 & words_1

    async def test_hard_split_long_segment(self):
        """Segments longer than chunk_size are force-split by character."""
        splitter = TextSplitter(chunk_size=10, overlap=0, separator="\n")
        text = "a" * 25  # single segment, no separator
        result = await splitter.run({"text": text})
        chunks = result["chunks"]
        assert len(chunks) >= 3
        assert all(len(c) <= 10 for c in chunks)
        # all content preserved
        assert "".join(chunks) == text

    async def test_hard_split_with_overlap(self):
        """Force-split respects overlap."""
        splitter = TextSplitter(chunk_size=10, overlap=3, separator="\n")
        text = "abcdefghijklmnopqrstuvwxyz"
        result = await splitter.run({"text": text})
        chunks = result["chunks"]
        assert all(len(c) <= 10 for c in chunks)
        # overlap means consecutive chunks share characters
        if len(chunks) >= 2:
            assert chunks[0][-3:] == chunks[1][:3]

    async def test_mixed_long_and_short_segments(self):
        """Mix of normal and oversized segments."""
        splitter = TextSplitter(chunk_size=10, overlap=0, separator="\n")
        text = "short\n" + "x" * 25 + "\ntiny"
        result = await splitter.run({"text": text})
        chunks = result["chunks"]
        assert all(len(c) <= 10 for c in chunks)

    async def test_force_split_resets_overlap_state(self):
        """Regression: after force-split, overlap must not use stale pre-long state."""
        splitter = TextSplitter(chunk_size=10, overlap=3, separator="\n")
        text = "abc\n" + "x" * 25 + "\nzzz"
        result = await splitter.run({"text": text})
        chunks = result["chunks"]
        # "abc" should NOT appear in the chunk after the long segment
        last_chunk = chunks[-1]
        assert "abc" not in last_chunk, (
            f"Stale overlap: last chunk {last_chunk!r} contains 'abc' from before force-split"
        )
        assert "zzz" in last_chunk
        assert all(len(c) <= 10 for c in chunks)

    async def test_every_chunk_respects_max_size(self):
        """Property: no chunk ever exceeds chunk_size."""
        splitter = TextSplitter(chunk_size=15, overlap=5, separator=" ")
        text = "the quick brown fox jumps over the lazy dog " + "x" * 30
        result = await splitter.run({"text": text})
        for chunk in result["chunks"]:
            assert len(chunk) <= 15, f"Chunk too long: {len(chunk)} chars: {chunk!r}"


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
