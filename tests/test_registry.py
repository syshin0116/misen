"""Tests for Registry."""

import pytest

from misen import Registry, RegistryKeyError, tool


@pytest.fixture
def sample_blocks():
    @tool(name="parse", description="Parse documents")
    def parse(input: dict) -> dict:
        return {"text": "parsed"}

    @tool(name="chunk", description="Chunk text")
    def chunk(input: dict) -> dict:
        return {"chunks": []}

    @tool(name="embed", description="Generate embeddings")
    def embed(input: dict) -> dict:
        return {"vectors": []}

    return parse, chunk, embed


class TestRegistry:
    def test_register_and_get(self, sample_blocks):
        parse, chunk, embed = sample_blocks
        reg = Registry()
        reg.register(parse)
        assert reg.get("parse") is parse

    def test_get_missing_raises(self):
        reg = Registry()
        with pytest.raises(RegistryKeyError):
            reg.get("nonexistent")

    def test_has(self, sample_blocks):
        parse, _, _ = sample_blocks
        reg = Registry()
        reg.register(parse)
        assert reg.has("parse")
        assert not reg.has("missing")

    def test_contains(self, sample_blocks):
        parse, _, _ = sample_blocks
        reg = Registry()
        reg.register(parse)
        assert "parse" in reg
        assert "missing" not in reg

    def test_len(self, sample_blocks):
        parse, chunk, embed = sample_blocks
        reg = Registry()
        assert len(reg) == 0
        reg.register(parse)
        reg.register(chunk)
        reg.register(embed)
        assert len(reg) == 3

    def test_register_with_custom_name(self, sample_blocks):
        parse, _, _ = sample_blocks
        reg = Registry()
        reg.register(parse, name="custom_parse")
        assert reg.has("custom_parse")
        assert not reg.has("parse")

    def test_register_with_tags(self, sample_blocks):
        parse, chunk, embed = sample_blocks
        reg = Registry()
        reg.register(parse, tags=["rag", "ingest"])
        reg.register(chunk, tags=["rag", "ingest"])
        reg.register(embed, tags=["rag", "ml"])

        results = reg.search(tags=["rag"])
        assert len(results) == 3

        results = reg.search(tags=["ingest"])
        assert len(results) == 2

        results = reg.search(tags=["rag", "ml"])
        assert len(results) == 1

    def test_search_by_name(self, sample_blocks):
        parse, chunk, embed = sample_blocks
        reg = Registry()
        reg.register(parse)
        reg.register(chunk)
        reg.register(embed)

        results = reg.search(name_contains="ch")
        assert len(results) == 1
        assert results[0]["name"] == "chunk"

    def test_search_case_insensitive(self, sample_blocks):
        parse, _, _ = sample_blocks
        reg = Registry()
        reg.register(parse)
        results = reg.search(name_contains="PARSE")
        assert len(results) == 1

    def test_list_all(self, sample_blocks):
        parse, chunk, embed = sample_blocks
        reg = Registry()
        reg.register(parse, tags=["a"])
        reg.register(chunk)
        all_blocks = reg.list()
        assert len(all_blocks) == 2
        names = {b["name"] for b in all_blocks}
        assert names == {"parse", "chunk"}

    def test_unregister(self, sample_blocks):
        parse, _, _ = sample_blocks
        reg = Registry()
        reg.register(parse)
        assert reg.has("parse")
        reg.unregister("parse")
        assert not reg.has("parse")

    def test_unregister_missing_is_noop(self):
        reg = Registry()
        reg.unregister("nonexistent")  # should not raise

    def test_from_blocks(self, sample_blocks):
        parse, chunk, embed = sample_blocks
        reg = Registry.from_blocks(parse, chunk, embed)
        assert len(reg) == 3
        assert reg.has("parse")
        assert reg.has("chunk")
        assert reg.has("embed")

    def test_repr(self, sample_blocks):
        parse, chunk, _ = sample_blocks
        reg = Registry.from_blocks(parse, chunk)
        assert "2" in repr(reg)

    def test_metadata_stored(self, sample_blocks):
        parse, _, _ = sample_blocks
        reg = Registry()
        reg.register(parse, tags=["rag"], metadata={"version": "1.0"})
        meta = reg.list()
        assert meta[0]["tags"] == ["rag"]
        assert meta[0]["version"] == "1.0"
        assert meta[0]["description"] == "Parse documents"
