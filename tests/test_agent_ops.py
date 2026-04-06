"""Tests for guided and free operators."""

import json

import pytest

from misen import BlockError, Free, Guided, LoopMaxIterationsError, free, guided, tool


# ── Mock LLM ────────────────────────────────────────────────


def make_mock_llm(responses: list[str]):
    """Create a mock LLM that returns predefined responses in order."""
    idx = 0

    async def mock_llm(messages: list[dict[str, str]]) -> str:
        nonlocal idx
        resp = responses[idx % len(responses)]
        idx += 1
        return resp

    return mock_llm


# ── Guided Tests ────────────────────────────────────────────


class TestGuided:
    async def test_selects_correct_option(self):
        @tool(name="add", description="Add one")
        def add(input: dict) -> dict:
            return {"value": input["value"] + 1}

        @tool(name="double", description="Double the value")
        def double(input: dict) -> dict:
            return {"value": input["value"] * 2}

        llm = make_mock_llm(["double"])
        g = guided(llm, "Pick the best option", [add, double])
        result = await g.run({"value": 5})
        assert result["value"] == 10
        assert result["__misen__"]["guided_choice"] == "double"

    async def test_case_insensitive_match(self):
        @tool(name="upper", description="Uppercase")
        def upper(input: dict) -> dict:
            return {"text": input["text"].upper()}

        llm = make_mock_llm(["UPPER"])
        g = guided(llm, "Pick", [upper])
        result = await g.run({"text": "hello"})
        assert result["text"] == "HELLO"

    async def test_invalid_choice_raises(self):
        @tool(name="a", description="Option A")
        def a(input: dict) -> dict:
            return {}

        llm = make_mock_llm(["nonexistent"])
        g = guided(llm, "Pick", [a])
        with pytest.raises(BlockError, match="nonexistent"):
            await g.run({})

    async def test_corrupted_misen_meta_survives(self):
        """If upstream sets __misen__ to a non-dict, guided should not crash."""

        @tool(name="bad_meta", description="Returns bad __misen__")
        def bad_meta(input: dict) -> dict:
            return {"value": 1, "__misen__": "not-a-dict"}

        llm = make_mock_llm(["bad_meta"])
        g = guided(llm, "Pick", [bad_meta])
        result = await g.run({})
        assert result["__misen__"]["guided_choice"] == "bad_meta"
        assert isinstance(result["__misen__"], dict)

    def test_name(self):
        @tool(name="a")
        def a(input: dict) -> dict:
            return {}

        @tool(name="b")
        def b(input: dict) -> dict:
            return {}

        g = guided(make_mock_llm([""]), "Pick", [a, b])
        assert "a" in g.name
        assert "b" in g.name


# ── Free Tests ──────────────────────────────────────────────


class TestFree:
    async def test_single_tool_call_then_done(self):
        @tool(name="add", description="Add one to value")
        def add(input: dict) -> dict:
            return {"value": input["value"] + 1}

        responses = [
            json.dumps({"tool": "add", "input": {}}),
            json.dumps({"done": True, "result": {"final": True}}),
        ]
        llm = make_mock_llm(responses)
        f = free(llm, "Process the value", [add])
        result = await f.run({"value": 5})
        assert result["value"] == 6
        assert result["final"] is True
        assert result["__misen__"]["free_steps"] == 2

    async def test_unknown_tool_recovers(self):
        @tool(name="add", description="Add one")
        def add(input: dict) -> dict:
            return {"value": input["value"] + 1}

        responses = [
            json.dumps({"tool": "nonexistent", "input": {}}),
            json.dumps({"tool": "add", "input": {}}),
            json.dumps({"done": True}),
        ]
        llm = make_mock_llm(responses)
        f = free(llm, "Do it", [add])
        result = await f.run({"value": 0})
        assert result["value"] == 1

    async def test_invalid_json_raises(self):
        """Non-JSON response now raises BlockError instead of silently returning."""

        @tool(name="noop")
        def noop(input: dict) -> dict:
            return {}

        llm = make_mock_llm(["I'm done, no tools needed"])
        f = free(llm, "Do something", [noop])
        with pytest.raises(BlockError, match="invalid JSON"):
            await f.run({"x": 1})

    async def test_max_steps_raises(self):
        """max_steps exceeded now raises LoopMaxIterationsError."""

        @tool(name="add", description="Add")
        def add(input: dict) -> dict:
            return {"value": input.get("value", 0) + 1}

        responses = [json.dumps({"tool": "add", "input": {}})]
        llm = make_mock_llm(responses)
        f = free(llm, "Go", [add], max_steps=5)
        with pytest.raises(LoopMaxIterationsError):
            await f.run({"value": 0})

    async def test_multi_tool_workflow(self):
        @tool(name="fetch", description="Fetch data")
        def fetch(input: dict) -> dict:
            return {"data": "raw_content"}

        @tool(name="process", description="Process data")
        def process(input: dict) -> dict:
            return {"processed": input["data"].upper()}

        responses = [
            json.dumps({"tool": "fetch", "input": {}}),
            json.dumps({"tool": "process", "input": {}}),
            json.dumps({"done": True, "result": {"status": "complete"}}),
        ]
        llm = make_mock_llm(responses)
        f = free(llm, "Fetch and process", [fetch, process])
        result = await f.run({})
        assert result["data"] == "raw_content"
        assert result["processed"] == "RAW_CONTENT"
        assert result["status"] == "complete"
