"""Tests for UX improvements: describe, mermaid, retry, tracing, kwargs tool, keyerror."""

import pytest

from misen import (
    BlockError,
    branch,
    loop,
    map_each,
    parallel,
    retry,
    sequential,
    tool,
)

# ── describe() ──────────────────────────────────────────────


class TestDescribe:
    def test_single_block_describe(self, add_one):
        desc = add_one.describe()
        assert "add_one" in desc

    def test_sequential_describe(self, add_one, double):
        pipe = sequential(add_one, double)
        desc = pipe.describe()
        assert "Sequential" in desc
        assert "add_one" in desc
        assert "double" in desc

    def test_parallel_describe(self, add_one, double):
        pipe = parallel(add_one, double)
        desc = pipe.describe()
        assert "Parallel" in desc
        assert "add_one" in desc
        assert "double" in desc

    def test_nested_describe(self, add_one, double, extract_length):
        pipe = sequential(add_one, parallel(double, extract_length))
        desc = pipe.describe()
        assert "Sequential" in desc
        assert "Parallel" in desc
        # Indentation should increase for nested blocks
        lines = desc.split("\n")
        assert len(lines) >= 3

    def test_branch_describe(self, add_one, double):
        pipe = branch(lambda d: d.get("x", 0) > 0, add_one, double)
        desc = pipe.describe()
        assert "Branch" in desc
        assert "if_true" in desc
        assert "if_false" in desc

    def test_loop_describe(self, add_one):
        pipe = loop(add_one, until=lambda d: d["value"] > 10)
        desc = pipe.describe()
        assert "Loop" in desc
        assert "max=100" in desc

    def test_map_each_describe(self, add_one):
        pipe = map_each(add_one, over_key="items")
        desc = pipe.describe()
        assert "MapEach" in desc
        assert "items" in desc


# ── to_mermaid() ────────────────────────────────────────────


class TestMermaid:
    def test_single_block_mermaid(self, add_one):
        mermaid = add_one.to_mermaid()
        assert "graph LR" in mermaid
        assert "add_one" in mermaid

    def test_sequential_mermaid(self, add_one, double):
        pipe = sequential(add_one, double)
        mermaid = pipe.to_mermaid()
        assert "graph LR" in mermaid
        assert "-->" in mermaid
        assert "add_one" in mermaid
        assert "double" in mermaid

    def test_parallel_mermaid(self, add_one, double):
        pipe = parallel(add_one, double)
        mermaid = pipe.to_mermaid()
        assert "graph LR" in mermaid
        assert "Parallel" in mermaid
        assert "join" in mermaid

    def test_branch_mermaid(self, add_one, double):
        pipe = branch(lambda d: d.get("x", 0) > 0, add_one, double)
        mermaid = pipe.to_mermaid()
        assert "Branch" in mermaid
        assert "true" in mermaid
        assert "false" in mermaid

    def test_nested_mermaid(self, add_one, double, extract_length):
        pipe = sequential(add_one, parallel(double, extract_length))
        mermaid = pipe.to_mermaid()
        assert "graph LR" in mermaid
        assert "-->" in mermaid
        assert "Parallel" in mermaid


# ── Retry ───────────────────────────────────────────────────


class TestRetry:
    @pytest.mark.asyncio
    async def test_retry_succeeds_first_try(self, add_one):
        block = retry(add_one, max_retries=3)
        result = await block.run({"value": 1})
        assert result["value"] == 2

    @pytest.mark.asyncio
    async def test_retry_succeeds_after_failures(self):
        call_count = 0

        @tool(name="flaky")
        def flaky(input: dict) -> dict:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise RuntimeError("temporary failure")
            return {"done": True}

        block = retry(flaky, max_retries=3, backoff=0.01)
        result = await block.run({})
        assert result["done"] is True
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_retry_exhausted(self):
        @tool(name="always_fail")
        def always_fail(input: dict) -> dict:
            raise RuntimeError("permanent failure")

        block = retry(always_fail, max_retries=2, backoff=0.01)
        with pytest.raises(BlockError, match="failed after 3 attempts"):
            await block.run({})

    @pytest.mark.asyncio
    async def test_retry_on_retry_callback(self):
        attempts = []

        @tool(name="flaky2")
        def flaky2(input: dict) -> dict:
            if len(attempts) < 2:
                attempts.append(1)
                raise RuntimeError("fail")
            return {"ok": True}

        block = retry(
            flaky2,
            max_retries=3,
            backoff=0.01,
            on_retry=lambda attempt, exc: attempts.append(f"retry-{attempt}"),
        )
        result = await block.run({})
        assert result["ok"] is True
        assert "retry-1" in attempts

    def test_retry_describe(self, add_one):
        block = retry(add_one, max_retries=5, backoff=2.0)
        desc = block.describe()
        assert "Retry" in desc
        assert "max=5" in desc

    def test_retry_mermaid(self, add_one):
        block = retry(add_one, max_retries=3)
        mermaid = block.to_mermaid()
        assert "Retry" in mermaid
        assert "fail" in mermaid


# ── KeyError improvement ────────────────────────────────────


class TestKeyError:
    @pytest.mark.asyncio
    async def test_missing_key_shows_block_name(self):
        @tool(name="needs_text")
        def needs_text(input: dict) -> dict:
            return {"upper": input["text"].upper()}

        with pytest.raises(BlockError, match="needs_text") as exc_info:
            await needs_text.run({"value": 1})
        assert "missing required key" in str(exc_info.value)
        assert "value" in str(exc_info.value)  # available keys listed

    @pytest.mark.asyncio
    async def test_missing_key_shows_available_keys(self):
        @tool(name="reader")
        def reader(input: dict) -> dict:
            return {"result": input["missing_key"]}

        with pytest.raises(BlockError, match="Available keys"):
            await reader.run({"a": 1, "b": 2})


# ── @tool keyword args ──────────────────────────────────────


class TestToolKwargs:
    @pytest.mark.asyncio
    async def test_kwargs_style_tool(self):
        @tool
        def add(value: int, increment: int) -> dict:
            return {"result": value + increment}

        result = await add.run({"value": 10, "increment": 5})
        assert result["result"] == 15

    @pytest.mark.asyncio
    async def test_kwargs_with_defaults(self):
        @tool
        def greet(name: str, greeting: str = "Hello") -> dict:
            return {"message": f"{greeting}, {name}!"}

        result = await greet.run({"name": "World"})
        assert result["message"] == "Hello, World!"

    @pytest.mark.asyncio
    async def test_kwargs_missing_required(self):
        @tool
        def needs_name(name: str) -> dict:
            return {"greeting": f"Hi {name}"}

        with pytest.raises(BlockError, match="missing required key"):
            await needs_name.run({"other": "value"})

    @pytest.mark.asyncio
    async def test_classic_dict_style_still_works(self):
        @tool
        def classic(input: dict) -> dict:
            return {"value": input["x"] + 1}

        result = await classic.run({"x": 5})
        assert result["value"] == 6

    @pytest.mark.asyncio
    async def test_kwargs_async(self):
        @tool
        async def async_add(a: int, b: int) -> dict:
            return {"sum": a + b}

        result = await async_add.run({"a": 3, "b": 4})
        assert result["sum"] == 7


# ── Execution tracing ───────────────────────────────────────


class TestTracing:
    @pytest.mark.asyncio
    async def test_trace_single_block(self, add_one):
        result = await add_one.run({"value": 1}, trace=True)
        assert result["value"] == 2
        trace = result["__misen__"]["trace"]
        assert len(trace) == 1
        assert trace[0]["block"] == "add_one"
        assert trace[0]["elapsed_s"] >= 0
        assert "value" in trace[0]["input_keys"]
        assert "value" in trace[0]["output_keys"]

    @pytest.mark.asyncio
    async def test_trace_disabled_by_default(self, add_one):
        result = await add_one.run({"value": 1})
        assert "__misen__" not in result

    @pytest.mark.asyncio
    async def test_trace_preserves_existing_meta(self):
        @tool(name="meta_block")
        def meta_block(input: dict) -> dict:
            return {"value": 1, "__misen__": {"custom": "data"}}

        result = await meta_block.run({}, trace=True)
        assert result["__misen__"]["custom"] == "data"
        assert "trace" in result["__misen__"]
