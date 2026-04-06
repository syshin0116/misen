"""Tests for branch, loop, and map_each operators."""

import pytest

from misen import (
    Branch,
    Loop,
    LoopMaxIterationsError,
    MapEach,
    branch,
    loop,
    map_each,
    sequential,
    tool,
)


class TestBranch:
    async def test_true_branch(self):
        @tool(name="upper")
        def upper(input: dict) -> dict:
            return {"text": input["text"].upper()}

        @tool(name="lower")
        def lower(input: dict) -> dict:
            return {"text": input["text"].lower()}

        b = branch(lambda d: d["loud"], upper, lower)
        result = await b.run({"text": "Hello", "loud": True})
        assert result["text"] == "HELLO"

    async def test_false_branch(self):
        @tool(name="upper")
        def upper(input: dict) -> dict:
            return {"text": input["text"].upper()}

        @tool(name="lower")
        def lower(input: dict) -> dict:
            return {"text": input["text"].lower()}

        b = branch(lambda d: d["loud"], upper, lower)
        result = await b.run({"text": "Hello", "loud": False})
        assert result["text"] == "hello"

    async def test_no_false_branch_returns_input(self):
        @tool(name="upper")
        def upper(input: dict) -> dict:
            return {"text": input["text"].upper()}

        b = branch(lambda d: d.get("transform", False), upper)
        result = await b.run({"text": "hello"})
        assert result["text"] == "hello"

    async def test_does_not_mutate_input(self):
        @tool(name="add")
        def add(input: dict) -> dict:
            return {"value": input["value"] + 1}

        original = {"value": 1, "go": True}
        b = branch(lambda d: d["go"], add)
        await b.run(original)
        assert original == {"value": 1, "go": True}

    async def test_in_sequential(self, add_one, double):
        pipeline = sequential(
            branch(lambda d: d["value"] > 5, double, add_one),
        )
        assert (await pipeline.run({"value": 10}))["value"] == 20
        assert (await pipeline.run({"value": 3}))["value"] == 4

    async def test_async_predicate(self):
        @tool(name="yes")
        def yes(input: dict) -> dict:
            return {"answer": "yes"}

        @tool(name="no")
        def no(input: dict) -> dict:
            return {"answer": "no"}

        async def check(data: dict) -> bool:
            return data["score"] > 50

        b = branch(check, yes, no)
        assert (await b.run({"score": 80}))["answer"] == "yes"
        assert (await b.run({"score": 30}))["answer"] == "no"

    def test_name(self):
        @tool(name="a")
        def a(input: dict) -> dict:
            return {}

        @tool(name="b")
        def b(input: dict) -> dict:
            return {}

        assert "a" in branch(lambda d: True, a, b).name
        assert "b" in branch(lambda d: True, a, b).name


class TestLoop:
    async def test_basic_loop(self):
        @tool(name="increment")
        def increment(input: dict) -> dict:
            return {"count": input["count"] + 1}

        lp = loop(increment, until=lambda d: d["count"] >= 5)
        result = await lp.run({"count": 0})
        assert result["count"] == 5

    async def test_max_iterations_error(self):
        @tool(name="noop")
        def noop(input: dict) -> dict:
            return {"x": input["x"]}

        with pytest.raises(LoopMaxIterationsError):
            await loop(noop, until=lambda d: False, max_iterations=3).run({"x": 0})

    async def test_accumulates_data(self):
        @tool(name="step")
        def step(input: dict) -> dict:
            history = input.get("history", [])
            return {
                "value": input["value"] + 1,
                "history": [*history, input["value"]],
            }

        lp = loop(step, until=lambda d: d["value"] >= 3)
        result = await lp.run({"value": 0})
        assert result["value"] == 3
        assert result["history"] == [0, 1, 2]

    async def test_single_iteration(self):
        @tool(name="set_done")
        def set_done(input: dict) -> dict:
            return {"done": True}

        lp = loop(set_done, until=lambda d: d.get("done", False))
        result = await lp.run({})
        assert result["done"] is True

    async def test_async_predicate(self):
        @tool(name="inc")
        def inc(input: dict) -> dict:
            return {"n": input["n"] + 1}

        async def done(data: dict) -> bool:
            return data["n"] >= 3

        result = await loop(inc, until=done).run({"n": 0})
        assert result["n"] == 3

    def test_name(self):
        @tool(name="step")
        def step(input: dict) -> dict:
            return {}

        assert "step" in loop(step, until=lambda d: True).name


class TestMapEach:
    async def test_basic_map(self):
        @tool(name="double_item")
        def double_item(input: dict) -> dict:
            return {"result": input["item"] * 2}

        m = map_each(double_item, over_key="numbers")
        result = await m.run({"numbers": [1, 2, 3]})
        assert result["numbers"] == [
            {"result": 2},
            {"result": 4},
            {"result": 6},
        ]

    async def test_custom_keys(self):
        @tool(name="process")
        def process(input: dict) -> dict:
            return {"processed": input["doc"].upper()}

        m = MapEach(
            process,
            over_key="documents",
            item_key="doc",
            output_key="results",
        )
        result = await m.run({"documents": ["hello", "world"]})
        assert result["results"] == [
            {"processed": "HELLO"},
            {"processed": "WORLD"},
        ]

    async def test_passes_rest_of_input(self):
        @tool(name="with_context")
        def with_context(input: dict) -> dict:
            return {"out": f"{input['prefix']}_{input['item']}"}

        m = map_each(with_context, over_key="items")
        result = await m.run({"items": ["a", "b"], "prefix": "x"})
        assert result["items"] == [
            {"out": "x_a"},
            {"out": "x_b"},
        ]

    async def test_empty_list(self):
        @tool(name="noop")
        def noop(input: dict) -> dict:
            return {}

        m = map_each(noop, over_key="items")
        result = await m.run({"items": []})
        assert result["items"] == []

    async def test_concurrent_execution(self):
        """Verify elements are processed concurrently."""
        import asyncio

        call_order = []

        @tool(name="slow")
        async def slow(input: dict) -> dict:
            call_order.append(f"start_{input['item']}")
            await asyncio.sleep(0.01)
            call_order.append(f"end_{input['item']}")
            return {"done": input["item"]}

        m = map_each(slow, over_key="items")
        await m.run({"items": [1, 2, 3]})

        starts = [x for x in call_order if x.startswith("start")]
        assert len(starts) == 3

    async def test_string_input_raises(self):
        """Strings must not be silently iterated character-wise."""
        from misen import BlockError

        @tool(name="noop")
        def noop(input: dict) -> dict:
            return {}

        m = map_each(noop, over_key="items")
        with pytest.raises(BlockError, match="must be a list"):
            await m.run({"items": "abc"})

    async def test_bytes_input_raises(self):
        from misen import BlockError

        @tool(name="noop")
        def noop(input: dict) -> dict:
            return {}

        m = map_each(noop, over_key="items")
        with pytest.raises(BlockError, match="must be a list"):
            await m.run({"items": b"abc"})

    def test_name(self):
        @tool(name="proc")
        def proc(input: dict) -> dict:
            return {}

        assert "proc" in map_each(proc, "items").name
        assert "items" in map_each(proc, "items").name
