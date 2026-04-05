"""Tests for Sequential and Parallel operators."""

import pytest

from misen import MergeConflictError, parallel, sequential, tool
from misen.core.operators import Parallel, Sequential


class TestSequential:
    async def test_chains_two_blocks(self, add_one, double):
        pipeline = sequential(add_one, double)
        result = await pipeline.run({"value": 3})
        # add_one: 3+1=4, double: 4*2=8
        assert result["value"] == 8

    async def test_accumulates_output(self, to_upper, extract_length):
        pipeline = sequential(to_upper, extract_length)
        result = await pipeline.run({"text": "hello"})
        assert result["text"] == "HELLO"
        assert result["length"] == 5

    async def test_does_not_mutate_input(self, add_one):
        original = {"value": 1}
        pipeline = sequential(add_one)
        await pipeline.run(original)
        assert original == {"value": 1}

    async def test_empty_sequential(self):
        pipeline = sequential()
        result = await pipeline.run({"x": 1})
        assert result == {"x": 1}

    async def test_three_blocks(self, add_one):
        pipeline = sequential(add_one, add_one, add_one)
        result = await pipeline.run({"value": 0})
        assert result["value"] == 3

    def test_name_auto_generated(self, add_one, double):
        pipeline = sequential(add_one, double)
        assert "add_one" in pipeline.name
        assert "double" in pipeline.name


class TestParallel:
    async def test_runs_blocks_merges_outputs(self, to_upper, extract_length):
        pipeline = parallel(to_upper, extract_length)
        result = await pipeline.run({"text": "hello"})
        assert result["text"] == "HELLO"
        assert result["length"] == 5

    async def test_all_blocks_receive_same_input(self):
        received = []

        @tool(name="spy_a")
        def spy_a(input: dict) -> dict:
            received.append(("a", dict(input)))
            return {"a": True}

        @tool(name="spy_b")
        def spy_b(input: dict) -> dict:
            received.append(("b", dict(input)))
            return {"b": True}

        pipeline = parallel(spy_a, spy_b)
        await pipeline.run({"x": 42})

        for name, data in received:
            assert data == {"x": 42}

    async def test_conflict_last_default(self):
        @tool(name="first")
        def first(input: dict) -> dict:
            return {"key": "from_first"}

        @tool(name="second")
        def second(input: dict) -> dict:
            return {"key": "from_second"}

        result = await parallel(first, second).run({})
        assert result["key"] == "from_second"

    async def test_conflict_first(self):
        @tool(name="first")
        def first(input: dict) -> dict:
            return {"key": "from_first"}

        @tool(name="second")
        def second(input: dict) -> dict:
            return {"key": "from_second"}

        result = await parallel(first, second, conflict="first").run({})
        assert result["key"] == "from_first"

    async def test_conflict_error(self):
        @tool(name="first")
        def first(input: dict) -> dict:
            return {"key": "from_first"}

        @tool(name="second")
        def second(input: dict) -> dict:
            return {"key": "from_second"}

        with pytest.raises(MergeConflictError):
            await parallel(first, second, conflict="error").run({})

    async def test_conflict_error_on_input_keys_too(self):
        """conflict='error' now applies to ALL keys, including input keys."""

        @tool(name="a")
        def a(input: dict) -> dict:
            return {"x": input["x"] + 1}

        @tool(name="b")
        def b(input: dict) -> dict:
            return {"x": input["x"] + 2}

        with pytest.raises(MergeConflictError):
            await parallel(a, b, conflict="error").run({"x": 0})

    async def test_does_not_mutate_input(self, to_upper):
        original = {"text": "hello"}
        await parallel(to_upper).run(original)
        assert original == {"text": "hello"}


class TestComposition:
    async def test_sequential_of_parallel(self, to_upper, extract_length, add_one):
        pipeline = sequential(
            parallel(to_upper, extract_length),
            add_one,
        )
        result = await pipeline.run({"text": "hello", "value": 0})
        assert result["text"] == "HELLO"
        assert result["length"] == 5
        assert result["value"] == 1

    async def test_parallel_of_sequential(self, add_one, double):
        branch_a = sequential(add_one, double)  # (v+1)*2
        branch_b = sequential(double, add_one)  # v*2+1

        @tool(name="get_a")
        def get_a(input: dict) -> dict:
            return {"a": input["value"]}

        @tool(name="get_b")
        def get_b(input: dict) -> dict:
            return {"b": input["value"]}

        pipeline = parallel(
            sequential(branch_a, get_a),
            sequential(branch_b, get_b),
        )
        result = await pipeline.run({"value": 3})
        assert result["a"] == 8  # (3+1)*2
        assert result["b"] == 7  # 3*2+1

    async def test_pipe_syntax_execution(self, add_one, double):
        pipeline = add_one | double
        result = await pipeline.run({"value": 3})
        assert result["value"] == 8
