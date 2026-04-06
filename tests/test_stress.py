"""Stress tests — push every operator to its limits."""

import asyncio
import json

import pytest

from misen import (
    Block,
    BlockError,
    LoopMaxIterationsError,
    MergeConflictError,
    branch,
    free,
    guided,
    loop,
    map_each,
    parallel,
    sequential,
    tool,
)

# ═══════════════════════════════════════════════════════════════
# Block contract — torture the dict→dict boundary
# ═══════════════════════════════════════════════════════════════


class TestContractAbuse:
    """Try every way to violate the Block contract."""

    async def test_return_none(self):
        @tool
        def bad(input: dict) -> dict:
            return None  # type: ignore[return-value]

        with pytest.raises(BlockError, match="expected dict output"):
            await bad.run({})

    async def test_return_list(self):
        @tool
        def bad(input: dict) -> dict:
            return [1, 2, 3]  # type: ignore[return-value]

        with pytest.raises(BlockError, match="expected dict output"):
            await bad.run({})

    async def test_return_string(self):
        @tool
        def bad(input: dict) -> dict:
            return "hello"  # type: ignore[return-value]

        with pytest.raises(BlockError, match="expected dict output"):
            await bad.run({})

    async def test_return_int(self):
        @tool
        def bad(input: dict) -> dict:
            return 42  # type: ignore[return-value]

        with pytest.raises(BlockError, match="expected dict output"):
            await bad.run({})

    async def test_return_bool(self):
        @tool
        def bad(input: dict) -> dict:
            return True  # type: ignore[return-value]

        with pytest.raises(BlockError, match="expected dict output"):
            await bad.run({})

    async def test_return_tuple(self):
        @tool
        def bad(input: dict) -> dict:
            return (1, 2)  # type: ignore[return-value]

        with pytest.raises(BlockError, match="expected dict output"):
            await bad.run({})

    async def test_input_none(self):
        @tool
        def noop(input: dict) -> dict:
            return {}

        with pytest.raises(BlockError, match="expected dict input"):
            await noop.run(None)  # type: ignore[arg-type]

    async def test_input_list(self):
        @tool
        def noop(input: dict) -> dict:
            return {}

        with pytest.raises(BlockError, match="expected dict input"):
            await noop.run([1, 2])  # type: ignore[arg-type]

    async def test_input_string(self):
        @tool
        def noop(input: dict) -> dict:
            return {}

        with pytest.raises(BlockError, match="expected dict input"):
            await noop.run("hello")  # type: ignore[arg-type]

    async def test_exception_becomes_block_error(self):
        @tool
        def explode(input: dict) -> dict:
            raise RuntimeError("boom")

        with pytest.raises(BlockError, match="boom"):
            await explode.run({})

    async def test_key_error_becomes_block_error(self):
        @tool
        def missing_key(input: dict) -> dict:
            return {"value": input["nonexistent"]}

        with pytest.raises(BlockError, match="nonexistent"):
            await missing_key.run({})

    async def test_type_error_becomes_block_error(self):
        @tool
        def type_fail(input: dict) -> dict:
            return {"value": input["x"] + "string"}

        with pytest.raises(BlockError):
            await type_fail.run({"x": 42})

    async def test_zero_division_becomes_block_error(self):
        @tool
        def div_zero(input: dict) -> dict:
            return {"value": input["x"] / 0}

        with pytest.raises(BlockError):
            await div_zero.run({"x": 1})

    async def test_misen_error_not_wrapped(self):
        """MisenError subclasses pass through without wrapping."""

        @tool
        def raise_merge(input: dict) -> dict:
            raise MergeConflictError("direct conflict")

        with pytest.raises(MergeConflictError, match="direct conflict"):
            await raise_merge.run({})

    async def test_empty_dict_input_ok(self):
        @tool
        def noop(input: dict) -> dict:
            return {"ok": True}

        result = await noop.run({})
        assert result == {"ok": True}

    async def test_empty_dict_output_ok(self):
        @tool
        def noop(input: dict) -> dict:
            return {}

        result = await noop.run({"x": 1})
        assert result == {}

    async def test_huge_dict(self):
        @tool
        def echo(input: dict) -> dict:
            return input

        big = {f"key_{i}": f"value_{i}" for i in range(10000)}
        result = await echo.run(big)
        assert len(result) == 10000

    async def test_nested_dict_values(self):
        @tool
        def echo(input: dict) -> dict:
            return input

        deep = {"a": {"b": {"c": {"d": {"e": [1, 2, 3]}}}}}
        result = await echo.run(deep)
        assert result["a"]["b"]["c"]["d"]["e"] == [1, 2, 3]

    async def test_none_values_in_dict(self):
        @tool
        def echo(input: dict) -> dict:
            return {"result": None, "also_none": None}

        result = await echo.run({})
        assert result["result"] is None

    async def test_unicode_keys_and_values(self):
        @tool
        def echo(input: dict) -> dict:
            return input

        result = await echo.run({"한글키": "한글값", "🔑": "🎉", "": "empty_key"})
        assert result["한글키"] == "한글값"
        assert result["🔑"] == "🎉"
        assert result[""] == "empty_key"


# ═══════════════════════════════════════════════════════════════
# Sequential — stress the chain
# ═══════════════════════════════════════════════════════════════


class TestSequentialStress:
    async def test_100_blocks_chain(self):
        """Chain 100 increment blocks."""
        blocks = []
        for i in range(100):

            @tool(name=f"add_{i}")
            def add(input: dict) -> dict:
                return {"value": input["value"] + 1}

            blocks.append(add)

        pipeline = sequential(*blocks)
        result = await pipeline.run({"value": 0})
        assert result["value"] == 100

    async def test_deep_nesting(self):
        """sequential(sequential(sequential(...))) 10 levels deep."""

        @tool(name="inc")
        def inc(input: dict) -> dict:
            return {"value": input["value"] + 1}

        pipeline = inc
        for _ in range(10):
            pipeline = sequential(pipeline, inc)

        result = await pipeline.run({"value": 0})
        assert result["value"] == 11  # 1 + 10 more

    async def test_error_in_middle_stops_chain(self):
        calls = []

        @tool(name="step1")
        def step1(input: dict) -> dict:
            calls.append(1)
            return {"step": 1}

        @tool(name="explode")
        def explode(input: dict) -> dict:
            raise ValueError("mid-chain failure")

        @tool(name="step3")
        def step3(input: dict) -> dict:
            calls.append(3)
            return {"step": 3}

        with pytest.raises(BlockError, match="mid-chain failure"):
            await sequential(step1, explode, step3).run({})

        assert 3 not in calls, "step3 should not have run"

    async def test_key_overwrite_in_chain(self):
        """Later blocks overwrite earlier blocks' keys."""

        @tool(name="set_a")
        def set_a(input: dict) -> dict:
            return {"x": "first"}

        @tool(name="set_b")
        def set_b(input: dict) -> dict:
            return {"x": "second"}

        result = await sequential(set_a, set_b).run({})
        assert result["x"] == "second"

    async def test_pipe_operator_20_blocks(self):
        @tool(name="inc")
        def inc(input: dict) -> dict:
            return {"value": input["value"] + 1}

        pipeline = inc
        for _ in range(19):
            pipeline = pipeline | inc

        assert len(pipeline.blocks) == 20  # should be flattened
        result = await pipeline.run({"value": 0})
        assert result["value"] == 20

    async def test_accumulator_carries_all_keys(self):
        @tool(name="a")
        def a(input: dict) -> dict:
            return {"from_a": True}

        @tool(name="b")
        def b(input: dict) -> dict:
            assert input["from_a"] is True
            return {"from_b": True}

        @tool(name="c")
        def c(input: dict) -> dict:
            assert input["from_a"] is True
            assert input["from_b"] is True
            return {"from_c": True}

        result = await sequential(a, b, c).run({})
        assert all(result[k] for k in ["from_a", "from_b", "from_c"])


# ═══════════════════════════════════════════════════════════════
# Parallel — concurrency edge cases
# ═══════════════════════════════════════════════════════════════


class TestParallelStress:
    async def test_20_concurrent_blocks(self):
        blocks = []
        for i in range(20):

            @tool(name=f"block_{i}")
            def make_block(input: dict, idx=i) -> dict:
                return {f"result_{idx}": idx}

            blocks.append(make_block)

        result = await parallel(*blocks).run({})
        for i in range(20):
            assert result[f"result_{i}"] == i

    async def test_one_fails_all_fail(self):
        @tool(name="ok")
        async def ok(input: dict) -> dict:
            await asyncio.sleep(0.01)
            return {"ok": True}

        @tool(name="explode")
        def explode(input: dict) -> dict:
            raise ValueError("parallel failure")

        with pytest.raises(BlockError, match="parallel failure"):
            await parallel(ok, explode).run({})

    async def test_all_same_key_conflict_error(self):
        blocks = []
        for i in range(5):

            @tool(name=f"writer_{i}")
            def writer(input: dict, idx=i) -> dict:
                return {"shared": idx}

            blocks.append(writer)

        with pytest.raises(MergeConflictError):
            await parallel(*blocks, conflict="error").run({})

    async def test_all_same_key_last_wins(self):
        @tool(name="a")
        def a(input: dict) -> dict:
            return {"x": "a"}

        @tool(name="b")
        def b(input: dict) -> dict:
            return {"x": "b"}

        @tool(name="c")
        def c(input: dict) -> dict:
            return {"x": "c"}

        result = await parallel(a, b, c, conflict="last").run({})
        assert result["x"] == "c"

    async def test_all_same_key_first_wins(self):
        @tool(name="a")
        def a(input: dict) -> dict:
            return {"x": "a"}

        @tool(name="b")
        def b(input: dict) -> dict:
            return {"x": "b"}

        result = await parallel(a, b, conflict="first").run({})
        assert result["x"] == "a"

    async def test_empty_parallel(self):
        result = await parallel().run({"x": 1})
        assert result == {"x": 1}

    async def test_single_block_parallel(self):
        @tool(name="solo")
        def solo(input: dict) -> dict:
            return {"result": 42}

        result = await parallel(solo).run({})
        assert result["result"] == 42

    async def test_parallel_of_parallels(self):
        @tool(name="a")
        def a(input: dict) -> dict:
            return {"a": 1}

        @tool(name="b")
        def b(input: dict) -> dict:
            return {"b": 2}

        @tool(name="c")
        def c(input: dict) -> dict:
            return {"c": 3}

        @tool(name="d")
        def d(input: dict) -> dict:
            return {"d": 4}

        result = await parallel(parallel(a, b), parallel(c, d)).run({})
        assert result == {"a": 1, "b": 2, "c": 3, "d": 4}

    async def test_actually_concurrent(self):
        """Prove blocks run concurrently, not sequentially."""
        import time

        @tool(name="sleep_a")
        async def sleep_a(input: dict) -> dict:
            await asyncio.sleep(0.1)
            return {"a": True}

        @tool(name="sleep_b")
        async def sleep_b(input: dict) -> dict:
            await asyncio.sleep(0.1)
            return {"b": True}

        start = time.monotonic()
        result = await parallel(sleep_a, sleep_b).run({})
        elapsed = time.monotonic() - start

        assert result["a"] is True
        assert result["b"] is True
        # Sequential would take ~0.2s. Parallel should be ~0.1s.
        assert elapsed < 0.18, f"Took {elapsed:.3f}s — likely not concurrent"


# ═══════════════════════════════════════════════════════════════
# Branch — edge cases
# ═══════════════════════════════════════════════════════════════


class TestBranchStress:
    async def test_nested_branches_5_deep(self):
        @tool(name="leaf")
        def leaf(input: dict) -> dict:
            return {"reached": input["value"]}

        def make_branch(depth: int) -> Block:
            if depth == 0:
                return leaf
            threshold = depth
            return branch(
                lambda d, t=threshold: d["value"] > t,
                make_branch(depth - 1),
                leaf,
            )

        b = make_branch(5)
        result = await b.run({"value": 10})
        assert result["reached"] == 10

    async def test_predicate_raises(self):
        @tool(name="noop")
        def noop(input: dict) -> dict:
            return {}

        def bad_pred(d: dict) -> bool:
            raise ValueError("predicate exploded")

        with pytest.raises(BlockError, match="predicate exploded"):
            await branch(bad_pred, noop).run({})

    async def test_async_predicate_raises(self):
        @tool(name="noop")
        def noop(input: dict) -> dict:
            return {}

        async def bad_pred(d: dict) -> bool:
            raise ValueError("async predicate exploded")

        with pytest.raises(BlockError, match="async predicate exploded"):
            await branch(bad_pred, noop).run({})

    async def test_both_branches_are_pipelines(self):
        @tool(name="inc")
        def inc(input: dict) -> dict:
            return {"value": input["value"] + 1}

        @tool(name="dec")
        def dec(input: dict) -> dict:
            return {"value": input["value"] - 1}

        @tool(name="double")
        def double(input: dict) -> dict:
            return {"value": input["value"] * 2}

        b = branch(
            lambda d: d["value"] > 0,
            sequential(inc, double),  # positive: +1 then *2
            sequential(dec, double),  # negative: -1 then *2
        )

        assert (await b.run({"value": 5}))["value"] == 12  # (5+1)*2
        assert (await b.run({"value": -3}))["value"] == -8  # (-3-1)*2

    async def test_branch_preserves_extra_keys(self):
        @tool(name="inc")
        def inc(input: dict) -> dict:
            return {"value": input["value"] + 1}

        result = await branch(lambda d: True, inc).run({"value": 1, "extra": "keep"})
        assert result["extra"] == "keep"


# ═══════════════════════════════════════════════════════════════
# Loop — boundary conditions
# ═══════════════════════════════════════════════════════════════


class TestLoopStress:
    async def test_exactly_at_max_iterations(self):
        """Condition met on the last allowed iteration."""
        counter = {"n": 0}

        @tool(name="inc")
        def inc(input: dict) -> dict:
            counter["n"] += 1
            return {"value": input["value"] + 1}

        lp = loop(inc, until=lambda d: d["value"] >= 10, max_iterations=10)
        result = await lp.run({"value": 0})
        assert result["value"] == 10

    async def test_one_past_max_iterations(self):
        """Condition would be met on iteration 11, but max is 10."""

        @tool(name="inc")
        def inc(input: dict) -> dict:
            return {"value": input["value"] + 1}

        with pytest.raises(LoopMaxIterationsError):
            await loop(inc, until=lambda d: d["value"] >= 11, max_iterations=10).run({"value": 0})

    async def test_max_iterations_1(self):
        @tool(name="inc")
        def inc(input: dict) -> dict:
            return {"value": input["value"] + 1}

        # Condition met after 1 iteration
        result = await loop(inc, until=lambda d: d["value"] >= 1, max_iterations=1).run(
            {"value": 0}
        )
        assert result["value"] == 1

    async def test_max_iterations_1_fails(self):
        @tool(name="inc")
        def inc(input: dict) -> dict:
            return {"value": input["value"] + 1}

        with pytest.raises(LoopMaxIterationsError):
            await loop(inc, until=lambda d: d["value"] >= 2, max_iterations=1).run({"value": 0})

    async def test_loop_with_growing_list(self):
        @tool(name="append")
        def append(input: dict) -> dict:
            items = input.get("items", [])
            return {"items": [*items, len(items)]}

        result = await loop(append, until=lambda d: len(d["items"]) >= 50, max_iterations=100).run(
            {}
        )
        assert result["items"] == list(range(50))

    async def test_nested_loop(self):
        @tool(name="inc_inner")
        def inc_inner(input: dict) -> dict:
            return {"inner": input["inner"] + 1}

        @tool(name="reset_and_inc")
        def reset_and_inc(input: dict) -> dict:
            return {"outer": input["outer"] + 1, "inner": 0}

        inner_loop = loop(inc_inner, until=lambda d: d["inner"] >= 3, max_iterations=10)
        outer_step = sequential(inner_loop, reset_and_inc)
        outer_loop = loop(outer_step, until=lambda d: d["outer"] >= 3, max_iterations=10)

        result = await outer_loop.run({"outer": 0, "inner": 0})
        assert result["outer"] == 3

    async def test_loop_predicate_error(self):
        @tool(name="noop")
        def noop(input: dict) -> dict:
            return {"x": 1}

        def bad_until(d: dict) -> bool:
            raise TypeError("until broke")

        with pytest.raises(BlockError, match="until broke"):
            await loop(noop, until=bad_until).run({})

    async def test_loop_block_error_propagates(self):
        call_count = {"n": 0}

        @tool(name="fail_on_3")
        def fail_on_3(input: dict) -> dict:
            call_count["n"] += 1
            if call_count["n"] == 3:
                raise ValueError("third time's the charm")
            return {"value": input["value"] + 1}

        with pytest.raises(BlockError, match="third time"):
            await loop(fail_on_3, until=lambda d: False, max_iterations=10).run({"value": 0})


# ═══════════════════════════════════════════════════════════════
# MapEach — fanout edge cases
# ═══════════════════════════════════════════════════════════════


class TestMapEachStress:
    async def test_100_elements(self):
        @tool(name="square")
        def square(input: dict) -> dict:
            return {"result": input["item"] ** 2}

        m = map_each(square, over_key="numbers")
        result = await m.run({"numbers": list(range(100))})
        assert len(result["numbers"]) == 100
        assert result["numbers"][99] == {"result": 99**2}

    async def test_error_in_one_element(self):
        @tool(name="fail_on_5")
        def fail_on_5(input: dict) -> dict:
            if input["item"] == 5:
                raise ValueError("five is bad")
            return {"ok": input["item"]}

        with pytest.raises(BlockError, match="five is bad"):
            await map_each(fail_on_5, over_key="items").run({"items": list(range(10))})

    async def test_nested_map_each(self):
        @tool(name="inc")
        def inc(input: dict) -> dict:
            return {"value": input["item"] + 1}

        inner = map_each(inc, over_key="sub_items")

        @tool(name="wrap")
        def wrap(input: dict) -> dict:
            return {"sub_items": input["item"]}

        pipeline = map_each(sequential(wrap, inner), over_key="matrix")
        result = await pipeline.run({"matrix": [[1, 2], [3, 4]]})
        assert len(result["matrix"]) == 2

    async def test_dict_input_raises(self):
        @tool(name="noop")
        def noop(input: dict) -> dict:
            return {}

        with pytest.raises(BlockError, match="must be a list"):
            await map_each(noop, over_key="items").run({"items": {"a": 1}})

    async def test_int_input_raises(self):
        @tool(name="noop")
        def noop(input: dict) -> dict:
            return {}

        with pytest.raises(BlockError, match="must be a list"):
            await map_each(noop, over_key="items").run({"items": 42})

    async def test_tuple_input_raises(self):
        @tool(name="noop")
        def noop(input: dict) -> dict:
            return {}

        with pytest.raises(BlockError, match="must be a list"):
            await map_each(noop, over_key="items").run({"items": (1, 2, 3)})

    async def test_single_element_list(self):
        @tool(name="double")
        def double(input: dict) -> dict:
            return {"result": input["item"] * 2}

        result = await map_each(double, over_key="items").run({"items": [7]})
        assert result["items"] == [{"result": 14}]

    async def test_context_isolation_between_elements(self):
        """Each element should get a clean copy of the base input."""

        @tool(name="mutator")
        def mutator(input: dict) -> dict:
            # Try to mutate - should not affect other elements
            input["shared"] = input.get("shared", 0) + 1
            return {"val": input["item"], "shared_seen": input["shared"]}

        result = await map_each(mutator, over_key="items").run({"items": [1, 2, 3], "shared": 0})
        # Each element should see shared=1 (their own mutation, not others')
        for r in result["items"]:
            assert r["shared_seen"] == 1


# ═══════════════════════════════════════════════════════════════
# Guided — LLM selection edge cases
# ═══════════════════════════════════════════════════════════════


def make_mock_llm(responses: list[str]):
    idx = 0

    async def mock_llm(messages: list[dict[str, str]]) -> str:
        nonlocal idx
        resp = responses[idx % len(responses)]
        idx += 1
        return resp

    return mock_llm


class TestGuidedStress:
    async def test_whitespace_in_response(self):
        @tool(name="pick_me", description="The one")
        def pick_me(input: dict) -> dict:
            return {"picked": True}

        llm = make_mock_llm(["  pick_me  \n"])
        g = guided(llm, "Pick", [pick_me])
        result = await g.run({})
        assert result["picked"] is True

    async def test_llm_returns_empty_string(self):
        @tool(name="a", description="A")
        def a(input: dict) -> dict:
            return {}

        llm = make_mock_llm([""])
        with pytest.raises(BlockError):
            await guided(llm, "Pick", [a]).run({})

    async def test_llm_returns_description_not_name(self):
        @tool(name="analyzer", description="Analyze documents")
        def analyzer(input: dict) -> dict:
            return {}

        llm = make_mock_llm(["Analyze documents"])
        with pytest.raises(BlockError):
            await guided(llm, "Pick", [analyzer]).run({})

    async def test_many_options(self):
        options = []
        for i in range(20):

            @tool(name=f"option_{i}", description=f"Option number {i}")
            def opt(input: dict, idx=i) -> dict:
                return {"chosen": idx}

            options.append(opt)

        llm = make_mock_llm(["option_17"])
        result = await guided(llm, "Pick one", options).run({})
        assert result["chosen"] == 17

    async def test_llm_raises(self):
        @tool(name="a", description="A")
        def a(input: dict) -> dict:
            return {}

        async def failing_llm(messages: list[dict[str, str]]) -> str:
            raise ConnectionError("LLM is down")

        with pytest.raises(BlockError, match="LLM is down"):
            await guided(failing_llm, "Pick", [a]).run({})

    async def test_unicode_option_name(self):
        @tool(name="분석기", description="Korean analyzer")
        def analyzer(input: dict) -> dict:
            return {"lang": "ko"}

        llm = make_mock_llm(["분석기"])
        result = await guided(llm, "Pick", [analyzer]).run({})
        assert result["lang"] == "ko"

    def test_duplicate_option_names_raises(self):
        @tool(name="same_name", description="First")
        def a(input: dict) -> dict:
            return {}

        @tool(name="same_name", description="Second")
        def b(input: dict) -> dict:
            return {}

        with pytest.raises(ValueError, match="duplicate block name"):
            guided(make_mock_llm([""]), "Pick", [a, b])


# ═══════════════════════════════════════════════════════════════
# Free — ReAct loop edge cases
# ═══════════════════════════════════════════════════════════════


class TestFreeStress:
    async def test_llm_calls_same_tool_repeatedly(self):
        call_count = {"n": 0}

        @tool(name="inc", description="Increment")
        def inc(input: dict) -> dict:
            call_count["n"] += 1
            return {"value": input.get("value", 0) + 1}

        responses = [json.dumps({"tool": "inc", "input": {}})] * 5 + [json.dumps({"done": True})]
        result = await free(make_mock_llm(responses), "Go", [inc], max_steps=10).run({})
        assert result["value"] == 5
        assert call_count["n"] == 5

    async def test_llm_returns_malformed_json(self):
        @tool(name="a", description="A")
        def a(input: dict) -> dict:
            return {}

        llm = make_mock_llm(["{broken json"])
        with pytest.raises(BlockError, match="invalid JSON"):
            await free(llm, "Go", [a]).run({})

    async def test_llm_returns_json_without_tool_or_done(self):
        @tool(name="a", description="A")
        def a(input: dict) -> dict:
            return {}

        responses = [
            json.dumps({"random": "data"}),  # no "tool" or "done"
            json.dumps({"done": True}),
        ]
        result = await free(make_mock_llm(responses), "Go", [a], max_steps=5).run({"x": 1})
        # Original input preserved, step count reflects both LLM calls
        assert result["x"] == 1
        assert result["__misen__"]["free_steps"] == 2
        # Should eventually finish

    async def test_tool_raises_error(self):
        @tool(name="explode", description="Will fail")
        def explode(input: dict) -> dict:
            raise ValueError("tool exploded")

        responses = [json.dumps({"tool": "explode", "input": {}})]
        with pytest.raises(BlockError, match="tool exploded"):
            await free(make_mock_llm(responses), "Go", [explode]).run({})

    async def test_llm_raises_on_first_call(self):
        @tool(name="a", description="A")
        def a(input: dict) -> dict:
            return {}

        async def failing_llm(messages: list[dict[str, str]]) -> str:
            raise ConnectionError("network error")

        with pytest.raises(BlockError, match="network error"):
            await free(failing_llm, "Go", [a]).run({})

    async def test_done_with_no_result_key(self):
        @tool(name="a", description="A")
        def a(input: dict) -> dict:
            return {}

        responses = [json.dumps({"done": True})]
        result = await free(make_mock_llm(responses), "Go", [a]).run({"x": 1})
        assert result["x"] == 1  # original input preserved

    async def test_done_with_non_dict_result_ignored(self):
        @tool(name="a", description="A")
        def a(input: dict) -> dict:
            return {}

        responses = [json.dumps({"done": True, "result": "not a dict"})]
        result = await free(make_mock_llm(responses), "Go", [a]).run({"x": 1})
        assert result["x"] == 1  # should not crash

    async def test_max_steps_1(self):
        @tool(name="a", description="A")
        def a(input: dict) -> dict:
            return {}

        responses = [json.dumps({"tool": "a", "input": {}})]
        with pytest.raises(LoopMaxIterationsError):
            await free(make_mock_llm(responses), "Go", [a], max_steps=1).run({})

    def test_duplicate_tool_names_raises(self):
        @tool(name="same", description="First")
        def a(input: dict) -> dict:
            return {}

        @tool(name="same", description="Second")
        def b(input: dict) -> dict:
            return {}

        with pytest.raises(ValueError, match="duplicate block name"):
            free(make_mock_llm([""]), "Go", [a, b])


# ═══════════════════════════════════════════════════════════════
# __misen__ metadata — accumulation and isolation
# ═══════════════════════════════════════════════════════════════


class TestMisenMetadata:
    async def test_guided_then_sequential_preserves_meta(self):
        @tool(name="pick", description="Pick")
        def pick(input: dict) -> dict:
            return {"picked": True}

        @tool(name="after")
        def after(input: dict) -> dict:
            return {"after": True}

        llm = make_mock_llm(["pick"])
        pipeline = sequential(guided(llm, "Go", [pick]), after)
        result = await pipeline.run({})
        assert result["__misen__"]["guided_choice"] == "pick"
        assert result["after"] is True

    async def test_two_guided_in_sequence(self):
        @tool(name="a", description="A")
        def a(input: dict) -> dict:
            return {"from_a": True}

        @tool(name="b", description="B")
        def b(input: dict) -> dict:
            return {"from_b": True}

        pipeline = sequential(
            guided(make_mock_llm(["a"]), "First", [a]),
            guided(make_mock_llm(["b"]), "Second", [b]),
        )
        result = await pipeline.run({})
        # Second guided overwrites __misen__.guided_choice
        assert result["__misen__"]["guided_choice"] == "b"
        assert result["from_a"] is True
        assert result["from_b"] is True

    async def test_free_meta_has_step_count(self):
        @tool(name="inc", description="Inc")
        def inc(input: dict) -> dict:
            return {"value": input.get("value", 0) + 1}

        responses = [
            json.dumps({"tool": "inc", "input": {}}),
            json.dumps({"tool": "inc", "input": {}}),
            json.dumps({"done": True}),
        ]
        result = await free(make_mock_llm(responses), "Go", [inc], max_steps=10).run({})
        assert result["__misen__"]["free_steps"] == 3
        assert result["value"] == 2


# ═══════════════════════════════════════════════════════════════
# Composition — complex real-world-ish scenarios
# ═══════════════════════════════════════════════════════════════


class TestCompositionStress:
    async def test_sequential_parallel_branch_loop_combined(self):
        """A pipeline that uses every operator type."""

        @tool(name="init")
        def init(input: dict) -> dict:
            return {"items": [1, 2, 3], "sum": 0, "count": 0}

        @tool(name="sum_item")
        def sum_item(input: dict) -> dict:
            return {"partial_sum": input["item"]}

        @tool(name="aggregate")
        def aggregate(input: dict) -> dict:
            total = sum(r["partial_sum"] for r in input["results"])
            return {"sum": total}

        @tool(name="inc_count")
        def inc_count(input: dict) -> dict:
            return {"count": input["count"] + 1}

        @tool(name="double_sum")
        def double_sum(input: dict) -> dict:
            return {"sum": input["sum"] * 2}

        pipeline = sequential(
            init,
            map_each(sum_item, over_key="items", output_key="results"),
            aggregate,
            branch(lambda d: d["sum"] > 5, double_sum, inc_count),
        )

        result = await pipeline.run({})
        assert result["sum"] == 12  # 1+2+3=6 > 5, so doubled to 12

    async def test_pipeline_reuse_as_block(self):
        """Use the same sub-pipeline in two different contexts."""

        @tool(name="inc")
        def inc(input: dict) -> dict:
            return {"value": input["value"] + 1}

        @tool(name="double")
        def double(input: dict) -> dict:
            return {"value": input["value"] * 2}

        sub = sequential(inc, double)  # (v+1)*2

        pipeline_a = sequential(sub, inc)  # (v+1)*2 + 1
        pipeline_b = sequential(inc, sub)  # ((v+1)+1)*2

        assert (await pipeline_a.run({"value": 3}))["value"] == 9  # (3+1)*2+1
        assert (await pipeline_b.run({"value": 3}))["value"] == 10  # ((3+1)+1)*2

    async def test_parallel_in_loop(self):
        @tool(name="inc_a")
        def inc_a(input: dict) -> dict:
            return {"a": input.get("a", 0) + 1}

        @tool(name="inc_b")
        def inc_b(input: dict) -> dict:
            return {"b": input.get("b", 0) + 1}

        body = parallel(inc_a, inc_b)
        lp = loop(body, until=lambda d: d.get("a", 0) >= 5, max_iterations=10)
        result = await lp.run({})
        assert result["a"] == 5
        assert result["b"] == 5

    async def test_map_each_in_branch(self):
        @tool(name="process")
        def process(input: dict) -> dict:
            return {"result": input["item"] * 10}

        @tool(name="noop")
        def noop(input: dict) -> dict:
            return {"results": []}

        pipeline = branch(
            lambda d: len(d["items"]) > 0,
            map_each(process, over_key="items", output_key="results"),
            noop,
        )

        result = await pipeline.run({"items": [1, 2, 3]})
        assert len(result["results"]) == 3

        result = await pipeline.run({"items": []})
        assert result["results"] == []

    async def test_20_level_pipe_flatten(self):
        """Verify | flattening doesn't break with extreme depth."""

        @tool(name="inc")
        def inc(input: dict) -> dict:
            return {"value": input["value"] + 1}

        pipeline = inc
        for _ in range(99):
            pipeline = pipeline | inc

        from misen.core.operators import Sequential

        assert isinstance(pipeline, Sequential)
        assert len(pipeline.blocks) == 100

        result = await pipeline.run({"value": 0})
        assert result["value"] == 100

    async def test_concurrent_run_of_same_pipeline(self):
        """Same pipeline instance called concurrently should not interfere."""

        @tool(name="slow_inc")
        async def slow_inc(input: dict) -> dict:
            await asyncio.sleep(0.01)
            return {"value": input["value"] + 1}

        pipeline = sequential(slow_inc, slow_inc, slow_inc)

        results = await asyncio.gather(
            pipeline.run({"value": 0}),
            pipeline.run({"value": 100}),
            pipeline.run({"value": 200}),
        )

        assert results[0]["value"] == 3
        assert results[1]["value"] == 103
        assert results[2]["value"] == 203
