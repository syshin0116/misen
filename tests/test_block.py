"""Tests for Block, FunctionBlock, and @tool."""

import pytest

from misen import Block, BlockError, FunctionBlock, tool
from misen.core.operators import Parallel, Sequential


class TestBlockABC:
    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            Block()  # type: ignore[abstract]

    def test_default_name_is_classname(self):
        class MyBlock(Block):
            async def execute(self, input):
                return input

        b = MyBlock()
        assert b.name == "MyBlock"

    def test_repr(self):
        class MyBlock(Block):
            async def execute(self, input):
                return input

        b = MyBlock(name="test")
        assert "test" in repr(b)


class TestDictContract:
    async def test_non_dict_input_raises(self):
        @tool
        def noop(input: dict) -> dict:
            return {}

        with pytest.raises(BlockError, match="expected dict input"):
            await noop.run("not a dict")  # type: ignore[arg-type]

    async def test_non_dict_output_raises(self):
        @tool
        def bad(input: dict) -> dict:
            return None  # type: ignore[return-value]

        with pytest.raises(BlockError, match="expected dict output"):
            await bad.run({})

    async def test_list_output_raises(self):
        @tool
        def bad(input: dict) -> dict:
            return [1, 2, 3]  # type: ignore[return-value]

        with pytest.raises(BlockError, match="expected dict output"):
            await bad.run({})

    async def test_exception_wrapped_in_block_error(self):
        @tool
        def failing(input: dict) -> dict:
            raise ValueError("something broke")

        with pytest.raises(BlockError, match="something broke"):
            await failing.run({})

    async def test_block_error_not_double_wrapped(self):
        @tool
        def failing(input: dict) -> dict:
            from misen.errors import BlockError

            raise BlockError("direct block error")

        with pytest.raises(BlockError, match="direct block error"):
            await failing.run({})


class TestFunctionBlock:
    async def test_sync_function(self):
        def add(input: dict) -> dict:
            return {"result": input["x"] + 1}

        block = FunctionBlock(add)
        result = await block.run({"x": 5})
        assert result == {"result": 6}

    async def test_async_function(self):
        async def add(input: dict) -> dict:
            return {"result": input["x"] + 1}

        block = FunctionBlock(add)
        result = await block.run({"x": 5})
        assert result == {"result": 6}

    def test_name_from_function(self):
        def my_func(input: dict) -> dict:
            return {}

        block = FunctionBlock(my_func)
        assert block.name == "my_func"

    def test_description_from_docstring(self):
        def my_func(input: dict) -> dict:
            """Does something."""
            return {}

        block = FunctionBlock(my_func)
        assert block.description == "Does something."


class TestToolDecorator:
    async def test_bare_decorator(self):
        @tool
        def add(input: dict) -> dict:
            return {"result": input["x"] + 1}

        assert isinstance(add, FunctionBlock)
        assert add.name == "add"
        result = await add.run({"x": 10})
        assert result == {"result": 11}

    async def test_decorator_with_args(self):
        @tool(name="custom", description="A custom block")
        def add(input: dict) -> dict:
            return {"result": input["x"] + 1}

        assert add.name == "custom"
        assert add.description == "A custom block"

    async def test_async_tool(self):
        @tool
        async def async_add(input: dict) -> dict:
            return {"result": input["x"] + 1}

        result = await async_add.run({"x": 3})
        assert result == {"result": 4}


class TestRunSync:
    def test_run_sync(self):
        @tool
        def add(input: dict) -> dict:
            return {"result": input["x"] + 1}

        result = add.run_sync({"x": 7})
        assert result == {"result": 8}

    def test_run_sync_default_input(self):
        @tool
        def noop(input: dict) -> dict:
            return {"ok": True}

        result = noop.run_sync()
        assert result == {"ok": True}


class TestOperatorSugar:
    def test_pipe_returns_sequential(self, add_one, double):
        pipeline = add_one | double
        assert isinstance(pipeline, Sequential)
        assert len(pipeline.blocks) == 2

    def test_ampersand_returns_parallel(self, add_one, double):
        pipeline = add_one & double
        assert isinstance(pipeline, Parallel)
        assert len(pipeline.blocks) == 2

    def test_pipe_flattens(self, add_one, double):
        @tool
        def triple(input: dict) -> dict:
            return {"value": input["value"] * 3}

        pipeline = (add_one | double) | triple
        assert isinstance(pipeline, Sequential)
        assert len(pipeline.blocks) == 3

    def test_ampersand_flattens(self, add_one, double):
        @tool
        def triple(input: dict) -> dict:
            return {"value": input["value"] * 3}

        pipeline = (add_one & double) & triple
        assert isinstance(pipeline, Parallel)
        assert len(pipeline.blocks) == 3
