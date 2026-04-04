"""Tests for Block, FunctionBlock, and @tool."""

import pytest

from misen import Block, FunctionBlock, tool
from misen.core.operators import Parallel, Sequential


class TestBlockABC:
    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            Block()  # type: ignore[abstract]

    def test_default_name_is_classname(self):
        class MyBlock(Block):
            async def run(self, input):
                return input

        b = MyBlock()
        assert b.name == "MyBlock"

    def test_repr(self):
        class MyBlock(Block):
            async def run(self, input):
                return input

        b = MyBlock(name="test")
        assert "test" in repr(b)


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
