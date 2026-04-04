"""Tests for the runner module."""

import pytest

from misen import run, run_sync, tool


class TestRunner:
    async def test_run_async(self):
        @tool
        def add(input: dict) -> dict:
            return {"result": input["x"] + 1}

        result = await run(add, {"x": 5})
        assert result == {"result": 6}

    async def test_run_with_none_input(self):
        @tool
        def noop(input: dict) -> dict:
            return {"ok": True}

        result = await run(noop)
        assert result == {"ok": True}

    def test_run_sync(self):
        @tool
        def add(input: dict) -> dict:
            return {"result": input["x"] + 1}

        result = run_sync(add, {"x": 5})
        assert result == {"result": 6}

    def test_run_sync_with_none_input(self):
        @tool
        def noop(input: dict) -> dict:
            return {"ok": True}

        result = run_sync(noop)
        assert result == {"ok": True}
