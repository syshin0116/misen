"""Runner — execute blocks."""

from __future__ import annotations

from typing import Any

from misen.core.block import Block


async def run(block: Block, input: dict[str, Any] | None = None) -> dict[str, Any]:
    """Execute a block asynchronously."""
    return await block.run(input or {})


def run_sync(block: Block, input: dict[str, Any] | None = None) -> dict[str, Any]:
    """Execute a block synchronously."""
    return block.run_sync(input or {})
