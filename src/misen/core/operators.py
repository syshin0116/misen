"""Operators — compose Blocks into larger Blocks."""

from __future__ import annotations

import asyncio
from typing import Any, Literal

from misen.core.block import Block
from misen.errors import MergeConflictError


class Sequential(Block):
    """Run blocks in order, accumulating outputs.

    Each block receives the merged dict of the original input
    plus all previous blocks' outputs.
    """

    def __init__(self, *blocks: Block, name: str = "", description: str = "") -> None:
        block_names = " | ".join(b.name for b in blocks)
        super().__init__(
            name=name or f"Sequential({block_names})",
            description=description,
        )
        self.blocks: tuple[Block, ...] = blocks

    async def run(self, input: dict[str, Any]) -> dict[str, Any]:
        data = dict(input)  # shallow copy — never mutate caller's dict
        for block in self.blocks:
            result = await block.run(data)
            data.update(result)
        return data


class Parallel(Block):
    """Run blocks concurrently, merge outputs.

    All blocks receive the same input. Outputs are merged with a
    configurable conflict strategy.

    Args:
        conflict: How to handle duplicate keys across block outputs.
            - ``"last"``  (default): later block wins (positional order).
            - ``"first"``: first block wins.
            - ``"error"``: raise ``MergeConflictError``.
    """

    def __init__(
        self,
        *blocks: Block,
        conflict: Literal["last", "first", "error"] = "last",
        name: str = "",
        description: str = "",
    ) -> None:
        block_names = " & ".join(b.name for b in blocks)
        super().__init__(
            name=name or f"Parallel({block_names})",
            description=description,
        )
        self.blocks: tuple[Block, ...] = blocks
        self.conflict = conflict

    async def run(self, input: dict[str, Any]) -> dict[str, Any]:
        results = await asyncio.gather(
            *(block.run(dict(input)) for block in self.blocks)
        )

        merged: dict[str, Any] = dict(input)
        seen_keys: dict[str, int] = {}  # key → index of block that first produced it

        for i, result in enumerate(results):
            for key, value in result.items():
                if key in seen_keys and key not in input:
                    if self.conflict == "error":
                        raise MergeConflictError(
                            f"Key {key!r} produced by both "
                            f"{self.blocks[seen_keys[key]].name!r} and "
                            f"{self.blocks[i].name!r}"
                        )
                    elif self.conflict == "first":
                        continue
                    # "last" → fall through to overwrite
                merged[key] = value
                if key not in input:
                    seen_keys.setdefault(key, i)

        return merged


# ── convenience functions ────────────────────────────────────


def sequential(*blocks: Block, **kwargs: Any) -> Sequential:
    """Create a Sequential block."""
    return Sequential(*blocks, **kwargs)


def parallel(*blocks: Block, **kwargs: Any) -> Parallel:
    """Create a Parallel block."""
    return Parallel(*blocks, **kwargs)
