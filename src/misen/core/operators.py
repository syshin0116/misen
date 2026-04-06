"""Operators — compose Blocks into larger Blocks."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable
from typing import Any, Literal

from misen.core.block import Block
from misen.errors import BlockError, LoopMaxIterationsError, MergeConflictError

Predicate = Callable[[dict[str, Any]], bool | Awaitable[bool]]
"""A sync or async function that takes a dict and returns bool."""


async def _call_predicate(pred: Predicate, data: dict[str, Any]) -> bool:
    result = pred(data)
    if inspect.isawaitable(result):
        return await result
    return result  # type: ignore[return-value]


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

    async def execute(self, input: dict[str, Any]) -> dict[str, Any]:
        data = dict(input)
        for block in self.blocks:
            result = await block.run(data)
            data.update(result)
        return data

    def describe(self, indent: int = 0) -> str:
        prefix = "  " * indent
        connector = "├─ " if indent > 0 else ""
        lines = [f"{prefix}{connector}Sequential"]
        for block in self.blocks:
            lines.append(block.describe(indent + 1))
        return "\n".join(lines)

    def to_mermaid(self) -> str:
        lines = ["graph LR"]
        counter = {"n": 0}
        self._mermaid_build(lines, counter)
        return "\n".join(lines)

    def _mermaid_build(
        self, lines: list[str], counter: dict[str, int]
    ) -> tuple[str, str]:
        prev_exit = None
        first_entry = None
        for block in self.blocks:
            if hasattr(block, "_mermaid_build"):
                entry, exit_ = block._mermaid_build(lines, counter)
            else:
                counter["n"] += 1
                nid = f"N{counter['n']}"
                lines.append(f'    {nid}["{block.name}"]')
                entry, exit_ = nid, nid
            if first_entry is None:
                first_entry = entry
            if prev_exit is not None:
                lines.append(f"    {prev_exit} --> {entry}")
            prev_exit = exit_
        return first_entry or "", prev_exit or ""


class Parallel(Block):
    """Run blocks concurrently, merge outputs.

    All blocks receive the same input. Outputs are merged with a
    configurable conflict strategy.

    Args:
        conflict: How to handle duplicate keys across block outputs.
            - ``"last"``  (default): later block wins (positional order).
            - ``"first"``: first block wins.
            - ``"error"``: raise ``MergeConflictError`` for ANY duplicate key,
              including keys that were in the original input.
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

    async def execute(self, input: dict[str, Any]) -> dict[str, Any]:
        results = await asyncio.gather(*(block.run(dict(input)) for block in self.blocks))

        merged: dict[str, Any] = dict(input)
        seen_keys: dict[str, int] = {}  # key → index of block that first produced it

        for i, result in enumerate(results):
            for key, value in result.items():
                if key in seen_keys:
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
                seen_keys.setdefault(key, i)

        return merged

    def describe(self, indent: int = 0) -> str:
        prefix = "  " * indent
        connector = "├─ " if indent > 0 else ""
        lines = [f"{prefix}{connector}Parallel"]
        for block in self.blocks:
            lines.append(block.describe(indent + 1))
        return "\n".join(lines)

    def _mermaid_build(
        self, lines: list[str], counter: dict[str, int]
    ) -> tuple[str, str]:
        counter["n"] += 1
        fork_id = f"N{counter['n']}"
        lines.append(f'    {fork_id}{{"Parallel"}}')
        counter["n"] += 1
        join_id = f"N{counter['n']}"
        lines.append(f'    {join_id}(("join"))')
        for block in self.blocks:
            if hasattr(block, "_mermaid_build"):
                entry, exit_ = block._mermaid_build(lines, counter)
            else:
                counter["n"] += 1
                nid = f"N{counter['n']}"
                lines.append(f'    {nid}["{block.name}"]')
                entry, exit_ = nid, nid
            lines.append(f"    {fork_id} --> {entry}")
            lines.append(f"    {exit_} --> {join_id}")
        return fork_id, join_id


class Branch(Block):
    """Pick one of two blocks based on a condition.

    Args:
        condition: Sync or async callable that returns bool.
        if_true: Block to run when condition is True.
        if_false: Block to run when condition is False (optional).
    """

    def __init__(
        self,
        condition: Predicate,
        if_true: Block,
        if_false: Block | None = None,
        *,
        name: str = "",
        description: str = "",
    ) -> None:
        super().__init__(
            name=name or f"Branch({if_true.name}, {if_false.name if if_false else 'None'})",
            description=description,
        )
        self.condition = condition
        self.if_true = if_true
        self.if_false = if_false

    async def execute(self, input: dict[str, Any]) -> dict[str, Any]:
        data = dict(input)
        if await _call_predicate(self.condition, input):
            result = await self.if_true.run(dict(input))
            data.update(result)
        elif self.if_false is not None:
            result = await self.if_false.run(dict(input))
            data.update(result)
        return data

    def describe(self, indent: int = 0) -> str:
        prefix = "  " * indent
        connector = "├─ " if indent > 0 else ""
        lines = [f"{prefix}{connector}Branch"]
        lines.append(f"{'  ' * (indent + 1)}├─ if_true: {self.if_true.name}")
        if self.if_false:
            lines.append(f"{'  ' * (indent + 1)}├─ if_false: {self.if_false.name}")
        return "\n".join(lines)

    def _mermaid_build(
        self, lines: list[str], counter: dict[str, int]
    ) -> tuple[str, str]:
        counter["n"] += 1
        cond_id = f"N{counter['n']}"
        lines.append(f'    {cond_id}{{"Branch"}}')
        counter["n"] += 1
        join_id = f"N{counter['n']}"
        lines.append(f'    {join_id}(("join"))')
        # if_true
        if hasattr(self.if_true, "_mermaid_build"):
            t_entry, t_exit = self.if_true._mermaid_build(lines, counter)
        else:
            counter["n"] += 1
            t_entry = t_exit = f"N{counter['n']}"
            lines.append(f'    {t_entry}["{self.if_true.name}"]')
        lines.append(f"    {cond_id} -->|true| {t_entry}")
        lines.append(f"    {t_exit} --> {join_id}")
        # if_false
        if self.if_false:
            if hasattr(self.if_false, "_mermaid_build"):
                f_entry, f_exit = self.if_false._mermaid_build(lines, counter)
            else:
                counter["n"] += 1
                f_entry = f_exit = f"N{counter['n']}"
                lines.append(f'    {f_entry}["{self.if_false.name}"]')
            lines.append(f"    {cond_id} -->|false| {f_entry}")
            lines.append(f"    {f_exit} --> {join_id}")
        else:
            lines.append(f"    {cond_id} -->|false| {join_id}")
        return cond_id, join_id


class Loop(Block):
    """Repeat a block until a condition is met.

    The block runs, its output becomes the next iteration's input.
    Stops when ``until(output)`` returns True or ``max_iterations`` is reached.

    Args:
        block: The block to repeat.
        until: Sync or async callable that returns True to stop.
        max_iterations: Safety limit (default 100).
    """

    def __init__(
        self,
        block: Block,
        until: Predicate,
        max_iterations: int = 100,
        *,
        name: str = "",
        description: str = "",
    ) -> None:
        super().__init__(
            name=name or f"Loop({block.name})",
            description=description,
        )
        self.block = block
        self.until = until
        self.max_iterations = max_iterations

    async def execute(self, input: dict[str, Any]) -> dict[str, Any]:
        data = dict(input)
        for _i in range(self.max_iterations):
            result = await self.block.run(data)
            data.update(result)
            if await _call_predicate(self.until, data):
                return data
        raise LoopMaxIterationsError(
            f"Loop({self.block.name}) exceeded {self.max_iterations} iterations"
        )

    def describe(self, indent: int = 0) -> str:
        prefix = "  " * indent
        connector = "├─ " if indent > 0 else ""
        lines = [f"{prefix}{connector}Loop (max={self.max_iterations})"]
        lines.append(self.block.describe(indent + 1))
        return "\n".join(lines)

    def _mermaid_build(
        self, lines: list[str], counter: dict[str, int]
    ) -> tuple[str, str]:
        counter["n"] += 1
        loop_id = f"N{counter['n']}"
        lines.append(f'    {loop_id}[/"Loop (max={self.max_iterations})"/]')
        if hasattr(self.block, "_mermaid_build"):
            entry, exit_ = self.block._mermaid_build(lines, counter)
        else:
            counter["n"] += 1
            entry = exit_ = f"N{counter['n']}"
            lines.append(f'    {entry}["{self.block.name}"]')
        lines.append(f"    {loop_id} --> {entry}")
        lines.append(f"    {exit_} -->|until| {loop_id}")
        return loop_id, loop_id


class MapEach(Block):
    """Apply a block to each element of a list key.

    Reads a list from ``over_key``, runs the block once per element
    (concurrently), and writes the results list to ``output_key``.

    Each element is passed as ``{item_key: element, **rest_of_input}``.
    """

    def __init__(
        self,
        block: Block,
        over_key: str = "items",
        *,
        item_key: str = "item",
        output_key: str | None = None,
        name: str = "",
        description: str = "",
    ) -> None:
        super().__init__(
            name=name or f"MapEach({block.name}, over={over_key!r})",
            description=description,
        )
        self.block = block
        self.over_key = over_key
        self.item_key = item_key
        self.output_key = output_key or over_key

    async def execute(self, input: dict[str, Any]) -> dict[str, Any]:
        items = input[self.over_key]
        if isinstance(items, (str, bytes)):
            raise BlockError(
                f"MapEach: {self.over_key!r} must be a list, got {type(items).__name__}. "
                f"Strings are not iterated element-wise."
            )
        if not isinstance(items, list):
            raise BlockError(
                f"MapEach: {self.over_key!r} must be a list, got {type(items).__name__}"
            )
        base = {k: v for k, v in input.items() if k != self.over_key}

        async def run_one(element: Any) -> dict[str, Any]:
            return await self.block.run({**base, self.item_key: element})

        results = await asyncio.gather(*(run_one(el) for el in items))
        return {self.output_key: list(results)}

    def describe(self, indent: int = 0) -> str:
        prefix = "  " * indent
        connector = "├─ " if indent > 0 else ""
        lines = [f"{prefix}{connector}MapEach(over={self.over_key!r})"]
        lines.append(self.block.describe(indent + 1))
        return "\n".join(lines)

    def _mermaid_build(
        self, lines: list[str], counter: dict[str, int]
    ) -> tuple[str, str]:
        counter["n"] += 1
        map_id = f"N{counter['n']}"
        lines.append(f'    {map_id}[/"MapEach(over={self.over_key!r})"/]')
        if hasattr(self.block, "_mermaid_build"):
            entry, exit_ = self.block._mermaid_build(lines, counter)
        else:
            counter["n"] += 1
            entry = exit_ = f"N{counter['n']}"
            lines.append(f'    {entry}["{self.block.name}"]')
        lines.append(f"    {map_id} --> {entry}")
        lines.append(f"    {exit_} --> {map_id}")
        return map_id, map_id


class Retry(Block):
    """Retry a block on failure with exponential backoff.

    Args:
        block: The block to retry.
        max_retries: Maximum number of retry attempts (default 3).
        backoff: Base delay in seconds between retries (default 1.0).
        on_retry: Optional callback ``(attempt, error) -> None`` for logging.
    """

    def __init__(
        self,
        block: Block,
        max_retries: int = 3,
        backoff: float = 1.0,
        *,
        on_retry: Callable[[int, Exception], Any] | None = None,
        name: str = "",
        description: str = "",
    ) -> None:
        super().__init__(
            name=name or f"Retry({block.name}, max={max_retries})",
            description=description,
        )
        self.block = block
        self.max_retries = max_retries
        self.backoff = backoff
        self.on_retry = on_retry

    async def execute(self, input: dict[str, Any]) -> dict[str, Any]:
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                return await self.block.run(dict(input))
            except Exception as exc:
                last_error = exc
                if attempt < self.max_retries:
                    if self.on_retry:
                        self.on_retry(attempt + 1, exc)
                    await asyncio.sleep(self.backoff * (2**attempt))
        raise BlockError(
            f"Retry({self.block.name}) failed after {self.max_retries + 1} attempts: {last_error}"
        ) from last_error

    def describe(self, indent: int = 0) -> str:
        prefix = "  " * indent
        connector = "├─ " if indent > 0 else ""
        lines = [f"{prefix}{connector}Retry (max={self.max_retries}, backoff={self.backoff}s)"]
        lines.append(self.block.describe(indent + 1))
        return "\n".join(lines)

    def _mermaid_build(
        self, lines: list[str], counter: dict[str, int]
    ) -> tuple[str, str]:
        counter["n"] += 1
        retry_id = f"N{counter['n']}"
        lines.append(f'    {retry_id}[/"Retry (max={self.max_retries})"/]')
        if hasattr(self.block, "_mermaid_build"):
            entry, exit_ = self.block._mermaid_build(lines, counter)
        else:
            counter["n"] += 1
            entry = exit_ = f"N{counter['n']}"
            lines.append(f'    {entry}["{self.block.name}"]')
        lines.append(f"    {retry_id} --> {entry}")
        lines.append(f"    {exit_} -->|fail| {retry_id}")
        return retry_id, retry_id


# ── convenience functions ────────────────────────────────────


def sequential(*blocks: Block, **kwargs: Any) -> Sequential:
    """Create a Sequential block."""
    return Sequential(*blocks, **kwargs)


def parallel(*blocks: Block, **kwargs: Any) -> Parallel:
    """Create a Parallel block."""
    return Parallel(*blocks, **kwargs)


def branch(
    condition: Predicate,
    if_true: Block,
    if_false: Block | None = None,
    **kwargs: Any,
) -> Branch:
    """Create a Branch block."""
    return Branch(condition, if_true, if_false, **kwargs)


def loop(
    block: Block,
    until: Predicate,
    max_iterations: int = 100,
    **kwargs: Any,
) -> Loop:
    """Create a Loop block."""
    return Loop(block, until, max_iterations, **kwargs)


def map_each(
    block: Block,
    over_key: str = "items",
    **kwargs: Any,
) -> MapEach:
    """Create a MapEach block."""
    return MapEach(block, over_key, **kwargs)


def retry(
    block: Block,
    max_retries: int = 3,
    backoff: float = 1.0,
    **kwargs: Any,
) -> Retry:
    """Create a Retry block."""
    return Retry(block, max_retries, backoff, **kwargs)
