"""Block — the universal building unit of misen.

Every Block is an async function: dict → dict.
Blocks compose via operators (|, &) and the result is always another Block.

Subclasses implement ``execute()``. The public ``run()`` method validates
the dict→dict contract and wraps errors in ``BlockError``.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import inspect
import time
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from typing import Any, overload

from misen.errors import BlockError, MisenError


class Block(ABC):
    """Abstract base for all misen blocks.

    Subclasses must implement ``async execute(input) -> dict``.
    Call ``run()`` to execute with contract validation.
    """

    name: str
    description: str

    def __init__(self, name: str = "", description: str = "") -> None:
        self.name = name or self.__class__.__name__
        self.description = description

    # ── execution ────────────────────────────────────────────

    async def run(self, input: dict[str, Any], *, trace: bool = False) -> dict[str, Any]:
        """Execute the block with dict→dict contract validation.

        Args:
            input: The input dictionary.
            trace: If True, record execution timing and key info under ``__misen__.trace``.
        """
        if not isinstance(input, dict):
            raise BlockError(f"{self.name}: expected dict input, got {type(input).__name__}")
        start = time.perf_counter() if trace else 0.0
        try:
            result = await self.execute(input)
        except KeyError as exc:
            available = ", ".join(sorted(input.keys())) or "(empty)"
            raise BlockError(
                f"{self.name}: missing required key {exc}. "
                f"Available keys: [{available}]"
            ) from exc
        except MisenError:
            raise
        except Exception as exc:
            raise BlockError(f"{self.name} failed: {exc}") from exc
        if not isinstance(result, dict):
            raise BlockError(f"{self.name}: expected dict output, got {type(result).__name__}")
        if trace:
            elapsed = time.perf_counter() - start
            existing = result.get("__misen__")
            meta = existing if isinstance(existing, dict) else {}
            parent_trace = meta.get("trace", [])
            parent_trace.append(
                {
                    "block": self.name,
                    "elapsed_s": round(elapsed, 4),
                    "input_keys": sorted(input.keys()),
                    "output_keys": sorted(k for k in result if k != "__misen__"),
                }
            )
            meta["trace"] = parent_trace
            result = {**result, "__misen__": meta}
        return result

    @abstractmethod
    async def execute(self, input: dict[str, Any]) -> dict[str, Any]:
        """Core logic. Subclasses must override this, not ``run()``."""
        ...

    def run_sync(self, input: dict[str, Any] | None = None) -> dict[str, Any]:
        """Synchronous convenience wrapper."""
        coro = self.run(input or {})
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None and loop.is_running():
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(asyncio.run, coro).result()
        return asyncio.run(coro)

    # ── operator sugar ───────────────────────────────────────

    def __or__(self, other: Block) -> Block:
        """``a | b`` → sequential(a, b). Flattens nested Sequential."""
        from misen.core.operators import Sequential

        blocks: list[Block] = []
        for b in (self, other):
            if isinstance(b, Sequential):
                blocks.extend(b.blocks)
            else:
                blocks.append(b)
        return Sequential(*blocks)

    def __and__(self, other: Block) -> Block:
        """``a & b`` → parallel(a, b). Flattens nested Parallel."""
        from misen.core.operators import Parallel

        blocks: list[Block] = []
        for b in (self, other):
            if isinstance(b, Parallel):
                blocks.extend(b.blocks)
            else:
                blocks.append(b)
        return Parallel(*blocks)

    # ── visualization ──────────────────────────────────────────

    def describe(self, indent: int = 0) -> str:
        """Return a tree-style description of this block/pipeline."""
        prefix = "  " * indent
        connector = "├─ " if indent > 0 else ""
        return f"{prefix}{connector}{self.name}"

    def to_mermaid(self) -> str:
        """Export pipeline as a Mermaid flowchart diagram."""
        lines = ["graph LR"]
        counter = {"n": 0}
        if hasattr(self, "_mermaid_build"):
            self._mermaid_build(lines, counter)
        else:
            counter["n"] += 1
            nid = f"N{counter['n']}"
            lines.append(f'    {nid}["{self.name}"]')
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"


# ── FunctionBlock ────────────────────────────────────────────


class FunctionBlock(Block):
    """Block wrapping a plain function (sync or async).

    Supports two calling conventions:
    - ``def f(input: dict) -> dict`` — classic dict→dict
    - ``def f(text: str, count: int) -> dict`` — keyword args extracted from input dict
    """

    def __init__(
        self,
        fn: Callable[..., dict[str, Any] | Awaitable[dict[str, Any]]],
        *,
        name: str = "",
        description: str = "",
    ) -> None:
        super().__init__(
            name=name or fn.__name__,
            description=description or fn.__doc__ or "",
        )
        self._fn = fn
        self._is_async = inspect.iscoroutinefunction(fn)
        # Detect if the function uses keyword-style arguments.
        # Classic style: first param is named "input" or annotated as dict.
        # Kwargs style: first param is NOT named "input" and NOT annotated as dict,
        #   AND all params with defaults are excluded from required keys.
        sig = inspect.signature(fn)
        params = list(sig.parameters.values())
        first = params[0] if params else None
        is_classic = first is not None and (
            first.name == "input"
            or first.annotation in (dict, dict[str, Any])
        )
        self._use_kwargs = not is_classic and len(params) > 0
        if self._use_kwargs:
            self._param_names = [p.name for p in params]
            self._has_defaults = {
                p.name for p in params if p.default is not inspect.Parameter.empty
            }

    async def execute(self, input: dict[str, Any]) -> dict[str, Any]:
        if self._use_kwargs:
            kwargs = {}
            for p in self._param_names:
                if p in input:
                    kwargs[p] = input[p]
                elif p not in self._has_defaults:
                    raise KeyError(p)
            if self._is_async:
                return await self._fn(**kwargs)  # type: ignore[misc]
            return self._fn(**kwargs)  # type: ignore[return-value]
        if self._is_async:
            return await self._fn(input)  # type: ignore[misc]
        return self._fn(input)  # type: ignore[return-value]


# ── @tool decorator ──────────────────────────────────────────

_F = Callable[[dict[str, Any]], dict[str, Any] | Awaitable[dict[str, Any]]]


@overload
def tool(fn: _F, /) -> FunctionBlock: ...


@overload
def tool(*, name: str = "", description: str = "") -> Callable[[_F], FunctionBlock]: ...


def tool(
    fn: _F | None = None,
    *,
    name: str = "",
    description: str = "",
) -> FunctionBlock | Callable[[_F], FunctionBlock]:
    """Create a Block from a function.

    Usage::

        @tool
        def my_block(input: dict) -> dict:
            return {"result": input["value"] + 1}

        @tool(name="custom")
        async def my_async_block(input: dict) -> dict:
            return {"result": input["value"] * 2}
    """

    def wrap(f: _F) -> FunctionBlock:
        return FunctionBlock(f, name=name, description=description)

    if fn is not None:
        return wrap(fn)
    return wrap
