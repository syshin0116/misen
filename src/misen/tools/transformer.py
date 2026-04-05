"""Transformer — apply a function to a specific key."""

from __future__ import annotations

from typing import Any, Callable

from misen.core.block import Block


class Transformer(Block):
    """Apply a transformation function to a single key.

    Config (constructor):
        fn: The transformation ``value -> value``.
        input_key / output_key: Dict keys to read from / write to.
    """

    def __init__(
        self,
        fn: Callable[[Any], Any],
        *,
        input_key: str = "text",
        output_key: str | None = None,
        name: str = "",
    ) -> None:
        super().__init__(name=name or f"Transformer({input_key})")
        self.fn = fn
        self.input_key = input_key
        self.output_key = output_key or input_key

    async def execute(self, input: dict[str, Any]) -> dict[str, Any]:
        return {self.output_key: self.fn(input[self.input_key])}
