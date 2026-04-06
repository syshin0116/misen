"""Registry — store and retrieve blocks by name."""

from __future__ import annotations

from typing import Any

from misen.core.block import Block
from misen.errors import RegistryKeyError


class Registry:
    """A named collection of blocks.

    Blocks can be registered directly or discovered from a directory
    of JSON metadata files.

    Usage::

        reg = Registry()
        reg.register(my_block)
        reg.register(other_block, tags=["rag", "ingest"])

        block = reg.get("my_block")
        results = reg.search(tags=["rag"])
    """

    def __init__(self) -> None:
        self._blocks: dict[str, Block] = {}
        self._metadata: dict[str, dict[str, Any]] = {}

    def register(
        self,
        block: Block,
        *,
        name: str | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Register a block.

        Args:
            block: The block to register.
            name: Override the block's name in the registry.
            tags: Searchable tags.
            metadata: Arbitrary extra metadata.
        """
        key = name or block.name
        self._blocks[key] = block
        self._metadata[key] = {
            "name": key,
            "description": block.description,
            "tags": tags or [],
            **(metadata or {}),
        }

    def get(self, name: str) -> Block:
        """Retrieve a block by name. Raises RegistryKeyError if not found."""
        try:
            return self._blocks[name]
        except KeyError:
            raise RegistryKeyError(
                f"Block {name!r} not found. Available: {list(self._blocks.keys())}"
            )

    def has(self, name: str) -> bool:
        """Check if a block is registered."""
        return name in self._blocks

    def list(self) -> list[dict[str, Any]]:
        """Return metadata for all registered blocks."""
        return list(self._metadata.values())

    def search(
        self,
        *,
        tags: list[str] | None = None,
        name_contains: str | None = None,
    ) -> list[dict[str, Any]]:
        """Search blocks by tags and/or name substring.

        Returns metadata dicts for matching blocks.
        """
        results = []
        for meta in self._metadata.values():
            if tags:
                block_tags = set(meta.get("tags", []))
                if not block_tags.issuperset(tags):
                    continue
            if name_contains and name_contains.lower() not in meta["name"].lower():
                continue
            results.append(meta)
        return results

    def unregister(self, name: str) -> None:
        """Remove a block from the registry."""
        self._blocks.pop(name, None)
        self._metadata.pop(name, None)

    @classmethod
    def from_blocks(cls, *blocks: Block, **kwargs: Any) -> Registry:
        """Create a registry pre-populated with blocks."""
        reg = cls()
        for block in blocks:
            reg.register(block, **kwargs)
        return reg

    def __len__(self) -> int:
        return len(self._blocks)

    def __contains__(self, name: str) -> bool:
        return self.has(name)

    def __repr__(self) -> str:
        return f"Registry({len(self)} blocks)"
