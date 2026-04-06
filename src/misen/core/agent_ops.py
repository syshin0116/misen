"""Agent operators — blocks that delegate decisions to an LLM."""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from typing import Any

from misen.core.block import Block
from misen.errors import BlockError, LoopMaxIterationsError

LLMCallable = Callable[[list[dict[str, str]]], Awaitable[str]]
"""An async function that takes chat messages and returns a string response.

Wrap any LLM client to match this signature::

    async def my_llm(messages: list[dict[str, str]]) -> str:
        resp = await client.chat.completions.create(
            model="gpt-4o", messages=messages,
        )
        return resp.choices[0].message.content
"""


# ── Guided ──────────────────────────────────────────────────


class Guided(Block):
    """LLM selects one block from a list of options.

    The LLM receives a system prompt describing the task, plus a list
    of options with names and descriptions. It returns the name of the
    chosen block, which is then executed.

    Runtime metadata (which option was chosen) is stored under the
    ``__misen__`` key to avoid polluting user data.
    """

    def __init__(
        self,
        llm: LLMCallable,
        prompt: str,
        options: list[Block],
        *,
        name: str = "",
        description: str = "",
    ) -> None:
        option_names = ", ".join(b.name for b in options)
        super().__init__(
            name=name or f"Guided([{option_names}])",
            description=description,
        )
        self.llm = llm
        self.prompt = prompt
        self.options: dict[str, Block] = {}
        for b in options:
            if b.name in self.options:
                raise ValueError(
                    f"Guided: duplicate block name {b.name!r}. Each option must have a unique name."
                )
            self.options[b.name] = b

    async def execute(self, input: dict[str, Any]) -> dict[str, Any]:
        options_desc = "\n".join(
            f"- {name}: {block.description or 'No description'}"
            for name, block in self.options.items()
        )

        messages = [
            {
                "role": "system",
                "content": (
                    f"{self.prompt}\n\n"
                    f"Available options:\n{options_desc}\n\n"
                    f"Respond with ONLY the name of the option you choose. "
                    f"Nothing else."
                ),
            },
            {
                "role": "user",
                "content": f"Current data:\n{json.dumps(input, default=str, ensure_ascii=False)}",
            },
        ]

        choice = (await self.llm(messages)).strip()

        # Fuzzy match: try exact, then case-insensitive
        block = self.options.get(choice)
        if block is None:
            for opt_name, opt_block in self.options.items():
                if opt_name.lower() == choice.lower():
                    block = opt_block
                    break

        if block is None:
            raise BlockError(
                f"Guided: LLM chose {choice!r}, but available options are: "
                f"{list(self.options.keys())}"
            )

        result = await block.run(dict(input))
        # Store runtime metadata separately from user data
        existing = result.get("__misen__")
        meta = existing if isinstance(existing, dict) else {}
        meta["guided_choice"] = choice
        return {**result, "__misen__": meta}


# ── Free ────────────────────────────────────────────────────


class Free(Block):
    """LLM uses tools freely in a ReAct-style loop.

    The LLM receives a prompt and a set of available tool-blocks.
    It can call tools by responding with JSON, inspect results,
    and repeat until it decides to finish.

    Finish signal: LLM responds with ``{"done": true, ...}``.

    Errors:
        - ``LoopMaxIterationsError`` if ``max_steps`` is exceeded.
        - ``BlockError`` if LLM returns invalid JSON (not a finish signal).
    """

    def __init__(
        self,
        llm: LLMCallable,
        prompt: str,
        tools: list[Block],
        *,
        max_steps: int = 20,
        name: str = "",
        description: str = "",
    ) -> None:
        tool_names = ", ".join(b.name for b in tools)
        super().__init__(
            name=name or f"Free([{tool_names}])",
            description=description,
        )
        self.llm = llm
        self.prompt = prompt
        self.tools: dict[str, Block] = {}
        for b in tools:
            if b.name in self.tools:
                raise ValueError(
                    f"Free: duplicate block name {b.name!r}. Each tool must have a unique name."
                )
            self.tools[b.name] = b
        self.max_steps = max_steps

    def _build_system_prompt(self) -> str:
        tools_desc = "\n".join(
            f"- {name}: {block.description or 'No description'}"
            for name, block in self.tools.items()
        )
        return (
            f"{self.prompt}\n\n"
            f"Available tools:\n{tools_desc}\n\n"
            f"To use a tool, respond with JSON:\n"
            f'{{"tool": "<tool_name>", "input": {{...}}}}\n\n'
            f"When you are done, respond with:\n"
            f'{{"done": true, "result": {{...}}}}\n\n'
            f"Always respond with valid JSON."
        )

    async def execute(self, input: dict[str, Any]) -> dict[str, Any]:
        messages: list[dict[str, str]] = [
            {"role": "system", "content": self._build_system_prompt()},
            {
                "role": "user",
                "content": f"Task input:\n{json.dumps(input, default=str, ensure_ascii=False)}",
            },
        ]

        data = dict(input)

        for step in range(self.max_steps):
            response = (await self.llm(messages)).strip()
            messages.append({"role": "assistant", "content": response})

            try:
                parsed = json.loads(response)
            except json.JSONDecodeError:
                # LLM may return multiple JSON objects or JSON + text.
                # Try parsing just the first line as JSON.
                first_line = response.split("\n", 1)[0].strip()
                try:
                    parsed = json.loads(first_line)
                except json.JSONDecodeError as exc:
                    raise BlockError(
                        f"Free({self.name}): LLM returned invalid JSON at step {step}: "
                        f"{response[:200]!r}"
                    ) from exc

            if parsed.get("done"):
                result = parsed.get("result", {})
                if isinstance(result, dict):
                    data.update(result)
                existing = data.get("__misen__")
                meta = existing if isinstance(existing, dict) else {}
                meta["free_steps"] = step + 1
                data["__misen__"] = meta
                return data

            tool_name = parsed.get("tool")
            tool_input = parsed.get("input", {})

            if tool_name not in self.tools:
                messages.append(
                    {
                        "role": "user",
                        "content": f"Error: Unknown tool {tool_name!r}. "
                        f"Available: {list(self.tools.keys())}",
                    }
                )
                continue

            block = self.tools[tool_name]
            tool_result = await block.run({**data, **tool_input})
            data.update(tool_result)

            messages.append(
                {
                    "role": "user",
                    "content": f"Tool {tool_name!r} result:\n"
                    f"{json.dumps(tool_result, default=str, ensure_ascii=False)}",
                }
            )

        raise LoopMaxIterationsError(
            f"Free({self.name}) exceeded {self.max_steps} steps without finishing"
        )


# ── convenience functions ────────────────────────────────────


def guided(
    llm: LLMCallable,
    prompt: str,
    options: list[Block],
    **kwargs: Any,
) -> Guided:
    """Create a Guided block."""
    return Guided(llm, prompt, options, **kwargs)


def free(
    llm: LLMCallable,
    prompt: str,
    tools: list[Block],
    **kwargs: Any,
) -> Free:
    """Create a Free block."""
    return Free(llm, prompt, tools, **kwargs)
