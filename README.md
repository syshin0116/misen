<div align="center">

# misen

**Reusable AI workflow blocks for Python.**

Define once, compose freely, plug in anywhere.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/syshin0116/misen/blob/main/LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-102%20passed-brightgreen.svg)]()

</div>

> **mise en place** (미즈 엉 플라스) — "everything in its place"

---

## Why misen?

AI agent systems reuse the same building blocks — parsing, chunking, embedding, vector search — but each project reimplements them slightly differently. Improvements in one place never reach the others. And pipelines built in LangGraph can't run in MCP, FastAPI, or n8n without being rewritten.

```
Project A:  HWP parsing → chunking → embedding → Qdrant
Project B:  PDF parsing → chunking → embedding → Qdrant
              ↑ nearly identical logic, implemented separately
```

**misen fixes this.** Every task is a `Block(dict → dict)`. Blocks compose into pipelines, pipelines are also blocks, and the whole thing runs anywhere — no platform lock-in.

### What makes it different

| | LangChain / LangGraph | misen |
|--|---|---|
| Unit of reuse | Chain / Node (framework-specific) | Block (`dict → dict`, platform-free) |
| Composition | Graph DSL | Operators (`\|`, `&`, `sequential`, `parallel`) |
| Platform | Locked to the framework | Adapters for LangGraph, MCP, FastAPI, n8n |
| Execution modes | Deterministic or agentic, not both | Mix forced / guided / free in one pipeline |
| Reuse across projects | Copy-paste | Import and compose |

---

## Install

```bash
pip install misen
```

Development:

```bash
git clone https://github.com/syshin0116/misen.git
cd misen
uv venv .venv && source .venv/bin/activate
uv pip install -e ".[dev]"
```

---

## Quick Start

### 1. Define blocks

```python
from misen import tool

@tool
def parse(input: dict) -> dict:
    return {"text": open(input["file"]).read()}

@tool
def chunk(input: dict) -> dict:
    return {"chunks": input["text"].split("\n\n")}

@tool
async def embed(input: dict) -> dict:
    vectors = await embedding_api(input["chunks"])
    return {"vectors": vectors}
```

### 2. Compose

```python
from misen import sequential, parallel

# Pipe syntax
pipeline = parse | chunk | embed

# Or explicit
pipeline = sequential(parse, chunk, embed)

# Nest freely — pipelines are blocks too
qa = sequential(ingest, search, generate_answer)
```

### 3. Run

```python
result = await pipeline.run({"file": "document.hwp"})

# or sync
result = pipeline.run_sync({"file": "document.hwp"})
```

---

## Operators

All operators return blocks, so they nest and compose recursively.

| Operator | What it does |
|---|---|
| `sequential(A, B, C)` | A → B → C, accumulating outputs |
| `parallel(A, B)` | Run concurrently, merge results |
| `branch(cond, A, B)` | Conditional routing (sync or async predicate) |
| `loop(A, until=cond)` | Repeat until condition is met |
| `map_each(A, over=key)` | Apply to each list element concurrently |
| `guided(llm, prompt, [A,B])` | LLM picks one block to run |
| `free(llm, prompt, tools)` | LLM uses tools in a ReAct loop |
| `a \| b` | Pipe syntax for `sequential` |
| `a & b` | Syntax for `parallel` |

### Deterministic

```python
from misen import branch, loop, map_each

router = branch(lambda d: d["doc_type"] == "table", table_analyzer, text_analyzer)
refiner = loop(improve, until=lambda d: d["score"] > 0.9)
batch = map_each(process_item, over_key="documents")
```

### LLM-driven

```python
from misen import guided, free

# LLM picks the best analysis method
router = guided(my_llm, "Choose analysis method", [stat_analysis, semantic_analysis])

# LLM uses tools freely in a ReAct loop
agent = free(my_llm, "Analyze this document", [search, summarize, extract])
```

The `llm` argument is any async callable `list[dict] → str`. Wrap any LLM client to match:

```python
async def my_llm(messages: list[dict[str, str]]) -> str:
    resp = await client.chat.completions.create(model="gpt-4o", messages=messages)
    return resp.choices[0].message.content
```

### Parallel conflict strategies

```python
parallel(a, b)                    # "last" (default) — later block wins
parallel(a, b, conflict="first")  # first block wins
parallel(a, b, conflict="error")  # raises MergeConflictError
```

---

## Built-in Tools

```python
from misen.tools import TextSplitter, Transformer

splitter = TextSplitter(chunk_size=1000, overlap=200)
counter = Transformer(len, input_key="chunks", output_key="chunk_count")

pipeline = splitter | counter
result = pipeline.run_sync({"text": long_text})
# → {"chunks": [...], "chunk_count": 5}
```

`TextSplitter` guarantees every chunk is at most `chunk_size` characters, including forced character-level splitting for oversized segments.

---

## How it works

### Block: the single abstraction

```
Block: dict → dict
```

Everything is a block — a tool, a pipeline, a nested composition. The `dict → dict` contract is enforced at runtime: non-dict inputs/outputs raise `BlockError`.

Subclasses implement `execute()`. The public `run()` method adds input/output validation and error normalization.

### Platform independent

The core knows nothing about platforms. Adapters translate between misen blocks and external systems.

```
misen Block (dict → dict)
    ├── LangGraph adapter → LangGraph node
    ├── MCP adapter → MCP tool
    ├── FastAPI adapter → REST endpoint
    └── n8n adapter → HTTP call
```

### Rust vs Python

| Area | Language | Rationale |
|---|---|---|
| Core (block, operators, runner) | **Python** | Orchestration, not CPU-bound |
| Adapters | **Python** | Platform integration, I/O-bound |
| Text splitting, token counting | Python → **Rust** | 10x for large batches (PyO3 + maturin) |
| Binary parsing (HWP) | **Rust** | Memory safety, speed |

The core stays Python. Rust is only for CPU-bound data processing, distributed as wheels. Falls back to Python when Rust extensions aren't installed.

---

## Tests

```bash
pytest tests/ -v
```

102 tests covering blocks, operators, agent ops, registry, tools, and composition.

---

## License

[MIT](LICENSE)
