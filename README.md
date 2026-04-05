# misen

> **mise en place** — "everything in its place"

A Python library for defining reusable AI workflow blocks, composing them freely, and plugging them in anywhere.

## Why

When building AI agent systems across multiple projects, core tasks (parsing, chunking, embedding, vector search, etc.) get reimplemented every time. Each project ends up with slightly different versions of the same logic. Improvements in one project never make it to the others.

```
Project A:  HWP parsing → chunking → embedding → Qdrant
Project B:  PDF parsing → chunking → embedding → Qdrant
              ↑ nearly identical logic, implemented separately
```

misen solves this with a single interface: **Block(`dict → dict`)**.

## Install

```bash
pip install misen
```

Development setup:

```bash
git clone https://github.com/syshin0116/misen.git
cd misen
uv venv .venv && source .venv/bin/activate
uv pip install -e ".[dev]"
```

## Quick Start

### Define blocks

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

### Compose blocks

```python
from misen import sequential, parallel, branch, loop, map_each

# Pipe syntax
pipeline = parse | chunk | embed

# Explicit
pipeline = sequential(parse, chunk, embed)

# Parallel execution
analysis = parallel(extract_keywords, generate_summary)

# Conditional branching
router = branch(lambda d: d["doc_type"] == "table", table_analyzer, text_analyzer)

# Repeat until done
refiner = loop(improve, until=lambda d: d["score"] > 0.9)

# Apply to each element in a list
batch = map_each(process_item, over_key="documents")

# Nest freely — pipelines are blocks too
qa_pipeline = sequential(
    ingest,           # a previously defined pipeline
    search,
    generate_answer,
)
```

### LLM-driven operators

```python
from misen import guided, free

# LLM picks one option
router = guided(my_llm, "Choose analysis method", [stat_analysis, semantic_analysis])

# LLM uses tools freely in a ReAct loop
agent = free(my_llm, "Analyze this document", [search, summarize, extract])
```

The `llm` argument is any async callable: `list[dict] → str`. Wrap any LLM client to match.

### Run

```python
# async
result = await pipeline.run({"file": "document.hwp"})

# sync
result = pipeline.run_sync({"file": "document.hwp"})
```

### Built-in tools

```python
from misen.tools import TextSplitter, Transformer

splitter = TextSplitter(chunk_size=1000, overlap=200)
counter = Transformer(len, input_key="chunks", output_key="chunk_count")

pipeline = splitter | counter
result = pipeline.run_sync({"text": long_text})
# → {"chunks": [...], "chunk_count": 5}
```

## Core Concepts

### Block

The universal building unit. Takes a `dict`, returns a `dict`.

```
Block: dict → dict
```

Tools, skills, and pipelines are all blocks. Composition results are also blocks — they nest recursively (closure property).

The `dict → dict` contract is enforced at runtime: non-dict inputs/outputs raise `BlockError`. Unhandled exceptions inside blocks are also wrapped in `BlockError`.

Subclasses implement `execute()`. The public `run()` method adds validation.

### Operators

| Operator | Description |
|---|---|
| `sequential(A, B, C)` | Run A → B → C in order |
| `parallel(A, B)` | Run concurrently, merge outputs |
| `branch(cond, A, B)` | Conditional (sync or async predicate) |
| `loop(A, until=cond)` | Repeat until condition met |
| `map_each(A, over=key)` | Apply to each list element (concurrent) |
| `guided(llm, prompt, opts)` | LLM picks from options |
| `free(llm, prompt, tools)` | LLM uses tools in a ReAct loop |
| `a \| b` | Pipe syntax for sequential |
| `a & b` | Syntax for parallel |

### Parallel conflict strategies

When two blocks output the same key:

```python
parallel(a, b)                    # "last" (default) — later block wins
parallel(a, b, conflict="first")  # first block wins
parallel(a, b, conflict="error")  # raises MergeConflictError
```

### Registry (optional)

Registry is not part of the core execution path. It's an optional catalog layer for organizing blocks when you have many of them.

```python
from misen import Registry

reg = Registry()
reg.register(parse, tags=["ingest"])
reg.register(chunk, tags=["ingest"])

block = reg.get("parse")
results = reg.search(tags=["ingest"])
```

Blocks compose and run without any registry. Use it when you need name-based lookup, tag search, or dynamic block resolution.

### Platform independent

The core knows nothing about platforms. Adapters handle the translation.

```
misen Block (dict → dict)
    ├── LangGraph adapter → LangGraph node
    ├── MCP adapter → MCP tool
    ├── FastAPI adapter → REST endpoint
    └── n8n adapter → HTTP call
```

## Tests

```bash
pytest tests/ -v
```

## License

MIT
