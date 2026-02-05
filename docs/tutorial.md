# KP3 Tutorial: From Zero to Knowledge Processing

This tutorial walks you through KP3's core concepts with hands-on examples. By the end, you'll understand how to store, search, organize, and process text using KP3's knowledge pipeline.

## What is KP3?

KP3 (Knowledge Processing Pipeline) is a system for:
- **Storing text** with automatic semantic embeddings
- **Searching** with hybrid search (keyword + semantic + recency)
- **Organizing** with tags, refs, scopes, and branches
- **Processing** with LLM-powered transformation pipelines
- **Tracking** provenance (what created what)

Think of it as a smart database for AI agents that need to remember, find, and transform information.

## Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager
- Podman or Docker
- OpenAI API key (for embeddings)

## Quick Start

```bash
# 1. Clone and enter the project
cd /path/to/kp3

# 2. Copy environment template
cp .env.example .env

# 3. Edit .env and add your OpenAI API key
# KP3_OPENAI_API_KEY=sk-your-key-here

# 4. Start PostgreSQL with pgvector
podman run -d --name kp3-postgres \
  -e POSTGRES_USER=kp3 \
  -e POSTGRES_PASSWORD=kp3 \
  -e POSTGRES_DB=kp3 \
  -p 5432:5432 \
  pgvector/pgvector:pg16

# 5. Install dependencies
uv sync

# 6. Run database migrations
uv run alembic upgrade head

# 7. Start the service
uv run kp3-service

# 8. (In another terminal) Run examples
uv run python examples/01_passages/semantic_notes.py
```

---

## Part 1: Passages - The Foundation

**Passages** are the fundamental unit of content in KP3. Every piece of text you store is a passage.

### Key Concepts

- **Content**: The text itself
- **Passage Type**: A category (e.g., "note", "document", "memory")
- **Embedding**: Automatically generated vector for semantic search
- **Agent ID**: Isolates data between users/agents

### Creating Passages

```python
import httpx

async with httpx.AsyncClient() as client:
    response = await client.post(
        "http://localhost:8080/passages",
        json={
            "content": "Python is a great language for data science.",
            "passage_type": "note",
        },
        headers={"X-Agent-ID": "my-agent"},
    )
    passage = response.json()
    print(f"Created: {passage['id']}")
```

### Searching Passages

KP3 uses **hybrid search** combining:
1. **Full-text search (FTS)**: Exact keyword matching
2. **Semantic search**: Meaning-based similarity via embeddings
3. **Recency boost**: Newer content ranked higher

```python
response = await client.get(
    "http://localhost:8080/passages/search",
    params={"query": "programming languages", "limit": 5},
    headers={"X-Agent-ID": "my-agent"},
)
results = response.json()["results"]
for r in results:
    print(f"[{r['passage_type']}] {r['content'][:50]}...")
```

### Try It

```bash
uv run python examples/01_passages/semantic_notes.py
```

---

## Part 2: Refs - Mutable Pointers

**Refs** are named pointers to passages, like git refs. They let you track "current state" while preserving history.

### Key Concepts

- **Name**: Hierarchical path (e.g., `docs/proposal/current`)
- **Points to**: A single passage ID
- **History**: Every change is recorded for auditing

### Use Cases

- Track "current version" of a document
- Create named bookmarks
- Build audit trails
- Implement state machines

### Working with Refs

```python
# Create/update a ref
await client.put(
    "http://localhost:8080/refs/docs/proposal/current",
    json={"passage_id": "abc-123-..."},
)

# Get current value
response = await client.get("http://localhost:8080/refs/docs/proposal/current")
ref = response.json()
print(f"Points to: {ref['passage_id']}")

# View history
response = await client.get(
    "http://localhost:8080/refs/docs/proposal/current/history",
    params={"limit": 10},
)
for entry in response.json()["history"]:
    print(f"{entry['changed_at']}: {entry['passage_id']}")
```

### Try It

```bash
uv run python examples/02_refs/mutable_pointers.py
```

---

## Part 3: Scopes - Versioned Memory

**Scopes** define versioned "working sets" of passages. They're like dynamic views that evolve over time.

### Key Concepts

- **Scope**: A named container with a versioned definition
- **Definition**: Which passages are "in" the scope
- **Version**: Every add/remove creates a new version
- **Scoped Search**: Search only within the scope

### Use Cases

- AI agent's working memory
- Project-specific context windows
- Experiment with knowledge subsets
- Time-travel through memory states

### Working with Scopes

```python
# Create a scope
response = await client.post(
    "http://localhost:8080/scopes",
    json={"name": "project-context", "description": "Current project memory"},
    headers={"X-Agent-ID": "my-agent"},
)

# Add passages to scope
await client.post(
    "http://localhost:8080/scopes/project-context/add",
    json={"passage_ids": ["id-1", "id-2", "id-3"]},
    headers={"X-Agent-ID": "my-agent"},
)

# Search within scope only
response = await client.get(
    "http://localhost:8080/scopes/project-context/search",
    params={"query": "implementation details"},
    headers={"X-Agent-ID": "my-agent"},
)

# View version history
response = await client.get(
    "http://localhost:8080/scopes/project-context/history",
    headers={"X-Agent-ID": "my-agent"},
)

# Revert to previous version
await client.post(
    "http://localhost:8080/scopes/project-context/revert",
    json={"target_version": 2},
    headers={"X-Agent-ID": "my-agent"},
)
```

### Try It

```bash
uv run python examples/03_scopes/versioned_memory.py
```

---

## Part 4: Branches - Safe Experimentation

**Branches** group refs together for atomic operations. They're designed for the "world model" pattern where you have human/persona/world refs that should move together.

### Key Concepts

- **Branch**: Groups 3 refs (human, persona, world) as a unit
- **HEAD**: The main/production branch (hooks enabled)
- **Experiment branches**: Safe sandboxes (hooks disabled)
- **Fork**: Copy refs to a new branch
- **Promote**: Copy refs from source to target

### Use Cases

- A/B test world model configurations
- Safe experimentation before production
- Compare processing strategies

### Working with Branches

```python
# Create main branch
await client.post(
    "http://localhost:8080/branches",
    json={
        "ref_prefix": "my-agent",
        "branch_name": "HEAD",
        "is_main": True,
    },
)

# Fork for experimentation
response = await client.post(
    "http://localhost:8080/branches/my-agent/HEAD/fork",
    json={"new_branch_name": "experiment-1"},
)

# ... make changes to experiment branch ...

# Promote experiment to production
await client.post(
    "http://localhost:8080/branches/my-agent/experiment-1/promote",
    json={"target_branch_name": "HEAD"},
)
```

### Try It

```bash
uv run python examples/04_branches/experimentation.py
```

---

## Part 5: Processing Runs - Transformation Pipelines

**Processing Runs** transform passages using configurable processors (like LLM summarization).

### Key Concepts

- **Input SQL**: Groups passages into processing batches
- **Processor**: Transforms each batch (e.g., `llm_prompt`)
- **Output**: New passages created with provenance links
- **Status**: pending → running → completed/failed

### Use Cases

- Hierarchical summarization (daily → weekly → monthly)
- Entity extraction
- Content classification
- Batch transformations

### Creating Runs

```python
# Create a summarization run
response = await client.post(
    "http://localhost:8080/runs",
    json={
        "input_sql": """
            SELECT array_agg(id) as passage_ids,
                   date_trunc('day', period_start) as group_key,
                   jsonb_build_object('date', date_trunc('day', period_start)) as group_metadata
            FROM passages
            WHERE passage_type = 'journal_entry'
            GROUP BY date_trunc('day', period_start)
        """,
        "processor_type": "llm_prompt",
        "processor_config": {
            "prompt": "Summarize these journal entries:\n\n{passages}",
            "output_type": "daily_summary",
        },
    },
    headers={"X-Agent-ID": "my-agent"},
)

run_id = response.json()["id"]

# Execute via CLI
# uv run kp3 run execute {run_id}
```

### Try It

```bash
uv run python examples/05_processing/summarization_pipeline.py
```

---

## Part 6: Provenance - Tracking Derivations

**Provenance** tracks which passages created which other passages.

### Key Concepts

- **Sources**: Input passages that created an output
- **Derived**: Output passages created from an input
- **Chain**: Full recursive derivation tree

### Use Cases

- Trace summaries back to source documents
- Verify information provenance
- Debug processing pipelines
- Build citation graphs

### Querying Provenance

```python
passage_id = "abc-123-..."

# Get direct sources
response = await client.get(f"http://localhost:8080/passages/{passage_id}/sources")
sources = response.json()["sources"]

# Get directly derived passages
response = await client.get(f"http://localhost:8080/passages/{passage_id}/derived")
derived = response.json()["derived"]

# Get full provenance chain
response = await client.get(
    f"http://localhost:8080/passages/{passage_id}/provenance",
    params={"max_depth": 10},
)
chain = response.json()["chain"]
```

### Try It

```bash
uv run python examples/06_provenance/derivation_chains.py
```

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        KP3 Architecture                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐         │
│  │   Passages  │───▶│    Refs     │───▶│   Scopes    │         │
│  │  (content)  │    │ (pointers)  │    │ (versions)  │         │
│  └─────────────┘    └─────────────┘    └─────────────┘         │
│         │                  │                  │                 │
│         │                  ▼                  │                 │
│         │           ┌─────────────┐           │                 │
│         │           │  Branches   │           │                 │
│         │           │   (groups)  │           │                 │
│         │           └─────────────┘           │                 │
│         │                                     │                 │
│         ▼                                     ▼                 │
│  ┌─────────────┐                      ┌─────────────┐          │
│  │ Processing  │─────────────────────▶│ Provenance  │          │
│  │    Runs     │                      │  (lineage)  │          │
│  └─────────────┘                      └─────────────┘          │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│                     PostgreSQL + pgvector                       │
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **Passages** store all text content with automatic embeddings
2. **Refs** point to passages (like git refs)
3. **Scopes** define versioned collections of passages
4. **Branches** group refs for atomic operations
5. **Processing Runs** transform passages into new passages
6. **Provenance** tracks the derivation relationships

---

## CLI Reference

KP3 includes a CLI for administrative tasks:

```bash
# Service management
uv run kp3-service              # Start the REST API server

# Passage operations
uv run kp3 passage list         # List passages
uv run kp3 passage search "query"  # Search passages

# Ref operations
uv run kp3 ref list             # List refs
uv run kp3 ref get <name>       # Get ref value
uv run kp3 ref set <name> <id>  # Set ref to passage

# Branch operations
uv run kp3 branch list          # List branches
uv run kp3 branch fork <src> <dst>  # Fork branch

# Run operations
uv run kp3 run list             # List processing runs
uv run kp3 run execute <id>     # Execute a run
```

---

## Next Steps

1. **Run all examples**: Each example in `examples/` demonstrates a specific feature
2. **Build an agent**: Use KP3 as memory for an AI agent
3. **Create pipelines**: Set up processing runs for your use case
4. **Explore the API**: Check `/docs` on the running service for OpenAPI docs

## Troubleshooting

### "Could not connect to KP3 service"
- Ensure the service is running: `pgrep -f kp3-service`
- Check logs: `tail /tmp/kp3-service.log`

### "Embedding generation failed"
- Verify your OpenAI API key in `.env`
- Check your API quota at platform.openai.com

### "Duplicate content" errors
- KP3 deduplicates by content hash
- The examples handle this gracefully
- Use search to find existing passages

### Database connection issues
- Ensure PostgreSQL is running: `podman ps`
- Check `KP3_DATABASE_URL` in `.env`
