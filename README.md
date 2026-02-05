# KP3 - Knowledge Processing Pipeline

[![PyPI version](https://img.shields.io/pypi/v/kp3.svg)](https://pypi.org/project/kp3/)
[![CI](https://github.com/marklubin/kp3/actions/workflows/ci.yml/badge.svg)](https://github.com/marklubin/kp3/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)

## The Problem

LLMs are stateless. Every conversation starts from zero. Your AI assistant doesn't remember that you prefer concise answers, that you're working on a Python project called "Atlas", or that you mentioned your dog's name last week.

Context windows help, but they're limited and expensive. RAG systems retrieve documents, but they don't *understand* users—they just search.

**AI agents need memory that evolves.**

## The Solution

KP3 gives AI agents persistent, searchable memory with automatic world model extraction:

- **Store conversations** as semantically searchable passages
- **Build understanding** by extracting what the agent learns about users, projects, and context over time
- **Retrieve intelligently** using hybrid search that combines keywords, meaning, and recency
- **Track provenance** so you always know where knowledge came from

Think of it as giving your AI agent a brain that remembers and learns.

## Key Concepts

| Concept | What it does |
|---------|--------------|
| **Passages** | Text content with embeddings, metadata, and provenance tracking |
| **Hybrid Search** | Combines full-text search + semantic similarity + recency scoring |
| **World Model** | LLM-extracted understanding of users, personas, and context that evolves over time |
| **Refs** | Git-like mutable pointers to passages with version history |
| **Scopes** | Dynamic memory partitions for context isolation |
| **Branches** | Experiment with different world model states without affecting production |

## Quick Start

### Install

```bash
# With uv (recommended)
uv add kp3

# Or with pip
pip install kp3
```

### Start PostgreSQL with pgvector

```bash
docker run -d --name kp3-postgres \
  -e POSTGRES_USER=kp3 \
  -e POSTGRES_PASSWORD=kp3 \
  -e POSTGRES_DB=kp3 \
  -p 5432:5432 \
  pgvector/pgvector:pg16
```

### Configure

```bash
export KP3_DATABASE_URL=postgresql+asyncpg://kp3:kp3@localhost:5432/kp3
export KP3_OPENAI_API_KEY=sk-your-key  # For embeddings
```

### Run

```bash
# Run migrations
kp3-migrate  # or: uv run alembic upgrade head

# Start the service
kp3-service

# Store a passage
curl -X POST http://localhost:8080/passages \
  -H "Content-Type: application/json" \
  -H "X-Agent-ID: my-agent" \
  -d '{"content": "User prefers concise responses", "passage_type": "memory"}'

# Search
curl "http://localhost:8080/passages/search?query=user+preferences" \
  -H "X-Agent-ID: my-agent"
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Your AI Agent                          │
├─────────────────────────────────────────────────────────────┤
│  KP3 Service (REST API + MCP)                               │
│  - Store passages    - Search memory    - Extract world model│
├─────────────────────────────────────────────────────────────┤
│  PostgreSQL + pgvector                                      │
│  - Full-text search  - Vector similarity  - ACID guarantees │
└─────────────────────────────────────────────────────────────┘
```

## When to Use KP3

**Good fit:**
- Personal AI assistants that should remember user preferences
- Customer support agents that need conversation history
- Research agents building domain knowledge over time
- Any system where AI should learn from interactions

**Not the right fit:**
- Simple document retrieval (use a vector DB directly)
- Static knowledge bases that don't evolve
- Systems that don't need provenance or versioning

## Documentation

- **[Tutorial](docs/tutorial.md)** - Comprehensive walkthrough from zero to production
- **[Examples](examples/)** - Hands-on demos for each feature
- **[CLI Reference](docs/cli-reference.md)** - All CLI commands
- **[API Reference](docs/api-reference.md)** - REST API endpoints

## Configuration

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `KP3_DATABASE_URL` | Yes | - | PostgreSQL connection URL |
| `KP3_OPENAI_API_KEY` | Yes | - | OpenAI API key for embeddings |
| `KP3_OPENAI_EMBEDDING_MODEL` | No | `text-embedding-3-large` | Embedding model |
| `DEEPSEEK_API_KEY` | No | - | For world model extraction |

See [.env.example](.env.example) for all options.

## Development

```bash
git clone https://github.com/marklubin/kp3.git
cd kp3
uv sync
uv run pytest
uv run ruff check
uv run pyright
```

## License

MIT License - see [LICENSE](LICENSE) for details.
