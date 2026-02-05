# CLAUDE.md - KP3 Project Guide

## Project Overview

KP3 (Knowledge Processing Pipeline) is a text processing system with semantic search, world model extraction, and provenance tracking. Built with PostgreSQL + pgvector.

## Development Commands

```bash
# Install dependencies
uv sync

# Run tests
uv run pytest

# Lint and format
uv run ruff check --fix
uv run ruff format

# Type check
uv run pyright

# Run migrations
uv run alembic upgrade head

# Start service
uv run kp3-service

# CLI help
uv run kp3 --help
```

## Docker Commands

```bash
# Start full stack
docker compose up -d

# Run migrations in container
docker compose exec kp3-service uv run alembic upgrade head

# Run CLI commands
docker compose exec kp3-service uv run kp3 <command>
```

## Code Style

- Python 3.12+, fully typed (pyright strict mode)
- Ruff for linting and formatting (line length: 100)
- Async/await for all database operations
- Pydantic for API schemas and validation

## Architecture

```
src/kp3/
├── cli.py              # Click CLI commands
├── config.py           # Pydantic settings
├── db/                 # SQLAlchemy models, engine
├── processors/         # Embedding, LLM, world model
├── services/           # Business logic (passages, refs, search)
├── query_service/      # FastAPI REST API + MCP
├── schemas/            # API request/response models
└── llm/                # OpenAI-compatible client
```

## Key Concepts

- **Passages**: Text content with embeddings, metadata, provenance
- **Refs**: Mutable pointers to passages (like git refs)
- **Branches**: Groups of refs (human/persona/world) as a unit
- **Processors**: Transform passages (embedding, world model extraction)
- **Shadow Tables**: Denormalized entity storage for fast queries
