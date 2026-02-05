# KP3 Examples

Hands-on demos that progressively showcase KP3's capabilities.

## Quick Start

```bash
# 1. Copy environment template and add your OpenAI API key
cp .env.example .env
# Edit .env: KP3_OPENAI_API_KEY=sk-your-key

# 2. Start PostgreSQL with pgvector
podman run -d --name kp3-postgres \
  -e POSTGRES_USER=kp3 \
  -e POSTGRES_PASSWORD=kp3 \
  -e POSTGRES_DB=kp3 \
  -p 5432:5432 \
  pgvector/pgvector:pg16

# 3. Install dependencies and run migrations
uv sync
uv run alembic upgrade head

# 4. Start the service
uv run kp3-service

# 5. Run any example (in another terminal)
uv run python examples/01_passages/semantic_notes.py
```

## Examples

| # | Example | What You'll Learn |
|---|---------|-------------------|
| 01 | [Passages](01_passages/) | Store text, hybrid search, tags |
| 02 | [Refs](02_refs/) | Mutable pointers with history |
| 03 | [Scopes](03_scopes/) | Versioned memory containers |
| 04 | [Branches](04_branches/) | Fork/promote for experimentation |
| 05 | [Processing](05_processing/) | Transformation pipelines |
| 06 | [Provenance](06_provenance/) | Track what created what |

## Architecture

### Track A: Core Data Primitives

```
Passages → Refs → Scopes → Branches
   │         │       │         │
   │         │       │         └── Groups of refs for experimentation
   │         │       └── Versioned search closures
   │         └── Mutable pointers to passages
   └── Fundamental unit of content
```

### Track B: Processing Layer

```
Processing Runs → Provenance
       │              │
       │              └── Track which passages created which
       └── Transform passages (summarize, extract, etc.)
```

## Running Examples

All examples use `python-dotenv` to load environment from `.env`:

```bash
# Just run directly - no shell tricks needed
uv run python examples/01_passages/semantic_notes.py
```

## Troubleshooting

### "Could not connect to KP3 service"
- Ensure the service is running: `pgrep -f kp3-service`
- Check it's healthy: `curl http://localhost:8080/health`

### "Embedding generation failed"
- Verify `KP3_OPENAI_API_KEY` is set in `.env`
- Check your OpenAI API quota

### "Duplicate content" errors
- Examples handle this gracefully (will find existing passages)
- This is expected on re-runs

## Using Docker Compose

If you prefer Docker Compose:

```bash
# Set your API key in .env first
docker compose up -d
docker compose exec kp3-service uv run alembic upgrade head
docker compose exec kp3-service uv run python examples/01_passages/semantic_notes.py
```

## Learn More

See the full [Tutorial](../docs/tutorial.md) for detailed explanations.
