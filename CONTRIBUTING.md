# Contributing to KP3

Thank you for your interest in contributing to KP3! This document provides guidelines for contributing to the project.

## Development Setup

### Prerequisites

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) - Fast Python package installer
- Docker (for running PostgreSQL with pgvector)

### Local Development

1. Clone the repository:
```bash
git clone https://github.com/marklubin/kp3.git
cd kp3
```

2. Install dependencies:
```bash
uv sync
```

3. Start PostgreSQL with pgvector:
```bash
docker run -d --name kp3-postgres \
    -e POSTGRES_USER=kp3 \
    -e POSTGRES_PASSWORD=kp3 \
    -e POSTGRES_DB=kp3 \
    -p 5432:5432 \
    pgvector/pgvector:pg16
```

4. Create your `.env` file:
```bash
cat > .env << EOF
KP3_DATABASE_URL=postgresql+asyncpg://kp3:kp3@localhost:5432/kp3
KP3_OPENAI_API_KEY=sk-your-openai-key
EOF
```

5. Run database migrations:
```bash
uv run alembic upgrade head
```

6. Seed default prompts:
```bash
uv run kp3 world-model seed-prompts
```

## Code Style

KP3 uses strict code quality standards:

### Formatting and Linting

We use [Ruff](https://docs.astral.sh/ruff/) for both linting and formatting:

```bash
# Format code
uv run ruff format

# Lint and auto-fix
uv run ruff check --fix

# Check without fixing
uv run ruff check
```

Configuration is in `pyproject.toml`. Line length is 100 characters.

### Type Checking

We use [Pyright](https://microsoft.github.io/pyright/) in strict mode:

```bash
uv run pyright
```

All code must be fully typed. No `Any` types unless absolutely necessary.

### Code Style Guidelines

- Use async/await for all database operations
- Use Pydantic for API schemas and validation
- Follow existing patterns in the codebase
- Keep functions focused and small
- Add docstrings to public functions and classes

## Testing

### Running Tests

```bash
# Run all tests
uv run pytest

# Run with verbose output
uv run pytest -v

# Run specific test file
uv run pytest tests/test_passages.py -v

# Run with coverage
uv run pytest --cov=kp3
```

### Writing Tests

- Use pytest fixtures (see `tests/conftest.py`)
- Use `@pytest.mark.asyncio` for async tests
- Mock external services (LLMs, embedding APIs)
- Test both success and error cases

## Pull Request Process

1. **Fork and clone** the repository
2. **Create a branch** for your feature or fix:
   ```bash
   git checkout -b feature/my-feature
   ```
3. **Make your changes** following the code style guidelines
4. **Add tests** for new functionality
5. **Run the test suite** and ensure all tests pass:
   ```bash
   uv run pytest
   uv run ruff check
   uv run pyright
   ```
6. **Commit your changes** with a clear commit message
7. **Push to your fork** and open a Pull Request

### Commit Message Guidelines

- Use the imperative mood ("Add feature" not "Added feature")
- Keep the first line under 72 characters
- Reference issues when relevant ("Fix #123")

### PR Description

Include in your PR description:
- What the change does
- Why the change is needed
- Any breaking changes
- Testing done

## Database Migrations

When modifying database models:

1. Make changes to `src/kp3/db/models.py`
2. Generate a migration:
   ```bash
   uv run alembic revision --autogenerate -m "Description of change"
   ```
3. Review the generated migration in `alembic/versions/`
4. Test the migration:
   ```bash
   uv run alembic upgrade head
   uv run alembic downgrade -1
   uv run alembic upgrade head
   ```

## Project Structure

```
kp3/
├── src/kp3/           # Main package
│   ├── cli.py         # CLI commands
│   ├── config.py      # Configuration settings
│   ├── db/            # Database models and engine
│   ├── processors/    # Data processors (embedding, LLM, world model)
│   ├── query_service/ # REST API and MCP server
│   ├── schemas/       # Pydantic schemas
│   ├── services/      # Business logic
│   └── llm/           # LLM client utilities
├── tests/             # Test suite
├── alembic/           # Database migrations
├── examples/          # Example code
└── docs/              # Documentation
```

## Questions?

If you have questions about contributing, feel free to open an issue for discussion.
