# KP3 Examples

This directory contains example code for extending and customizing KP3.

## Importers

The `importers/` directory contains example code for importing data from various sources into KP3.

### SQLite Importer

`importers/sqlite_importer.py` demonstrates how to import data from a SQLite database. It expects a schema with:

- `memory_shards` table with columns: `id`, `uid`, `contents`, `created_at`, `agent_id`
- `agents` table with columns: `id`, `name`

You can use this as a template for creating your own importers.

#### Usage

```python
import asyncio
from pathlib import Path
from kp3.db.engine import async_session
from examples.importers.sqlite_importer import import_memory_shards

async def main():
    async with async_session() as session:
        async with session.begin():
            stats = await import_memory_shards(session, Path("backup.db"))
            print(f"Imported: {stats.imported}")
            print(f"Duplicates skipped: {stats.skipped_duplicate}")
            print(f"Empty skipped: {stats.skipped_empty}")

asyncio.run(main())
```

## Creating Your Own Importer

To create a custom importer:

1. Create a new file in `importers/` (e.g., `my_importer.py`)
2. Import the passage service: `from kp3.services.passages import create_passage`
3. Implement your data loading logic
4. Use `create_passage()` to insert passages with appropriate metadata
5. Use `get_passage_by_external_id()` for deduplication

Key fields when creating passages:
- `content`: The text content
- `passage_type`: Category of the passage (e.g., "memory_shard", "document")
- `source_system`: Identifier for the import source (for deduplication)
- `source_external_id`: Unique ID from the source system
- `metadata`: Additional structured data about the passage
- `period_start`/`period_end`: Time range the passage covers
