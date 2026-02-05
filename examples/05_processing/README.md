# B1: Processing Runs - Hierarchical Summarization

This example demonstrates KP3's Processing Runs API for executing transformation pipelines that take passages as input and produce new passages as output.

## What You'll Learn

- **Create processing runs** with SQL grouping and processor config
- **Monitor run status** and progress
- **Understand passage grouping** via SQL
- **Track provenance** (which inputs created which outputs)

## Running the Example

```bash
# Start the KP3 stack
docker compose up -d

# Run the example
docker compose exec kp3-service uv run python examples/05_processing/summarization_pipeline.py
```

## Key Concepts

### What is a Processing Run?

A processing run transforms passages through a pipeline:

```
┌─────────────────┐     ┌───────────────┐     ┌─────────────────┐
│   Input SQL     │────>│   Processor   │────>│  Output         │
│   (grouping)    │     │   (transform) │     │  (new passages) │
└─────────────────┘     └───────────────┘     └─────────────────┘
```

### Input SQL

The input SQL groups passages into batches for processing:

```sql
SELECT
    array_agg(id ORDER BY period_start) as passage_ids,
    'week-' || date_trunc('week', period_start)::date as group_key,
    jsonb_build_object('week_start', ...) as group_metadata
FROM passages
WHERE passage_type = 'journal_entry'
GROUP BY date_trunc('week', period_start)
```

Required columns:
- `passage_ids` - Array of UUIDs to process together
- `group_key` - Identifier for the group
- `group_metadata` - Optional JSONB with context

### Processor Types

| Type | Description |
|------|-------------|
| `llm_prompt` | Call LLM with configured prompt |
| `embedding` | Generate embeddings |
| `world_model` | Extract world model entities |

### Run Status

```
pending → running → completed
                └── failed
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/runs` | Create a new run |
| GET | `/runs` | List runs (with status filter) |
| GET | `/runs/{id}` | Get run details |

## Example: Hierarchical Summarization

### Daily → Weekly → Monthly

```python
# 1. Summarize daily entries into weekly summaries
weekly_run = await create_run(
    client,
    input_sql="""
        SELECT array_agg(id) as passage_ids,
               date_trunc('week', period_start) as group_key,
               jsonb_build_object() as group_metadata
        FROM passages WHERE passage_type = 'daily_entry'
        GROUP BY date_trunc('week', period_start)
    """,
    processor_type="llm_prompt",
    processor_config={
        "prompt_name": "summarize_weekly",
        "output_passage_type": "weekly_summary",
    },
)

# 2. Summarize weekly summaries into monthly
monthly_run = await create_run(
    client,
    input_sql="""
        SELECT array_agg(id) as passage_ids,
               date_trunc('month', period_start) as group_key,
               jsonb_build_object() as group_metadata
        FROM passages WHERE passage_type = 'weekly_summary'
        GROUP BY date_trunc('month', period_start)
    """,
    processor_type="llm_prompt",
    processor_config={
        "prompt_name": "summarize_monthly",
        "output_passage_type": "monthly_summary",
    },
)
```

### Execute Runs

```bash
# Execute via CLI
uv run kp3 run execute <run_id>

# List all runs
uv run kp3 run list

# Check run status
uv run kp3 run show <run_id>
```

## Processor Configuration

### LLM Prompt Processor

```python
processor_config = {
    "prompt_name": "summarize",      # Name of configured prompt
    "output_passage_type": "summary", # Type for output passages
    "model": "gpt-4o-mini",           # LLM model to use
}
```

### Required Prompts

Configure prompts in the database:

```sql
INSERT INTO extraction_prompts (name, version, is_active, system_prompt, user_prompt_template, field_descriptions)
VALUES (
    'summarize', 1, true,
    'You are a summarization assistant.',
    'Summarize the following passages:\n\n{{passages}}',
    '{}'::jsonb
);
```

## Provenance Tracking

After execution, provenance records link inputs to outputs:

```
daily_entry_1 ─┐
daily_entry_2 ─┼──> weekly_summary_1
daily_entry_3 ─┘

weekly_summary_1 ─┐
weekly_summary_2 ─┼──> monthly_summary_1
weekly_summary_3 ─┘
weekly_summary_4 ─┘
```

See B2 (Provenance) for querying these relationships.

## Next Steps

- Learn how to query derivation chains with Provenance (B2)
- Use branches (A4) to experiment with different processing configs
- Combine with scopes (A3) to build curated knowledge bases
