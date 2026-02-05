# Example 07: Hierarchical Memory System

A complete implementation of a **hierarchical summarization memory** for AI agents. This is the pattern you'd use to give an AI assistant long-term memory that stays relevant and searchable over time.

## The Problem

AI agents accumulate conversations rapidly. After weeks of use, you have thousands of conversation fragments. Naive approaches fail:

- **Vector search alone**: Returns random snippets without context
- **Full conversation logs**: Too long for context windows
- **Simple summarization**: Loses important details

## The Solution: Hierarchical Memory

Organize memory into layers that progressively compress information while preserving what matters:

```
┌─────────────────────────────────────────────────────────────┐
│  Layer 4: World Model (always in context)                   │
│  - Human understanding (who they are, preferences)          │
│  - Persona (agent's evolving character)                     │
│  - Active projects, entities, themes                        │
├─────────────────────────────────────────────────────────────┤
│  Layer 3: Monthly summaries (~12/year)                      │
│  - High-level patterns and milestones                       │
│  - Searchable for "what happened in October"                │
├─────────────────────────────────────────────────────────────┤
│  Layer 2: Weekly summaries (~52/year)                       │
│  - Key events and decisions                                 │
│  - Good for "what did we work on last week"                 │
├─────────────────────────────────────────────────────────────┤
│  Layer 1: Daily summaries (~365/year)                       │
│  - Condensed daily context                                  │
│  - Preserves temporal ordering                              │
├─────────────────────────────────────────────────────────────┤
│  Layer 0: Raw conversations (thousands)                     │
│  - Full detail, searchable by content                       │
│  - Archived but accessible when needed                      │
└─────────────────────────────────────────────────────────────┘
```

## How It Works

### 1. Ingest Conversations

Every conversation becomes a passage with temporal metadata:

```python
passage = await client.post("/passages", json={
    "content": "User asked about Python decorators. Explained @property...",
    "passage_type": "conversation",
    "period_start": "2025-02-04T10:00:00Z",
    "period_end": "2025-02-04T10:30:00Z",
})
```

### 2. Scheduled Summarization

Cron jobs run processing pipelines that roll up content:

```bash
# Daily at midnight: summarize yesterday's conversations
kp3 run fold "
  SELECT id FROM passages
  WHERE passage_type = 'conversation'
    AND period_start >= NOW() - INTERVAL '1 day'
  ORDER BY period_start
" -p llm_prompt -c '{"prompt_name": "daily_summary"}'

# Weekly on Sunday: summarize last week's daily summaries
kp3 run fold "
  SELECT id FROM passages
  WHERE passage_type = 'daily_summary'
    AND period_start >= NOW() - INTERVAL '7 days'
  ORDER BY period_start
" -p llm_prompt -c '{"prompt_name": "weekly_summary"}'
```

### 3. World Model Extraction

Hooks automatically update the world model when refs change:

```python
# When human_ref updates, extract new understanding
hook = await create_ref_hook(
    session,
    ref_name="agent/human/HEAD",
    action_type="world_model_update",
    config={"extract_to": ["human", "projects", "themes"]}
)
```

### 4. Scoped Search

At query time, search the appropriate layer:

```python
# For "what's their favorite color?" - check world model first
world_model = await client.get("/scopes/world-model/search", params={
    "query": "favorite color preferences"
})

# For "what did we discuss last Tuesday?" - search daily summaries
results = await client.get("/passages/search", params={
    "query": "discussion Tuesday",
    "passage_type": "daily_summary",
    "period_start_after": "2025-01-28",
    "period_end_before": "2025-01-29",
})

# For deep detail - search raw conversations
detail = await client.get("/passages/search", params={
    "query": "Python decorators @property",
    "passage_type": "conversation",
})
```

## Files in This Example

| File | Purpose |
|------|---------|
| `hierarchical_memory.py` | Complete working implementation |
| `README.md` | This documentation |

## Running the Example

```bash
# Ensure service is running
docker compose up -d

# Run the example
uv run python examples/07_hierarchical_memory/hierarchical_memory.py
```

## Key Patterns Demonstrated

### Pattern 1: Temporal Bucketing

Group passages by time period for summarization:

```sql
SELECT
    array_agg(id ORDER BY period_start) as passage_ids,
    date_trunc('day', period_start)::date as group_key
FROM passages
WHERE passage_type = 'conversation'
GROUP BY date_trunc('day', period_start)
```

### Pattern 2: Provenance-Aware Search

When you find a summary, trace back to sources:

```python
# Get derivations for a passage
derivations = await client.get(f"/passages/{summary_id}/derivations")
# Returns the original conversations that created this summary
```

### Pattern 3: Ref-Based State Management

Use refs to track "current" state at each level:

```
agent/daily/HEAD      -> Latest daily summary
agent/weekly/HEAD     -> Latest weekly summary
agent/human/HEAD      -> Current human understanding
agent/persona/HEAD    -> Current agent persona
```

### Pattern 4: Hooks for Cascading Updates

When a ref updates, trigger downstream processing:

```python
# When daily summary updates, trigger scope refresh
await create_ref_hook(session, "agent/daily/HEAD", "refresh_scope", {
    "scope_name": "recent-context"
})
```

## Architecture Diagram

```
Conversations (real-time)
        │
        ▼
┌───────────────┐     ┌─────────────────┐
│  Raw Passages │────▶│  Daily Summary  │
│  (Layer 0)    │     │  (Layer 1)      │
└───────────────┘     └────────┬────────┘
                               │
                               ▼
                      ┌─────────────────┐
                      │ Weekly Summary  │
                      │ (Layer 2)       │
                      └────────┬────────┘
                               │
                               ▼
                      ┌─────────────────┐
                      │ Monthly Summary │
                      │ (Layer 3)       │
                      └────────┬────────┘
                               │
                               ▼
                      ┌─────────────────┐
                      │   World Model   │◀── Hooks update on each layer
                      │   (Layer 4)     │
                      └─────────────────┘
                               │
                               ▼
                      ┌─────────────────┐
                      │  Agent Context  │  (What goes in the prompt)
                      │  Window         │
                      └─────────────────┘
```

## Production Considerations

1. **Scheduling**: Use cron/systemd timers for summarization jobs
2. **Idempotency**: Processing runs track state; safe to re-run
3. **Monitoring**: Check `/runs` endpoint for failed/stuck runs
4. **Pruning**: Archive old raw conversations after summarization
5. **Cost**: Use smaller models (gpt-4o-mini) for routine summarization

## Related Examples

- [05_processing](../05_processing/) - Processing runs basics
- [06_provenance](../06_provenance/) - Tracking derivation chains
- [03_scopes](../03_scopes/) - Memory scopes for context isolation
