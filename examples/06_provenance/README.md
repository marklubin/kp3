# B2: Provenance - Derivation Chain Tracking

This example demonstrates KP3's Provenance API for tracking which passages derived from which other passages.

## What You'll Learn

- **Get sources** - Find inputs that created a passage
- **Get derived** - Find outputs created from a passage
- **Full provenance chain** - Recursive derivation tracking

## Running the Example

```bash
# Start the KP3 stack
docker compose up -d

# Run the example
docker compose exec kp3-service uv run python examples/06_provenance/derivation_chains.py
```

## Key Concepts

### What is Provenance?

Provenance tracks how passages relate through processing:

```
source_passage_1 ─┐
source_passage_2 ─┼──> derived_passage
source_passage_3 ─┘
```

When a processing run creates output passages, KP3 records:
- Which input passages (sources) were used
- Which output passages (derived) were created
- Which processing run performed the transformation

### Derivation Relationships

```
Passage A (source)      Passage B (source)
     │                       │
     └───────────┬───────────┘
                 │
                 ▼
         Processing Run
                 │
                 ▼
         Passage C (derived)
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/passages/{id}/sources` | Get direct source passages |
| GET | `/passages/{id}/derived` | Get directly derived passages |
| GET | `/passages/{id}/provenance` | Get full chain (recursive) |

## Example: Tracing a Summary

### Get Sources (Depth 1)

```python
# "What passages created this summary?"
sources = await get_passage_sources(client, summary_id)

# Returns:
# {
#   "passage_id": "summary-uuid",
#   "sources": [
#     {"id": "src1", "content": "...", "passage_type": "journal_entry"},
#     {"id": "src2", "content": "...", "passage_type": "journal_entry"},
#   ],
#   "count": 2
# }
```

### Get Derived (Depth 1)

```python
# "What was created from this journal entry?"
derived = await get_passage_derived(client, entry_id)

# Returns:
# {
#   "passage_id": "entry-uuid",
#   "derived": [
#     {"id": "sum1", "content": "...", "passage_type": "weekly_summary"},
#   ],
#   "count": 1
# }
```

### Get Full Chain

```python
# "Show me the complete derivation tree"
chain = await get_passage_provenance(client, final_summary_id, max_depth=10)

# Returns entries at each depth level:
# depth=1: weekly_summary_1, weekly_summary_2
# depth=2: daily_entry_1, daily_entry_2, daily_entry_3, ...
```

## Use Cases

### Verify Information Sources

```python
# User asks: "Where did this fact come from?"
summary = await get_passage(client, summary_id)
sources = await get_passage_sources(client, summary_id)

print("This summary was derived from:")
for src in sources["sources"]:
    print(f"  - {src['passage_type']}: {src['content'][:100]}...")
```

### Build Citation Graphs

```python
# Track information flow through system
provenance = await get_passage_provenance(client, doc_id, max_depth=5)

# Group by depth
by_depth = {}
for entry in provenance["chain"]:
    depth = entry["depth"]
    if depth not in by_depth:
        by_depth[depth] = []
    by_depth[depth].append(entry["source_passage_id"])

print("Citation tree:")
for depth, ids in sorted(by_depth.items()):
    print(f"  Level {depth}: {len(ids)} passages")
```

### Debug Processing Pipelines

```python
# Check if a specific input was used
sources = await get_passage_sources(client, output_id)
input_ids = [s["id"] for s in sources["sources"]]

if expected_input_id in input_ids:
    print("Expected input was used")
else:
    print("Expected input NOT found in sources")
```

### Find All Outputs from a Source

```python
# "What summaries used this document?"
derived = await get_passage_derived(client, document_id)

summaries = [d for d in derived["derived"] if d["passage_type"] == "summary"]
print(f"Document used in {len(summaries)} summaries")
```

## Provenance Chain Structure

The `/provenance` endpoint returns a flat list with depth information:

```json
{
  "passage_id": "final-summary-uuid",
  "chain": [
    {
      "derived_passage_id": "final-summary-uuid",
      "source_passage_id": "weekly-summary-1",
      "processing_run_id": "run-456",
      "depth": 1
    },
    {
      "derived_passage_id": "weekly-summary-1",
      "source_passage_id": "daily-entry-1",
      "processing_run_id": "run-123",
      "depth": 2
    },
    // ... more entries
  ],
  "count": 10
}
```

## Next Steps

- Combine with processing runs (B1) to create derivation chains
- Use refs (A2) to track "current" versions in chains
- Build scopes (A3) from provenance-traced passages
