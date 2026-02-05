# A1: Passages - Semantic Note Search

This example demonstrates KP3's core Passages API for building a personal note-taking app where you can dump thoughts and find them later using natural language search.

## What You'll Learn

- **Create passages** (notes) with automatic embedding generation
- **Hybrid search** combining full-text search (FTS) with semantic similarity
- **Tag-based organization** for flexible categorization
- **Agent isolation** for multi-user scenarios

## Running the Example

```bash
# Start the KP3 stack
docker compose up -d

# Run the example
docker compose exec kp3-service uv run python examples/01_passages/semantic_notes.py
```

## Key Concepts

### Passages

Passages are the fundamental unit in KP3. Every passage:
- Contains text content
- Is automatically embedded for semantic search
- Has a `passage_type` for categorization
- Belongs to an agent (via `X-Agent-ID` header)
- Can have optional metadata and time periods

### Search Modes

| Mode | Description | Best For |
|------|-------------|----------|
| `fts` | Full-text search using PostgreSQL tsvector | Exact word matching |
| `semantic` | Vector similarity using embeddings | Conceptually related content |
| `hybrid` | Combines FTS + semantic with RRF fusion | General use (default) |

### Tags

Tags provide flexible categorization:
- Create tags with names and optional descriptions
- Tags have their own embeddings for semantic matching
- Search by tags to find related passages
- Use `search_type=tags` to search tag names instead of content

## API Endpoints Used

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/passages` | Create a new passage |
| GET | `/passages/{id}` | Retrieve a passage by ID |
| GET | `/passages/search` | Search passages |
| POST | `/tags` | Create a new tag |
| POST | `/passages/{id}/tags` | Attach tags to a passage |

## Example Output

```
Step 2: Semantic Search
Query: 'machine learning and AI concepts'
  - [learning] (score: 0.812) Learned about vector embeddings today...

Query: 'problems with caching'
  - [incident] (score: 0.756) Bug in production: users are seeing stale data...
```

## Next Steps

- Try modifying the search queries to see how semantic search finds related content
- Add more notes with different topics
- Experiment with different search modes to understand their strengths
