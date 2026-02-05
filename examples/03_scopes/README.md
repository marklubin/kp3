# A3: Scopes - Versioned Memory Containers

This example demonstrates KP3's Memory Scopes API for creating versioned search closures that evolve over time.

## What You'll Learn

- **Create scopes** as memory containers
- **Add/remove passages** to modify scope contents
- **Scoped search** - only search within the scope
- **Version history** - see how scope evolved
- **Revert** - restore to any previous state

## Running the Example

```bash
# Start the KP3 stack
docker compose up -d

# Run the example
docker compose exec kp3-service uv run python examples/03_scopes/versioned_memory.py
```

## Key Concepts

### What is a Scope?

A scope defines a "working set" of passages for search. Think of it as:
- A dynamic folder that can grow/shrink
- A versioned bookmark collection
- An AI agent's working memory

```
scope: agent-memory
├── passage-1 (user preferences)
├── passage-2 (current project)
└── passage-3 (recent context)
```

### Versioning

Every scope operation creates a new version:

```
v1: Created scope (empty)
v2: Added passages [1, 2, 3]
v3: Added passage [4]
v4: Removed passage [1]
v5: Reverted to v3
```

### Scoped Search

Normal search returns all passages. Scoped search only returns passages in the scope:

```python
# Global search - returns everything matching
results = await search_passages(client, "programming")

# Scoped search - only returns matches within scope
results = await search_in_scope(client, "agent-memory", "programming")
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/scopes` | Create a new scope |
| GET | `/scopes` | List all scopes |
| GET | `/scopes/{name}` | Get scope details |
| DELETE | `/scopes/{name}` | Delete a scope |
| POST | `/scopes/{name}/passages` | Create passage in scope (atomic) |
| POST | `/scopes/{name}/add` | Add passages to scope |
| POST | `/scopes/{name}/remove` | Remove passages from scope |
| GET | `/scopes/{name}/search` | Search within scope |
| GET | `/scopes/{name}/history` | Get version history |
| POST | `/scopes/{name}/revert` | Revert to previous version |

## Common Patterns

### AI Agent Working Memory

```python
# Create memory scope for agent
scope = await create_scope(client, "agent-123/memory")

# Add relevant context during conversation
await add_passages_to_scope(client, "agent-123/memory", [context_passage_id])

# Search only in agent's memory
results = await search_in_scope(client, "agent-123/memory", query)

# Clean up stale context
await remove_from_scope(client, "agent-123/memory", [old_passage_ids])
```

### Project Context Window

```python
# Create scope for project
scope = await create_scope(client, "project-xyz/context")

# Add project-related knowledge
await add_passages_to_scope(client, "project-xyz/context", [
    requirements_passage_id,
    design_doc_passage_id,
    code_doc_passage_id,
])

# Search only project context
results = await search_in_scope(client, "project-xyz/context", "authentication")
```

### Experiment with Knowledge Subsets

```python
# Create experiment scope
scope = await create_scope(client, "experiment/subset-a")

# Add curated passages
await add_passages_to_scope(client, "experiment/subset-a", curated_ids)

# Test retrieval quality
results = await search_in_scope(client, "experiment/subset-a", test_query)

# Didn't work? Revert and try different subset
await revert_scope(client, "experiment/subset-a", to_version=1)
await add_passages_to_scope(client, "experiment/subset-a", different_ids)
```

## Scope Contents

Scopes can contain:

1. **Literal passage IDs** - Directly added passages
2. **Refs** - Named pointers (resolved at search time)

```python
# Add both passages and refs
await add_passages_to_scope(client, scope_name, passage_ids)
await add_refs_to_scope(client, scope_name, ["world/human/HEAD"])
```

## Next Steps

- Combine scopes with refs for dynamic content
- Use branches (A4) for isolated experimentation
- Explore how processing runs can populate scopes
