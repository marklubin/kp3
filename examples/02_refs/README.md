# A2: Refs - Mutable Pointers

This example demonstrates KP3's Refs API for tracking "current state" with mutable pointers.

## What You'll Learn

- **Create/update refs** - Named pointers to passages
- **View ref history** - Full audit trail of changes
- **Use prefixes** - Organize refs hierarchically
- **Bookmark patterns** - Track current versions, states, etc.

## Running the Example

```bash
# Start the KP3 stack
docker compose up -d

# Run the example
docker compose exec kp3-service uv run python examples/02_refs/mutable_pointers.py
```

## Key Concepts

### What is a Ref?

A ref is a named, mutable pointer to a passage. Think of it like:
- A git ref (branch/tag) pointing to a commit
- A symbolic link pointing to a file
- A bookmark that can be updated

```
docs/proposal/current  -->  passage-uuid-123
                              (v3: Final version)
```

### Ref Naming Conventions

Use slashes to create hierarchical namespaces:

```
docs/proposal/current     # Current version of proposal
docs/proposal/draft       # Working draft
state/auth/session        # Auth session state
world/human/HEAD          # World model refs (used by branches)
```

### History Tracking

Every ref change is recorded:

```python
history = await get_ref_history(client, "docs/proposal/current")
# [
#   {passage_id: "abc", previous_passage_id: "xyz", changed_at: "..."},
#   {passage_id: "xyz", previous_passage_id: None,  changed_at: "..."},
# ]
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/refs` | List refs (with optional prefix filter) |
| GET | `/refs/{name}` | Get a specific ref |
| PUT | `/refs/{name}` | Set/update a ref |
| DELETE | `/refs/{name}` | Delete a ref |
| GET | `/refs/{name}/history` | Get change history |

## Common Patterns

### State Machine

```python
# Track state transitions
await set_ref(client, "order/123/state", pending_passage_id)
# ... later ...
await set_ref(client, "order/123/state", processing_passage_id)
# ... later ...
await set_ref(client, "order/123/state", completed_passage_id)
```

### Version Tracking

```python
# Track document versions
await set_ref(client, "doc/readme/v1", v1_passage_id)
await set_ref(client, "doc/readme/v2", v2_passage_id)
await set_ref(client, "doc/readme/latest", v2_passage_id)
```

### Named Bookmarks

```python
# Create meaningful bookmarks
await set_ref(client, "saved/important-insight", passage_id)
await set_ref(client, "saved/daily-summary/2024-01-15", passage_id)
```

## Next Steps

- Explore how refs integrate with branches (A4)
- See how refs work with scopes (A3) for versioned search
- Use refs in processing pipelines (B1)
