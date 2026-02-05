# A4: Branches - Experimentation Workflow

This example demonstrates KP3's Branches API for safe experimentation with world models.

## What You'll Learn

- **Create branches** that group 3 refs as a unit
- **Fork branches** to create experiment copies
- **Experiment safely** with hooks disabled
- **Promote** winning branches to production

## Running the Example

```bash
# Start the KP3 stack
docker compose up -d

# Run the example
docker compose exec kp3-service uv run python examples/04_branches/experimentation.py
```

## Key Concepts

### What is a Branch?

A branch groups the 3 world model refs (human/persona/world) as a unit:

```
Branch: corindel/HEAD
├── human_ref:   corindel/human/HEAD   -> passage-abc
├── persona_ref: corindel/persona/HEAD -> passage-def
└── world_ref:   corindel/world/HEAD   -> passage-ghi
```

### Why Branches?

1. **Atomic operations** - Update all 3 refs together
2. **Safe experimentation** - Experiment branches don't fire hooks
3. **Easy promotion** - Copy refs from experiment to production
4. **Lineage tracking** - Know which branch forked from which

### Hook Control

| Branch Type | hooks_enabled | Use Case |
|-------------|---------------|----------|
| Main (HEAD) | `true` | Production - updates trigger integrations |
| Experiment | `false` | Testing - no side effects |

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/branches` | List branches (with optional prefix filter) |
| POST | `/branches` | Create a new branch |
| GET | `/branches/{name}` | Get branch details |
| DELETE | `/branches/{name}` | Delete a branch |
| POST | `/branches/{name}/fork` | Fork to a new branch |
| POST | `/branches/{name}/promote` | Promote to target branch |

## Workflow

### 1. Create HEAD Branch (Production)

```python
head = await create_branch(
    client,
    ref_prefix="corindel",
    branch_name="HEAD",
    is_main=True,  # hooks_enabled defaults to True
)
```

### 2. Initialize HEAD Refs

```python
await set_ref(client, head["human_ref"], human_passage_id)
await set_ref(client, head["persona_ref"], persona_passage_id)
await set_ref(client, head["world_ref"], world_passage_id)
```

### 3. Fork for Experimentation

```python
experiment = await fork_branch(
    client,
    "corindel/HEAD",
    "experiment-1",
    description="Testing new persona",
)
# Copies current refs, hooks_enabled=False
```

### 4. Update Experiment

```python
# Safe - no hooks fire
await set_ref(client, experiment["human_ref"], new_human_id)
await set_ref(client, experiment["persona_ref"], new_persona_id)
```

### 5. Promote Winner

```python
# Copies refs from experiment to HEAD, fires hooks on HEAD
await promote_branch(client, "corindel/experiment-1", "HEAD")
```

### 6. Clean Up

```python
await delete_branch(client, "corindel/experiment-1", delete_refs=True)
```

## Common Patterns

### A/B Testing

```python
# Create two experiment branches
exp_a = await fork_branch(client, "corindel/HEAD", "exp-a")
exp_b = await fork_branch(client, "corindel/HEAD", "exp-b")

# Configure differently
await set_ref(client, exp_a["persona_ref"], persona_a_id)
await set_ref(client, exp_b["persona_ref"], persona_b_id)

# Test both, promote winner
await promote_branch(client, "corindel/exp-a", "HEAD")
```

### Safe Rollback

```python
# Before major change, create backup
backup = await fork_branch(client, "corindel/HEAD", "backup-2024-01-15")

# Make changes to HEAD...
# If something goes wrong:
await promote_branch(client, "corindel/backup-2024-01-15", "HEAD")
```

## Next Steps

- Use branches with processing runs (B1) for pipeline experimentation
- Combine with scopes (A3) for isolated search contexts
- Track provenance (B2) of passages created on different branches
