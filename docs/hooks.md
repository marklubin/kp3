# Ref Hooks

Hooks are actions that automatically trigger when a ref is updated. They enable cascading updates, external integrations, and reactive workflows.

## Overview

When you call `set_ref()` to update a ref, KP3 checks for any hooks registered to that ref name and executes them. This is how you build reactive systems where changes propagate automatically.

```
set_ref("agent/human/HEAD", passage_id)
        │
        ▼
┌───────────────────┐
│ Check hooks table │
│ for ref_name      │
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│ Execute matching  │
│ hooks in order    │
└───────────────────┘
```

## Database Schema

Hooks are stored in the `passage_ref_hooks` table:

```sql
CREATE TABLE passage_ref_hooks (
    id UUID PRIMARY KEY,
    ref_name TEXT NOT NULL,           -- Which ref triggers this hook
    action_type TEXT NOT NULL,        -- What action to take
    config JSONB NOT NULL,            -- Action-specific configuration
    enabled BOOLEAN DEFAULT TRUE,     -- Can disable without deleting
    created_at TIMESTAMPTZ
);
```

## Creating Hooks

### Via SQL

```sql
INSERT INTO passage_ref_hooks (ref_name, action_type, config) VALUES
('agent/daily/HEAD', 'webhook', '{"url": "https://example.com/notify"}'),
('agent/human/HEAD', 'refresh_scope', '{"scope_name": "working-memory"}');
```

### Via Python

```python
from kp3.services.refs import create_ref_hook

hook = await create_ref_hook(
    session,
    ref_name="agent/human/HEAD",
    action_type="webhook",
    config={"url": "https://example.com/on-human-update"},
    enabled=True,
)
```

## Hook Types

### Webhook (Future)

POST to an external URL when the ref updates:

```python
config = {
    "url": "https://your-service.com/kp3-webhook",
    "headers": {"Authorization": "Bearer token"},
    "include_passage": True,  # Include passage content in payload
}
```

### Custom Hooks

Implement custom hooks by:

1. Adding a new `action_type` value
2. Implementing the handler in `refs.py` `_execute_hook_action()`

```python
async def _execute_hook_action(hook: PassageRefHook, passage: Passage) -> None:
    if hook.action_type == "my_custom_action":
        # Your custom logic here
        await do_something(hook.config, passage)
```

## Controlling Hook Execution

### Per-Call Control

```python
# Skip hooks for this update
await set_ref(session, "agent/test/HEAD", passage_id, fire_hooks=False)
```

### Branch-Level Control

Branches can disable hooks to prevent cascading during experiments:

```python
branch = WorldModelBranch(
    name="experiment-1",
    hooks_enabled=False,  # Hooks won't fire for refs in this branch
)
```

### Disabling a Hook

```sql
UPDATE passage_ref_hooks SET enabled = FALSE WHERE id = '...';
```

## Use Cases

### 1. Scope Refresh

Keep a scope up-to-date when its underlying data changes:

```python
await create_ref_hook(
    session,
    ref_name="agent/daily/HEAD",
    action_type="refresh_scope",
    config={"scope_name": "recent-context"},
)
```

### 2. External Notifications

Notify external systems when important refs update:

```python
await create_ref_hook(
    session,
    ref_name="agent/human/HEAD",
    action_type="webhook",
    config={
        "url": "https://slack.webhook.com/...",
        "template": "Human model updated for agent {agent_id}",
    },
)
```

### 3. Cascading Processing

Trigger downstream processing when upstream completes:

```python
# When daily summary updates, trigger world model extraction
await create_ref_hook(
    session,
    ref_name="agent/daily/HEAD",
    action_type="trigger_run",
    config={"processor_type": "world_model"},
)
```

### 4. Audit Logging

Log all changes to sensitive refs:

```python
await create_ref_hook(
    session,
    ref_name="agent/human/HEAD",
    action_type="audit_log",
    config={"destination": "audit_passages"},
)
```

## Best Practices

1. **Idempotency**: Design hooks to be safe to re-run
2. **Error Handling**: Hooks that fail will raise exceptions; handle appropriately
3. **Performance**: Keep hooks fast; offload heavy work to async queues
4. **Testing**: Use `fire_hooks=False` in tests to isolate behavior
5. **Monitoring**: Log hook executions for debugging

## Listing Hooks

```python
from kp3.services.refs import list_ref_hooks

# All enabled hooks
hooks = await list_ref_hooks(session)

# Hooks for a specific ref
hooks = await list_ref_hooks(session, ref_name="agent/human/HEAD")

# Include disabled hooks
hooks = await list_ref_hooks(session, enabled_only=False)
```

## Hook Execution Order

Hooks are executed in the order they were created (by `created_at`). If you need specific ordering, create hooks in the desired sequence or add an `order` column to the schema.

## Error Behavior

If a hook fails:
1. The exception is logged
2. The exception is re-raised
3. The transaction may be rolled back (depending on caller)

Design hooks to be resilient, or use try/catch in callers if hooks are non-critical.
