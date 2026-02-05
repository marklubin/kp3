# REST API Reference

The kp3-service exposes a REST API on port 8080.

## Health Check
```
GET /health
```

## Passages

### Search Passages
```
GET /passages/search?query=...&mode=hybrid&search_type=content&limit=5
Header: X-Agent-ID: <agent-id>

Query Parameters:
- query: Search query text (required)
- mode: fts | semantic | hybrid (default: hybrid)
- search_type: content | tags (default: content)
- limit: Max results (default: 5, max: 50)
- period_start_after: Filter by period_start >= value
- period_end_before: Filter by period_end <= value
- created_after: Filter by created_at >= value
- created_before: Filter by created_at <= value

Response:
{
  "query": "search query",
  "mode": "hybrid",
  "search_type": "content",
  "results": [
    {
      "id": "uuid",
      "content": "passage content",
      "passage_type": "memory_shard",
      "score": 0.85
    }
  ],
  "count": 1
}
```

### Create Passage
```
POST /passages
Header: X-Agent-ID: <agent-id>
Content-Type: application/json

{
  "content": "passage content",
  "passage_type": "memory_shard",
  "metadata": {},
  "period_start": "2024-01-01T00:00:00Z",
  "period_end": "2024-01-01T23:59:59Z"
}

Response:
{
  "id": "uuid",
  "content": "passage content",
  "passage_type": "memory_shard"
}
```

## Tags

### Create Tag
```
POST /tags
Header: X-Agent-ID: <agent-id>
Content-Type: application/json

{
  "name": "Python",
  "description": "Python programming content"
}

Response: { "id": "uuid", "name": "Python", ... }
```

### List Tags
```
GET /tags?limit=100&offset=0&order_by=name
Header: X-Agent-ID: <agent-id>

Response: { "tags": [...], "count": N }
```

### Get/Delete Tag
```
GET /tags/{tag_id}
DELETE /tags/{tag_id}
Header: X-Agent-ID: <agent-id>
```

### Attach/Detach Tags to Passage
```
POST /passages/{passage_id}/tags
DELETE /passages/{passage_id}/tags
Header: X-Agent-ID: <agent-id>
Content-Type: application/json

{ "tag_ids": ["uuid1", "uuid2"] }
```

### Get Passage Tags
```
GET /passages/{passage_id}/tags
Header: X-Agent-ID: <agent-id>
```

## Memory Scopes

Scopes define dynamic search closures using refs and literal passage IDs.

### Create Scope
```
POST /scopes
Header: X-Agent-ID: <agent-id>
Content-Type: application/json

{
  "name": "working-memory",
  "description": "Active conversation context"
}

Response:
{
  "id": "uuid",
  "name": "working-memory",
  "agent_id": "agent-id",
  "head_ref": "agent-id/scope/working-memory/HEAD",
  "description": "Active conversation context",
  "created_at": "...",
  "updated_at": "..."
}
```

### List Scopes
```
GET /scopes?limit=100&offset=0
Header: X-Agent-ID: <agent-id>

Response: { "scopes": [...], "count": N }
```

### Get/Delete Scope
```
GET /scopes/{name}
DELETE /scopes/{name}
Header: X-Agent-ID: <agent-id>
```

### Create Passage in Scope
```
POST /scopes/{name}/passages
Header: X-Agent-ID: <agent-id>
Content-Type: application/json

{
  "content": "Important context to remember",
  "passage_type": "memory"
}

Response:
{
  "passage_id": "uuid",
  "content": "...",
  "passage_type": "memory",
  "scope_version": 2
}
```

### Add to Scope
```
POST /scopes/{name}/add
Header: X-Agent-ID: <agent-id>
Content-Type: application/json

{
  "passage_ids": ["uuid1", "uuid2"],
  "refs": ["agent/human/HEAD", "agent/persona/HEAD"]
}

Response: { "scope_version": 3, "modified_count": 4 }
```

### Remove from Scope
```
POST /scopes/{name}/remove
Header: X-Agent-ID: <agent-id>
Content-Type: application/json

{
  "passage_ids": ["uuid1"],
  "refs": ["old/ref"]
}

Response: { "scope_version": 4, "modified_count": 2 }
```

### Get Scope History
```
GET /scopes/{name}/history?limit=100
Header: X-Agent-ID: <agent-id>

Response:
{
  "history": [
    { "version": 4, "changed_at": "...", "passage_id": "def-uuid" },
    { "version": 3, "changed_at": "...", "passage_id": "prev-uuid" },
    ...
  ],
  "count": 4
}
```

### Revert Scope
```
POST /scopes/{name}/revert
Header: X-Agent-ID: <agent-id>
Content-Type: application/json

{ "to_version": 2 }

Response: { "scope_version": 5, "reverted_from": 4 }
```

### Search in Scope
```
GET /scopes/{name}/search?query=...&mode=hybrid&limit=10
Header: X-Agent-ID: <agent-id>

Response:
{
  "query": "...",
  "mode": "hybrid",
  "search_type": "content",
  "results": [...],
  "count": N,
  "scope": "working-memory",
  "scope_version": 5
}
```

## Prompts

### Get Prompt
```
GET /prompts/{name}

Response:
{
  "id": "uuid",
  "name": "human",
  "version": 1,
  "system_prompt": "...",
  "user_prompt_template": "...",
  "field_descriptions": {}
}
```
