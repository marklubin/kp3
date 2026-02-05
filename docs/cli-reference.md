# CLI Reference

## Passages

```bash
# Create a passage
kp3 passage create "Your content here" -t passage_type

# List passages
kp3 passage ls [--type TYPE] [--limit N]

# Search passages (requires running kp3-service)
kp3 passage search "query" --agent AGENT_ID [-m fts|semantic|hybrid] [-n LIMIT]
```

## Processing Runs

```bash
# Create and execute a processing run
kp3 run create "SELECT id FROM passages WHERE ..." -p processor_type [-c '{"config": "json"}']

# List runs
kp3 run ls [--status STATUS] [--limit N]

# Execute fold operation (sequential processing with state)
kp3 run fold "SELECT id FROM passages ORDER BY created_at" -p world_model -c '{"config": "json"}'
```

## Refs (Mutable Pointers)

```bash
# List refs
kp3 refs list [--prefix PREFIX]

# Get ref details
kp3 refs get REF_NAME

# Set ref to point to passage
kp3 refs set REF_NAME PASSAGE_ID [--no-hooks]

# View ref history
kp3 refs history REF_NAME [--limit N]
```

## World Model

```bash
# Seed default extraction prompts
kp3 world-model seed-prompts

# Process a single passage
kp3 world-model step PASSAGE_ID --ref-prefix agent_name --agent-id AGENT_ID

# Process multiple passages with fold semantics
kp3 world-model fold "SELECT id FROM passages ORDER BY created_at" \
    --ref-prefix agent_name \
    --agent-id AGENT_ID

# Backfill world model from historical passages
kp3 world-model backfill [--branch BRANCH] [--limit N] [--dry-run]
```

## Branches

```bash
# Create a new branch
kp3 world-model branch create prefix/branch_name [--main] [-d "description"]

# Fork from existing branch
kp3 world-model branch fork prefix/source prefix/new_branch

# List branches
kp3 world-model branch list [--prefix PREFIX]

# Show branch details
kp3 world-model branch show prefix/branch_name

# Promote branch (copy refs to target)
kp3 world-model branch promote prefix/source [--to TARGET]

# Delete branch
kp3 world-model branch delete prefix/branch_name [--delete-refs] [--force]
```

## Prompts

```bash
# List prompts
kp3 prompts list [--name NAME]

# Show prompt details
kp3 prompts show NAME [--version VERSION]

# Create new prompt version
kp3 prompts create NAME -s system.txt -t template.txt [-f fields.json] [--activate]

# Activate prompt version
kp3 prompts activate NAME VERSION
```

## Utilities

```bash
# Execute raw SQL
kp3 sql "SELECT * FROM passages LIMIT 5"
```
