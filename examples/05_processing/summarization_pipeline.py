#!/usr/bin/env python3
"""
Example: Processing Runs - Hierarchical Summarization
=====================================================

Demonstrates KP3's Processing Runs API for executing transformation pipelines.
Processing runs take passages as input and produce new passages as output,
with full provenance tracking.

Features showcased:
- Create and monitor processing runs
- View run status and progress
- Understand passage grouping via SQL
- Track which passages created which outputs

Use Cases:
- Hierarchical summarization (daily -> weekly -> monthly)
- Entity extraction pipelines
- Content transformation workflows

Note: This example creates runs but does not execute them.
Execution requires the CLI: `uv run kp3 run execute <run_id>`

Usage:
    docker compose exec kp3-service uv run python examples/05_processing/summarization_pipeline.py
"""

import asyncio
import sys
from datetime import datetime, timedelta, timezone

import httpx
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configuration
BASE_URL = "http://localhost:8080"
AGENT_ID = "demo-agent"


async def create_passage(
    client: httpx.AsyncClient,
    content: str,
    passage_type: str = "journal_entry",
    period_start: datetime | None = None,
    period_end: datetime | None = None,
) -> dict:
    """Create a new passage, or return existing if duplicate."""
    payload = {
        "content": content,
        "passage_type": passage_type,
    }
    if period_start:
        payload["period_start"] = period_start.isoformat()
    if period_end:
        payload["period_end"] = period_end.isoformat()

    response = await client.post(
        f"{BASE_URL}/passages",
        json=payload,
        headers={"X-Agent-ID": AGENT_ID},
    )
    if response.status_code == 409:
        # Passage already exists, search for it
        search_response = await client.get(
            f"{BASE_URL}/passages/search",
            params={"query": content[:100], "limit": 5},
            headers={"X-Agent-ID": AGENT_ID},
        )
        search_response.raise_for_status()
        results = search_response.json()["results"]
        for result in results:
            if result["content"] == content:
                return result
        raise ValueError("Could not find existing passage after 409")
    response.raise_for_status()
    return response.json()


async def create_run(
    client: httpx.AsyncClient,
    input_sql: str,
    processor_type: str,
    processor_config: dict,
) -> dict:
    """Create a new processing run."""
    response = await client.post(
        f"{BASE_URL}/runs",
        json={
            "input_sql": input_sql,
            "processor_type": processor_type,
            "processor_config": processor_config,
        },
    )
    response.raise_for_status()
    return response.json()


async def get_run(client: httpx.AsyncClient, run_id: str) -> dict:
    """Get a run by ID."""
    response = await client.get(f"{BASE_URL}/runs/{run_id}")
    response.raise_for_status()
    return response.json()


async def list_runs(client: httpx.AsyncClient, status: str | None = None) -> list[dict]:
    """List runs, optionally filtered by status."""
    params = {}
    if status:
        params["status"] = status
    response = await client.get(f"{BASE_URL}/runs", params=params)
    response.raise_for_status()
    return response.json()["runs"]


async def main():
    print("=" * 60)
    print("KP3 Demo: Processing Runs - Hierarchical Summarization")
    print("=" * 60)
    print()

    async with httpx.AsyncClient(timeout=30.0) as client:
        # Step 1: Create sample journal entries
        print("Step 1: Creating sample journal entries...")
        print("-" * 40)

        now = datetime.now(timezone.utc)
        entries = [
            {
                "content": "Monday: Started the day with a team standup. "
                "Discussed progress on the API redesign. Had a productive coding session "
                "implementing the new authentication flow.",
                "day_offset": 0,
            },
            {
                "content": "Tuesday: Reviewed pull requests from the team. "
                "Fixed a bug in the caching layer that was causing stale data. "
                "Had lunch with the design team to discuss UI improvements.",
                "day_offset": 1,
            },
            {
                "content": "Wednesday: Deep work day - no meetings. "
                "Made significant progress on the database migration. "
                "Wrote documentation for the new API endpoints.",
                "day_offset": 2,
            },
            {
                "content": "Thursday: Sprint planning meeting in the morning. "
                "Estimated stories for the next sprint. Paired with junior dev on testing.",
                "day_offset": 3,
            },
            {
                "content": "Friday: Wrapped up the authentication feature. "
                "Wrote integration tests. Did a knowledge share session on the new architecture.",
                "day_offset": 4,
            },
        ]

        passages = []
        for entry in entries:
            day = now - timedelta(days=4 - entry["day_offset"])
            try:
                passage = await create_passage(
                    client,
                    entry["content"],
                    passage_type="journal_entry",
                    period_start=day.replace(hour=0, minute=0, second=0),
                    period_end=day.replace(hour=23, minute=59, second=59),
                )
                passages.append(passage)
                day_name = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"][
                    entry["day_offset"]
                ]
                print(f"  Created {day_name} entry: {passage['id'][:8]}...")
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 409:
                    print("  Entry already exists, skipping...")
                else:
                    raise

        print()

        # Step 2: Create a summarization run
        print("Step 2: Creating a summarization run...")
        print("-" * 40)

        # SQL that groups passages by week
        input_sql = """
        SELECT
            array_agg(id ORDER BY period_start) as passage_ids,
            'week-' || date_trunc('week', period_start)::date as group_key,
            jsonb_build_object(
                'week_start', date_trunc('week', period_start)::date,
                'entry_count', count(*)
            ) as group_metadata
        FROM passages
        WHERE passage_type = 'journal_entry'
          AND period_start IS NOT NULL
        GROUP BY date_trunc('week', period_start)
        ORDER BY date_trunc('week', period_start) DESC
        """

        processor_config = {
            "prompt_name": "summarize",  # Would need this prompt configured
            "output_passage_type": "weekly_summary",
            "model": "gpt-4o-mini",
        }

        run = await create_run(
            client,
            input_sql=input_sql,
            processor_type="llm_prompt",
            processor_config=processor_config,
        )

        print(f"  Created run: {run['id']}")
        print(f"  Status: {run['status']}")
        print(f"  Processor: {run['processor_type']}")
        print()
        print("  Input SQL groups entries by week:")
        print("    - array_agg(id) -> passage_ids")
        print("    - 'week-' || date -> group_key")
        print("    - jsonb_build_object -> group_metadata")

        print()

        # Step 3: Check run status
        print("Step 3: Checking run status...")
        print("-" * 40)

        run = await get_run(client, run["id"])
        print(f"  ID: {run['id']}")
        print(f"  Status: {run['status']}")
        print(f"  Created: {run['created_at']}")
        print(f"  Total groups: {run['total_groups'] or 'Not yet determined'}")
        print(f"  Processed: {run['processed_groups']}/{run['total_groups'] or '?'}")
        print(f"  Outputs: {run['output_count']}")

        print()

        # Step 4: List all runs
        print("Step 4: Listing all runs...")
        print("-" * 40)

        runs = await list_runs(client)
        print(f"  Found {len(runs)} total runs:")
        for r in runs[:5]:  # Show first 5
            print(f"    - {r['id'][:8]}... [{r['status']}] {r['processor_type']}")

        print()

        # Step 5: List pending runs
        print("Step 5: Listing pending runs...")
        print("-" * 40)

        pending = await list_runs(client, status="pending")
        print(f"  Found {len(pending)} pending runs:")
        for r in pending[:3]:
            print(f"    - {r['id'][:8]}... processor={r['processor_type']}")

        print()

        # Step 6: Explain the processing flow
        print("Step 6: Understanding the Processing Flow")
        print("-" * 40)
        print()
        print("  Processing runs work in 3 steps:")
        print()
        print("  1. INPUT SQL")
        print("     - Groups passages into processing batches")
        print("     - Returns: passage_ids[], group_key, group_metadata")
        print()
        print("  2. PROCESSOR")
        print("     - Receives each group with its passages")
        print("     - Transforms content (e.g., LLM summarization)")
        print("     - Returns: action (create/update/pass)")
        print()
        print("  3. OUTPUT")
        print("     - Creates new passages from processor results")
        print("     - Records provenance (which inputs created which output)")
        print("     - Updates run progress counters")
        print()
        print("  To execute the run:")
        print(f"    uv run kp3 run execute {run['id']}")

        print()
        print("=" * 60)
        print("Demo complete!")
        print()
        print("Key takeaways:")
        print("  - Runs define: input SQL + processor type + config")
        print("  - Input SQL groups passages into batches")
        print("  - Processors transform batches into outputs")
        print("  - Provenance tracks input->output relationships")
        print("  - Use CLI to execute runs: `kp3 run execute <id>`")
        print("=" * 60)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except httpx.ConnectError:
        print("Error: Could not connect to KP3 service at", BASE_URL)
        print("Make sure the service is running: docker compose up -d")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nInterrupted")
        sys.exit(0)
