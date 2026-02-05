#!/usr/bin/env python3
"""
Example: Hierarchical Memory System
====================================

A complete implementation of hierarchical summarization memory for AI agents.
This demonstrates how to build a production-ready memory system that:

1. Ingests conversations as passages with temporal metadata
2. Summarizes daily -> weekly -> monthly using processing runs
3. Maintains a world model that evolves over time
4. Uses scopes to define what context an agent sees
5. Uses hooks to trigger cascading updates

Architecture:
    Layer 0: Raw conversations (searchable, archived)
    Layer 1: Daily summaries (one per day)
    Layer 2: Weekly summaries (rolled up from dailies)
    Layer 3: Monthly summaries (rolled up from weeklies)
    Layer 4: World model (human/persona/world blocks)

Usage:
    uv run python examples/07_hierarchical_memory/hierarchical_memory.py
"""

import asyncio
import sys
from datetime import datetime, timedelta, timezone

import httpx
from dotenv import load_dotenv

load_dotenv()

BASE_URL = "http://localhost:8080"
AGENT_ID = "memory-demo-agent"


async def create_passage(
    client: httpx.AsyncClient,
    content: str,
    passage_type: str,
    period_start: datetime | None = None,
    period_end: datetime | None = None,
    metadata: dict | None = None,
) -> dict:
    """Create a passage, handling duplicates gracefully."""
    payload = {
        "content": content,
        "passage_type": passage_type,
        "metadata": metadata or {},
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
        # Already exists - search for it
        search = await client.get(
            f"{BASE_URL}/passages/search",
            params={"query": content[:50], "limit": 1},
            headers={"X-Agent-ID": AGENT_ID},
        )
        if search.json()["results"]:
            return search.json()["results"][0]
    response.raise_for_status()
    return response.json()


async def search_passages(
    client: httpx.AsyncClient,
    query: str,
    passage_type: str | None = None,
    limit: int = 5,
) -> list[dict]:
    """Search passages with optional type filter."""
    params = {"query": query, "limit": limit}
    response = await client.get(
        f"{BASE_URL}/passages/search",
        params=params,
        headers={"X-Agent-ID": AGENT_ID},
    )
    response.raise_for_status()
    results = response.json()["results"]
    if passage_type:
        results = [r for r in results if r.get("passage_type") == passage_type]
    return results


async def create_scope(client: httpx.AsyncClient, name: str, description: str) -> dict:
    """Create a memory scope."""
    response = await client.post(
        f"{BASE_URL}/scopes",
        json={"name": name, "description": description},
        headers={"X-Agent-ID": AGENT_ID},
    )
    if response.status_code == 409:
        # Already exists
        get_response = await client.get(
            f"{BASE_URL}/scopes/{name}",
            headers={"X-Agent-ID": AGENT_ID},
        )
        return get_response.json()
    response.raise_for_status()
    return response.json()


async def add_to_scope(
    client: httpx.AsyncClient,
    scope_name: str,
    passage_ids: list[str] | None = None,
    refs: list[str] | None = None,
) -> dict:
    """Add passages or refs to a scope."""
    response = await client.post(
        f"{BASE_URL}/scopes/{scope_name}/add",
        json={"passage_ids": passage_ids or [], "refs": refs or []},
        headers={"X-Agent-ID": AGENT_ID},
    )
    response.raise_for_status()
    return response.json()


async def search_scope(
    client: httpx.AsyncClient,
    scope_name: str,
    query: str,
    limit: int = 5,
) -> list[dict]:
    """Search within a scope."""
    response = await client.get(
        f"{BASE_URL}/scopes/{scope_name}/search",
        params={"query": query, "limit": limit},
        headers={"X-Agent-ID": AGENT_ID},
    )
    response.raise_for_status()
    return response.json()["results"]


async def create_run(
    client: httpx.AsyncClient,
    input_sql: str,
    processor_type: str,
    processor_config: dict,
) -> dict:
    """Create a processing run."""
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


def print_section(title: str) -> None:
    """Print a section header."""
    print()
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)
    print()


def print_step(step: int, title: str) -> None:
    """Print a step header."""
    print(f"Step {step}: {title}")
    print("-" * 40)


async def main():
    print_section("KP3 Hierarchical Memory System Demo")
    print("This demo shows how to build a production memory system")
    print("with hierarchical summarization for AI agents.")

    async with httpx.AsyncClient(timeout=30.0) as client:
        # Check service health
        try:
            health = await client.get(f"{BASE_URL}/health")
            health.raise_for_status()
        except httpx.ConnectError:
            print(f"\nError: Cannot connect to {BASE_URL}")
            print("Start the service: docker compose up -d")
            return

        # ============================================================
        # LAYER 0: Raw Conversations
        # ============================================================
        print_section("Layer 0: Ingesting Raw Conversations")

        print_step(1, "Creating sample conversations over 2 weeks")

        now = datetime.now(timezone.utc)
        conversations = [
            # Week 1
            {
                "day": 0,
                "content": "User asked about setting up a Python virtual environment. "
                "Explained venv vs conda. They're working on a data science project. "
                "Mentioned they prefer VS Code as their editor.",
            },
            {
                "day": 1,
                "content": "Continued helping with Python setup. Installed pandas and numpy. "
                "User mentioned they're analyzing sales data for their small business. "
                "They have about 2 years of transaction history.",
            },
            {
                "day": 2,
                "content": "User had questions about pandas DataFrames. Showed groupby operations. "
                "They want to see monthly revenue trends. Seemed excited about the visualizations.",
            },
            {
                "day": 3,
                "content": "Debugging session - their CSV had encoding issues. Fixed with utf-8. "
                "Also helped with date parsing. User mentioned deadline is end of month.",
            },
            {
                "day": 4,
                "content": "Created matplotlib charts for the sales analysis. User requested "
                "specific colors matching their brand (blue and orange). "
                "They'll present this to their business partner.",
            },
            # Week 2
            {
                "day": 7,
                "content": "User came back after the presentation - it went well! "
                "Partner wants to see customer segmentation analysis. "
                "Discussed RFM analysis approach.",
            },
            {
                "day": 8,
                "content": "Implemented RFM (Recency, Frequency, Monetary) segmentation. "
                "Found they have about 500 active customers. "
                "User surprised by how many are 'at risk' of churning.",
            },
            {
                "day": 9,
                "content": "Created a simple dashboard using Streamlit. User loved the interactivity. "
                "They want to share it with their team. Discussed deployment options.",
            },
            {
                "day": 10,
                "content": "Helped deploy the Streamlit app to their local network. "
                "User mentioned they might want to learn more Python after this project. "
                "Recommended some learning resources.",
            },
        ]

        passage_ids_by_day: dict[int, str] = {}
        for conv in conversations:
            day = now - timedelta(days=14 - conv["day"])
            passage = await create_passage(
                client,
                conv["content"],
                passage_type="conversation",
                period_start=day.replace(hour=10, minute=0, second=0),
                period_end=day.replace(hour=11, minute=0, second=0),
            )
            passage_ids_by_day[conv["day"]] = passage["id"]
            day_label = f"Day {conv['day']:2d}"
            print(f"  {day_label}: {passage['id'][:8]}... ({len(conv['content'])} chars)")

        print(f"\n  Total: {len(conversations)} conversations ingested")

        # ============================================================
        # LAYER 1: Daily Summaries
        # ============================================================
        print_section("Layer 1: Daily Summarization")

        print_step(2, "Creating daily summary processing run")
        print()
        print("  In production, this would run via cron at end of each day.")
        print("  The SQL groups conversations by date:")
        print()

        daily_sql = """
        SELECT
            array_agg(id ORDER BY period_start) as passage_ids,
            'daily-' || date_trunc('day', period_start)::date as group_key,
            jsonb_build_object(
                'date', date_trunc('day', period_start)::date,
                'conversation_count', count(*)
            ) as group_metadata
        FROM passages
        WHERE passage_type = 'conversation'
          AND agent_id = 'memory-demo-agent'
          AND period_start IS NOT NULL
        GROUP BY date_trunc('day', period_start)
        ORDER BY date_trunc('day', period_start)
        """

        print("    SELECT array_agg(id) as passage_ids,")
        print("           date_trunc('day', period_start) as group_key")
        print("    FROM passages")
        print("    WHERE passage_type = 'conversation'")
        print("    GROUP BY date_trunc('day', period_start)")

        daily_run = await create_run(
            client,
            input_sql=daily_sql,
            processor_type="llm_prompt",
            processor_config={
                "prompt_template": "Summarize these conversations into a daily digest. "
                "Focus on: what was accomplished, user sentiment, and any follow-ups needed.",
                "output_passage_type": "daily_summary",
            },
        )
        print(f"\n  Created run: {daily_run['id'][:8]}...")
        print(f"  Execute with: kp3 run execute {daily_run['id']}")

        # ============================================================
        # LAYER 2: Weekly Summaries
        # ============================================================
        print_section("Layer 2: Weekly Summarization")

        print_step(3, "Creating weekly summary processing run")
        print()
        print("  This rolls up daily summaries into weekly summaries.")
        print("  Runs via cron every Sunday night.")

        weekly_sql = """
        SELECT
            array_agg(id ORDER BY period_start) as passage_ids,
            'weekly-' || date_trunc('week', period_start)::date as group_key,
            jsonb_build_object(
                'week_start', date_trunc('week', period_start)::date,
                'daily_count', count(*)
            ) as group_metadata
        FROM passages
        WHERE passage_type = 'daily_summary'
          AND agent_id = 'memory-demo-agent'
          AND period_start IS NOT NULL
        GROUP BY date_trunc('week', period_start)
        ORDER BY date_trunc('week', period_start)
        """

        weekly_run = await create_run(
            client,
            input_sql=weekly_sql,
            processor_type="llm_prompt",
            processor_config={
                "prompt_template": "Create a weekly summary from these daily digests. "
                "Highlight: major themes, progress on projects, and relationship developments.",
                "output_passage_type": "weekly_summary",
            },
        )
        print(f"\n  Created run: {weekly_run['id'][:8]}...")

        # ============================================================
        # LAYER 3: Monthly Summaries
        # ============================================================
        print_section("Layer 3: Monthly Summarization")

        print_step(4, "Creating monthly summary processing run")
        print()
        print("  Rolls up weekly summaries into monthly overviews.")
        print("  Good for 'what did we do last month' queries.")

        monthly_sql = """
        SELECT
            array_agg(id ORDER BY period_start) as passage_ids,
            'monthly-' || date_trunc('month', period_start)::date as group_key,
            jsonb_build_object(
                'month', date_trunc('month', period_start)::date,
                'weekly_count', count(*)
            ) as group_metadata
        FROM passages
        WHERE passage_type = 'weekly_summary'
          AND agent_id = 'memory-demo-agent'
          AND period_start IS NOT NULL
        GROUP BY date_trunc('month', period_start)
        ORDER BY date_trunc('month', period_start)
        """

        monthly_run = await create_run(
            client,
            input_sql=monthly_sql,
            processor_type="llm_prompt",
            processor_config={
                "prompt_template": "Create a monthly overview from these weekly summaries. "
                "Focus on: big picture progress, evolving relationship, and strategic insights.",
                "output_passage_type": "monthly_summary",
            },
        )
        print(f"\n  Created run: {monthly_run['id'][:8]}...")

        # ============================================================
        # LAYER 4: World Model
        # ============================================================
        print_section("Layer 4: World Model Extraction")

        print_step(5, "World model: persistent understanding of the user")
        print()
        print("  The world model extracts durable facts from summaries:")
        print("    - Human block: who they are, preferences, context")
        print("    - Persona block: how the agent should behave with them")
        print("    - World block: projects, entities, themes")
        print()
        print("  Example human understanding from our conversations:")
        print()

        human_understanding = """User Profile:
- Small business owner analyzing sales data
- Technical level: beginner Python, learning data science
- Preferred tools: VS Code, pandas, matplotlib
- Brand colors: blue and orange
- Working with: ~500 customers, 2 years of transaction data
- Current project: Customer segmentation dashboard (Streamlit)
- Deadline pressure: end of month presentation
- Learning motivation: wants to continue learning Python
- Collaboration: has a business partner who reviews analyses"""

        human_passage = await create_passage(
            client,
            human_understanding,
            passage_type="world_model_human",
        )
        print(f"  Created human block: {human_passage['id'][:8]}...")

        persona_understanding = """Agent Persona for this User:
- Communication style: patient, educational, encouraging
- Technical depth: explain concepts, don't just give code
- Proactive: suggest best practices and learning resources
- Remember: they respond well to visualizations
- Deadline-aware: be mindful of their end-of-month timeline
- Celebrate wins: acknowledge their progress and successes"""

        persona_passage = await create_passage(
            client,
            persona_understanding,
            passage_type="world_model_persona",
        )
        print(f"  Created persona block: {persona_passage['id'][:8]}...")

        # ============================================================
        # SCOPES: Defining Agent Context
        # ============================================================
        print_section("Scopes: Defining What the Agent Sees")

        print_step(6, "Creating memory scopes for different contexts")
        print()
        print("  Scopes define which passages are 'visible' for a query.")
        print("  They use refs (mutable pointers) for dynamic resolution.")

        # Create working memory scope
        working_memory = await create_scope(
            client,
            "working-memory",
            "Current conversation context and recent summaries",
        )
        print(f"\n  Created 'working-memory' scope: {working_memory['id'][:8]}...")

        # Add world model passages to working memory
        await add_to_scope(
            client,
            "working-memory",
            passage_ids=[human_passage["id"], persona_passage["id"]],
        )
        print("  Added world model blocks to working-memory")

        # Create long-term memory scope
        long_term = await create_scope(
            client,
            "long-term-memory",
            "All summaries and historical context",
        )
        print(f"  Created 'long-term-memory' scope: {long_term['id'][:8]}...")

        # ============================================================
        # HOOKS: Cascading Updates
        # ============================================================
        print_section("Hooks: Automatic Cascading Updates")

        print_step(7, "Understanding ref hooks")
        print()
        print("  Hooks trigger actions when refs are updated.")
        print("  Use cases:")
        print("    - Refresh scope when daily summary updates")
        print("    - Update world model when weekly summary changes")
        print("    - Send webhook to external systems")
        print()
        print("  Hook configuration (stored in passage_ref_hooks table):")
        print()
        print("    ref_name: 'agent/daily/HEAD'")
        print("    action_type: 'refresh_scope'")
        print("    config: {'scope_name': 'working-memory'}")
        print("    enabled: true")
        print()
        print("  The refs service automatically fires hooks on set_ref().")

        # ============================================================
        # SEARCH: Querying the Memory System
        # ============================================================
        print_section("Search: Querying Hierarchical Memory")

        print_step(8, "Demonstrating multi-layer search")
        print()

        # Search raw conversations
        print("  Query: 'What editor do they use?'")
        print("  Strategy: Check world model first, then search if needed")
        print()

        results = await search_passages(client, "editor preference VS Code")
        if results:
            print(f"  Found in passage: {results[0]['id'][:8]}...")
            print(f"  Type: {results[0].get('passage_type', 'unknown')}")
            print(f"  Content preview: {results[0]['content'][:100]}...")
        print()

        # Search for project info
        print("  Query: 'What project are they working on?'")
        results = await search_passages(client, "project dashboard Streamlit")
        if results:
            print(f"  Found {len(results)} results")
            for r in results[:2]:
                print(f"    - {r['id'][:8]}... ({r.get('passage_type', 'unknown')})")

        # ============================================================
        # PRODUCTION SETUP
        # ============================================================
        print_section("Production Setup")

        print("Cron schedule for hierarchical summarization:")
        print()
        print("  # Daily digest at 11:59 PM")
        print("  59 23 * * * kp3 run fold 'SELECT...' -p daily_summary")
        print()
        print("  # Weekly rollup on Sunday at 11:59 PM")
        print("  59 23 * * 0 kp3 run fold 'SELECT...' -p weekly_summary")
        print()
        print("  # Monthly overview on 1st at midnight")
        print("  0 0 1 * * kp3 run fold 'SELECT...' -p monthly_summary")
        print()
        print("  # World model update after each weekly summary")
        print("  0 1 * * 1 kp3 world-model backfill --limit 7")

        # ============================================================
        # SUMMARY
        # ============================================================
        print_section("Summary")

        print("This demo created a hierarchical memory system:")
        print()
        print(f"  Conversations:     {len(conversations)} raw passages")
        print(f"  Processing runs:   3 (daily, weekly, monthly)")
        print(f"  World model:       2 blocks (human, persona)")
        print(f"  Scopes:            2 (working-memory, long-term-memory)")
        print()
        print("Key patterns:")
        print("  1. Temporal bucketing via SQL GROUP BY date_trunc()")
        print("  2. Fold semantics for stateful processing")
        print("  3. Refs for mutable 'HEAD' pointers")
        print("  4. Hooks for cascading updates")
        print("  5. Scopes for context isolation")
        print()
        print("Next steps:")
        print("  - Execute runs: kp3 run execute <run_id>")
        print("  - Set up prompts: kp3 prompts create ...")
        print("  - Configure hooks in the database")
        print("  - Schedule cron jobs for production")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except httpx.ConnectError:
        print(f"Error: Could not connect to KP3 service at {BASE_URL}")
        print("Make sure the service is running: docker compose up -d")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nInterrupted")
        sys.exit(0)
