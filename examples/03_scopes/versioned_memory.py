#!/usr/bin/env python3
"""
Example: Scopes - Versioned Memory Containers
=============================================

Demonstrates KP3's Memory Scopes API for creating versioned search closures.
Scopes define a "working set" of passages that evolves over time, with full
version history and revert capability.

Features showcased:
- Create scopes as memory containers
- Add/remove passages from scope
- Scoped search (only search within scope)
- Version history and revert to previous state

Use Cases:
- AI agent's working memory that evolves
- Project-specific context windows
- Experiment with different knowledge subsets
- Time-travel through memory states

Usage:
    docker compose exec kp3-service uv run python examples/03_scopes/versioned_memory.py
"""

import asyncio
import sys
import time

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
    passage_type: str = "memory",
) -> dict:
    """Create a new passage, or return existing if duplicate."""
    response = await client.post(
        f"{BASE_URL}/passages",
        json={"content": content, "passage_type": passage_type},
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


async def create_scope(
    client: httpx.AsyncClient,
    name: str,
    description: str | None = None,
) -> dict:
    """Create a new memory scope, or return existing if duplicate."""
    response = await client.post(
        f"{BASE_URL}/scopes",
        json={"name": name, "description": description},
        headers={"X-Agent-ID": AGENT_ID},
    )
    if response.status_code in (409, 500):
        # Scope may already exist, try to fetch it
        get_response = await client.get(
            f"{BASE_URL}/scopes/{name}",
            headers={"X-Agent-ID": AGENT_ID},
        )
        if get_response.status_code == 200:
            return get_response.json()
    response.raise_for_status()
    return response.json()


async def get_scope(client: httpx.AsyncClient, name: str) -> dict | None:
    """Get a scope by name."""
    response = await client.get(
        f"{BASE_URL}/scopes/{name}",
        headers={"X-Agent-ID": AGENT_ID},
    )
    if response.status_code == 404:
        return None
    response.raise_for_status()
    return response.json()


async def add_passages_to_scope(
    client: httpx.AsyncClient,
    scope_name: str,
    passage_ids: list[str],
) -> dict:
    """Add passages to a scope."""
    response = await client.post(
        f"{BASE_URL}/scopes/{scope_name}/add",
        json={"passage_ids": passage_ids},
        headers={"X-Agent-ID": AGENT_ID},
    )
    response.raise_for_status()
    return response.json()


async def remove_from_scope(
    client: httpx.AsyncClient,
    scope_name: str,
    passage_ids: list[str],
) -> dict:
    """Remove passages from a scope."""
    response = await client.post(
        f"{BASE_URL}/scopes/{scope_name}/remove",
        json={"passage_ids": passage_ids},
        headers={"X-Agent-ID": AGENT_ID},
    )
    response.raise_for_status()
    return response.json()


async def search_in_scope(
    client: httpx.AsyncClient,
    scope_name: str,
    query: str,
    mode: str = "hybrid",
    limit: int = 5,
) -> dict:
    """Search passages within a scope."""
    response = await client.get(
        f"{BASE_URL}/scopes/{scope_name}/search",
        params={"query": query, "mode": mode, "limit": limit},
        headers={"X-Agent-ID": AGENT_ID},
    )
    response.raise_for_status()
    return response.json()


async def get_scope_history(
    client: httpx.AsyncClient,
    scope_name: str,
    limit: int = 10,
) -> list[dict]:
    """Get version history of a scope."""
    response = await client.get(
        f"{BASE_URL}/scopes/{scope_name}/history",
        params={"limit": limit},
        headers={"X-Agent-ID": AGENT_ID},
    )
    response.raise_for_status()
    return response.json()["history"]


async def revert_scope(
    client: httpx.AsyncClient,
    scope_name: str,
    to_version: int,
) -> dict:
    """Revert a scope to a previous version."""
    response = await client.post(
        f"{BASE_URL}/scopes/{scope_name}/revert",
        json={"to_version": to_version},
        headers={"X-Agent-ID": AGENT_ID},
    )
    response.raise_for_status()
    return response.json()


async def create_passage_in_scope(
    client: httpx.AsyncClient,
    scope_name: str,
    content: str,
    passage_type: str = "memory",
) -> dict | None:
    """Create a passage and add it to scope atomically.

    Returns None if passage already exists (409).
    """
    response = await client.post(
        f"{BASE_URL}/scopes/{scope_name}/passages",
        json={"content": content, "passage_type": passage_type},
        headers={"X-Agent-ID": AGENT_ID},
    )
    if response.status_code == 409:
        return None  # Passage already exists
    response.raise_for_status()
    return response.json()


async def main():
    print("=" * 60)
    print("KP3 Demo: Scopes - Versioned Memory Containers")
    print("=" * 60)
    print()

    async with httpx.AsyncClient(timeout=30.0) as client:
        # Step 1: Create a memory scope
        print("Step 1: Creating a memory scope...")
        print("-" * 40)

        # Use timestamp to ensure unique scope name for demo re-runs
        scope_name = f"agent-memory-{int(time.time())}"
        try:
            scope = await create_scope(
                client,
                scope_name,
                description="Working memory for demo agent",
            )
            print(f"  Created scope: {scope['name']}")
            print(f"  Description: {scope['description']}")
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 409:
                print(f"  Scope '{scope_name}' already exists, using it...")
                scope = await get_scope(client, scope_name)
            else:
                raise

        print()

        # Step 2: Create some memory passages
        print("Step 2: Creating memory passages...")
        print("-" * 40)

        memories = [
            "User prefers dark mode interfaces and minimal animations.",
            "User is working on a Python project using FastAPI and PostgreSQL.",
            "User mentioned they have a meeting at 3pm with the design team.",
            "User asked about vector databases yesterday - showed interest in pgvector.",
            "User's favorite programming language is Python, also knows TypeScript.",
        ]

        passages = []
        for i, content in enumerate(memories, 1):
            try:
                passage = await create_passage(client, content)
                passages.append(passage)
                print(f"  [{i}] Created: {passage['id'][:8]}...")
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 409:
                    print(f"  [{i}] Already exists, skipping...")
                else:
                    raise

        print()

        # Step 3: Add passages to scope
        print("Step 3: Adding passages to scope...")
        print("-" * 40)

        if passages:
            # Add first 3 passages
            result = await add_passages_to_scope(
                client, scope_name, [p["id"] for p in passages[:3]]
            )
            print(f"  Added {result['modified_count']} passages")
            print(f"  Scope version: {result['scope_version']}")

        print()

        # Step 4: Search within scope
        print("Step 4: Searching within scope...")
        print("-" * 40)

        print("\nQuery: 'programming languages'")
        results = await search_in_scope(client, scope_name, "programming languages", limit=3)
        print(f"  Scope version: {results['scope_version']}")
        print(f"  Results in scope: {results['count']}")
        for r in results["results"]:
            snippet = r["content"][:50] + "..." if len(r["content"]) > 50 else r["content"]
            print(f"    - (score: {r['score']:.3f}) {snippet}")

        print()

        # Step 5: Add more passages
        print("Step 5: Adding more passages to scope...")
        print("-" * 40)

        if len(passages) >= 5:
            result = await add_passages_to_scope(
                client, scope_name, [p["id"] for p in passages[3:5]]
            )
            print(f"  Added {result['modified_count']} more passages")
            print(f"  Scope version: {result['scope_version']}")

        # Search again
        print("\nQuery: 'programming languages' (after adding more)")
        results = await search_in_scope(client, scope_name, "programming languages", limit=3)
        print(f"  Results in scope: {results['count']}")
        for r in results["results"]:
            snippet = r["content"][:50] + "..." if len(r["content"]) > 50 else r["content"]
            print(f"    - (score: {r['score']:.3f}) {snippet}")

        print()

        # Step 6: View scope history
        print("Step 6: Viewing scope history...")
        print("-" * 40)

        history = await get_scope_history(client, scope_name)
        print(f"  Total versions: {len(history)}")
        for entry in history[:5]:  # Show last 5
            print(f"    v{entry['version']}: {entry['changed_at'][:19]}")

        print()

        # Step 7: Remove a passage
        print("Step 7: Removing a passage from scope...")
        print("-" * 40)

        if passages:
            result = await remove_from_scope(client, scope_name, [passages[0]["id"]])
            print(f"  Removed {result['modified_count']} passage")
            print(f"  Scope version: {result['scope_version']}")

        print()

        # Step 8: Revert to a previous version
        print("Step 8: Reverting to previous version...")
        print("-" * 40)

        history = await get_scope_history(client, scope_name)
        if len(history) >= 2:
            # Revert to version before the removal
            target_version = history[1]["version"]
            result = await revert_scope(client, scope_name, target_version)
            print(f"  Reverted from v{result['reverted_from']} to v{target_version}")
            print(f"  New version: {result['scope_version']}")
            print("  (Note: Revert creates a new version, history is preserved)")

        print()

        # Step 9: Create passage directly in scope
        print("Step 9: Creating passage directly in scope...")
        print("-" * 40)

        result = await create_passage_in_scope(
            client,
            scope_name,
            "User mentioned they're interested in learning Rust next.",
        )
        if result:
            print(f"  Created passage: {result['passage_id'][:8]}...")
            print(f"  Scope version: {result['scope_version']}")
            print("  (Passage created and added atomically)")
        else:
            print("  Passage already exists, skipping...")
            print("  (This is expected on demo re-runs)")

        print()
        print("=" * 60)
        print("Demo complete!")
        print()
        print("Key takeaways:")
        print("  - Scopes define versioned search closures")
        print("  - Every add/remove creates a new version")
        print("  - Scoped search only returns passages in the scope")
        print("  - Full history allows reverting to any previous state")
        print("  - Revert is non-destructive (creates new version)")
        print("  - Use scopes for agent working memory, project context, etc.")
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
