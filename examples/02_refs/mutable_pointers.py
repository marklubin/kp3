#!/usr/bin/env python3
"""
Example: Refs - Mutable Pointers
================================

Demonstrates KP3's Refs API for tracking "current state" with mutable pointers.
Refs work like git refs - named pointers that can be updated to point to
different passages over time, with full history tracking.

Features showcased:
- Create/update refs
- Get current passage a ref points to
- View ref history (audit trail)
- Use refs as bookmarks to important passages

Use Cases:
- Track "current version" of a document
- Implement state machines
- Create named bookmarks
- Build audit trails

Usage:
    docker compose exec kp3-service uv run python examples/02_refs/mutable_pointers.py
"""

import asyncio
import sys

import httpx
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configuration
BASE_URL = "http://localhost:8080"
AGENT_ID = "demo-user"


async def create_passage(client: httpx.AsyncClient, content: str, passage_type: str) -> dict:
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


async def set_ref(
    client: httpx.AsyncClient,
    name: str,
    passage_id: str,
    metadata: dict | None = None,
) -> dict:
    """Set a ref to point to a passage."""
    response = await client.put(
        f"{BASE_URL}/refs/{name}",
        json={"passage_id": passage_id, "metadata": metadata or {}},
    )
    response.raise_for_status()
    return response.json()


async def get_ref(client: httpx.AsyncClient, name: str) -> dict | None:
    """Get a ref by name."""
    response = await client.get(f"{BASE_URL}/refs/{name}")
    if response.status_code == 404:
        return None
    response.raise_for_status()
    return response.json()


async def list_refs(client: httpx.AsyncClient, prefix: str | None = None) -> list[dict]:
    """List refs, optionally filtered by prefix."""
    params = {}
    if prefix:
        params["prefix"] = prefix
    response = await client.get(f"{BASE_URL}/refs", params=params)
    response.raise_for_status()
    return response.json()["refs"]


async def get_ref_history(client: httpx.AsyncClient, name: str, limit: int = 10) -> list[dict]:
    """Get history of changes for a ref."""
    response = await client.get(f"{BASE_URL}/refs/{name}/history", params={"limit": limit})
    response.raise_for_status()
    return response.json()["history"]


async def delete_ref(client: httpx.AsyncClient, name: str) -> bool:
    """Delete a ref."""
    response = await client.delete(f"{BASE_URL}/refs/{name}")
    return response.status_code == 204


async def main():
    print("=" * 60)
    print("KP3 Demo: Refs - Mutable Pointers")
    print("=" * 60)
    print()

    async with httpx.AsyncClient(timeout=30.0) as client:
        # Step 1: Create some passages to reference
        print("Step 1: Creating passages to reference...")
        print("-" * 40)

        passages = []
        versions = [
            "Draft v1: Initial project proposal for the new authentication system.",
            "Draft v2: Added OAuth2 support and security considerations.",
            "Draft v3: Final version with implementation timeline and milestones.",
        ]

        for i, content in enumerate(versions, 1):
            try:
                passage = await create_passage(client, content, "document")
                passages.append(passage)
                print(f"  Created passage {i}: {passage['id'][:8]}...")
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 409:
                    print(f"  Passage {i} already exists, skipping...")
                else:
                    raise

        if len(passages) < 3:
            print("\n  Note: Some passages already existed. Continuing with demo...")
            print("  (Clear the database to see full progression)")

        print()

        # Step 2: Create a ref pointing to the first version
        print("Step 2: Creating a ref pointing to v1...")
        print("-" * 40)

        if passages:
            ref = await set_ref(
                client,
                "docs/proposal/current",
                passages[0]["id"],
                metadata={"version": "v1", "author": "alice"},
            )
            print(f"  Ref: {ref['name']}")
            print(f"  Points to: {ref['passage_id'][:8]}...")
            print(f"  Updated: {ref['updated_at']}")

        print()

        # Step 3: Update the ref to point to newer versions
        print("Step 3: Updating ref through versions...")
        print("-" * 40)

        if len(passages) >= 3:
            # Update to v2
            ref = await set_ref(
                client,
                "docs/proposal/current",
                passages[1]["id"],
                metadata={"version": "v2", "author": "alice"},
            )
            print(f"  Updated to v2: {ref['passage_id'][:8]}...")

            # Update to v3
            ref = await set_ref(
                client,
                "docs/proposal/current",
                passages[2]["id"],
                metadata={"version": "v3", "author": "bob"},
            )
            print(f"  Updated to v3: {ref['passage_id'][:8]}...")

        print()

        # Step 4: View the ref history
        print("Step 4: Viewing ref history (audit trail)...")
        print("-" * 40)

        history = await get_ref_history(client, "docs/proposal/current")
        for entry in history:
            prev = entry["previous_passage_id"][:8] if entry["previous_passage_id"] else "None"
            print(f"  {entry['changed_at'][:19]}")
            print(f"    Current: {entry['passage_id'][:8]}...")
            print(f"    Previous: {prev}")
            print()

        # Step 5: Create multiple refs with a common prefix
        print("Step 5: Creating refs with common prefix...")
        print("-" * 40)

        if len(passages) >= 3:
            # Create refs for different stages
            await set_ref(client, "docs/proposal/draft", passages[0]["id"])
            await set_ref(client, "docs/proposal/review", passages[1]["id"])
            await set_ref(client, "docs/proposal/approved", passages[2]["id"])
            print("  Created: docs/proposal/draft")
            print("  Created: docs/proposal/review")
            print("  Created: docs/proposal/approved")

        print()

        # Step 6: List refs by prefix
        print("Step 6: Listing refs by prefix...")
        print("-" * 40)

        refs = await list_refs(client, prefix="docs/proposal/")
        print(f"  Found {len(refs)} refs with prefix 'docs/proposal/':")
        for ref in refs:
            print(f"    - {ref['name']} -> {ref['passage_id'][:8]}...")

        print()

        # Step 7: Get a specific ref
        print("Step 7: Getting specific ref...")
        print("-" * 40)

        ref = await get_ref(client, "docs/proposal/current")
        if ref:
            print(f"  Name: {ref['name']}")
            print(f"  Points to: {ref['passage_id']}")
            print(f"  Metadata: {ref['metadata']}")

        print()

        # Step 8: Clean up a ref
        print("Step 8: Deleting a ref...")
        print("-" * 40)

        deleted = await delete_ref(client, "docs/proposal/draft")
        if deleted:
            print("  Deleted: docs/proposal/draft")
            print("  (Note: The passage still exists, only the ref was deleted)")

        print()
        print("=" * 60)
        print("Demo complete!")
        print()
        print("Key takeaways:")
        print("  - Refs are mutable pointers to passages (like git refs)")
        print("  - Every change is recorded in history for auditing")
        print("  - Use prefixes to organize refs (e.g., 'docs/', 'state/')")
        print("  - Deleting a ref doesn't delete the underlying passage")
        print("  - Refs can store metadata (version, author, etc.)")
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
