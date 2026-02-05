#!/usr/bin/env python3
"""
Example: Semantic Note Search
=============================

Demonstrates KP3's core Passages API for building a personal note-taking app
where you can dump thoughts and find them later using natural language search.

Features showcased:
- Create passages (notes)
- Hybrid search (FTS + semantic + recency)
- Tag-based organization
- Agent isolation (multi-user)

Usage:
    # With docker compose running:
    docker compose exec kp3-service uv run python examples/01_passages/semantic_notes.py

    # Or directly:
    uv run python examples/01_passages/semantic_notes.py
"""

import asyncio
import sys
from uuid import UUID

import httpx
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configuration
BASE_URL = "http://localhost:8080"
AGENT_ID = "demo-user"


async def create_note(
    client: httpx.AsyncClient,
    content: str,
    passage_type: str = "note",
    metadata: dict | None = None,
) -> dict:
    """Create a new note (passage), or return existing if duplicate."""
    response = await client.post(
        f"{BASE_URL}/passages",
        json={
            "content": content,
            "passage_type": passage_type,
            "metadata": metadata or {},
        },
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


async def get_note(client: httpx.AsyncClient, passage_id: UUID | str) -> dict:
    """Retrieve a note by ID."""
    response = await client.get(
        f"{BASE_URL}/passages/{passage_id}",
        headers={"X-Agent-ID": AGENT_ID},
    )
    response.raise_for_status()
    return response.json()


async def search_notes(
    client: httpx.AsyncClient,
    query: str,
    mode: str = "hybrid",
    limit: int = 5,
) -> list[dict]:
    """Search notes using hybrid search (FTS + semantic + recency)."""
    response = await client.get(
        f"{BASE_URL}/passages/search",
        params={"query": query, "mode": mode, "limit": limit},
        headers={"X-Agent-ID": AGENT_ID},
    )
    response.raise_for_status()
    return response.json()["results"]


async def create_tag(client: httpx.AsyncClient, name: str, description: str | None = None) -> dict:
    """Create a new tag for organizing notes."""
    response = await client.post(
        f"{BASE_URL}/tags",
        json={"name": name, "description": description},
        headers={"X-Agent-ID": AGENT_ID},
    )
    response.raise_for_status()
    return response.json()


async def attach_tags(client: httpx.AsyncClient, passage_id: str, tag_ids: list[str]) -> dict:
    """Attach tags to a note."""
    response = await client.post(
        f"{BASE_URL}/passages/{passage_id}/tags",
        json={"tag_ids": tag_ids},
        headers={"X-Agent-ID": AGENT_ID},
    )
    response.raise_for_status()
    return response.json()


async def search_by_tags(
    client: httpx.AsyncClient,
    query: str,
    mode: str = "hybrid",
    limit: int = 5,
) -> list[dict]:
    """Search for notes by matching tag names/descriptions."""
    response = await client.get(
        f"{BASE_URL}/passages/search",
        params={"query": query, "search_type": "tags", "mode": mode, "limit": limit},
        headers={"X-Agent-ID": AGENT_ID},
    )
    response.raise_for_status()
    return response.json()["results"]


async def main():
    print("=" * 60)
    print("KP3 Demo: Semantic Note Search")
    print("=" * 60)
    print()

    async with httpx.AsyncClient(timeout=30.0) as client:
        # Step 1: Create some notes
        print("Step 1: Creating notes...")
        print("-" * 40)

        notes_data = [
            {
                "content": "Meeting with the design team about the new dashboard layout. "
                "They proposed using a card-based design with drag-and-drop widgets. "
                "Need to prototype this by Friday.",
                "type": "meeting",
            },
            {
                "content": "Learned about vector embeddings today. They convert text into "
                "high-dimensional numerical representations that capture semantic meaning. "
                "Cosine similarity measures how similar two vectors are.",
                "type": "learning",
            },
            {
                "content": "Bug in production: users are seeing stale data after login. "
                "Traced it to aggressive caching in the API gateway. "
                "Quick fix: add cache-busting headers. Long-term: implement cache invalidation.",
                "type": "incident",
            },
            {
                "content": "Recipe idea: pasta with sun-dried tomatoes, fresh basil, "
                "pine nuts, and parmesan. Light olive oil base, garlic, salt, pepper.",
                "type": "personal",
            },
            {
                "content": "PostgreSQL pgvector extension enables vector similarity search "
                "directly in the database. Supports IVFFlat and HNSW indexes. "
                "Much simpler than running a separate vector database.",
                "type": "learning",
            },
        ]

        created_notes = []
        for i, note in enumerate(notes_data, 1):
            try:
                result = await create_note(client, note["content"], passage_type=note["type"])
                created_notes.append(result)
                print(f"  [{i}] Created {note['type']} note: {result['id'][:8]}...")
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 409:
                    print(f"  [{i}] Note already exists (duplicate content), skipping...")
                else:
                    raise

        print()

        # Step 2: Semantic search - find conceptually related notes
        print("Step 2: Semantic Search")
        print("-" * 40)

        queries = [
            "machine learning and AI concepts",  # Should find embedding notes
            "problems with caching",  # Should find the bug note
            "food and cooking",  # Should find the recipe
            "user interface design",  # Should find the dashboard meeting
        ]

        for query in queries:
            print(f"\nQuery: '{query}'")
            results = await search_notes(client, query, mode="semantic", limit=2)
            for r in results:
                snippet = r["content"][:60] + "..." if len(r["content"]) > 60 else r["content"]
                print(f"  - [{r['passage_type']}] (score: {r['score']:.3f}) {snippet}")

        print()

        # Step 3: Full-text search - exact word matching
        print("Step 3: Full-Text Search (FTS)")
        print("-" * 40)

        fts_queries = [
            "pgvector",  # Exact match
            "dashboard widgets",  # Multiple words
        ]

        for query in fts_queries:
            print(f"\nQuery: '{query}'")
            results = await search_notes(client, query, mode="fts", limit=2)
            if results:
                for r in results:
                    snippet = r["content"][:60] + "..." if len(r["content"]) > 60 else r["content"]
                    print(f"  - [{r['passage_type']}] (score: {r['score']:.3f}) {snippet}")
            else:
                print("  (no results)")

        print()

        # Step 4: Hybrid search - best of both worlds
        print("Step 4: Hybrid Search (FTS + Semantic)")
        print("-" * 40)

        print("\nQuery: 'vector database'")
        print("  Combines exact matches ('vector') with semantic understanding ('database')")
        results = await search_notes(client, "vector database", mode="hybrid", limit=3)
        for r in results:
            snippet = r["content"][:60] + "..." if len(r["content"]) > 60 else r["content"]
            print(f"  - [{r['passage_type']}] (score: {r['score']:.3f}) {snippet}")

        print()

        # Step 5: Tag-based organization
        print("Step 5: Tags for Organization")
        print("-" * 40)

        # Create tags
        tags_data = [
            ("work", "Work-related notes and tasks"),
            ("learning", "Things I've learned or want to remember"),
            ("tech", "Technical notes about programming and infrastructure"),
        ]

        created_tags = []
        for name, desc in tags_data:
            try:
                tag = await create_tag(client, name, desc)
                created_tags.append(tag)
                print(f"  Created tag: '{name}' ({tag['id'][:8]}...)")
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 409:
                    print(f"  Tag '{name}' already exists, skipping...")
                else:
                    raise

        # Attach tags to notes (if we created notes)
        if created_notes and created_tags:
            print("\n  Attaching tags to notes...")
            # Attach 'tech' and 'learning' tags to the learning notes
            for note in created_notes:
                if note.get("passage_type") == "learning":
                    tech_tag = next((t for t in created_tags if t["name"] == "tech"), None)
                    learn_tag = next((t for t in created_tags if t["name"] == "learning"), None)
                    if tech_tag and learn_tag:
                        try:
                            await attach_tags(client, note["id"], [tech_tag["id"], learn_tag["id"]])
                            print(f"    Tagged note {note['id'][:8]}... with: tech, learning")
                        except httpx.HTTPStatusError:
                            pass

        # Search by tag description
        print("\n  Searching by tag 'programming'...")
        results = await search_by_tags(client, "programming", mode="semantic", limit=3)
        if results:
            for r in results:
                snippet = r["content"][:50] + "..." if len(r["content"]) > 50 else r["content"]
                print(f"    - [{r['passage_type']}] {snippet}")
        else:
            print("    (no results - tags may not be attached yet)")

        print()

        # Step 6: Retrieve a specific note
        print("Step 6: Retrieve Note by ID")
        print("-" * 40)

        if created_notes:
            note_id = created_notes[0]["id"]
            note = await get_note(client, note_id)
            print(f"  ID: {note['id']}")
            print(f"  Type: {note['passage_type']}")
            print(f"  Created: {note['created_at']}")
            print(f"  Content: {note['content'][:80]}...")

        print()
        print("=" * 60)
        print("Demo complete!")
        print()
        print("Key takeaways:")
        print("  - Passages store any text content with automatic embedding")
        print("  - Hybrid search combines exact matching (FTS) with semantic understanding")
        print("  - Tags enable flexible categorization with their own semantic search")
        print("  - X-Agent-ID header isolates data between different users/agents")
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
