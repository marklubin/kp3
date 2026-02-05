#!/usr/bin/env python3
"""
Example: Provenance - Derivation Chain Tracking
===============================================

Demonstrates KP3's Provenance API for tracking which passages derived
from which other passages. This enables answering questions like:
- "What sources were used to create this summary?"
- "What outputs were generated from this input?"
- "What's the full derivation chain?"

Features showcased:
- Get direct source passages
- Get directly derived passages
- Traverse full provenance chain (recursive)

Use Cases:
- Trace summaries back to source documents
- Verify information provenance
- Debug processing pipelines
- Build citation graphs

Usage:
    docker compose exec kp3-service uv run python examples/06_provenance/derivation_chains.py
"""

import asyncio
import sys

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
    passage_type: str = "note",
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


async def get_passage_sources(client: httpx.AsyncClient, passage_id: str) -> dict:
    """Get immediate source passages for a derived passage."""
    response = await client.get(f"{BASE_URL}/passages/{passage_id}/sources")
    response.raise_for_status()
    return response.json()


async def get_passage_derived(client: httpx.AsyncClient, passage_id: str) -> dict:
    """Get passages derived from this passage."""
    response = await client.get(f"{BASE_URL}/passages/{passage_id}/derived")
    response.raise_for_status()
    return response.json()


async def get_passage_provenance(
    client: httpx.AsyncClient, passage_id: str, max_depth: int = 10
) -> dict:
    """Get full provenance chain for a passage."""
    response = await client.get(
        f"{BASE_URL}/passages/{passage_id}/provenance",
        params={"max_depth": max_depth},
    )
    response.raise_for_status()
    return response.json()


async def search_passages(
    client: httpx.AsyncClient,
    query: str,
    passage_type: str | None = None,
) -> list[dict]:
    """Search for passages."""
    params = {"query": query, "limit": 10}
    response = await client.get(
        f"{BASE_URL}/passages/search",
        params=params,
        headers={"X-Agent-ID": AGENT_ID},
    )
    response.raise_for_status()
    return response.json()["results"]


async def main():
    print("=" * 60)
    print("KP3 Demo: Provenance - Derivation Chain Tracking")
    print("=" * 60)
    print()

    async with httpx.AsyncClient(timeout=30.0) as client:
        # Step 1: Explain the concept
        print("Step 1: Understanding Provenance")
        print("-" * 40)
        print()
        print("  Provenance tracks derivation relationships:")
        print()
        print("    source_1 ─┐")
        print("    source_2 ─┼──> derived_passage")
        print("    source_3 ─┘")
        print()
        print("  When processing runs create outputs, KP3 records")
        print("  which inputs (sources) produced which outputs (derived).")
        print()

        # Step 2: Search for existing derived passages
        print("Step 2: Finding Derived Passages")
        print("-" * 40)

        # Look for summaries or processed passages
        print("\n  Searching for summary passages...")
        summaries = await search_passages(client, "summary")

        derived_passage = None
        for result in summaries:
            # Try to get sources for this passage
            try:
                sources_response = await get_passage_sources(client, result["id"])
                if sources_response["count"] > 0:
                    derived_passage = result
                    print(f"\n  Found derived passage: {result['id'][:8]}...")
                    print(f"    Type: {result['passage_type']}")
                    print(f"    Has {sources_response['count']} sources")
                    break
            except httpx.HTTPStatusError:
                continue

        if not derived_passage:
            print("\n  No derived passages found in the database.")
            print("  (Run a processing pipeline to create derivation links)")
            print()
            print("  Showing API usage with hypothetical passage IDs...")

        print()

        # Step 3: Get sources for a derived passage
        print("Step 3: Getting Source Passages")
        print("-" * 40)

        if derived_passage:
            sources_response = await get_passage_sources(client, derived_passage["id"])
            print(f"\n  Sources for passage {derived_passage['id'][:8]}...:")
            print(f"  Total: {sources_response['count']}")
            for src in sources_response["sources"][:5]:
                content = src["content"]
                snippet = content[:50] + "..." if len(content) > 50 else content
                print(f"    - {src['id'][:8]}... [{src['passage_type']}]")
                print(f"      \"{snippet}\"")
        else:
            print("\n  Example API call:")
            print('    GET /passages/{passage_id}/sources')
            print()
            print("  Returns:")
            print("    {")
            print('      "passage_id": "abc123...",')
            print('      "sources": [')
            print('        {"id": "src1...", "content": "...", "passage_type": "..."},')
            print('        {"id": "src2...", "content": "...", "passage_type": "..."}')
            print("      ],")
            print('      "count": 2')
            print("    }")

        print()

        # Step 4: Get derived passages
        print("Step 4: Getting Derived Passages")
        print("-" * 40)

        if derived_passage:
            # Get what the sources themselves have produced
            sources_response = await get_passage_sources(client, derived_passage["id"])
            if sources_response["sources"]:
                first_source = sources_response["sources"][0]
                derived_response = await get_passage_derived(client, first_source["id"])
                print(f"\n  Derived from source {first_source['id'][:8]}...:")
                print(f"  Total: {derived_response['count']}")
                for drv in derived_response["derived"][:5]:
                    snippet = (
                        drv["content"][:50] + "..." if len(drv["content"]) > 50 else drv["content"]
                    )
                    print(f"    - {drv['id'][:8]}... [{drv['passage_type']}]")
                    print(f"      \"{snippet}\"")
        else:
            print("\n  Example API call:")
            print('    GET /passages/{passage_id}/derived')
            print()
            print("  Returns passages that used this one as input.")

        print()

        # Step 5: Get full provenance chain
        print("Step 5: Getting Full Provenance Chain")
        print("-" * 40)

        if derived_passage:
            provenance = await get_passage_provenance(client, derived_passage["id"])
            print(f"\n  Provenance chain for {derived_passage['id'][:8]}...:")
            print(f"  Total entries: {provenance['count']}")
            print()
            print("  Chain entries (depth = levels from starting passage):")
            for entry in provenance["chain"][:10]:
                run = entry["processing_run_id"]
                run_id = run[:8] + "..." if run else "N/A"
                src = entry["source_passage_id"][:8]
                drv = entry["derived_passage_id"][:8]
                print(f"    Depth {entry['depth']}:")
                print(f"      {src}... -> {drv}...")
                print(f"      Run: {run_id}")
        else:
            print("\n  Example API call:")
            print('    GET /passages/{passage_id}/provenance?max_depth=10')
            print()
            print("  Returns the recursive derivation chain:")
            print()
            print("    depth=1: direct sources")
            print("    depth=2: sources of sources")
            print("    depth=N: N levels deep")

        print()

        # Step 6: Visualize a hypothetical chain
        print("Step 6: Visualizing Derivation Chains")
        print("-" * 40)
        print()
        print("  Example: Hierarchical Summarization Chain")
        print()
        print("  daily_entry_1 ─┐")
        print("  daily_entry_2 ─┼──> weekly_summary_1 ─┐")
        print("  daily_entry_3 ─┘                      │")
        print("                                        ├──> monthly_summary")
        print("  daily_entry_4 ─┐                      │")
        print("  daily_entry_5 ─┼──> weekly_summary_2 ─┘")
        print("  daily_entry_6 ─┘")
        print()
        print("  Starting from monthly_summary:")
        print("    depth=1: weekly_summary_1, weekly_summary_2")
        print("    depth=2: daily_entry_1..6")

        print()
        print("=" * 60)
        print("Demo complete!")
        print()
        print("Key takeaways:")
        print("  - /sources: Get inputs that created a passage")
        print("  - /derived: Get outputs created from a passage")
        print("  - /provenance: Get full recursive chain")
        print("  - Provenance is created by processing runs")
        print("  - Use for traceability, debugging, citations")
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
