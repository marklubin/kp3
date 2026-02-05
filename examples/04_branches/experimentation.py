#!/usr/bin/env python3
"""
Example: Branches - Experimentation Workflow
============================================

Demonstrates KP3's Branches API for safe experimentation with world models.
Branches group the 3 world model refs (human/persona/world) as a unit,
allowing you to fork, experiment, and promote without affecting production.

Features showcased:
- Create branches (groups of refs)
- Fork from existing branch
- Work on experiment branch without side effects (hooks disabled)
- Promote winning branch to HEAD

Use Cases:
- A/B test different world model configurations
- Safe experimentation before promoting to production
- Compare alternative processing strategies

Usage:
    docker compose exec kp3-service uv run python examples/04_branches/experimentation.py
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
    passage_type: str = "world_model",
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


async def create_branch(
    client: httpx.AsyncClient,
    ref_prefix: str,
    branch_name: str,
    *,
    description: str | None = None,
    is_main: bool = False,
) -> dict:
    """Create a new branch."""
    response = await client.post(
        f"{BASE_URL}/branches",
        json={
            "ref_prefix": ref_prefix,
            "branch_name": branch_name,
            "description": description,
            "is_main": is_main,
        },
    )
    response.raise_for_status()
    return response.json()


async def get_branch(client: httpx.AsyncClient, name: str) -> dict | None:
    """Get a branch by full name."""
    response = await client.get(f"{BASE_URL}/branches/{name}")
    if response.status_code == 404:
        return None
    response.raise_for_status()
    return response.json()


async def list_branches(client: httpx.AsyncClient, ref_prefix: str | None = None) -> list[dict]:
    """List branches, optionally filtered by prefix."""
    params = {}
    if ref_prefix:
        params["ref_prefix"] = ref_prefix
    response = await client.get(f"{BASE_URL}/branches", params=params)
    response.raise_for_status()
    return response.json()["branches"]


async def fork_branch(
    client: httpx.AsyncClient,
    source_name: str,
    new_branch_name: str,
    description: str | None = None,
) -> dict:
    """Fork a branch to a new branch."""
    response = await client.post(
        f"{BASE_URL}/branches/{source_name}/fork",
        json={"new_branch_name": new_branch_name, "description": description},
    )
    response.raise_for_status()
    return response.json()


async def promote_branch(
    client: httpx.AsyncClient,
    source_name: str,
    target_branch: str = "HEAD",
) -> dict:
    """Promote a branch to another branch."""
    response = await client.post(
        f"{BASE_URL}/branches/{source_name}/promote",
        json={"target_branch": target_branch},
    )
    response.raise_for_status()
    return response.json()


async def delete_branch(client: httpx.AsyncClient, name: str, delete_refs: bool = False) -> bool:
    """Delete a branch."""
    response = await client.delete(
        f"{BASE_URL}/branches/{name}",
        params={"delete_refs": delete_refs},
    )
    return response.status_code == 204


async def set_ref(client: httpx.AsyncClient, name: str, passage_id: str) -> dict:
    """Set a ref to point to a passage."""
    response = await client.put(
        f"{BASE_URL}/refs/{name}",
        json={"passage_id": passage_id},
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


async def main():
    print("=" * 60)
    print("KP3 Demo: Branches - Experimentation Workflow")
    print("=" * 60)
    print()

    async with httpx.AsyncClient(timeout=30.0) as client:
        prefix = "demo-entity"

        # Step 1: Create world model passages
        print("Step 1: Creating world model passages...")
        print("-" * 40)

        passages = {}
        world_model_data = {
            "human_v1": "Human model v1: User prefers concise responses.",
            "human_v2": "Human model v2: User prefers detailed, thorough responses with examples.",
            "persona_v1": "Persona v1: Helpful assistant with friendly tone.",
            "persona_v2": "Persona v2: Expert consultant with professional tone.",
            "world_v1": "World model v1: Basic context about user's projects.",
            "world_v2": "World model v2: Enhanced context with user preferences and history.",
        }

        for key, content in world_model_data.items():
            try:
                passage = await create_passage(client, content)
                passages[key] = passage
                print(f"  Created {key}: {passage['id'][:8]}...")
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 409:
                    print(f"  {key} already exists, skipping...")
                else:
                    raise

        if len(passages) < 6:
            print("\n  Note: Some passages already existed.")
            print("  Demo may show partial results.")

        print()

        # Step 2: Create HEAD branch (production)
        print("Step 2: Creating HEAD branch (production)...")
        print("-" * 40)

        try:
            head_branch = await create_branch(
                client,
                ref_prefix=prefix,
                branch_name="HEAD",
                description="Production world model",
                is_main=True,
            )
            print(f"  Created: {head_branch['name']}")
            print(f"  Hooks enabled: {head_branch['hooks_enabled']}")
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 409:
                print(f"  Branch '{prefix}/HEAD' already exists")
                head_branch = await get_branch(client, f"{prefix}/HEAD")
            else:
                raise

        print()

        # Step 3: Initialize HEAD refs with v1 models
        print("Step 3: Initializing HEAD with v1 models...")
        print("-" * 40)

        if passages and head_branch:
            if "human_v1" in passages:
                await set_ref(client, head_branch["human_ref"], passages["human_v1"]["id"])
                print(f"  Set {head_branch['human_ref']}")
            if "persona_v1" in passages:
                await set_ref(client, head_branch["persona_ref"], passages["persona_v1"]["id"])
                print(f"  Set {head_branch['persona_ref']}")
            if "world_v1" in passages:
                await set_ref(client, head_branch["world_ref"], passages["world_v1"]["id"])
                print(f"  Set {head_branch['world_ref']}")

        print()

        # Step 4: Fork to experiment branch
        print("Step 4: Forking to experiment branch...")
        print("-" * 40)

        try:
            exp_branch = await fork_branch(
                client,
                f"{prefix}/HEAD",
                "experiment-1",
                description="Testing v2 models",
            )
            print(f"  Forked: {exp_branch['name']}")
            print(f"  Hooks enabled: {exp_branch['hooks_enabled']} (safe for experiments)")
            print(f"  Parent: {exp_branch['parent_branch_id']}")
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 409:
                print(f"  Branch '{prefix}/experiment-1' already exists")
                exp_branch = await get_branch(client, f"{prefix}/experiment-1")
            else:
                raise

        print()

        # Step 5: Update experiment branch with v2 models
        print("Step 5: Updating experiment branch with v2 models...")
        print("-" * 40)

        if passages and exp_branch:
            if "human_v2" in passages:
                await set_ref(client, exp_branch["human_ref"], passages["human_v2"]["id"])
                print(f"  Updated {exp_branch['human_ref']}")
            if "persona_v2" in passages:
                await set_ref(client, exp_branch["persona_ref"], passages["persona_v2"]["id"])
                print(f"  Updated {exp_branch['persona_ref']}")
            if "world_v2" in passages:
                await set_ref(client, exp_branch["world_ref"], passages["world_v2"]["id"])
                print(f"  Updated {exp_branch['world_ref']}")
            print("  (No hooks fired - experiment branch has hooks_enabled=False)")

        print()

        # Step 6: Compare branches
        print("Step 6: Comparing HEAD vs experiment...")
        print("-" * 40)

        if head_branch and exp_branch:
            print("\n  HEAD (production):")
            head_human = await get_ref(client, head_branch["human_ref"])
            if head_human:
                print(f"    human: {head_human['passage_id'][:8]}...")

            print("\n  experiment-1:")
            exp_human = await get_ref(client, exp_branch["human_ref"])
            if exp_human:
                print(f"    human: {exp_human['passage_id'][:8]}...")

        print()

        # Step 7: List all branches
        print("Step 7: Listing all branches for prefix...")
        print("-" * 40)

        branches = await list_branches(client, ref_prefix=prefix)
        print(f"  Found {len(branches)} branches:")
        for b in branches:
            hooks = "hooks=on" if b["hooks_enabled"] else "hooks=off"
            main = " (main)" if b["is_main"] else ""
            print(f"    - {b['name']}{main} [{hooks}]")

        print()

        # Step 8: Promote experiment to HEAD
        print("Step 8: Promoting experiment to HEAD...")
        print("-" * 40)

        if exp_branch:
            try:
                result = await promote_branch(client, f"{prefix}/experiment-1", "HEAD")
                print(f"  Promoted to: {result['name']}")
                print("  (Hooks would fire on HEAD - production updated)")
            except httpx.HTTPStatusError as e:
                print(f"  Promotion failed: {e.response.text}")

        print()

        # Step 9: Verify HEAD was updated
        print("Step 9: Verifying HEAD was updated...")
        print("-" * 40)

        if head_branch:
            head_human = await get_ref(client, head_branch["human_ref"])
            if head_human and exp_branch:
                exp_human = await get_ref(client, exp_branch["human_ref"])
                if exp_human and head_human["passage_id"] == exp_human["passage_id"]:
                    print("  HEAD now points to same passages as experiment")
                    print(f"    human: {head_human['passage_id'][:8]}...")

        print()

        # Step 10: Clean up experiment branch
        print("Step 10: Cleaning up experiment branch...")
        print("-" * 40)

        deleted = await delete_branch(client, f"{prefix}/experiment-1", delete_refs=True)
        if deleted:
            print(f"  Deleted: {prefix}/experiment-1")
            print("  (Refs also deleted since delete_refs=True)")

        print()
        print("=" * 60)
        print("Demo complete!")
        print()
        print("Key takeaways:")
        print("  - Branches group 3 refs (human/persona/world) as a unit")
        print("  - HEAD branch is production with hooks_enabled=True")
        print("  - Experiment branches have hooks_enabled=False (safe)")
        print("  - Fork copies current refs to a new branch")
        print("  - Promote copies refs from source to target")
        print("  - Use for A/B testing world model configurations")
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
