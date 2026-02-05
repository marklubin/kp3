"""REST API router for passage search and management."""

import logging
from datetime import datetime
from uuid import UUID

from fastapi import APIRouter, Header, HTTPException, Query
from sqlalchemy.exc import IntegrityError

from kp3.db.engine import async_session
from kp3.processors.embedding import generate_embedding
from kp3.schemas.api import (
    BranchCreateRequest,
    BranchForkRequest,
    BranchListResponse,
    BranchPromoteRequest,
    BranchResponse,
    PassageCreate,
    PassageCreateResponse,
    PassageResponse,
    PassageResult,
    PassageTagsRequest,
    PassageTagsResponse,
    PromptResponse,
    ProvenanceChainEntry,
    ProvenanceChainResponse,
    ProvenanceDerivedResponse,
    ProvenancePassage,
    ProvenanceSourcesResponse,
    RefHistoryEntry,
    RefHistoryResponse,
    RefListResponse,
    RefResponse,
    RefSetRequest,
    RunCreateRequest,
    RunListResponse,
    RunResponse,
    SearchResponse,
    TagCreate,
    TagListResponse,
    TagResponse,
)
from kp3.schemas.scope import (
    MemoryScopeCreate,
    MemoryScopeListResponse,
    MemoryScopeResponse,
    ScopeAddRequest,
    ScopedPassageCreate,
    ScopedPassageCreateResponse,
    ScopedSearchResponse,
    ScopeHistoryEntry,
    ScopeHistoryResponse,
    ScopeModifyResponse,
    ScopeRemoveRequest,
    ScopeRevertRequest,
    ScopeRevertResponse,
)
from kp3.services.branches import (
    BranchError,
    BranchExistsError,
    BranchNotFoundError,
    create_branch,
    delete_branch,
    fork_branch,
    get_branch_by_name,
    list_branches,
    promote_branch,
)
from kp3.services.derivations import get_derived, get_full_provenance, get_sources
from kp3.services.passages import create_passage, get_passage
from kp3.services.prompts import get_active_prompt
from kp3.services.refs import delete_ref, get_ref_history, list_refs, set_ref
from kp3.services.runs import create_run, get_run, list_runs
from kp3.services.scopes import (
    add_passages_to_scope,
    add_refs_to_scope,
    create_passage_in_scope,
    create_scope,
    delete_scope,
    get_current_version,
    get_scope,
    get_scope_history,
    list_scopes,
    remove_from_scope,
    resolve_scope,
    revert_scope,
)
from kp3.services.search import SearchMode, SearchType, search_passages
from kp3.services.tags import (
    attach_tags_to_passage,
    create_tag,
    delete_tag,
    detach_tags_from_passage,
    get_passage_tags,
    get_tag,
    list_tags,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["passages"])


@router.get("/passages/search", response_model=SearchResponse)
async def search(
    query: str = Query(min_length=1, description="Search query text"),
    search_type: SearchType = Query(
        default="content",
        description="Search type: content (search passage text) or tags (search by tag names)",
    ),
    mode: SearchMode = Query(
        default="hybrid",
        description="Search mode: fts, semantic, or hybrid",
    ),
    limit: int = Query(default=5, ge=1, le=50, description="Maximum results"),
    period_start_after: datetime | None = Query(
        default=None, description="Filter: period_start >= this value"
    ),
    period_end_before: datetime | None = Query(
        default=None, description="Filter: period_end <= this value"
    ),
    created_after: datetime | None = Query(
        default=None, description="Filter: created_at >= this value"
    ),
    created_before: datetime | None = Query(
        default=None, description="Filter: created_at <= this value"
    ),
    x_agent_id: str = Header(..., alias="X-Agent-ID"),
) -> SearchResponse:
    """Search passages using full-text, semantic, or hybrid search.

    **Search types:**
    - **content**: Search passage content directly (default)
    - **tags**: Search by tag names/descriptions, return passages with matching tags

    **Search modes (apply to both types):**
    - **fts**: Full-text search using PostgreSQL tsvector
    - **semantic**: Vector similarity search using embeddings
    - **hybrid**: Reciprocal Rank Fusion combining both methods (default)

    **Timestamp filters** can be applied to both search types.

    Requires X-Agent-ID header. Only returns passages for that specific agent.
    """
    async with async_session() as session:
        results = await search_passages(
            session,
            query,
            search_type=search_type,
            mode=mode,
            limit=limit,
            agent_id=x_agent_id,
            period_start_after=period_start_after,
            period_end_before=period_end_before,
            created_after=created_after,
            created_before=created_before,
        )

    return SearchResponse(
        query=query,
        mode=mode,
        search_type=search_type,
        results=[
            PassageResult(
                id=r.id,
                content=r.content,
                passage_type=r.passage_type,
                score=r.score,
            )
            for r in results
        ],
        count=len(results),
    )


@router.post("/passages", response_model=PassageCreateResponse)
async def create_new_passage(
    payload: PassageCreate,
    x_agent_id: str = Header(..., alias="X-Agent-ID"),
) -> PassageCreateResponse:
    """Create a new passage.

    All passages are automatically embedded for semantic search.
    Duplicate content (by SHA256 hash) will be rejected.

    Requires X-Agent-ID header to scope passage to an agent.
    """
    # Generate embedding (required for all passages)
    try:
        embedding = await generate_embedding(payload.content)
        logger.info(
            "Generated embedding for %s passage (%d dims)",
            payload.passage_type,
            len(embedding),
        )
    except Exception as e:
        logger.exception("Failed to generate embedding")
        raise HTTPException(status_code=502, detail=f"Embedding generation failed: {e}") from e

    async with async_session() as session:
        try:
            passage = await create_passage(
                session,
                content=payload.content,
                passage_type=payload.passage_type,
                metadata=payload.metadata,
                period_start=payload.period_start,
                period_end=payload.period_end,
                embedding_openai=embedding,
                agent_id=x_agent_id,
            )
            await session.commit()
        except IntegrityError as e:
            # Duplicate content - return 409 with info about finding the existing passage
            await session.rollback()
            if "content_hash" in str(e):
                raise HTTPException(
                    status_code=409,
                    detail="Passage with identical content already exists. Use search to find it.",
                ) from e
            raise

        return PassageCreateResponse(
            id=passage.id,
            content=passage.content,
            passage_type=passage.passage_type,
        )


@router.get("/passages/{passage_id}", response_model=PassageResponse)
async def get_passage_by_id(
    passage_id: UUID,
    x_agent_id: str = Header(..., alias="X-Agent-ID"),
) -> PassageResponse:
    """Get a passage by ID.

    Returns the full passage including metadata and timestamps.

    Requires X-Agent-ID header. Only returns the passage if it belongs to that agent.
    """
    async with async_session() as session:
        passage = await get_passage(session, passage_id)

    if not passage:
        raise HTTPException(status_code=404, detail=f"Passage '{passage_id}' not found")

    # Verify agent ownership
    if passage.agent_id != x_agent_id:
        raise HTTPException(status_code=404, detail=f"Passage '{passage_id}' not found")

    return PassageResponse(
        id=passage.id,
        content=passage.content,
        passage_type=passage.passage_type,
        metadata=passage.metadata_ or {},
        period_start=passage.period_start,
        period_end=passage.period_end,
        created_at=passage.created_at,
    )


@router.get("/passages/{passage_id}/sources", response_model=ProvenanceSourcesResponse)
async def get_passage_sources(passage_id: UUID) -> ProvenanceSourcesResponse:
    """Get the immediate source passages for a derived passage.

    Returns passages that were used as inputs to create this passage.
    """
    async with async_session() as session:
        passage = await get_passage(session, passage_id)
        if not passage:
            raise HTTPException(status_code=404, detail=f"Passage '{passage_id}' not found")

        sources = await get_sources(session, passage_id)

        return ProvenanceSourcesResponse(
            passage_id=passage_id,
            sources=[
                ProvenancePassage(
                    id=p.id,
                    content=p.content,
                    passage_type=p.passage_type,
                    created_at=p.created_at,
                )
                for p in sources
            ],
            count=len(sources),
        )


@router.get("/passages/{passage_id}/derived", response_model=ProvenanceDerivedResponse)
async def get_passage_derived(passage_id: UUID) -> ProvenanceDerivedResponse:
    """Get passages that were derived from this passage.

    Returns passages that used this passage as an input.
    """
    async with async_session() as session:
        passage = await get_passage(session, passage_id)
        if not passage:
            raise HTTPException(status_code=404, detail=f"Passage '{passage_id}' not found")

        derived = await get_derived(session, passage_id)

        return ProvenanceDerivedResponse(
            passage_id=passage_id,
            derived=[
                ProvenancePassage(
                    id=p.id,
                    content=p.content,
                    passage_type=p.passage_type,
                    created_at=p.created_at,
                )
                for p in derived
            ],
            count=len(derived),
        )


@router.get("/passages/{passage_id}/provenance", response_model=ProvenanceChainResponse)
async def get_passage_provenance(
    passage_id: UUID,
    max_depth: int = Query(default=10, ge=1, le=100, description="Maximum chain depth"),
) -> ProvenanceChainResponse:
    """Get the full provenance chain for a passage.

    Recursively finds all source passages up to max_depth.
    """
    async with async_session() as session:
        passage = await get_passage(session, passage_id)
        if not passage:
            raise HTTPException(status_code=404, detail=f"Passage '{passage_id}' not found")

        chain = await get_full_provenance(session, passage_id, max_depth=max_depth)

        return ProvenanceChainResponse(
            passage_id=passage_id,
            chain=[
                ProvenanceChainEntry(
                    derived_passage_id=UUID(str(c["derived_passage_id"])),
                    source_passage_id=UUID(str(c["source_passage_id"])),
                    processing_run_id=UUID(str(c["processing_run_id"]))
                    if c["processing_run_id"]
                    else None,
                    depth=int(c["depth"]),
                )
                for c in chain
            ],
            count=len(chain),
        )


@router.get("/prompts/{name}", response_model=PromptResponse)
async def get_prompt_by_name(name: str) -> PromptResponse:
    """Get the active prompt by name.

    Returns the currently active version of the named prompt.
    """
    async with async_session() as session:
        prompt = await get_active_prompt(session, name)

    if not prompt:
        raise HTTPException(status_code=404, detail=f"Prompt '{name}' not found")

    return PromptResponse(
        id=str(prompt.id),
        name=prompt.name,
        version=prompt.version,
        system_prompt=prompt.system_prompt,
        user_prompt_template=prompt.user_prompt_template,
        field_descriptions=prompt.field_descriptions or {},
    )


# =============================================================================
# Tag endpoints
# =============================================================================


@router.post("/tags", response_model=TagResponse, status_code=201)
async def create_new_tag(
    payload: TagCreate,
    x_agent_id: str = Header(..., alias="X-Agent-ID"),
) -> TagResponse:
    """Create a new tag.

    Tags are automatically embedded for semantic search.
    Tags are deduplicated by canonical name (lowercase, normalized whitespace).

    Requires X-Agent-ID header to scope tag to an agent.
    """
    async with async_session() as session:
        try:
            tag = await create_tag(
                session,
                name=payload.name,
                agent_id=x_agent_id,
                description=payload.description,
            )
            await session.commit()
        except ValueError as e:
            raise HTTPException(status_code=409, detail=str(e)) from e

        return TagResponse(
            id=tag.id,
            name=tag.name,
            description=tag.description,
            passage_count=tag.passage_count,
            created_at=tag.created_at,
            updated_at=tag.updated_at,
        )


@router.get("/tags", response_model=TagListResponse)
async def list_all_tags(
    limit: int = Query(default=100, ge=1, le=1000, description="Maximum results"),
    offset: int = Query(default=0, ge=0, description="Offset for pagination"),
    order_by: str = Query(
        default="name",
        description="Order by: name, created_at, passage_count",
    ),
    x_agent_id: str = Header(..., alias="X-Agent-ID"),
) -> TagListResponse:
    """List all tags for the agent.

    Requires X-Agent-ID header. Only returns tags for that specific agent.
    """
    async with async_session() as session:
        tags = await list_tags(
            session,
            agent_id=x_agent_id,
            limit=limit,
            offset=offset,
            order_by=order_by,
        )

        return TagListResponse(
            tags=[
                TagResponse(
                    id=t.id,
                    name=t.name,
                    description=t.description,
                    passage_count=t.passage_count,
                    created_at=t.created_at,
                    updated_at=t.updated_at,
                )
                for t in tags
            ],
            count=len(tags),
        )


@router.get("/tags/{tag_id}", response_model=TagResponse)
async def get_tag_by_id(
    tag_id: UUID,
    x_agent_id: str = Header(..., alias="X-Agent-ID"),
) -> TagResponse:
    """Get a tag by ID.

    Requires X-Agent-ID header.
    """
    async with async_session() as session:
        tag = await get_tag(session, tag_id, x_agent_id)

    if not tag:
        raise HTTPException(status_code=404, detail=f"Tag '{tag_id}' not found")

    return TagResponse(
        id=tag.id,
        name=tag.name,
        description=tag.description,
        passage_count=tag.passage_count,
        created_at=tag.created_at,
        updated_at=tag.updated_at,
    )


@router.delete("/tags/{tag_id}", status_code=204)
async def delete_tag_by_id(
    tag_id: UUID,
    x_agent_id: str = Header(..., alias="X-Agent-ID"),
) -> None:
    """Delete a tag.

    This will also remove the tag from all passages it was attached to.

    Requires X-Agent-ID header.
    """
    async with async_session() as session:
        deleted = await delete_tag(session, tag_id, x_agent_id)
        if not deleted:
            raise HTTPException(status_code=404, detail=f"Tag '{tag_id}' not found")
        await session.commit()


# =============================================================================
# Passage-tag endpoints
# =============================================================================


@router.get("/passages/{passage_id}/tags", response_model=PassageTagsResponse)
async def get_tags_for_passage(
    passage_id: UUID,
    x_agent_id: str = Header(..., alias="X-Agent-ID"),
) -> PassageTagsResponse:
    """Get all tags attached to a passage.

    Requires X-Agent-ID header.
    """
    async with async_session() as session:
        tags = await get_passage_tags(session, passage_id, x_agent_id)

        return PassageTagsResponse(
            passage_id=passage_id,
            tags=[
                TagResponse(
                    id=t.id,
                    name=t.name,
                    description=t.description,
                    passage_count=t.passage_count,
                    created_at=t.created_at,
                    updated_at=t.updated_at,
                )
                for t in tags
            ],
        )


@router.post("/passages/{passage_id}/tags", response_model=PassageTagsResponse)
async def attach_tags_to_passage_endpoint(
    passage_id: UUID,
    payload: PassageTagsRequest,
    x_agent_id: str = Header(..., alias="X-Agent-ID"),
) -> PassageTagsResponse:
    """Attach tags to a passage.

    Tags that are already attached will be silently ignored.

    Requires X-Agent-ID header.
    """
    async with async_session() as session:
        await attach_tags_to_passage(session, passage_id, payload.tag_ids, x_agent_id)
        await session.commit()

        tags = await get_passage_tags(session, passage_id, x_agent_id)

        return PassageTagsResponse(
            passage_id=passage_id,
            tags=[
                TagResponse(
                    id=t.id,
                    name=t.name,
                    description=t.description,
                    passage_count=t.passage_count,
                    created_at=t.created_at,
                    updated_at=t.updated_at,
                )
                for t in tags
            ],
        )


@router.delete("/passages/{passage_id}/tags", response_model=PassageTagsResponse)
async def detach_tags_from_passage_endpoint(
    passage_id: UUID,
    payload: PassageTagsRequest,
    x_agent_id: str = Header(..., alias="X-Agent-ID"),
) -> PassageTagsResponse:
    """Detach tags from a passage.

    Tags that are not attached will be silently ignored.

    Requires X-Agent-ID header.
    """
    async with async_session() as session:
        await detach_tags_from_passage(session, passage_id, payload.tag_ids, x_agent_id)
        await session.commit()

        tags = await get_passage_tags(session, passage_id, x_agent_id)

        return PassageTagsResponse(
            passage_id=passage_id,
            tags=[
                TagResponse(
                    id=t.id,
                    name=t.name,
                    description=t.description,
                    passage_count=t.passage_count,
                    created_at=t.created_at,
                    updated_at=t.updated_at,
                )
                for t in tags
            ],
        )


# =============================================================================
# Ref endpoints
# =============================================================================


@router.get("/refs", response_model=RefListResponse)
async def list_refs_endpoint(
    prefix: str | None = Query(default=None, description="Filter refs by prefix"),
    limit: int = Query(default=100, ge=1, le=1000, description="Maximum results"),
) -> RefListResponse:
    """List all refs, optionally filtered by prefix.

    Refs are mutable pointers to passages, similar to git refs.
    Use prefix to filter (e.g., "world/human/" for all human world model refs).
    """
    async with async_session() as session:
        refs = await list_refs(session, prefix=prefix, limit=limit)

        return RefListResponse(
            refs=[
                RefResponse(
                    name=r["name"],
                    passage_id=r["passage_id"],
                    updated_at=r["updated_at"],
                    metadata=r["metadata"] or {},
                )
                for r in refs
            ],
            count=len(refs),
        )


# NOTE: /history endpoint must come BEFORE the generic {name:path} endpoint
# because FastAPI matches routes in order and {name:path} would capture "name/history"
@router.get("/refs/{name:path}/history", response_model=RefHistoryResponse)
async def get_ref_history_endpoint(
    name: str,
    limit: int = Query(default=100, ge=1, le=1000, description="Maximum entries"),
) -> RefHistoryResponse:
    """Get the history of changes for a ref.

    Returns history entries ordered by most recent first.
    """
    async with async_session() as session:
        history = await get_ref_history(session, name, limit=limit)

        if not history:
            # Check if ref exists (might have no history if just created)
            refs = await list_refs(session, prefix=name, limit=1)
            ref = next((r for r in refs if r["name"] == name), None)
            if not ref:
                raise HTTPException(status_code=404, detail=f"Ref '{name}' not found")

        return RefHistoryResponse(
            history=[
                RefHistoryEntry(
                    id=h["id"],
                    ref_name=h["ref_name"],
                    passage_id=h["passage_id"],
                    previous_passage_id=h["previous_passage_id"],
                    changed_at=h["changed_at"],
                    metadata=h["metadata"] or {},
                )
                for h in history
            ],
            count=len(history),
        )


@router.get("/refs/{name:path}", response_model=RefResponse)
async def get_ref_endpoint(name: str) -> RefResponse:
    """Get a ref by name.

    Returns the passage ID that the ref currently points to.
    The name can include slashes (e.g., "world/human/HEAD").
    """
    async with async_session() as session:
        refs = await list_refs(session, prefix=name, limit=1)

        # Find exact match (prefix search might return partial matches)
        ref = next((r for r in refs if r["name"] == name), None)
        if not ref:
            raise HTTPException(status_code=404, detail=f"Ref '{name}' not found")

        return RefResponse(
            name=ref["name"],
            passage_id=ref["passage_id"],
            updated_at=ref["updated_at"],
            metadata=ref["metadata"] or {},
        )


@router.put("/refs/{name:path}", response_model=RefResponse, status_code=200)
async def set_ref_endpoint(name: str, payload: RefSetRequest) -> RefResponse:
    """Set a ref to point to a passage.

    Creates the ref if it doesn't exist, updates it if it does.
    Records history for auditing.
    """
    async with async_session() as session:
        # Verify passage exists
        passage = await get_passage(session, payload.passage_id)
        if not passage:
            raise HTTPException(
                status_code=404, detail=f"Passage '{payload.passage_id}' not found"
            )

        ref = await set_ref(
            session, name, payload.passage_id, metadata=payload.metadata, fire_hooks=True
        )
        await session.commit()

        return RefResponse(
            name=ref.name,
            passage_id=ref.passage_id,
            updated_at=ref.updated_at,
            metadata=ref.metadata_ or {},
        )


@router.delete("/refs/{name:path}", status_code=204)
async def delete_ref_endpoint(name: str) -> None:
    """Delete a ref.

    Note: This does not delete the passage it points to or the ref history.
    """
    async with async_session() as session:
        deleted = await delete_ref(session, name)
        if not deleted:
            raise HTTPException(status_code=404, detail=f"Ref '{name}' not found")
        await session.commit()


# =============================================================================
# Branch endpoints
# =============================================================================


@router.get("/branches", response_model=BranchListResponse)
async def list_branches_endpoint(
    ref_prefix: str | None = Query(default=None, description="Filter by ref prefix"),
    limit: int = Query(default=100, ge=1, le=1000, description="Maximum results"),
) -> BranchListResponse:
    """List all branches, optionally filtered by ref prefix.

    Branches group the 3 world model refs (human/persona/world) as a unit.
    """
    async with async_session() as session:
        branches = await list_branches(session, ref_prefix=ref_prefix, limit=limit)

        return BranchListResponse(
            branches=[
                BranchResponse(
                    id=b.id,
                    name=b.name,
                    ref_prefix=b.ref_prefix,
                    branch_name=b.branch_name,
                    human_ref=b.human_ref,
                    persona_ref=b.persona_ref,
                    world_ref=b.world_ref,
                    parent_branch_id=b.parent_branch_id,
                    is_main=b.is_main,
                    hooks_enabled=b.hooks_enabled,
                    description=b.description,
                    created_at=b.created_at,
                )
                for b in branches
            ],
            count=len(branches),
        )


@router.post("/branches", response_model=BranchResponse, status_code=201)
async def create_branch_endpoint(payload: BranchCreateRequest) -> BranchResponse:
    """Create a new branch.

    Creates a new branch with empty refs. Use fork endpoint to derive from an existing branch.
    """
    async with async_session() as session:
        try:
            branch = await create_branch(
                session,
                ref_prefix=payload.ref_prefix,
                branch_name=payload.branch_name,
                description=payload.description,
                is_main=payload.is_main,
                hooks_enabled=payload.hooks_enabled,
            )
            await session.commit()
        except BranchExistsError as e:
            raise HTTPException(status_code=409, detail=str(e)) from e

        return BranchResponse(
            id=branch.id,
            name=branch.name,
            ref_prefix=branch.ref_prefix,
            branch_name=branch.branch_name,
            human_ref=branch.human_ref,
            persona_ref=branch.persona_ref,
            world_ref=branch.world_ref,
            parent_branch_id=branch.parent_branch_id,
            is_main=branch.is_main,
            hooks_enabled=branch.hooks_enabled,
            description=branch.description,
            created_at=branch.created_at,
        )


@router.get("/branches/{name:path}", response_model=BranchResponse)
async def get_branch_endpoint(name: str) -> BranchResponse:
    """Get a branch by its full name (e.g., 'corindel/experiment-1')."""
    async with async_session() as session:
        branch = await get_branch_by_name(session, name)

    if not branch:
        raise HTTPException(status_code=404, detail=f"Branch '{name}' not found")

    return BranchResponse(
        id=branch.id,
        name=branch.name,
        ref_prefix=branch.ref_prefix,
        branch_name=branch.branch_name,
        human_ref=branch.human_ref,
        persona_ref=branch.persona_ref,
        world_ref=branch.world_ref,
        parent_branch_id=branch.parent_branch_id,
        is_main=branch.is_main,
        hooks_enabled=branch.hooks_enabled,
        description=branch.description,
        created_at=branch.created_at,
    )


@router.delete("/branches/{name:path}", status_code=204)
async def delete_branch_endpoint(
    name: str,
    delete_refs: bool = Query(
        default=False, description="Also delete the underlying refs"
    ),
) -> None:
    """Delete a branch.

    Cannot delete main branches. Optionally also deletes the underlying refs.
    """
    async with async_session() as session:
        branch = await get_branch_by_name(session, name)
        if not branch:
            raise HTTPException(status_code=404, detail=f"Branch '{name}' not found")

        try:
            deleted = await delete_branch(
                session,
                branch.ref_prefix,
                branch.branch_name,
                delete_refs=delete_refs,
            )
            if not deleted:
                raise HTTPException(status_code=404, detail=f"Branch '{name}' not found")
            await session.commit()
        except BranchError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e


@router.post("/branches/{name:path}/fork", response_model=BranchResponse, status_code=201)
async def fork_branch_endpoint(name: str, payload: BranchForkRequest) -> BranchResponse:
    """Fork a branch, copying its current refs to a new branch.

    The new branch starts with the same passage IDs as the source.
    """
    async with async_session() as session:
        source_branch = await get_branch_by_name(session, name)
        if not source_branch:
            raise HTTPException(status_code=404, detail=f"Branch '{name}' not found")

        try:
            new_branch = await fork_branch(
                session,
                ref_prefix=source_branch.ref_prefix,
                source_branch=source_branch.branch_name,
                new_branch_name=payload.new_branch_name,
                description=payload.description,
            )
            await session.commit()
        except BranchNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e
        except BranchExistsError as e:
            raise HTTPException(status_code=409, detail=str(e)) from e

        return BranchResponse(
            id=new_branch.id,
            name=new_branch.name,
            ref_prefix=new_branch.ref_prefix,
            branch_name=new_branch.branch_name,
            human_ref=new_branch.human_ref,
            persona_ref=new_branch.persona_ref,
            world_ref=new_branch.world_ref,
            parent_branch_id=new_branch.parent_branch_id,
            is_main=new_branch.is_main,
            hooks_enabled=new_branch.hooks_enabled,
            description=new_branch.description,
            created_at=new_branch.created_at,
        )


@router.post("/branches/{name:path}/promote", response_model=BranchResponse)
async def promote_branch_endpoint(name: str, payload: BranchPromoteRequest) -> BranchResponse:
    """Promote a branch to another branch (typically HEAD).

    Copies the current passage IDs from source refs to target refs.
    Fires hooks on the target refs if the target branch has hooks_enabled.
    """
    async with async_session() as session:
        source_branch = await get_branch_by_name(session, name)
        if not source_branch:
            raise HTTPException(status_code=404, detail=f"Branch '{name}' not found")

        try:
            target_branch = await promote_branch(
                session,
                ref_prefix=source_branch.ref_prefix,
                source_branch=source_branch.branch_name,
                target_branch=payload.target_branch,
            )
            await session.commit()
        except BranchNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e

        return BranchResponse(
            id=target_branch.id,
            name=target_branch.name,
            ref_prefix=target_branch.ref_prefix,
            branch_name=target_branch.branch_name,
            human_ref=target_branch.human_ref,
            persona_ref=target_branch.persona_ref,
            world_ref=target_branch.world_ref,
            parent_branch_id=target_branch.parent_branch_id,
            is_main=target_branch.is_main,
            hooks_enabled=target_branch.hooks_enabled,
            description=target_branch.description,
            created_at=target_branch.created_at,
        )


# =============================================================================
# Processing Run endpoints
# =============================================================================


@router.get("/runs", response_model=RunListResponse)
async def list_runs_endpoint(
    status: str | None = Query(default=None, description="Filter by status"),
    limit: int = Query(default=100, ge=1, le=1000, description="Maximum results"),
) -> RunListResponse:
    """List processing runs, optionally filtered by status.

    Status values: pending, running, completed, failed
    """
    async with async_session() as session:
        runs = await list_runs(session, status=status, limit=limit)

        return RunListResponse(
            runs=[
                RunResponse(
                    id=r.id,
                    input_sql=r.input_sql,
                    processor_type=r.processor_type,
                    processor_config=r.processor_config,
                    status=r.status,
                    total_groups=r.total_groups,
                    processed_groups=r.processed_groups,
                    output_count=r.output_count,
                    error_message=r.error_message,
                    started_at=r.started_at,
                    completed_at=r.completed_at,
                    created_at=r.created_at,
                )
                for r in runs
            ],
            count=len(runs),
        )


@router.post("/runs", response_model=RunResponse, status_code=201)
async def create_run_endpoint(payload: RunCreateRequest) -> RunResponse:
    """Create a new processing run.

    The run is created in 'pending' status. Use the CLI to execute it.
    """
    async with async_session() as session:
        run = await create_run(
            session,
            input_sql=payload.input_sql,
            processor_type=payload.processor_type,
            processor_config=payload.processor_config,
        )
        await session.commit()

        return RunResponse(
            id=run.id,
            input_sql=run.input_sql,
            processor_type=run.processor_type,
            processor_config=run.processor_config,
            status=run.status,
            total_groups=run.total_groups,
            processed_groups=run.processed_groups,
            output_count=run.output_count,
            error_message=run.error_message,
            started_at=run.started_at,
            completed_at=run.completed_at,
            created_at=run.created_at,
        )


@router.get("/runs/{run_id}", response_model=RunResponse)
async def get_run_endpoint(run_id: UUID) -> RunResponse:
    """Get a processing run by ID."""
    async with async_session() as session:
        run = await get_run(session, run_id)

    if not run:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")

    return RunResponse(
        id=run.id,
        input_sql=run.input_sql,
        processor_type=run.processor_type,
        processor_config=run.processor_config,
        status=run.status,
        total_groups=run.total_groups,
        processed_groups=run.processed_groups,
        output_count=run.output_count,
        error_message=run.error_message,
        started_at=run.started_at,
        completed_at=run.completed_at,
        created_at=run.created_at,
    )


# =============================================================================
# Memory Scope endpoints
# =============================================================================


@router.post("/scopes", response_model=MemoryScopeResponse, status_code=201)
async def create_new_scope(
    payload: MemoryScopeCreate,
    x_agent_id: str = Header(..., alias="X-Agent-ID"),
) -> MemoryScopeResponse:
    """Create a new memory scope.

    Scopes define dynamic search closures using refs and literal passage IDs.
    The scope is initialized with an empty definition.

    Requires X-Agent-ID header to scope the scope to an agent.
    """
    async with async_session() as session:
        try:
            scope = await create_scope(
                session,
                name=payload.name,
                agent_id=x_agent_id,
                description=payload.description,
            )
            await session.commit()
        except ValueError as e:
            raise HTTPException(status_code=409, detail=str(e)) from e

        return MemoryScopeResponse(
            id=scope.id,
            name=scope.name,
            agent_id=scope.agent_id,
            head_ref=scope.head_ref,
            description=scope.description,
            created_at=scope.created_at,
            updated_at=scope.updated_at,
        )


@router.get("/scopes", response_model=MemoryScopeListResponse)
async def list_all_scopes(
    limit: int = Query(default=100, ge=1, le=1000, description="Maximum results"),
    offset: int = Query(default=0, ge=0, description="Offset for pagination"),
    x_agent_id: str = Header(..., alias="X-Agent-ID"),
) -> MemoryScopeListResponse:
    """List all memory scopes for the agent.

    Requires X-Agent-ID header. Only returns scopes for that specific agent.
    """
    async with async_session() as session:
        scopes = await list_scopes(session, x_agent_id, limit=limit, offset=offset)

        return MemoryScopeListResponse(
            scopes=[
                MemoryScopeResponse(
                    id=s.id,
                    name=s.name,
                    agent_id=s.agent_id,
                    head_ref=s.head_ref,
                    description=s.description,
                    created_at=s.created_at,
                    updated_at=s.updated_at,
                )
                for s in scopes
            ],
            count=len(scopes),
        )


@router.get("/scopes/{name}", response_model=MemoryScopeResponse)
async def get_scope_by_name(
    name: str,
    x_agent_id: str = Header(..., alias="X-Agent-ID"),
) -> MemoryScopeResponse:
    """Get a scope by name.

    Requires X-Agent-ID header.
    """
    async with async_session() as session:
        scope = await get_scope(session, name, x_agent_id)

    if not scope:
        raise HTTPException(status_code=404, detail=f"Scope '{name}' not found")

    return MemoryScopeResponse(
        id=scope.id,
        name=scope.name,
        agent_id=scope.agent_id,
        head_ref=scope.head_ref,
        description=scope.description,
        created_at=scope.created_at,
        updated_at=scope.updated_at,
    )


@router.delete("/scopes/{name}", status_code=204)
async def delete_scope_by_name(
    name: str,
    x_agent_id: str = Header(..., alias="X-Agent-ID"),
) -> None:
    """Delete a scope.

    Note: This does not delete the scope definition passages or ref history.

    Requires X-Agent-ID header.
    """
    async with async_session() as session:
        deleted = await delete_scope(session, name, x_agent_id)
        if not deleted:
            raise HTTPException(status_code=404, detail=f"Scope '{name}' not found")
        await session.commit()


@router.post("/scopes/{name}/passages", response_model=ScopedPassageCreateResponse, status_code=201)
async def create_passage_in_scope_endpoint(
    name: str,
    payload: ScopedPassageCreate,
    x_agent_id: str = Header(..., alias="X-Agent-ID"),
) -> ScopedPassageCreateResponse:
    """Create a passage and add it to the scope atomically.

    The passage is automatically embedded for semantic search.

    Requires X-Agent-ID header.
    """
    async with async_session() as session:
        scope = await get_scope(session, name, x_agent_id)
        if not scope:
            raise HTTPException(status_code=404, detail=f"Scope '{name}' not found")

        # Generate embedding
        try:
            embedding = await generate_embedding(payload.content)
        except Exception as e:
            logger.exception("Failed to generate embedding")
            raise HTTPException(status_code=502, detail=f"Embedding generation failed: {e}") from e

        try:
            passage, new_version = await create_passage_in_scope(
                session,
                scope,
                content=payload.content,
                passage_type=payload.passage_type,
                metadata=payload.metadata,
                period_start=payload.period_start,
                period_end=payload.period_end,
                embedding_openai=embedding,
            )
            await session.commit()
        except IntegrityError as e:
            await session.rollback()
            if "content_hash" in str(e):
                raise HTTPException(
                    status_code=409,
                    detail="Passage with identical content already exists. Use search to find it.",
                ) from e
            raise

        return ScopedPassageCreateResponse(
            passage_id=passage.id,
            content=passage.content,
            passage_type=passage.passage_type,
            scope_version=new_version,
        )


@router.post("/scopes/{name}/add", response_model=ScopeModifyResponse)
async def add_to_scope(
    name: str,
    payload: ScopeAddRequest,
    x_agent_id: str = Header(..., alias="X-Agent-ID"),
) -> ScopeModifyResponse:
    """Add passages and/or refs to a scope.

    Passages that don't exist are silently ignored.
    Refs are added without validation (resolved at search time).

    Requires X-Agent-ID header.
    """
    async with async_session() as session:
        scope = await get_scope(session, name, x_agent_id)
        if not scope:
            raise HTTPException(status_code=404, detail=f"Scope '{name}' not found")

        total_added = 0
        new_version = await get_current_version(session, scope)

        if payload.passage_ids:
            new_version, passages_added = await add_passages_to_scope(
                session, scope, payload.passage_ids
            )
            total_added += passages_added

        if payload.refs:
            new_version, refs_added = await add_refs_to_scope(session, scope, payload.refs)
            total_added += refs_added

        await session.commit()

        return ScopeModifyResponse(
            scope_version=new_version,
            modified_count=total_added,
        )


@router.post("/scopes/{name}/remove", response_model=ScopeModifyResponse)
async def remove_from_scope_endpoint(
    name: str,
    payload: ScopeRemoveRequest,
    x_agent_id: str = Header(..., alias="X-Agent-ID"),
) -> ScopeModifyResponse:
    """Remove passages and/or refs from a scope.

    Items that are not in the scope are silently ignored.

    Requires X-Agent-ID header.
    """
    async with async_session() as session:
        scope = await get_scope(session, name, x_agent_id)
        if not scope:
            raise HTTPException(status_code=404, detail=f"Scope '{name}' not found")

        new_version, total_removed = await remove_from_scope(
            session,
            scope,
            passage_ids=payload.passage_ids if payload.passage_ids else None,
            refs=payload.refs if payload.refs else None,
        )
        await session.commit()

        return ScopeModifyResponse(
            scope_version=new_version,
            modified_count=total_removed,
        )


@router.get("/scopes/{name}/history", response_model=ScopeHistoryResponse)
async def get_scope_history_endpoint(
    name: str,
    limit: int = Query(default=100, ge=1, le=1000, description="Maximum entries"),
    x_agent_id: str = Header(..., alias="X-Agent-ID"),
) -> ScopeHistoryResponse:
    """Get the history of scope changes.

    Returns history entries ordered by most recent first.

    Requires X-Agent-ID header.
    """
    async with async_session() as session:
        scope = await get_scope(session, name, x_agent_id)
        if not scope:
            raise HTTPException(status_code=404, detail=f"Scope '{name}' not found")

        history = await get_scope_history(session, scope, limit=limit)

        return ScopeHistoryResponse(
            history=[
                ScopeHistoryEntry(
                    version=h["version"],
                    changed_at=h["changed_at"],
                    passage_id=h["passage_id"],
                )
                for h in history
            ],
            count=len(history),
        )


@router.post("/scopes/{name}/revert", response_model=ScopeRevertResponse)
async def revert_scope_endpoint(
    name: str,
    payload: ScopeRevertRequest,
    x_agent_id: str = Header(..., alias="X-Agent-ID"),
) -> ScopeRevertResponse:
    """Revert a scope to a previous version.

    Creates a new version with the same definition as the target version.
    This is non-destructive - history is preserved.

    Requires X-Agent-ID header.
    """
    async with async_session() as session:
        scope = await get_scope(session, name, x_agent_id)
        if not scope:
            raise HTTPException(status_code=404, detail=f"Scope '{name}' not found")

        try:
            new_version, reverted_from = await revert_scope(session, scope, payload.to_version)
            await session.commit()
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e

        return ScopeRevertResponse(
            scope_version=new_version,
            reverted_from=reverted_from,
        )


@router.get("/scopes/{name}/search", response_model=ScopedSearchResponse)
async def search_in_scope(
    name: str,
    query: str = Query(min_length=1, description="Search query text"),
    search_type: SearchType = Query(
        default="content",
        description="Search type: content or tags",
    ),
    mode: SearchMode = Query(
        default="hybrid",
        description="Search mode: fts, semantic, or hybrid",
    ),
    limit: int = Query(default=10, ge=1, le=50, description="Maximum results"),
    period_start_after: datetime | None = Query(
        default=None, description="Filter: period_start >= this value"
    ),
    period_end_before: datetime | None = Query(
        default=None, description="Filter: period_end <= this value"
    ),
    created_after: datetime | None = Query(
        default=None, description="Filter: created_at >= this value"
    ),
    created_before: datetime | None = Query(
        default=None, description="Filter: created_at <= this value"
    ),
    x_agent_id: str = Header(..., alias="X-Agent-ID"),
) -> ScopedSearchResponse:
    """Search passages within a scope.

    Only searches passages that are part of the scope (via literal IDs or resolved refs).

    Requires X-Agent-ID header.
    """
    async with async_session() as session:
        scope = await get_scope(session, name, x_agent_id)
        if not scope:
            raise HTTPException(status_code=404, detail=f"Scope '{name}' not found")

        # Resolve scope to passage IDs
        scope_ids = await resolve_scope(session, scope)
        scope_version = await get_current_version(session, scope)

        # If scope is empty, return empty results
        if not scope_ids:
            return ScopedSearchResponse(
                query=query,
                mode=mode,
                search_type=search_type,
                results=[],
                count=0,
                scope=name,
                scope_version=scope_version,
            )

        # Search within scope
        results = await search_passages(
            session,
            query,
            search_type=search_type,
            mode=mode,
            limit=limit,
            agent_id=x_agent_id,
            scope_ids=scope_ids,
            period_start_after=period_start_after,
            period_end_before=period_end_before,
            created_after=created_after,
            created_before=created_before,
        )

        return ScopedSearchResponse(
            query=query,
            mode=mode,
            search_type=search_type,
            results=[
                PassageResult(
                    id=r.id,
                    content=r.content,
                    passage_type=r.passage_type,
                    score=r.score,
                )
                for r in results
            ],
            count=len(results),
            scope=name,
            scope_version=scope_version,
        )
