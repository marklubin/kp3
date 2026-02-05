"""Pydantic schemas for KP3 API.

These types define the API request/response models for the KP3 service.
"""

from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, Field

# Search mode type
SearchMode = Literal["fts", "semantic", "hybrid"]

# Search type - content search vs tag search
SearchType = Literal["content", "tags"]


class PromptResponse(BaseModel):
    """Response from GET /prompts/{name} endpoint.

    Contains the active extraction prompt for a given name.
    """

    id: str
    name: str
    version: int
    system_prompt: str
    user_prompt_template: str
    field_descriptions: dict[str, Any]


class PassageResult(BaseModel):
    """A single passage in search results."""

    id: UUID
    content: str
    passage_type: str
    score: float = Field(description="Relevance score (higher is better)")


class SearchResponse(BaseModel):
    """Response from GET /passages/search endpoint."""

    query: str
    mode: str
    search_type: str = Field(default="content", description="Search type: content or tags")
    results: list[PassageResult]
    count: int = Field(description="Number of results returned")


class PassageCreate(BaseModel):
    """Request body for POST /passages endpoint."""

    content: str
    passage_type: str
    metadata: dict[str, Any] | None = None
    period_start: datetime | None = None
    period_end: datetime | None = None


class PassageCreateResponse(BaseModel):
    """Response from POST /passages endpoint."""

    id: UUID
    content: str
    passage_type: str


class PassageResponse(BaseModel):
    """Response from GET /passages/{id} endpoint."""

    id: UUID
    content: str
    passage_type: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    period_start: datetime | None = None
    period_end: datetime | None = None
    created_at: datetime


# =============================================================================
# Tag schemas
# =============================================================================


class TagCreate(BaseModel):
    """Request body for POST /tags endpoint."""

    name: str = Field(min_length=1, max_length=256, description="Tag display name")
    description: str | None = Field(
        default=None, max_length=1024, description="Optional description"
    )


class TagResponse(BaseModel):
    """Response for tag endpoints."""

    id: UUID
    name: str
    description: str | None
    passage_count: int
    created_at: datetime
    updated_at: datetime


class TagListResponse(BaseModel):
    """Response from GET /tags endpoint."""

    tags: list[TagResponse]
    count: int


class PassageTagsRequest(BaseModel):
    """Request body for attaching/detaching tags to/from passages."""

    tag_ids: list[UUID] = Field(min_length=1, description="List of tag UUIDs")


class PassageTagsResponse(BaseModel):
    """Response for passage tags endpoints."""

    passage_id: UUID
    tags: list[TagResponse]


# =============================================================================
# Ref schemas
# =============================================================================


class RefResponse(BaseModel):
    """Response for ref endpoints."""

    name: str
    passage_id: UUID
    updated_at: datetime
    metadata: dict[str, Any] = Field(default_factory=dict)


class RefListResponse(BaseModel):
    """Response from GET /refs endpoint."""

    refs: list[RefResponse]
    count: int


class RefSetRequest(BaseModel):
    """Request body for PUT /refs/{name} endpoint."""

    passage_id: UUID
    metadata: dict[str, Any] | None = None


class RefHistoryEntry(BaseModel):
    """A single entry in ref history."""

    id: UUID
    ref_name: str
    passage_id: UUID
    previous_passage_id: UUID | None
    changed_at: datetime
    metadata: dict[str, Any] = Field(default_factory=dict)


class RefHistoryResponse(BaseModel):
    """Response from GET /refs/{name}/history endpoint."""

    history: list[RefHistoryEntry]
    count: int


# =============================================================================
# Branch schemas
# =============================================================================


class BranchResponse(BaseModel):
    """Response for branch endpoints."""

    id: UUID
    name: str
    ref_prefix: str
    branch_name: str
    human_ref: str
    persona_ref: str
    world_ref: str
    parent_branch_id: UUID | None
    is_main: bool
    hooks_enabled: bool
    description: str | None
    created_at: datetime


class BranchListResponse(BaseModel):
    """Response from GET /branches endpoint."""

    branches: list[BranchResponse]
    count: int


class BranchCreateRequest(BaseModel):
    """Request body for POST /branches endpoint."""

    ref_prefix: str = Field(description="Agent/entity prefix (e.g., 'corindel')")
    branch_name: str = Field(description="Branch name (e.g., 'experiment-1' or 'HEAD')")
    description: str | None = None
    is_main: bool = Field(default=False, description="Whether this is the main/production branch")
    hooks_enabled: bool | None = Field(
        default=None, description="Whether hooks fire. Defaults to True for main, False otherwise."
    )


class BranchForkRequest(BaseModel):
    """Request body for POST /branches/{name}/fork endpoint."""

    new_branch_name: str = Field(description="Name for the new branch")
    description: str | None = None


class BranchPromoteRequest(BaseModel):
    """Request body for POST /branches/{name}/promote endpoint."""

    target_branch: str = Field(default="HEAD", description="Target branch to promote to")


# =============================================================================
# Processing Run schemas
# =============================================================================


class RunResponse(BaseModel):
    """Response for run endpoints."""

    id: UUID
    input_sql: str
    processor_type: str
    processor_config: dict[str, Any]
    status: str
    total_groups: int | None
    processed_groups: int
    output_count: int
    error_message: str | None
    started_at: datetime | None
    completed_at: datetime | None
    created_at: datetime


class RunListResponse(BaseModel):
    """Response from GET /runs endpoint."""

    runs: list[RunResponse]
    count: int


class RunCreateRequest(BaseModel):
    """Request body for POST /runs endpoint."""

    input_sql: str = Field(description="SQL query returning passage_ids, group_key, group_metadata")
    processor_type: str = Field(description="Processor type (e.g., 'llm_prompt')")
    processor_config: dict[str, Any] = Field(description="Processor-specific configuration")


# =============================================================================
# Provenance schemas
# =============================================================================


class ProvenancePassage(BaseModel):
    """A passage in provenance results."""

    id: UUID
    content: str
    passage_type: str
    created_at: datetime


class ProvenanceSourcesResponse(BaseModel):
    """Response from GET /passages/{id}/sources endpoint."""

    passage_id: UUID
    sources: list[ProvenancePassage]
    count: int


class ProvenanceDerivedResponse(BaseModel):
    """Response from GET /passages/{id}/derived endpoint."""

    passage_id: UUID
    derived: list[ProvenancePassage]
    count: int


class ProvenanceChainEntry(BaseModel):
    """An entry in the provenance chain."""

    derived_passage_id: UUID
    source_passage_id: UUID
    processing_run_id: UUID | None
    depth: int


class ProvenanceChainResponse(BaseModel):
    """Response from GET /passages/{id}/provenance endpoint."""

    passage_id: UUID
    chain: list[ProvenanceChainEntry]
    count: int
