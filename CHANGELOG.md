# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2025-02-04

### Added
- Initial release
- **Passages API** with hybrid search combining full-text search (PostgreSQL tsvector) and semantic search (pgvector embeddings) using Reciprocal Rank Fusion
- **Tags system** with semantic search on tag names and descriptions
- **Refs system** - Git-like mutable pointers to passages with version history
- **Memory Scopes** - Dynamic search closures with versioning and revert support
- **Branches** for world model experimentation and isolation
- **Processing Runs** for transformation pipelines with fold semantics
- **Provenance tracking** for derivation chains between passages
- **World Model Extraction** - LLM-based extraction of human/persona/world state from conversations
- **REST API** (FastAPI) with comprehensive endpoints for all features
- **MCP server** for Model Context Protocol integration
- **CLI tools** for all operations (`kp3` command)
- **Docker and Podman support** with compose configuration
- **6 example demos** with documentation covering all major features

[Unreleased]: https://github.com/marklubin/kp3/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/marklubin/kp3/releases/tag/v0.1.0
