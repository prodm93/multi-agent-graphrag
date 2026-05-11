"""Serialisable ontology models inferred from the corpus.

These types describe **type definitions**, not extracted graph instances.
Graphiti creates the actual entity and relationship instances during
``add_episode`` ingestion. The job of `OntologyAgent` is to produce a
serialisable description of *what kinds of things exist* in the corpus and
*how they may be connected*; that description is then converted by
`GraphAgent` into Graphiti's ``add_episode(..., entity_types=...,
edge_types=..., edge_type_map=...)`` keyword arguments.

`type_name` on `SchemaFieldDefinition` is a string (e.g. ``'str'``,
``'str | None'``) so the whole ontology survives `model_dump()` round-trips
through pipeline state without dragging Python type objects along.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class SchemaFieldDefinition(BaseModel):
    """Field definition for a generated ontology type."""

    name: str = Field(..., description="Field name")
    type_name: str = Field(
        ...,
        description="Python-style type name, e.g. 'str' or 'str | None'",
    )
    description: str = Field(..., description="Field meaning")
    required: bool = Field(default=True, description="Whether the field is required")


class EntityTypeDefinition(BaseModel):
    """Serialisable definition of an entity type, not an extracted entity instance."""

    name: str = Field(
        ...,
        description="Entity type name, e.g. 'Protein', 'Company', or 'Entity'",
    )
    description: str = Field(..., description="What this entity type represents")
    fields: list[SchemaFieldDefinition] = Field(default_factory=list)


class EdgeTypeDefinition(BaseModel):
    """Serialisable definition of an edge type, not an extracted relationship instance."""

    name: str = Field(
        ...,
        description="Edge type name, e.g. 'TREATS' or 'ASSOCIATED_WITH'",
    )
    description: str = Field(..., description="What this relationship type represents")
    fields: list[SchemaFieldDefinition] = Field(default_factory=list)


class EdgeTypeMapping(BaseModel):
    """Allowed edge types between source and target entity types."""

    source_entity_type: str = Field(..., description="Source entity type name")
    target_entity_type: str = Field(..., description="Target entity type name")
    edge_type_names: list[str] = Field(
        ...,
        description="Allowed edge type names for this pair",
    )


class OntologyDefinition(BaseModel):
    """Serialisable ontology inferred from the corpus for Graphiti custom ontology.

    `GraphAgent` converts this into Graphiti's ``add_episode`` keyword
    arguments at ingestion time:

    * ``entity_types``: ``dict[str, type[BaseModel]]`` — built from
      ``self.entity_types`` by compiling each `EntityTypeDefinition` into a
      runtime `BaseModel` subclass keyed by ``name``.
    * ``edge_types``: ``dict[str, type[BaseModel]]`` — built from
      ``self.edge_types`` the same way.
    * ``edge_type_map``: ``dict[tuple[str, str], list[str]]`` — built from
      ``self.edge_type_map`` by mapping each entry to
      ``(source_entity_type, target_entity_type) -> edge_type_names``.
    """

    entity_types: list[EntityTypeDefinition] = Field(default_factory=list)
    edge_types: list[EdgeTypeDefinition] = Field(default_factory=list)
    edge_type_map: list[EdgeTypeMapping] = Field(default_factory=list)
