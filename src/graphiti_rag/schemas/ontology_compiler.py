"""Compile a serialisable :class:`OntologyDefinition` into Graphiti's runtime dicts.

Graphiti's ``add_episode`` accepts:

* ``entity_types: dict[str, type[BaseModel]]``
* ``edge_types: dict[str, type[BaseModel]]``
* ``edge_type_map: dict[tuple[str, str], list[str]]``

The ontology coming out of :class:`OntologyAgent` is a plain Pydantic
description (so it survives JSON round-trips through pipeline state). This
module orchestrates the compile by delegating:

* type-string → Python type resolution to :mod:`ontology_type_resolution`
* identifier sanitisation to :mod:`ontology_name_sanitiser`

then assembles the runtime dicts, dropping or renaming anything the LLM
emitted that would break Graphiti, and adding the documented
``("Entity", "Entity")`` catch-all so custom edge types are considered for
every extracted pair (otherwise unclassified pairs degrade to RELATES_TO).
"""
from __future__ import annotations

import logging
from typing import Any

from graphiti_core.nodes import EntityNode
from pydantic import BaseModel, Field, create_model

from graphiti_rag.schemas.ontology import (
    OntologyDefinition,
    SchemaFieldDefinition,
)
from graphiti_rag.schemas.ontology_name_sanitiser import (
    sanitise_class_name,
    sanitise_field_name,
)
from graphiti_rag.schemas.ontology_type_resolution import resolve_type

logger = logging.getLogger(__name__)

# Graphiti rejects any entity-type field that collides with an EntityNode
# attribute (name, uuid, summary, group_id, labels, created_at,
# name_embedding, attributes). Read the set from Graphiti itself so it
# tracks future additions instead of going stale.
_PROTECTED_ENTITY_FIELD_NAMES: frozenset[str] = frozenset(
    EntityNode.model_fields.keys()
)


def _fields_to_pydantic(
    fields: list[SchemaFieldDefinition],
    *,
    owner: str,
    protected: frozenset[str] = frozenset(),
) -> dict[str, tuple[Any, Any]]:
    pydantic_fields: dict[str, tuple[Any, Any]] = {}
    seen: set[str] = set()
    for field_def in fields:
        sanitised = sanitise_field_name(field_def.name)
        if sanitised is None:
            logger.warning(
                "ontology_compiler: dropping field on %s — empty/invalid name %r",
                owner,
                field_def.name,
            )
            continue
        # Avoid collisions with attributes the host framework reserves
        # (Graphiti's EntityNode fields). Suffix until unique.
        while sanitised in protected:
            sanitised = f"{sanitised}_"
        if sanitised in seen:
            logger.warning(
                "ontology_compiler: dropping duplicate field %r on %s "
                "(after sanitising %r)",
                sanitised,
                owner,
                field_def.name,
            )
            continue
        seen.add(sanitised)

        if sanitised != field_def.name:
            logger.warning(
                "ontology_compiler: renamed field on %s: %r -> %r",
                owner,
                field_def.name,
                sanitised,
            )

        py_type = resolve_type(field_def.type_name)
        description = field_def.description
        if sanitised != field_def.name:
            description = f"{description} (original name: {field_def.name})"

        if field_def.required and "None" not in field_def.type_name:
            default: Any = Field(..., description=description)
        else:
            default = Field(default=None, description=description)
            if py_type is not type(None) and "None" not in str(py_type):
                py_type = py_type | None

        pydantic_fields[sanitised] = (py_type, default)
    return pydantic_fields


def _compile_model(
    *,
    sanitised_name: str,
    original_name: str,
    description: str,
    fields: list[SchemaFieldDefinition],
    protected: frozenset[str] = frozenset(),
) -> type[BaseModel]:
    pydantic_fields = _fields_to_pydantic(
        fields, owner=sanitised_name, protected=protected
    )
    model = create_model(sanitised_name, **pydantic_fields)
    if sanitised_name != original_name:
        model.__doc__ = f"{description} (original name: {original_name})"
    else:
        model.__doc__ = description
    return model


def compile_ontology(
    ontology: OntologyDefinition,
) -> tuple[
    dict[str, type[BaseModel]],
    dict[str, type[BaseModel]],
    dict[tuple[str, str], list[str]],
]:
    """Return ``(entity_types, edge_types, edge_type_map)`` ready for Graphiti."""
    entity_name_map: dict[str, str] = {}
    entity_types: dict[str, type[BaseModel]] = {}
    for entity in ontology.entity_types:
        sanitised = sanitise_class_name(entity.name)
        if sanitised is None:
            logger.warning(
                "ontology_compiler: dropping entity type with invalid name %r",
                entity.name,
            )
            continue
        if sanitised in entity_types:
            logger.warning(
                "ontology_compiler: dropping duplicate entity type %r "
                "(after sanitising %r)",
                sanitised,
                entity.name,
            )
            continue
        if sanitised != entity.name:
            logger.warning(
                "ontology_compiler: renamed entity type %r -> %r",
                entity.name,
                sanitised,
            )
        try:
            entity_types[sanitised] = _compile_model(
                sanitised_name=sanitised,
                original_name=entity.name,
                description=entity.description,
                fields=entity.fields,
                protected=_PROTECTED_ENTITY_FIELD_NAMES,
            )
        except Exception:
            logger.exception(
                "ontology_compiler: failed to compile entity type %s", sanitised
            )
            continue
        entity_name_map[entity.name] = sanitised

    edge_name_map: dict[str, str] = {}
    edge_types: dict[str, type[BaseModel]] = {}
    for edge in ontology.edge_types:
        sanitised = sanitise_class_name(edge.name)
        if sanitised is None:
            logger.warning(
                "ontology_compiler: dropping edge type with invalid name %r",
                edge.name,
            )
            continue
        if sanitised in edge_types:
            logger.warning(
                "ontology_compiler: dropping duplicate edge type %r "
                "(after sanitising %r)",
                sanitised,
                edge.name,
            )
            continue
        if sanitised != edge.name:
            logger.warning(
                "ontology_compiler: renamed edge type %r -> %r",
                edge.name,
                sanitised,
            )
        try:
            edge_types[sanitised] = _compile_model(
                sanitised_name=sanitised,
                original_name=edge.name,
                description=edge.description,
                fields=edge.fields,
            )
        except Exception:
            logger.exception(
                "ontology_compiler: failed to compile edge type %s", sanitised
            )
            continue
        edge_name_map[edge.name] = sanitised

    edge_type_map: dict[tuple[str, str], list[str]] = {}
    for mapping in ontology.edge_type_map:
        src = entity_name_map.get(mapping.source_entity_type)
        tgt = entity_name_map.get(mapping.target_entity_type)
        if src is None or tgt is None:
            logger.warning(
                "ontology_compiler: dropping edge_type_map entry "
                "(%r -> %r) — missing entity type after sanitisation",
                mapping.source_entity_type,
                mapping.target_entity_type,
            )
            continue
        resolved_edges: list[str] = []
        for edge_name in mapping.edge_type_names:
            sanitised_edge = edge_name_map.get(edge_name)
            if sanitised_edge is None:
                logger.warning(
                    "ontology_compiler: dropping edge name %r from mapping "
                    "(%r -> %r) — missing after sanitisation",
                    edge_name,
                    mapping.source_entity_type,
                    mapping.target_entity_type,
                )
                continue
            resolved_edges.append(sanitised_edge)
        if resolved_edges:
            edge_type_map[(src, tgt)] = resolved_edges

    # Documented catch-all: ("Entity", "Entity") applies an edge type to any
    # entity pair. Without it, edges between entities Graphiti couldn't
    # classify into our custom types fall back to the generic RELATES_TO.
    # https://help.getzep.com/graphiti/core-concepts/custom-entity-and-edge-types
    if edge_types:
        edge_type_map[("Entity", "Entity")] = list(edge_types.keys())

    return entity_types, edge_types, edge_type_map
