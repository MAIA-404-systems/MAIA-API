"""
Model discovery and aggregation across online remote nodes for MAIA_API.
"""

from __future__ import annotations

from datetime import datetime, timezone
import logging
from typing import Any, Dict, List

from nodes import NodeRegistry
import config

logger = logging.getLogger("maia.model_registry")


def get_all_available_models(registry: NodeRegistry) -> List[Dict[str, Any]]:
    """
    Returns a consolidated list of unique models found across all online remote nodes.
    If no remote nodes are online or no models are found, returns an empty list.
    """
    models_map: Dict[str, List[Dict[str, Any]]] = {}

    nodes = registry.list_nodes()
    logger.debug("[MODEL_REGISTRY] Extraction des modèles parmi %d nœud(s)...", len(nodes))

    for node in nodes:
        if not node.online or not node.enabled:
            continue

        for raw_model_name in node.models:
            if config.is_vision_helper(raw_model_name):
                continue
            clean_name = config.clean_model_name(raw_model_name)
            if not clean_name:
                continue
            if clean_name not in models_map:
                models_map[clean_name] = []
            models_map[clean_name].append({
                "node_id": node.id,
                "node_name": node.name,
                "host": node.host,
                "type": node.type,
                "latency_ms": node.latency_ms,
            })

    result = []
    for model_name, nodes_info in models_map.items():
        result.append({
            "id": model_name,
            "name": model_name,
            "nodes": nodes_info,
            "node_count": len(nodes_info),
        })

    result_sorted = sorted(result, key=lambda x: x["id"])
    logger.debug("[MODEL_REGISTRY] Total de modèles agrégés : %d", len(result_sorted))
    return result_sorted


def format_openai_models_list(registry: NodeRegistry) -> Dict[str, Any]:
    models = get_all_available_models(registry)
    now_ts = int(datetime.now(timezone.utc).timestamp())
    return {
        "object": "list",
        "data": [
            {
                "id": m["id"],
                "object": "model",
                "created": now_ts,
                "owned_by": "maia-network",
            }
            for m in models
        ],
    }


def format_ollama_tags_list(registry: NodeRegistry) -> Dict[str, Any]:
    models = get_all_available_models(registry)
    now_iso = datetime.now(timezone.utc).isoformat()
    return {
        "models": [
            {
                "name": m["id"],
                "model": m["id"],
                "modified_at": now_iso,
                "details": {
                    "format": "remote",
                    "family": "remote",
                    "parameter_size": "unknown",
                },
            }
            for m in models
        ]
    }
