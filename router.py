"""
Distributed routing and load balancing logic for MAIA_API (100% Remote Architecture).
Supports transparent remote model activation on MAIA Beacon GPU workers.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import logging
import time
from typing import Any, Dict, Optional

import httpx

import config
from nodes import NodeRegistry, NodeState

logger = logging.getLogger("maia.router")


@dataclass
class RoutingDecision:
    requested_model: str
    target_model: str
    target_node_id: Optional[str]
    target_host: Optional[str]
    target_type: Optional[str]  # "ollama", "llama.cpp", or "beacon"
    route_kind: str             # "load_balanced" or "unavailable"
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "requested_model": self.requested_model,
            "target_model": self.target_model,
            "target_node_id": self.target_node_id,
            "target_host": self.target_host,
            "target_type": self.target_type,
            "route_kind": self.route_kind,
            "reason": self.reason,
        }


async def ensure_beacon_model_active(
    node: NodeState,
    requested_model: str,
    registry: Optional[NodeRegistry] = None,
    client: Optional[httpx.AsyncClient] = None,
    context_size: int = 16384,
    thinking: bool = False,
    thinking_effort: str = "medium",
) -> bool:
    """If the target node is a MAIA Beacon worker, ensures the requested model is activated over HTTP."""
    if node.type != "beacon":
        return True

    canon_req = config.canonical_model_name(requested_model)
    clean_req = config.clean_model_name(requested_model)

    if node.active_model and config.canonical_model_name(node.active_model) == canon_req:
        return True

    target_file = clean_req
    for m in node.models:
        if config.canonical_model_name(m) == canon_req:
            target_file = m
            break
        elif len(canon_req) > 5 and (canon_req in config.canonical_model_name(m) or config.canonical_model_name(m) in canon_req):
            target_file = m

    should_close = False
    if client is None or client.is_closed:
        client = httpx.AsyncClient(timeout=config.NODE_SCAN_TIMEOUT)
        should_close = True

    try:
        # Check current status first
        st_resp = await client.get(f"{node.host}/api/status", timeout=config.NODE_SCAN_TIMEOUT)
        if st_resp.status_code == 200:
            st_data = st_resp.json()
            cur_active = st_data.get("active_model")
            if st_data.get("status") == "running" and cur_active and config.canonical_model_name(cur_active) == canon_req:
                node.active_model = cur_active
                if registry and node.id in registry._nodes:
                    registry._nodes[node.id].active_model = cur_active
                return True

        logger.info("[BEACON REMOTE ACTIVATION] Envoi de l'ordre de chargement du modèle '%s' à %s...", target_file, node.name)
        sel_resp = await client.post(
            f"{node.host}/api/select",
            json={
                "model": target_file,
                "context_size": context_size,
                "thinking": thinking,
                "thinking_effort": thinking_effort
            },
            timeout=5.0,
        )
        if sel_resp.status_code in (200, 202):
            start_wait = time.time()
            while time.time() - start_wait < 120.0:
                await asyncio.sleep(1.0)
                poll_resp = await client.get(f"{node.host}/api/status", timeout=3.0)
                if poll_resp.status_code == 200:
                    poll_data = poll_resp.json()
                    poll_act = str(poll_data.get("active_model") or "")
                    if poll_data.get("status") == "running" and config.canonical_model_name(poll_act) == canon_req:
                        logger.info("[BEACON REMOTE ACTIVATION] Modèle '%s' prêt et actif sur %s !", target_file, node.name)
                        node.active_model = poll_act
                        if registry and node.id in registry._nodes:
                            registry._nodes[node.id].active_model = poll_act
                        return True
                    elif poll_data.get("status") == "error":
                        logger.warning("[BEACON REMOTE ACTIVATION] Erreur signalée par Beacon : %s", poll_data.get("error_message"))
                        return False
    except Exception as exc:
        logger.warning("[BEACON REMOTE ACTIVATION] Échec de la communication avec Beacon '%s' : %s", node.name, exc)
        return False
    finally:
        if should_close and client:
            await client.aclose()

    return True


async def route_request(
    requested_model: str,
    registry: NodeRegistry,
    strategy: str = config.DEFAULT_LOAD_BALANCING_STRATEGY,
    require_vision: bool = False,
    client: Optional[httpx.AsyncClient] = None,
    context_size: int = 16384,
    thinking: bool = False,
    thinking_effort: str = "medium",
) -> RoutingDecision:
    canonical_model = config.canonical_model_name(requested_model)
    target_node = registry.find_best_node_for_model(requested_model, strategy=strategy, require_vision=require_vision)

    if not target_node:
        # Trigger an on-demand refresh sweep in case a new remote node just came online
        await registry.refresh_all()
        target_node = registry.find_best_node_for_model(requested_model, strategy=strategy, require_vision=require_vision)

    if target_node:
        # If node is a MAIA Beacon, trigger remote activation via HTTP
        if target_node.type == "beacon":
            activated = await ensure_beacon_model_active(
                target_node, requested_model, registry=registry, client=client, context_size=context_size,
                thinking=thinking, thinking_effort=thinking_effort
            )
            if not activated:
                return RoutingDecision(
                    requested_model=requested_model,
                    target_model=canonical_model,
                    target_node_id=target_node.id,
                    target_host=target_node.host,
                    target_type="beacon",
                    route_kind="unavailable",
                    reason=f"Failed to remotely activate model '{requested_model}' on Beacon node '{target_node.name}'",
                )

        matched_model = target_node.models[0] if target_node.models else requested_model
        for m in target_node.models:
            if config.canonical_model_name(m) == canonical_model:
                matched_model = m
                break
            elif len(canonical_model) > 5 and (canonical_model in config.canonical_model_name(m) or config.canonical_model_name(m) in canonical_model):
                matched_model = m

        if target_node.type == "ollama" and require_vision:
            if not config.is_vision_model(matched_model):
                for m in target_node.models:
                    if config.is_vision_model(m):
                        matched_model = m
                        break

        decision = RoutingDecision(
            requested_model=requested_model,
            target_model=matched_model,
            target_node_id=target_node.id,
            target_host=target_node.host,
            target_type=target_node.type,
            route_kind="load_balanced",
            reason=f"Selected remote node '{target_node.name}' ({target_node.id}) via {strategy}",
        )
        logger.info(
            "route_request requested=%s -> target_model=%s node=%s (%s) host=%s strategy=%s",
            requested_model,
            matched_model,
            target_node.id,
            target_node.type,
            target_node.host,
            strategy,
        )
        return decision

    logger.warning("route_request model=%s unavailable across all remote nodes", requested_model)
    return RoutingDecision(
        requested_model=requested_model,
        target_model=canonical_model,
        target_node_id=None,
        target_host=None,
        target_type=None,
        route_kind="unavailable",
        reason=f"No online remote node currently serving model '{requested_model}'",
    )
