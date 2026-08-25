"""
Node discovery, network scanning, and health probe logic for distributed MAIA_API (100% Remote).
Supports Ollama (11434), raw llama.cpp (8080), and MAIA Beacon GPU workers (configurable port).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
import logging
import time
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import httpx

import config

logger = logging.getLogger("maia.nodes")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_own_ips() -> set[str]:
    """Retrieves all local IPv4 addresses of the current machine to avoid self-scanning."""
    import socket
    ips = {"127.0.0.1", "localhost", "0.0.0.0"}
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(0.5)
        s.connect(("8.8.8.8", 80))
        primary = s.getsockname()[0]
        s.close()
        if primary:
            ips.add(primary)
    except Exception:
        pass

    try:
        hostname = socket.gethostname()
        for info in socket.getaddrinfo(hostname, None):
            ip = info[4][0]
            if ":" not in ip:
                ips.add(ip)
    except Exception:
        pass
    return ips



@dataclass
class NodeState:
    id: str
    name: str
    host: str
    type: str  # "ollama", "llama.cpp", or "beacon"
    enabled: bool = True
    priority: int = 1
    online: bool = False
    last_refresh: Optional[str] = None
    models: List[str] = field(default_factory=list)
    active_model: Optional[str] = None
    last_error: Optional[str] = None
    latency_ms: Optional[float] = None
    active_requests: int = 0
    gpu_vram: Optional[Dict[str, Any]] = None
    details: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "host": self.host,
            "type": self.type,
            "enabled": self.enabled,
            "priority": self.priority,
            "online": self.online,
            "last_refresh": self.last_refresh,
            "models": list(self.models),
            "active_model": self.active_model,
            "last_error": self.last_error,
            "latency_ms": self.latency_ms,
            "active_requests": self.active_requests,
            "gpu_vram": self.gpu_vram,
            "details": self.details,
        }


class NodeRegistry:
    def __init__(
        self,
        nodes_config: Optional[Dict[str, Dict[str, Any]]] = None,
        refresh_interval: Optional[int] = None,
        client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        self._nodes: Dict[str, NodeState] = {}
        if nodes_config:
            for node_id, data in nodes_config.items():
                self._nodes[node_id] = self._dict_to_node(node_id, data)

        self._refresh_interval = refresh_interval or config.NODE_REFRESH_INTERVAL
        self._client = client
        self._lock = asyncio.Lock()
        self._refresh_lock = asyncio.Lock()
        self._rr_indices: Dict[str, int] = {}
        self._task: Optional[asyncio.Task] = None

    @staticmethod
    def _dict_to_node(node_id: str, data: Dict[str, Any]) -> NodeState:
        return NodeState(
            id=node_id,
            name=str(data.get("name") or node_id),
            host=str(data.get("host") or "").rstrip("/"),
            type=str(data.get("type") or "ollama").lower(),
            enabled=bool(data.get("enabled", True)),
            priority=int(data.get("priority", 1)),
            active_model=data.get("active_model"),
            gpu_vram=data.get("gpu_vram"),
            details=data.get("details"),
        )

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(config.NODE_SCAN_TIMEOUT, connect=config.CONNECT_TIMEOUT),
                limits=httpx.Limits(max_connections=100, max_keepalive_connections=30),
            )
        return self._client

    async def _probe_host(
        self,
        host_url: str,
        expected_type: Optional[str] = None,
        client: Optional[httpx.AsyncClient] = None,
    ) -> Optional[NodeState]:
        """Probes a single remote host URL to determine if it is a Beacon, Ollama, or llama.cpp server."""
        client_to_use = client or self._get_client()
        clean_host = host_url.rstrip("/")
        started = time.perf_counter()

        parsed = urlparse(clean_host)
        port = parsed.port

        # Determine target probing order based on port
        check_beacon = expected_type in (None, "beacon") if port != 11434 and port != 8080 else (port == config.MAIA_BEACON_PORT or expected_type == "beacon")
        check_ollama = expected_type in (None, "ollama") if port != config.MAIA_BEACON_PORT and port != 8080 else (port == 11434 or expected_type == "ollama")
        check_llamacpp = expected_type in (None, "llama.cpp") if port != config.MAIA_BEACON_PORT and port != 11434 else (port == 8080 or expected_type == "llama.cpp")

        # Try MAIA Beacon worker endpoint (/api/status)
        if check_beacon:
            try:
                resp = await client_to_use.get(f"{clean_host}/api/status", timeout=config.NODE_SCAN_TIMEOUT)
                if resp.status_code == 200:
                    payload = resp.json()
                    if isinstance(payload, dict) and ("available_models" in payload or payload.get("type") == "beacon"):
                        avail = payload.get("available_models", [])
                        models: List[str] = []
                        for m in avail:
                            if isinstance(m, str):
                                clean_m = m[:-5] if m.lower().endswith(".gguf") else m
                                if clean_m not in models:
                                    models.append(clean_m)
                        active_model = payload.get("active_model")
                        if active_model:
                            clean_act = str(active_model)
                            if clean_act.lower().endswith(".gguf"):
                                clean_act = clean_act[:-5]
                            if clean_act not in models:
                                models.append(clean_act)
                        latency = round((time.perf_counter() - started) * 1000.0, 2)
                        node_id = f"beacon-{clean_host.replace('http://', '').replace('https://', '').replace(':', '-')}"
                        return NodeState(
                            id=node_id,
                            name=f"MAIA Beacon ({clean_host})",
                            host=clean_host,
                            type="beacon",
                            online=True,
                            last_refresh=utc_now_iso(),
                            models=models,
                            active_model=str(active_model) if active_model else None,
                            latency_ms=latency,
                            gpu_vram=payload.get("gpu_vram"),
                            details=payload,
                        )
            except Exception:
                pass

        # Try Ollama endpoint (/api/tags)
        if check_ollama:
            try:
                resp = await client_to_use.get(f"{clean_host}/api/tags", timeout=config.NODE_SCAN_TIMEOUT)
                if resp.status_code == 200:
                    payload = resp.json()
                    models = sorted(
                        {
                            model.get("name")
                            for model in payload.get("models", [])
                            if isinstance(model, dict) and model.get("name")
                        }
                    )
                    latency = round((time.perf_counter() - started) * 1000.0, 2)
                    node_id = f"ollama-{clean_host.replace('http://', '').replace('https://', '').replace(':', '-')}"
                    return NodeState(
                        id=node_id,
                        name=f"Ollama ({clean_host})",
                        host=clean_host,
                        type="ollama",
                        online=True,
                        last_refresh=utc_now_iso(),
                        models=models,
                        latency_ms=latency,
                    )
            except Exception:
                pass

        # Try raw llama.cpp server endpoint (/v1/models)
        if check_llamacpp:
            try:
                resp = await client_to_use.get(f"{clean_host}/v1/models", timeout=config.NODE_SCAN_TIMEOUT)
                if resp.status_code == 200:
                    payload = resp.json()
                    models = []
                    raw_data = payload.get("data", [])
                    if isinstance(raw_data, list):
                        for m in raw_data:
                            if isinstance(m, dict) and m.get("id"):
                                m_id = str(m["id"])
                                if m_id.lower().endswith(".gguf"):
                                    m_id = m_id[:-5]
                                if m_id not in models:
                                    models.append(m_id)
                    latency = round((time.perf_counter() - started) * 1000.0, 2)
                    node_id = f"llamacpp-{clean_host.replace('http://', '').replace('https://', '').replace(':', '-')}"
                    return NodeState(
                        id=node_id,
                        name=f"llama.cpp ({clean_host})",
                        host=clean_host,
                        type="llama.cpp",
                        online=True,
                        last_refresh=utc_now_iso(),
                        models=models,
                        latency_ms=latency,
                    )
            except Exception:
                pass

        return None

    async def scan_network(self) -> List[NodeState]:
        """Asynchronously sweeps the local subnet, known active nodes, and configured static remote hosts."""
        client = self._get_client()
        priority_targets: List[str] = []
        subnet_targets: List[str] = []

        # Add static remote hosts
        for h in config.STATIC_REMOTE_HOSTS:
            if h and h not in priority_targets:
                priority_targets.append(h)

        # Add local loopback (localhost) for local services auto-detection
        for port in config.SCAN_PORTS:
            loopback_target = f"http://127.0.0.1:{port}"
            if loopback_target not in priority_targets:
                priority_targets.append(loopback_target)

        # Add previously known online nodes to priority probes
        for node in self._nodes.values():
            if node.online and node.host not in priority_targets:
                priority_targets.append(node.host)

        # Add local subnet sweep
        own_ips = get_own_ips()
        prefix = config.SUBNET_PREFIX
        if prefix:
            for i in range(1, 255):
                ip = f"{prefix}.{i}"
                if ip in own_ips:
                    continue
                for port in config.SCAN_PORTS:
                    target = f"http://{ip}:{port}"
                    if target not in priority_targets:
                        subnet_targets.append(target)

        # Probe priority targets first with fast gathering
        priority_results = await asyncio.gather(
            *[self._probe_host(t, client=client) for t in priority_targets],
            return_exceptions=True,
        )

        discovered_map: Dict[str, NodeState] = {}
        for res in priority_results:
            if isinstance(res, NodeState) and res.online:
                discovered_map[res.id] = res

        # Probe subnet with controlled concurrency to prevent socket exhaustion
        semaphore = asyncio.Semaphore(32)

        async def _probe_subnet(target: str) -> Optional[NodeState]:
            async with semaphore:
                return await self._probe_host(target, client=client)

        logger.debug("[SCAN] Démarrage du balayage de %d cibles réseau...", len(subnet_targets))
        subnet_results = await asyncio.gather(
            *[_probe_subnet(t) for t in subnet_targets],
            return_exceptions=True,
        )

        for res in subnet_results:
            if isinstance(res, NodeState) and res.online:
                discovered_map[res.id] = res

        # Filter out raw llama.cpp nodes if a Beacon is running on the same host/IP
        # to prevent duplicate registration of the Beacon's underlying llama-server.
        beacon_hosts = set()
        for node in discovered_map.values():
            if node.type == "beacon":
                try:
                    parsed = urlparse(node.host)
                    host = parsed.hostname or parsed.path
                    if host:
                        host = host.lower()
                        if host == "localhost":
                            host = "127.0.0.1"
                        beacon_hosts.add(host)
                except Exception:
                    pass

        filtered_nodes = []
        for node in discovered_map.values():
            if node.type == "llama.cpp":
                try:
                    parsed = urlparse(node.host)
                    host = parsed.hostname or parsed.path
                    if host:
                        host = host.lower()
                        if host == "localhost":
                            host = "127.0.0.1"
                        if host in beacon_hosts:
                            logger.info("[SCAN] Ignoré le nœud llama.cpp sur %s car un MAIA Beacon y est déjà actif.", node.host)
                            continue
                except Exception:
                    pass
            filtered_nodes.append(node)

        return filtered_nodes

    async def refresh_all(self) -> List[NodeState]:
        """Refreshes all node states by performing a protected network sweep."""
        # Use refresh lock so overlapping requests wait for current scan rather than storming network
        async with self._refresh_lock:
            logger.info("[REFRESH] Balayage réseau et rafraîchissement des nœuds distants...")
            discovered = await self.scan_network()

            async with self._lock:
                new_nodes: Dict[str, NodeState] = {}
                for node in discovered:
                    if node.id in self._nodes:
                        node.active_requests = self._nodes[node.id].active_requests
                        node.enabled = self._nodes[node.id].enabled
                        node.priority = self._nodes[node.id].priority
                    new_nodes[node.id] = node

                self._nodes = new_nodes
                logger.info("[REFRESH] %d nœud(s) distant(s) actif(s) détecté(s).", len(self._nodes))
                return [self._clone_node(n) for n in self._nodes.values()]

    def list_nodes(self) -> List[NodeState]:
        return [self._clone_node(n) for n in self._nodes.values()]

    def list_nodes_dict(self) -> List[Dict[str, Any]]:
        return [n.to_dict() for n in self._nodes.values()]

    def get_node(self, node_id: str) -> Optional[NodeState]:
        n = self._nodes.get(node_id)
        return self._clone_node(n) if n else None

    def find_best_node_for_model(
        self,
        requested_model: str,
        strategy: str = config.DEFAULT_LOAD_BALANCING_STRATEGY,
        require_vision: bool = False,
    ) -> Optional[NodeState]:
        canonical_req = config.canonical_model_name(requested_model)
        eligible_nodes: List[NodeState] = []

        for node in self._nodes.values():
            if not node.online or not node.enabled:
                continue

            if require_vision and not config.node_supports_vision(node.models):
                continue

            has_model = False
            for m in node.models:
                canon_m = config.canonical_model_name(m)
                if (
                    canon_m == canonical_req
                    or canon_m.startswith(canonical_req + ":")
                    or canonical_req.startswith(canon_m + ":")
                    or canonical_req.split(":")[0] == canon_m.split(":")[0]
                    or (len(canonical_req) > 5 and (canonical_req in canon_m or canon_m in canonical_req))
                ):
                    has_model = True
                    break

            if has_model:
                eligible_nodes.append(node)

        if not eligible_nodes and require_vision:
            for node in self._nodes.values():
                if node.online and node.enabled and config.node_supports_vision(node.models):
                    eligible_nodes.append(node)

        if not eligible_nodes:
            return None

        # Sort based on strategy
        if strategy == "least_latency":
            eligible_nodes.sort(key=lambda n: (n.latency_ms if n.latency_ms is not None else 9999, n.active_requests))
            return self._clone_node(eligible_nodes[0])

        elif strategy == "least_connections":
            eligible_nodes.sort(key=lambda n: (n.active_requests, n.priority))
            return self._clone_node(eligible_nodes[0])

        else:
            # Prioritize nodes where model is already active if Beacon
            eligible_nodes.sort(
                key=lambda n: (
                    0 if n.type == "beacon" and n.active_model and config.canonical_model_name(n.active_model) == canonical_req else 1,
                    n.priority,
                )
            )
            idx = self._rr_indices.get(canonical_req, 0) % len(eligible_nodes)
            self._rr_indices[canonical_req] = idx + 1
            return self._clone_node(eligible_nodes[idx])

    def increment_active_requests(self, node_id: str) -> None:
        if node_id in self._nodes:
            self._nodes[node_id].active_requests += 1

    def decrement_active_requests(self, node_id: str) -> None:
        if node_id in self._nodes:
            self._nodes[node_id].active_requests = max(0, self._nodes[node_id].active_requests - 1)

    def start_background_refresh(self) -> None:
        if self._task and not self._task.done():
            return
        self._task = asyncio.create_task(self._refresh_loop(), name="maia-node-refresh")

    async def stop_background_refresh(self) -> None:
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def close(self) -> None:
        await self.stop_background_refresh()
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    async def _refresh_loop(self) -> None:
        logger.info("[REFRESH LOOP] Démarrage du scan initial au lancement...")
        try:
            await self.refresh_all()
        except Exception as exc:
            logger.warning("[REFRESH LOOP] Erreur lors du scan initial : %s", exc)

        while True:
            await asyncio.sleep(self._refresh_interval)
            try:
                await self.refresh_all()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.warning("[REFRESH LOOP] Erreur lors du rafraîchissement périodique : %s", exc)

    @staticmethod
    def _clone_node(node: NodeState) -> NodeState:
        return NodeState(
            id=node.id,
            name=node.name,
            host=node.host,
            type=node.type,
            enabled=node.enabled,
            priority=node.priority,
            online=node.online,
            last_refresh=node.last_refresh,
            models=list(node.models),
            active_model=node.active_model,
            last_error=node.last_error,
            latency_ms=node.latency_ms,
            active_requests=node.active_requests,
            gpu_vram=node.gpu_vram,
            details=node.details,
        )
