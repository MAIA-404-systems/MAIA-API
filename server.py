"""
HTTP transport and load balancing proxy layer for MAIA_API using FastAPI (100% Remote Architecture).
"""

from __future__ import annotations

from contextlib import asynccontextmanager
import copy
from datetime import datetime, timezone
import json
import logging
import time
from typing import Any, AsyncGenerator, Dict, List, Optional
import uuid

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
import httpx
import uvicorn

import config
import model_registry
from nodes import NodeRegistry
import router

logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL, logging.INFO),
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("maia.server")

# In-memory metrics & log store
MAX_LOG_ENTRIES = 100
RECENT_REQUEST_LOGS: List[Dict[str, Any]] = []


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def log_request_event(
    model: str,
    target_node_id: Optional[str],
    target_host: Optional[str],
    route_kind: str,
    status_code: int,
    duration_ms: float,
    error: Optional[str] = None,
) -> None:
    event = {
        "id": str(uuid.uuid4())[:8],
        "timestamp": utc_now_iso(),
        "model": model,
        "target_node": target_node_id or "none",
        "target_host": target_host or "none",
        "route_kind": route_kind,
        "status_code": status_code,
        "duration_ms": round(duration_ms, 2),
        "error": error,
    }
    RECENT_REQUEST_LOGS.append(event)
    if len(RECENT_REQUEST_LOGS) > MAX_LOG_ENTRIES:
        RECENT_REQUEST_LOGS.pop(0)


def has_image_in_payload(data: Dict[str, Any], is_openai: bool) -> bool:
    messages = data.get("messages", [])

    if not is_openai and "images" in data and isinstance(data["images"], list) and data["images"]:
        return True

    if isinstance(messages, list):
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            if is_openai:
                content = msg.get("content")
                if isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "image_url":
                            return True
            else:
                if "images" in msg and isinstance(msg["images"], list) and msg["images"]:
                    return True
    return False


def estimate_context_size(data: Dict[str, Any]) -> int:
    try:
        user_num_ctx = data.get("options", {}).get("num_ctx")
        if user_num_ctx:
            return int(user_num_ctx)

        max_tokens = int(data.get("max_tokens", data.get("options", {}).get("num_predict", 4096)))
        messages_str = json.dumps(data.get("messages", []))
        estimated_input_tokens = len(messages_str) // 3
        total_tokens = estimated_input_tokens + max_tokens + 1024

        power = 4096
        while power < total_tokens:
            power *= 2
        return power
    except Exception:
        return 16384


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize NodeRegistry and background refresh sweep
    client = httpx.AsyncClient(timeout=config.REQUEST_TIMEOUT)
    registry = NodeRegistry(client=client)
    registry.start_background_refresh()
    app.state.registry = registry
    app.state.client = client
    logger.info("MAIA_API 100%% Remote Load Balancer initialized on %s:%s.", config.HOST, config.PORT)
    yield
    # Shutdown: Clean up background tasks and close client
    logger.info("Shutting down MAIA_API...")
    await registry.close()
    await client.aclose()


def create_app() -> FastAPI:
    app = FastAPI(title="MAIA_API", version=config.APP_VERSION, lifespan=lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Node Discovery & Status REST API

    @app.get("/api/nodes")
    async def list_nodes():
        reg: NodeRegistry = app.state.registry
        return reg.list_nodes_dict()

    @app.post("/api/refresh")
    @app.post("/api/nodes/refresh")
    @app.get("/api/refresh")
    async def refresh_nodes():
        reg: NodeRegistry = app.state.registry
        refreshed = await reg.refresh_all()
        return [n.to_dict() for n in refreshed]

    @app.get("/api/status")
    async def get_status():
        reg: NodeRegistry = app.state.registry
        nodes = reg.list_nodes()
        online_count = sum(1 for n in nodes if n.online and n.enabled)
        total_models = len(model_registry.get_all_available_models(reg))
        total_active_requests = sum(n.active_requests for n in nodes)

        active_model = None
        gpu_vram = {"total_mib": 0, "used_mib": 0, "free_mib": 0}
        for n in nodes:
            if n.online:
                if not active_model and n.active_model:
                    active_model = n.active_model
                # Also copy GPU stats if available
                node_vram = getattr(n, "gpu_vram", None)
                node_details = getattr(n, "details", None)
                if node_vram:
                    gpu_vram = node_vram
                elif isinstance(node_details, dict) and "gpu_vram" in node_details:
                    gpu_vram = node_details["gpu_vram"]

        return {
            "version": config.APP_VERSION,
            "status": "online" if online_count > 0 else "offline",
            "subnet": config.SUBNET_PREFIX,
            "scan_ports": config.SCAN_PORTS,
            "total_nodes": len(nodes),
            "online_nodes": online_count,
            "total_models": total_models,
            "active_requests": total_active_requests,
            "strategy": config.DEFAULT_LOAD_BALANCING_STRATEGY,
            "active_model": active_model,
            "gpu_vram": gpu_vram,
        }

    @app.get("/api/models")
    async def list_models_aggregated():
        reg: NodeRegistry = app.state.registry
        res = model_registry.get_all_available_models(reg)
        return res

    @app.get("/api/logs")
    async def get_logs():
        return list(reversed(RECENT_REQUEST_LOGS))

    @app.post("/api/select")
    async def select_model(request: Request):
        reg: NodeRegistry = app.state.registry
        client: httpx.AsyncClient = app.state.client
        try:
            data = await request.json() or {}
        except Exception:
            data = {}

        requested_model = data.get("model")
        if not requested_model:
            raise HTTPException(status_code=400, detail="Nom de modèle requis")

        context_size = int(data.get("context_size", 8192))
        thinking = bool(data.get("thinking", True))
        thinking_effort = data.get("thinking_effort", "low")

        decision = await router.route_request(
            requested_model,
            reg,
            client=client,
            context_size=context_size,
            thinking=thinking,
            thinking_effort=thinking_effort
        )

        if decision.route_kind == "unavailable" or not decision.target_host:
            raise HTTPException(status_code=503, detail=decision.reason)

        return {
            "message": f"Modèle sélectionné et routé vers le noeud '{decision.target_node_id}'",
            "status": "success",
            "active_model": decision.target_model,
            "node_id": decision.target_node_id,
            "route_kind": decision.route_kind
        }

    # OpenAI Compatible Proxy Endpoints

    @app.get("/v1/models")
    async def openai_models():
        reg: NodeRegistry = app.state.registry
        return model_registry.format_openai_models_list(reg)

    @app.post("/v1/chat/completions")
    async def openai_chat_completions(request: Request):
        reg: NodeRegistry = app.state.registry
        client: httpx.AsyncClient = app.state.client
        data = await request.json() or {}
        requested_model = str(data.get("model", "default"))
        is_stream = bool(data.get("stream", False))

        require_vision = has_image_in_payload(data, is_openai=True)
        context_size = min(estimate_context_size(data), config.MAX_CONTEXT_SIZE)
        decision = await router.route_request(
            requested_model, reg, require_vision=require_vision, client=client, context_size=context_size
        )
        if decision.route_kind == "unavailable" or not decision.target_host:
            log_request_event(requested_model, None, None, "unavailable", 503, 0.0, decision.reason)
            return JSONResponse(
                status_code=503,
                content={
                    "error": {
                        "message": decision.reason,
                        "type": "server_error",
                        "code": 503,
                    }
                },
            )

        target_payload = copy.deepcopy(data)
        target_payload["model"] = decision.target_model

        start_time = time.perf_counter()
        target_node_id = decision.target_node_id
        target_url = f"{decision.target_host}/v1/chat/completions"

        if not is_stream:
            reg.increment_active_requests(target_node_id)
            try:
                resp = await client.post(
                    target_url,
                    json=target_payload,
                    timeout=httpx.Timeout(config.REQUEST_TIMEOUT, connect=config.CONNECT_TIMEOUT),
                )
                reg.decrement_active_requests(target_node_id)
                duration_ms = (time.perf_counter() - start_time) * 1000.0
                log_request_event(
                    requested_model, target_node_id, decision.target_host,
                    decision.route_kind, resp.status_code, duration_ms
                )
                try:
                    return JSONResponse(status_code=resp.status_code, content=resp.json())
                except Exception:
                    return Response(content=resp.content, status_code=resp.status_code, media_type=resp.headers.get("content-type"))
            except Exception as exc:
                reg.decrement_active_requests(target_node_id)
                duration_ms = (time.perf_counter() - start_time) * 1000.0
                log_request_event(requested_model, target_node_id, decision.target_host, decision.route_kind, 502, duration_ms, str(exc))
                return JSONResponse(
                    status_code=502,
                    content={
                        "error": {
                            "message": f"Failed to connect to remote node '{target_node_id}': {exc}",
                            "type": "upstream_error",
                            "code": 502,
                        }
                    },
                )

        # Streaming Response
        async def stream_generator() -> AsyncGenerator[bytes, None]:
            reg.increment_active_requests(target_node_id)
            status_code = 500
            try:
                async with client.stream(
                    "POST",
                    target_url,
                    json=target_payload,
                    timeout=httpx.Timeout(config.REQUEST_TIMEOUT, connect=config.CONNECT_TIMEOUT),
                ) as resp:
                    status_code = resp.status_code
                    async for chunk in resp.aiter_bytes():
                        yield chunk
            except Exception as exc:
                logger.error("Error streaming from remote node %s: %s", target_node_id, exc)
            finally:
                reg.decrement_active_requests(target_node_id)
                duration_ms = (time.perf_counter() - start_time) * 1000.0
                log_request_event(
                    requested_model, target_node_id, decision.target_host,
                    decision.route_kind, status_code, duration_ms
                )

        return StreamingResponse(stream_generator(), media_type="text/event-stream")

    # Ollama Compatible Proxy Endpoints

    @app.get("/api/tags")
    async def ollama_tags():
        reg: NodeRegistry = app.state.registry
        return model_registry.format_ollama_tags_list(reg)

    @app.post("/api/chat")
    async def ollama_chat(request: Request):
        reg: NodeRegistry = app.state.registry
        client: httpx.AsyncClient = app.state.client
        data = await request.json() or {}
        requested_model = str(data.get("model", "default"))
        is_stream = bool(data.get("stream", True))

        require_vision = has_image_in_payload(data, is_openai=False)
        context_size = min(estimate_context_size(data), config.MAX_CONTEXT_SIZE)
        decision = await router.route_request(
            requested_model, reg, require_vision=require_vision, client=client, context_size=context_size
        )
        if decision.route_kind == "unavailable" or not decision.target_host:
            log_request_event(requested_model, None, None, "unavailable", 503, 0.0, decision.reason)
            return JSONResponse(status_code=503, content={"error": decision.reason})

        target_payload = copy.deepcopy(data)
        target_payload["model"] = decision.target_model

        start_time = time.perf_counter()
        target_node_id = decision.target_node_id
        target_url = f"{decision.target_host}/api/chat"

        if not is_stream:
            reg.increment_active_requests(target_node_id)
            try:
                resp = await client.post(
                    target_url,
                    json=target_payload,
                    timeout=httpx.Timeout(config.REQUEST_TIMEOUT, connect=config.CONNECT_TIMEOUT),
                )
                reg.decrement_active_requests(target_node_id)
                duration_ms = (time.perf_counter() - start_time) * 1000.0
                log_request_event(
                    requested_model, target_node_id, decision.target_host,
                    decision.route_kind, resp.status_code, duration_ms
                )
                try:
                    return JSONResponse(status_code=resp.status_code, content=resp.json())
                except Exception:
                    return Response(content=resp.content, status_code=resp.status_code)
            except Exception as exc:
                reg.decrement_active_requests(target_node_id)
                duration_ms = (time.perf_counter() - start_time) * 1000.0
                log_request_event(requested_model, target_node_id, decision.target_host, decision.route_kind, 502, duration_ms, str(exc))
                return JSONResponse(status_code=502, content={"error": f"Upstream remote node error: {exc}"})

        async def stream_generator() -> AsyncGenerator[bytes, None]:
            reg.increment_active_requests(target_node_id)
            status_code = 500
            try:
                async with client.stream(
                    "POST",
                    target_url,
                    json=target_payload,
                    timeout=httpx.Timeout(config.REQUEST_TIMEOUT, connect=config.CONNECT_TIMEOUT),
                ) as resp:
                    status_code = resp.status_code
                    async for chunk in resp.aiter_bytes():
                        yield chunk
            except Exception as exc:
                logger.error("Error streaming from remote node %s: %s", target_node_id, exc)
            finally:
                reg.decrement_active_requests(target_node_id)
                duration_ms = (time.perf_counter() - start_time) * 1000.0
                log_request_event(
                    requested_model, target_node_id, decision.target_host,
                    decision.route_kind, status_code, duration_ms
                )

        return StreamingResponse(stream_generator(), media_type="application/x-ndjson")

    @app.post("/api/generate")
    async def ollama_generate(request: Request):
        reg: NodeRegistry = app.state.registry
        client: httpx.AsyncClient = app.state.client
        data = await request.json() or {}
        requested_model = str(data.get("model", "default"))
        is_stream = bool(data.get("stream", True))

        require_vision = has_image_in_payload(data, is_openai=False)
        context_size = min(estimate_context_size(data), config.MAX_CONTEXT_SIZE)
        decision = await router.route_request(
            requested_model, reg, require_vision=require_vision, client=client, context_size=context_size
        )
        if decision.route_kind == "unavailable" or not decision.target_host:
            log_request_event(requested_model, None, None, "unavailable", 503, 0.0, decision.reason)
            return JSONResponse(status_code=503, content={"error": decision.reason})

        target_payload = copy.deepcopy(data)
        target_payload["model"] = decision.target_model

        start_time = time.perf_counter()
        target_node_id = decision.target_node_id
        target_url = f"{decision.target_host}/api/generate"

        if not is_stream:
            reg.increment_active_requests(target_node_id)
            try:
                resp = await client.post(
                    target_url,
                    json=target_payload,
                    timeout=httpx.Timeout(config.REQUEST_TIMEOUT, connect=config.CONNECT_TIMEOUT),
                )
                reg.decrement_active_requests(target_node_id)
                duration_ms = (time.perf_counter() - start_time) * 1000.0
                log_request_event(
                    requested_model, target_node_id, decision.target_host,
                    decision.route_kind, resp.status_code, duration_ms
                )
                try:
                    return JSONResponse(status_code=resp.status_code, content=resp.json())
                except Exception:
                    return Response(content=resp.content, status_code=resp.status_code)
            except Exception as exc:
                reg.decrement_active_requests(target_node_id)
                duration_ms = (time.perf_counter() - start_time) * 1000.0
                log_request_event(requested_model, target_node_id, decision.target_host, decision.route_kind, 502, duration_ms, str(exc))
                return JSONResponse(status_code=502, content={"error": f"Upstream remote node error: {exc}"})

        async def stream_generator() -> AsyncGenerator[bytes, None]:
            reg.increment_active_requests(target_node_id)
            status_code = 500
            try:
                async with client.stream(
                    "POST",
                    target_url,
                    json=target_payload,
                    timeout=httpx.Timeout(config.REQUEST_TIMEOUT, connect=config.CONNECT_TIMEOUT),
                ) as resp:
                    status_code = resp.status_code
                    async for chunk in resp.aiter_bytes():
                        yield chunk
            except Exception as exc:
                logger.error("Error streaming from remote node %s: %s", target_node_id, exc)
            finally:
                reg.decrement_active_requests(target_node_id)
                duration_ms = (time.perf_counter() - start_time) * 1000.0
                log_request_event(
                    requested_model, target_node_id, decision.target_host,
                    decision.route_kind, status_code, duration_ms
                )

        return StreamingResponse(stream_generator(), media_type="application/x-ndjson")

    return app


app = create_app()

if __name__ == "__main__":
    GREEN = "\033[92m"
    RESET = "\033[0m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"

    LOGO = fr"""
{GREEN}      ###**###      {RESET}
{GREEN}   ##.        .##   {RESET}
{GREEN} ##              ## {RESET}
{GREEN} #.              .# {RESET}   {CYAN}{BOLD} __  __    _    ___    _          _    ____ ___  {RESET}
{GREEN}#= :####:  :####: =#{RESET}   {CYAN}{BOLD}|  \/  |  / \  |_ _|  / \        / \  |  _ \_ _| {RESET}
{GREEN}#: ######  ###### :#{RESET}   {CYAN}{BOLD}| |\/| | / _ \  | |  / _ \      / _ \ | |_) | |  {RESET}
{GREEN}#= :####:  :####: =#{RESET}   {CYAN}{BOLD}| |  | |/ ___ \ | | / ___ \    / ___ \|  __/| |  {RESET}
{GREEN} #                # {RESET}   {CYAN}{BOLD}|_|  |_/_/   \_\___/_/   \_\  /_/   \_\_|  |___| {RESET}
{GREEN} ##              ## {RESET}
{GREEN}   ##          ##   {RESET}
{GREEN}     ####++####     {RESET}
"""
    print(LOGO)
    print(f"[*] Starting MAIA_API 100% Remote Load Balancer on {config.HOST}:{config.PORT}...")
    uvicorn.run(app, host=config.HOST, port=config.PORT, log_level="info")
