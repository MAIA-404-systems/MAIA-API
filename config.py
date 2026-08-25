"""
Centralized configuration for MAIA_API Distributed Load Balancer (100% Remote Architecture).
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import socket
from typing import Any, Dict, List

ROOT_DIR = Path(__file__).resolve().parent
ENV_PATH = ROOT_DIR / ".env"

APP_VERSION = "3.2.0"


def load_env() -> None:
    if not ENV_PATH.exists():
        return
    try:
        with open(ENV_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())
    except Exception:
        pass


load_env()


def get_local_subnet_prefix() -> str:
    """Auto-detects the machine's primary local network IP prefix (e.g. '192.168.1')."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(0.5)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        parts = local_ip.split(".")
        if len(parts) == 4 and parts[0] != "127":
            return f"{parts[0]}.{parts[1]}.{parts[2]}"
    except Exception:
        pass
    return "192.168.1"


# Network scan & subnet configuration
raw_subnet = os.getenv("SUBNET_PREFIX", "auto").strip()
if raw_subnet.lower() in ("auto", "", "none"):
    SUBNET_PREFIX = get_local_subnet_prefix()
else:
    if "/" in raw_subnet:
        raw_subnet = raw_subnet.split("/")[0]
    parts = raw_subnet.split(".")
    if len(parts) >= 3:
        SUBNET_PREFIX = f"{parts[0]}.{parts[1]}.{parts[2]}"
    else:
        SUBNET_PREFIX = raw_subnet

MAIA_BEACON_PORT = int(os.getenv("MAIA_BEACON_PORT", "11343"))

# Scan ports: 11434 (Ollama), 8080 (llama.cpp raw), configured MAIA Beacon port (default 11343)
raw_ports = os.getenv("SCAN_PORTS", f"11434,8080,{MAIA_BEACON_PORT}").split(",")
SCAN_PORTS: List[int] = []
for p in raw_ports:
    p = p.strip()
    if p.isdigit():
        SCAN_PORTS.append(int(p))
if not SCAN_PORTS:
    SCAN_PORTS = [11434, 8080, MAIA_BEACON_PORT]

raw_static = os.getenv("STATIC_REMOTE_HOSTS", "").split(",")
STATIC_REMOTE_HOSTS: List[str] = [h.strip().rstrip("/") for h in raw_static if h.strip()]

# Server defaults
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 11345

HOST = os.getenv("MAIA_API_HOST", os.getenv("MAIA_HOST", DEFAULT_HOST))
PORT = int(os.getenv("MAIA_API_PORT", os.getenv("MAIA_PORT", str(DEFAULT_PORT))))
LOG_LEVEL = os.getenv("MAIA_LOG_LEVEL", "INFO").upper()

# Timeouts & intervals
SCAN_INTERVAL = int(os.getenv("NODE_REFRESH_INTERVAL", os.getenv("SCAN_INTERVAL", "60")))
SCAN_TIMEOUT = float(os.getenv("NODE_SCAN_TIMEOUT", os.getenv("SCAN_TIMEOUT", "1.5")))
NODE_REFRESH_INTERVAL = SCAN_INTERVAL
NODE_SCAN_TIMEOUT = SCAN_TIMEOUT
CONNECT_TIMEOUT = float(os.getenv("CONNECT_TIMEOUT", "2.0"))
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "180.0"))
MAX_CONTEXT_SIZE = int(os.getenv("MAX_CONTEXT_SIZE", "256000"))

# Load balancing default strategy: "round_robin", "least_latency", or "least_connections"
DEFAULT_LOAD_BALANCING_STRATEGY = os.getenv("DEFAULT_LOAD_BALANCING_STRATEGY", "round_robin")


def clean_model_name(raw_name: str) -> str:
    """Strips client prefixes and .gguf file extensions while preserving case."""
    if not raw_name:
        return ""
    name = str(raw_name).strip()
    prefixes = ["ollama/", "openai/", "llama/", "custom/", "local/"]
    for p in prefixes:
        if name.lower().startswith(p):
            name = name[len(p):]
            break
    if name.lower().endswith(".gguf"):
        name = name[:-5]
    return name


def canonical_model_name(raw_name: str) -> str:
    """Returns a lowercased, prefix-stripped, and extension-stripped canonical model identifier."""
    return clean_model_name(raw_name).lower()


def is_vision_helper(model_name: str) -> bool:
    if not model_name:
        return False
    name = str(model_name).lower()
    return "mmproj" in name or "projector" in name


def is_vision_model(model_name: str) -> bool:
    if not model_name:
        return False
    name = str(model_name).lower()
    if is_vision_helper(name):
        return False
    return "vision" in name or "vl" in name or "llava" in name or "bakllava" in name


def node_supports_vision(models_list: list[str]) -> bool:
    for m in models_list:
        if is_vision_helper(m) or is_vision_model(m):
            return True
    return False
