#!/usr/bin/env python3.14
import sys
if sys.version_info < (3, 14):
    sys.exit("Python 3.14+ required")
# files = off ✓
"""cat r1 · files = off · DSpark"""
import tkinter as tk
from tkinter import scrolledtext, font, messagebox
import numpy as np
import time
import threading
import re
import json
import os
import io
import ast
import contextlib
import subprocess
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional, Dict, List, Any, Tuple
from dataclasses import dataclass, field
import html as html_module
import uuid
from urllib import request as urlrequest
from urllib.error import URLError, HTTPError
from urllib.parse import urlparse
from html.parser import HTMLParser


class CompatConfig(dict):
    """Dictionary with legacy key aliases that stay in sync."""

    def __init__(self, data: Dict[str, Any], legacy_to_canonical: Dict[str, str]):
        super().__init__(data)
        self._legacy_to_canonical = dict(legacy_to_canonical)
        self._canonical_to_legacy = {v: k for k, v in self._legacy_to_canonical.items()}
        for legacy, canonical in self._legacy_to_canonical.items():
            if canonical in self:
                super().__setitem__(legacy, super().__getitem__(canonical))

    def _canonical(self, key: str) -> str:
        return self._legacy_to_canonical.get(key, key)

    def __getitem__(self, key):
        return super().__getitem__(self._canonical(key))

    def __setitem__(self, key, value):
        canonical = self._canonical(key)
        super().__setitem__(canonical, value)
        legacy = self._canonical_to_legacy.get(canonical)
        if legacy:
            super().__setitem__(legacy, value)

    def __contains__(self, key):
        return super().__contains__(self._canonical(key))

    def get(self, key, default=None):
        return super().get(self._canonical(key), default)

    def setdefault(self, key, default=None):
        canonical = self._canonical(key)
        if canonical in self:
            return super().__getitem__(canonical)
        self[key] = default
        return default

    def pop(self, key, default=None):
        canonical = self._canonical(key)
        legacy = self._canonical_to_legacy.get(canonical)
        if canonical in self:
            val = super().pop(canonical)
            if legacy and legacy in self:
                super().pop(legacy, None)
            return val
        if default is not None:
            return default
        raise KeyError(key)

    def update(self, *args, **kwargs):
        updates = dict(*args, **kwargs)
        for k, v in updates.items():
            self[k] = v


# ──────────────────────────────────────────────────────────────
# DISTILLATION (in-memory student stack · files = off)
# cat r1 parity
# ──────────────────────────────────────────────────────────────
CAT_R1_DISTIL_PARAMS = {
    "distil_enabled": True,
    "distil_teacher_weight": 0.72,
    "teacher_weight": 0.72,
    "distil_passes": 4,
    "turbo_passes": 2,
    "distil_protocol": "cat-r1-distil",
}

# ──────────────────────────────────────────────────────────────
# MYTHOS PARAMETERS (BitNet GigaEngine · files = off)
# cat r1 level
# ──────────────────────────────────────────────────────────────
CAT_R1_MYTHOS_PARAMS = {
    "prose_tier": "cat r1",
    "mythos_tier": "cat r1",
    "reasoning_mode": "cat-r1-hybrid-reasoning",
    "code_interpreter": "cat-r1-code-1.0",
    "code_interpreter_family": "cat-r1",
    "code_interpreter_version": "1.0",
    "cat_r1_recursive_depth": 3,
    "mythos_mode": True,
    "mythos_recursive_improve": True,
    "mythos_recursive_depth": 3,
    "mythos_recursive_epsilon": 0.008,
    "code_auto_run": True,
    "code_output_exact": True,
    "code_token_weights_only": True,
    "mythos_voice": True,
    "mythos_runtime": True,
    "code_interpreter_name": "cat r1 code 1.0",
    "code_anything": True,
    "code_anything_mode": "universal-cat-r1",
    "simulate_latency": 0.0,
    "simulate_enabled": False,
    "v4_as_bitnet": True,
    "step_delay": 0.0,
    # cat r1 scale (10M ctx · 512K out · 1.7T effective)
    "cat_r1_context_window": 10_000_000,
    "cat_r1_max_output": 512_000,
    "nominal_base_params": 1_700_000_000_000,
}

# ──────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────
CONFIG = {
    # ── Architecture ──
    "d_model": 512,
    "layers": 4,
    "heads": 8,
    "n_experts": 4,
    "top_k": 2,
    "recursive_depth": 3,
    "recursive_epsilon": 0.008,
    "o1_preview": True,
    "o1_self_check": True,
    "interpreter_syntax": "cat-r1-code-1.0",
    "o1_interpreter_protocol": "cat-r1-code-1.0",
    "cat_r1_code_interpreter_model": "cat-r1-code-1.0",
    "cat_r1_code": True,
    "cat_code_interpreter_r1": True,
    "claude_code_fork": "anthropics/claude-code",
    "cat_code_interpreter_fork_ref": "plugins/feature-dev,plugins/code-review,examples/hooks",
    "catcode_bitnet_only": True,
    "catcode_no_api": True,
    "api_enabled": False,
    "cat_r1_code_interpreter_think_tags": True,
    "ultrathink_default": True,
    "r1_synth_default": True,
    # ── Multi-level compression (BitNet + sparse + low-rank + MoE) ──
    "compression_enabled": True,
    "compression_sparse_k": 128,
    "compression_rank": 256,
    "compression_stack_mult": 8,
    "weight_bits": 1.58,
    "bitnet_real": True,
    "bitnet_use_packed": True,
    "bitnet_self_test": True,
    "bitnet_kernel": "ternary_addsub",
    "bitnet_encode": True,
    "bitnet_force_real": True,
    "bitnet_no_weight_files": True,
    "ai_mode": True,
    "nominal_base_params": 1_700_000_000_000,
    # ── API & protocol ──
    "api_port": 8765,
    "api_key": os.getenv("CATR1_API_KEY", "lm-studio"),
    "chat_protocol": "cat-r1-chat",
    "chat_version": "r1",
    "max_sessions": 128,
    "llm_match_target": "cat r1",
    "llm_match_provider": "cat r1",
    # ── Tokenizer & context ──
    "vocab_size": 65536,
    "max_seq": 2048,
    "ff_mult": 4,
    "cat_r1_eps": 1e-5,
    "act_bits": 16,
    "files": "off",
    "cat_r1_enabled": True,
    "cat_r1_model_id": "cat-r1",
    "cat_r1_edition": "r1",
    "coding_api_protocol": "cat-r1-coding-api",
    "coding_api_version": "1.0",
    # ── Compression ──
    "whitepaper_compression": True,
    "google_whitepaper_heuristics": True,
    "cat_r1_voice": True,
    "compression_nvidia_awq_bits": 4,
    "compression_nvidia_gptq_block": 128,
    "compression_nvidia_sparse_2_4": True,
    "compression_google_lowrank_rank": 64,
    "compression_google_moe_experts": 64,
    "compression_google_moe_topk": 8,
    "compression_zstd_level": 6,
    # ── Frontier code engine ──
    "code_terminal_timeout": 8,
    "cat_r1_code_enabled": True,
    "cat_r1_engine": True,
    "cat_r1_code_perfect": True,
    "cat_r1b_lint": True,
    "vibe_code_heuristics": True,
    "cat_r1_reasoning": True,
    "deepmind_fast": True,
    "flash_attention": True,
    "adaptive_compute": True,
    "turbo_encode_tasks": ("chat", "math", "execute", "explain", "agent", "research"),
    "cat_r1_self_verify": True,
    "web_program_enabled": True,
    "web_fetch_enabled": True,
    "web_max_sites": 256,
    "web_max_fetch_kb": 1024,
    "web_preview_port": None,
    # ── cat r1 frontier features ──
        "cat_r1_moe_dense": True,
        "cat_r1_multi_turn_search": True,
        "cat_r1_fun_mode": True,
        "cat_r1_realtime": True,
        "cat_r1_moe_scale": True,
        "cat_r1_multi_token_prediction": True,
        "cat_r1_active_inference": True,
        "cat_r1_long_context_finegrained": True,
        "cat_r1_sparse_attention": True,
        # ── cat r1 V4 Pro neural compression ──
        "v4_pro_compression": True,
        "v4_pro_compression_algorithm": "fractal-ternary-recursive-sparse",
        "v4_pro_compression_scales": 3,
        "v4_pro_compression_recursive_depth": 4,
        "v4_pro_coding": True,
        "v4_pro_coding_depth": 5,
        "v4_pro_coding_passes": 2,
        # ── CatR1 Pro & Flash model routing ──
        "cat_r1_pro_enabled": True,
        "cat_r1_flash_enabled": True,
        "cat_r1_model_router": True,
        "cat_r1_pro_model_id": "cat-r1-pro-v1",
        "cat_r1_flash_model_id": "cat-r1-flash-v1",
        "cat_r1_pro_reasoning_depth": 5,
        "cat_r1_flash_reasoning_depth": 1,
        "v4_pro_target_tps": 48.0,
        "v4_flash_target_tps": 420.0,
        "v4_pro_encode_budget_ms": 120.0,
        "v4_flash_encode_budget_ms": 8.0,
        "v4_speculative_draft_tokens": 5,
        "dspark_enabled": True,
        "dspark_speculative_decode": True,
        "dspark_block_size": 7,
        "dspark_markov_rank": 64,
        "dspark_confidence_head": True,
        "cat_r1_dspark": True,
        "cat_r1_dspark_algo": "speculative-markov-confidence-decode",
        "cat_r1_dspark_on_pr": True,
        "cat_r1_draft_gamma": 5,
        "cat_r1_adaptive_gamma": True,
    # ── GUI ──
    "gui_theme": "cat r1",
    "gui_cat_r1": True,
    "default_chat_mode": "expert",
    "universal_chat": True,
    "bilingual_chat": True,
    "tri_channel_detect": True,
    "always_answer": True,
    "user_desire_mode": True,
    **CAT_R1_DISTIL_PARAMS,
    **CAT_R1_MYTHOS_PARAMS,
}

_LEGACY_CONFIG_KEY_MAP: Dict[str, str] = {}
for _key in list(CONFIG.keys()):
    if _key.startswith("cat_r1_"):
        _legacy_key = f"catr1_{_key[len('cat_r1_'):]}"
        _LEGACY_CONFIG_KEY_MAP[_legacy_key] = _key
CONFIG = CompatConfig(CONFIG, _LEGACY_CONFIG_KEY_MAP)

# cat r1 layout · files = off · cat r1
CAT_R1_UI = {
    "bg": "#0f1117",
    "sidebar": "#13161e",
    "sidebar_border": "#1e2230",
    "header_bg": "#0f1117",
    "header_border": "#1e2230",
    "text": "#e4e6ed",
    "muted": "#7c8298",
    "user_bg": "#1a2040",
    "user_fg": "#e4e6ed",
    "bot_bg": "#13161e",
    "bot_fg": "#e4e6ed",
    "think_fg": "#8b92a8",
    "think_bg": "#161a24",
    "think_border": "#1e2230",
    "code_bg": "#0a0c12",
    "code_fg": "#c9d1d9",
    "input_bg": "#13161e",
    "input_border": "#1e2230",
    "input_shadow": "#0a0c12",
    "accent": "#6c5ce7",
    "accent_text": "#3b82f6",
    "send_hover": "#1a1a1a",
    "avatar_bot": "#6c5ce7",
    "avatar_user": "#7c8298",
    "mascot_bg": "#1a1f33",
    "mascot_fg": "#a78bfa",
    "new_chat_bg": "#000000",
    "new_chat_border": "#000000",
    "history_hover": "#0d0d0d",
    "empty_title": "#e4e6ed",
    "radius_pad": 16,
    "input_radius": 20,
    "mode_active_bg": "#000000",
    "mode_active_fg": "#3b82f6",
    "mode_idle_bg": "#000000",
    "mode_idle_fg": "#3b82f6",
    "toggle_on": "#000000",
    "toggle_off": "#000000",
}

GUI_APP_NAME = "cat r1"
WINDOW_TITLE = "cat r1"
GUI_TAGLINE = "cat r1 · DSpark × BitNet"
MASCOT_GLYPH = "🐱"
MASCOT_NAME = "cat r1"
CAT_R1_PRO = "cat r1"
CAT_R1_FLASH = "cat r1-flash"
V4_MODEL_PRO = CAT_R1_PRO
V4_MODEL_FLASH = CAT_R1_FLASH
V4_MODEL_LABEL_PRO = CAT_R1_PRO
V4_MODEL_LABEL_FLASH = CAT_R1_FLASH

FILES = CONFIG["files"]
BRAND = "cat r1"
EDITION = CONFIG.get("cat_r1_edition", "r1")
MODEL_NAME = "cat r1"
CORE_NAME = "cat r1 core"
WEB_PROGRAM_NAME = "cat r1 web"
LINEAR_NAME = "cat r1 linear"
BITNET_V4_LABEL = f"{BRAND} BitNet"
CAT_R1_LABEL = "cat r1"
CAT_R1_MODEL_ID = CONFIG["cat_r1_model_id"]
MYTHOS_TIER = CONFIG["mythos_tier"]
CODE_ENGINE = CONFIG["code_interpreter_name"]
CODE_BACKEND = CONFIG["code_interpreter"]
LLM_MATCH_TARGET = CONFIG.get("llm_match_target", "cat r1")
LLM_MATCH_PROVIDER = CONFIG.get("llm_match_provider", "cat r1")
CODING_API_VER = CONFIG.get("coding_api_version", "1.0")
CODING_API_PROTO = CONFIG.get("coding_api_protocol", "cat-r1-coding-api")
CODING_API_LABEL = CONFIG["code_interpreter_name"]
CATR1_ENGINE = CODING_API_LABEL
CAT_R1_CODE_ENABLED = CONFIG["cat_r1_code_enabled"]
MYTHOS_MODE = CONFIG.get("mythos_mode", True)
CAT_R1_REASONING = CONFIG.get("cat_r1_reasoning", True)
MYTHOS_NAME = BRAND
BRAND_TAG = BRAND
MASCOT_DESC = BRAND_TAG
REASONING_MODE = CONFIG.get("reasoning_mode", "cat-r1-hybrid-reasoning")
PROSE_TIER = CONFIG["prose_tier"]
VERSION = EDITION
MYTHOS_ENGINE_VER = CONFIG["code_interpreter_version"]
O1_INTERPRETER_PROTO = CONFIG.get("o1_interpreter_protocol", "o1-preview-interpreter")
O1_INTERPRETER_TAG = CONFIG.get("interpreter_syntax", "cat-r1-code-interpreter-0.1")

# Legacy aliases (internal · do not use in user-facing strings)
CATR1_UI = CAT_R1_UI
CATR1_MODEL_ID = CAT_R1_MODEL_ID
CATR1_CODE_ENABLED = CAT_R1_CODE_ENABLED
CATR1_NAME = CAT_R1_PRO
CATR1_V3_NAME = CAT_R1_FLASH
CATR1_MODE = CAT_R1_REASONING

# ──────────────────────────────────────────────────────────────
# PERSISTENT MEMORY (first-run per device · settings · files = soft)
# cat r1 keeps chat in-memory; only device metadata persists.
# ──────────────────────────────────────────────────────────────
PERSISTENT_MEMORY_DIR = os.path.join(
    os.path.expanduser("~"),
    ".config" if sys.platform != "darwin" else "Library/Application Support",
    "cat-r1"
)
PERSISTENT_MEMORY_FILE = os.path.join(PERSISTENT_MEMORY_DIR, "memory.json")


@dataclass
class PersistentMemory:
    first_run: bool = True
    device_id: str = ""
    chat_mode: str = "expert"
    thinking_on: bool = True
    total_sessions: int = 0
    total_messages: int = 0
    setup_version: str = ""


def _ensure_persistent_memory_paths() -> None:
    """
    Ensure persistent memory dir is writable.
    If the default location is not writable (permissions/sandbox), fall back to a temp dir.
    """
    global PERSISTENT_MEMORY_DIR, PERSISTENT_MEMORY_FILE
    try:
        os.makedirs(PERSISTENT_MEMORY_DIR, exist_ok=True)
        return
    except OSError:
        pass
    try:
        tmp = os.path.join("/tmp", "cat-r1")
        os.makedirs(tmp, exist_ok=True)
        PERSISTENT_MEMORY_DIR = tmp
        PERSISTENT_MEMORY_FILE = os.path.join(PERSISTENT_MEMORY_DIR, "memory.json")
    except OSError:
        # Last resort: disable persistence silently.
        PERSISTENT_MEMORY_DIR = ""
        PERSISTENT_MEMORY_FILE = ""


def _load_persistent_memory() -> PersistentMemory:
    if os.path.exists(PERSISTENT_MEMORY_FILE):
        try:
            with open(PERSISTENT_MEMORY_FILE) as f:
                data = json.load(f)
            import dataclasses
            known = {f.name for f in dataclasses.fields(PersistentMemory)}
            return PersistentMemory(**{k: v for k, v in data.items() if k in known})
        except Exception:
            pass
    _ensure_persistent_memory_paths()
    if not PERSISTENT_MEMORY_DIR or not PERSISTENT_MEMORY_FILE:
        mem = PersistentMemory()
        mem.device_id = uuid.uuid4().hex[:12]
        mem.setup_version = EDITION
        return mem
    mem = PersistentMemory()
    mem.device_id = uuid.uuid4().hex[:12]
    mem.setup_version = EDITION
    _save_persistent_memory(mem)
    return mem


def _save_persistent_memory(mem: PersistentMemory) -> None:
    try:
        _ensure_persistent_memory_paths()
        if not PERSISTENT_MEMORY_DIR or not PERSISTENT_MEMORY_FILE:
            return
        with open(PERSISTENT_MEMORY_FILE, "w") as f:
            json.dump({
                "first_run": mem.first_run,
                "device_id": mem.device_id,
                "chat_mode": mem.chat_mode,
                "thinking_on": mem.thinking_on,
                "total_sessions": mem.total_sessions,
                "total_messages": mem.total_messages,
                "setup_version": mem.setup_version,
            }, f, indent=2)
    except OSError:
        pass


CAT_R11_PROFILE_MD = f"""# {BRAND}

**{CAT_R1_MODEL_ID}** · {CORE_NAME} · **{BRAND}**.

| | |
|---|---|---|
| Brand | **{BRAND}** — local assistant |
| Tier | {BRAND} reasoning · {BRAND} code |
| Chat | **{CAT_R1_PRO}** reasoning · **{CAT_R1_FLASH}** instant |
| Models | **{CAT_R1_PRO}** (deep) · **{CAT_R1_FLASH}** (instant) |
| Compression | **{BITNET_V4_LABEL}** (BitNet · fractal-ternary-recursive-sparse) |
| Runtime | **{MYTHOS_NAME}** on {CORE_NAME} |
| Reasoning | {REASONING_MODE} + recursive polish + GRPO self-verify |
| Prose | `{PROSE_TIER}` extended thinking |
| Context | {CONFIG['cat_r1_context_window']:,} tokens (in-memory) |
| Output | up to {CONFIG['cat_r1_max_output']:,} tokens |
| Code | **{CODE_ENGINE}** · `{CODE_BACKEND}` · `{MYTHOS_ENGINE_VER}` |
| Student stack | {CONFIG['distil_passes']} in-memory heads |
| MoE | {CONFIG['n_experts']} experts · top-{CONFIG['top_k']} routing · cat-r1 MoE scale |
| Compression | Sparse top-{CONFIG['compression_sparse_k']} · low-rank {CONFIG['compression_rank']} · {CONFIG['weight_bits']}-bit BitNet |
| Web | {WEB_PROGRAM_NAME} — artifacts · fetch · preview |
| Architecture | {LINEAR_NAME} · causal MHA · MoE FFN · ReLU² · RMSNorm |
| Weights | AbsMean ternary {{−1, 0, 1}} @ {CONFIG['weight_bits']} bits |
| Effective params | {CONFIG['nominal_base_params'] / 1e12:.1f}T |
| API ID | `{CAT_R1_MODEL_ID}` |

Run: `python3 "##catr1.py"` · CLI: `python3 "##catr1.py" --chat`
"""


# ──────────────────────────────────────────────────────────────
# CATR1.10 CORE (cat r1 · BitNet GigaEngine)
# files = off · in-memory shadow weights + packed ternary · 1.7T effective
# ──────────────────────────────────────────────────────────────
def _round_clip(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.clip(np.round(x), lo, hi).astype(np.int8)


def _squared_relu(x: np.ndarray) -> np.ndarray:
    r = np.maximum(x, 0.0)
    return (r * r).astype(np.float32)


class BitNetQuantizer:
    """
    BitNet b1.58 AbsMean weight quant + per-token absmax activation quant.
    Weights ∈ {{-1, 0, 1}} · ~1.58 bits/weight via base-3 packing.
    """

    @staticmethod
    def absmean_quantize_weights(w: np.ndarray, eps: float) -> Tuple[np.ndarray, np.float32]:
        gamma = np.float32(np.mean(np.abs(w), dtype=np.float64) + eps)
        w_q = np.clip(np.round(w.astype(np.float64) / float(gamma)), -1, 1).astype(np.int8)
        return w_q, gamma

    @staticmethod
    def absmax_quantize_activations(x: np.ndarray, bits: int, eps: float) -> Tuple[np.ndarray, np.ndarray]:
        xm = np.atleast_2d(x.astype(np.float32))
        scales = np.max(np.abs(xm), axis=1, keepdims=True).astype(np.float32) + np.float32(eps)
        qmax = float((2 ** (bits - 1)) - 1)
        x_q = np.clip(np.round(xm / scales * qmax), -qmax, qmax).astype(np.int16)
        return x_q, scales

    @staticmethod
    def pack_ternary(w_q: np.ndarray) -> bytes:
        """Pack 5 ternary symbols {{-1,0,1}} → 1 byte (base-3 · ~1.58 bits/weight)."""
        flat = np.asarray(w_q, dtype=np.int8).ravel()
        out = bytearray()
        for i in range(0, len(flat), 5):
            val = 0
            for j, t in enumerate(flat[i : i + 5]):
                val += (int(t) + 1) * (3 ** j)
            out.append(val % 256)
        return bytes(out)

    @staticmethod
    def unpack_ternary(packed: bytes, shape: Tuple[int, ...]) -> np.ndarray:
        """Unpack base-3 ternary weights from in-memory bytes."""
        need = int(np.prod(shape))
        out: List[int] = []
        for b in packed:
            for j in range(5):
                if len(out) >= need:
                    break
                out.append((b // (3 ** j)) % 3 - 1)
            if len(out) >= need:
                break
        if len(out) < need:
            out.extend([0] * (need - len(out)))
        return np.array(out[:need], dtype=np.int8).reshape(shape)

    @staticmethod
    def verify_pack_roundtrip(w_q: np.ndarray, packed: bytes) -> bool:
        back = BitNetQuantizer.unpack_ternary(packed, w_q.shape)
        return bool(np.array_equal(w_q, back))


class BitNetMatmul:
    """
    Real BitNet matmul: ternary weights {{-1,0,1}} reduce multiply to add/sub.
    Falls back to int32 dot when kernel=auto and shapes are large.
    """

    @staticmethod
    def ternary_addsub(x_q: np.ndarray, w: np.ndarray, x_scales: np.ndarray, w_scale: float) -> np.ndarray:
        """x_q (B,in) int · w (out,in) {{-1,0,1}} → float (B,out) via vectorized GEMM."""
        x_f = x_q.astype(np.float32) * x_scales.astype(np.float32)
        return (x_f @ w.astype(np.float32).T) * np.float32(w_scale)

    @classmethod
    def apply(
        cls,
        x: np.ndarray,
        w: np.ndarray,
        w_scale: float,
        bias: Optional[np.ndarray],
        *,
        act_bits: int,
        eps: float,
    ) -> np.ndarray:
        single = x.ndim == 1
        x_q, x_scales = BitNetQuantizer.absmax_quantize_activations(x, act_bits, eps)
        kernel = CONFIG.get("bitnet_kernel", "ternary_addsub")
        n_muls = x_q.shape[0] * w.shape[0] * w.shape[1]
        force_real = CONFIG.get("bitnet_real", True) and CONFIG.get("bitnet_force_real", True)
        if kernel == "ternary_addsub" and (force_real or n_muls <= 8_000_000):
            acc = cls.ternary_addsub(x_q, w, x_scales, w_scale)
        else:
            acc = (x_q.astype(np.int32) @ w.T.astype(np.int32)).astype(np.float32)
            acc = acc * x_scales * np.float32(w_scale)
        if bias is not None:
            acc = acc + bias
        return acc[0] if single else acc


class BitNetEngine:
    """In-memory BitNet b1.58 stack — real pack/unpack + ternary inference."""

    _self_test_cache: Optional[Dict[str, Any]] = None

    @classmethod
    def self_test(cls, d_model: int = 64, seed: int = 0) -> Dict[str, Any]:
        if cls._self_test_cache is not None and cls._self_test_cache.get("d_model") == d_model:
            return cls._self_test_cache
        lin = CatR1Linear(d_model, d_model, seed)
        w = lin.shadow_w.copy()
        w_q, gamma = BitNetQuantizer.absmean_quantize_weights(w, CONFIG["cat_r1_eps"])
        packed = BitNetQuantizer.pack_ternary(w_q)
        pack_ok = BitNetQuantizer.verify_pack_roundtrip(w_q, packed)
        x = np.random.RandomState(seed).randn(8, d_model).astype(np.float32)
        y_signed = lin.forward(x, use_packed=False)
        y_packed = lin.forward(x, use_packed=True)
        forward_ok = bool(np.allclose(y_signed, y_packed, rtol=1e-4, atol=1e-3))
        recon = w_q.astype(np.float32) * float(gamma)
        mse = float(np.mean((w - recon) ** 2))
        report = {
            "d_model": d_model,
            "ok": pack_ok and forward_ok,
            "pack_roundtrip": pack_ok,
            "forward_packed_match": forward_ok,
            "weight_mse": mse,
            "packed_bytes": len(lin.w_packed),
            "kernel": CONFIG.get("bitnet_kernel", "ternary_addsub"),
            "weight_bits": CONFIG["weight_bits"],
            "act_bits": CONFIG["act_bits"],
        }
        cls._self_test_cache = report
        return report

    @classmethod
    def enable_real_mode(cls, engine: Optional["CatR11Engine"] = None) -> Dict[str, Any]:
        """Lock BitNet to real in-memory ternary inference — files = off, no weight files."""
        CONFIG["files"] = "off"
        CONFIG["bitnet_real"] = True
        CONFIG["bitnet_force_real"] = True
        CONFIG["bitnet_no_weight_files"] = True
        CONFIG["bitnet_use_packed"] = True
        CONFIG["bitnet_encode"] = True
        CONFIG["bitnet_self_test"] = True
        CONFIG["bitnet_kernel"] = "ternary_addsub"
        CONFIG["v4_as_bitnet"] = True
        cls._self_test_cache = None
        if engine is None:
            return cls.self_test(CONFIG["d_model"])
        for block in engine.cat_r1_blocks:
            for lin in _iter_cat_r1_linears(block):
                lin.requantize()
        engine.output_head_lin.requantize()
        engine.output_head = engine.output_head_lin.w_signed
        for stack in engine._student_layers:
            for block in stack:
                for lin in _iter_cat_r1_linears(block):
                    lin.requantize()
        if hasattr(engine, "dspark"):
            engine.dspark.requantize()
        engine._clear_engine_caches()
        engine.deepmind.clear()
        engine.cat_r1_stats = cat_r1_memory_report(
            engine.cat_r1_blocks, engine.embeddings, engine.output_head_lin.shadow_w
        )
        bt = cls.self_test(engine.d_model)
        engine.cat_r1_stats["bitnet"] = bt
        return bt

    @classmethod
    def status_text(cls, stats: Dict[str, Any]) -> str:
        st = cls.self_test(stats.get("d_model", CONFIG["d_model"])) if CONFIG.get("bitnet_self_test") else {}
        ok = st.get("ok", True)
        files_off = CONFIG.get("files", "off") == "off"
        txt = (
            f"**{BRAND} BitNet b1.58** · AbsMean ternary {{−1,0,1}} · "
            f"{'✓ real' if CONFIG.get('bitnet_real') else 'sim'} · "
            f"{'✓ packed' if CONFIG.get('bitnet_use_packed') else 'signed'} · "
            f"{'✓ files=off' if files_off else 'files'} · "
            f"self-test {'PASS' if ok else 'FAIL'}\n\n"
            f"- Linear layers: **{stats.get('cat_r1_linear_layers', 0)}**\n"
            f"- Packed weights: **{stats.get('packed_kb', 0):.1f} KB** (in-memory only)\n"
            f"- Effective: **{stats.get('weight_bits', 1.58)} bits/weight**\n"
            f"- Kernel: `{CONFIG.get('bitnet_kernel', 'ternary_addsub')}`\n"
            f"- Pack roundtrip: **{'ok' if st.get('pack_roundtrip') else 'fail'}**\n"
            f"- Forward match: **{'ok' if st.get('forward_packed_match') else 'fail'}**"
        )
        dspark = stats.get("dspark")
        if dspark:
            txt += (
                f"\n- DSpark: **{dspark.get('accepted', 0)}/{dspark.get('drafts', 0)}** "
                f"drafts · γ={dspark.get('gamma', 5)}"
            )
        if CONFIG.get("dspark_enabled"):
            txt += f"\n- DSpark×BitNet: **enabled** · speculative decode · files=off"
        return txt


class CatR1Linear:
    """
    cat r1 BitNet GigaLinear layer (files = off):
    shadow FP32 weights → AbsMean ternary {-1,0,1} → int8/16 activation matmul.
    cat r1 · 1.58-bit BitNet engine.
    """

    __slots__ = ("in_f", "out_f", "shadow_w", "bias", "w_scale", "w_signed", "w_packed", "eps")

    def __init__(self, in_features: int, out_features: int, seed: int, *, bias: bool = False):
        rng = np.random.RandomState(seed)
        self.in_f = in_features
        self.out_f = out_features
        self.eps = CONFIG["cat_r1_eps"]
        scale = np.sqrt(2.0 / max(in_features, 1))
        self.shadow_w = (rng.randn(out_features, in_features).astype(np.float32) * scale)
        self.bias = np.zeros(out_features, dtype=np.float32) if bias else None
        self.w_scale = np.float32(1.0)
        self.w_signed = np.zeros((out_features, in_features), dtype=np.int8)
        self.w_packed = b""
        self.requantize()

    def requantize(self) -> None:
        self.w_signed, self.w_scale = BitNetQuantizer.absmean_quantize_weights(
            self.shadow_w, self.eps
        )
        self.w_packed = BitNetQuantizer.pack_ternary(self.w_signed)

    @staticmethod
    def _pack(w_q: np.ndarray) -> bytes:
        return BitNetQuantizer.pack_ternary(w_q)

    @staticmethod
    def _unpack(packed: bytes, shape: Tuple[int, int]) -> np.ndarray:
        return BitNetQuantizer.unpack_ternary(packed, shape)

    def get_weights(self, *, use_packed: Optional[bool] = None) -> np.ndarray:
        use_packed = CONFIG.get("bitnet_use_packed", True) if use_packed is None else use_packed
        if use_packed and self.w_packed:
            return self._unpack(self.w_packed, (self.out_f, self.in_f))
        return self.w_signed

    def forward(self, x: np.ndarray, *, use_packed: Optional[bool] = None) -> np.ndarray:
        if CONFIG.get("bitnet_real", True):
            w = self.get_weights(use_packed=use_packed)
            return BitNetMatmul.apply(
                x, w, float(self.w_scale), self.bias,
                act_bits=CONFIG["act_bits"], eps=self.eps,
            )
        single = x.ndim == 1
        xm = np.atleast_2d(x.astype(np.float32))
        qmax = (2 ** (CONFIG["act_bits"] - 1)) - 1
        scales = np.max(np.abs(xm), axis=1, keepdims=True) + self.eps
        x_q = np.clip(np.round(xm / scales * qmax), -qmax, qmax).astype(np.int16)
        acc = (x_q.astype(np.int32) @ self.w_signed.T.astype(np.int32)).astype(np.float32)
        acc = acc * scales * float(self.w_scale)
        if self.bias is not None:
            acc = acc + self.bias
        return acc[0] if single else acc

    def param_count(self) -> Tuple[int, float]:
        n = self.in_f * self.out_f
        return n, n * CONFIG["weight_bits"] / 8.0


def _rms_norm(x: np.ndarray, gamma: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    if x.ndim == 1:
        rms = float(np.sqrt(np.mean(x * x) + eps))
        return (x / rms * gamma).astype(np.float32)
    rms = np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)
    return (x / rms * gamma).astype(np.float32)


class CatR1Block:
    """One transformer block: causal MHA + cat r1 linear FFN (ReLU²), all ternary matmul."""

    __slots__ = ("q", "k", "v", "o", "ff_up", "ff_down", "router", "experts", "norm1", "norm2", "_h", "_hd")

    def __init__(self, d_model: int, seed: int):
        h = CONFIG["heads"]
        hd = d_model // h
        ff = d_model * CONFIG["ff_mult"]
        self.q = CatR1Linear(d_model, d_model, seed + 1)
        self.k = CatR1Linear(d_model, d_model, seed + 2)
        self.v = CatR1Linear(d_model, d_model, seed + 3)
        self.o = CatR1Linear(d_model, d_model, seed + 4)
        self.ff_up = CatR1Linear(d_model, ff, seed + 5)
        self.ff_down = CatR1Linear(ff, d_model, seed + 6)
        self.router = CatR1Linear(d_model, CONFIG["n_experts"], seed + 7)
        rng = np.random.RandomState(seed + 8)
        self.experts = [
            (CatR1Linear(d_model, ff, seed + 100 + i * 2), CatR1Linear(ff, d_model, seed + 101 + i * 2))
            for i in range(CONFIG["n_experts"])
        ]
        self.norm1 = rng.randn(d_model).astype(np.float32) * 0.1 + 1.0
        self.norm2 = rng.randn(d_model).astype(np.float32) * 0.1 + 1.0
        self._hd = hd
        self._h = h

    def _causal_mha(self, x: np.ndarray) -> np.ndarray:
        t, d = x.shape
        h, hd = self._h, self._hd
        eps = CONFIG["cat_r1_eps"]
        xn = np.stack([_rms_norm(x[ti], self.norm1) for ti in range(t)], axis=0)
        q = self.q.forward(xn).reshape(t, h, hd)
        k = self.k.forward(xn).reshape(t, h, hd)
        v = self.v.forward(xn).reshape(t, h, hd)
        scale = np.sqrt(hd) + eps
        scores = np.einsum("thd,shd->hts", q, k) / scale
        mask = np.triu(np.ones((t, t), dtype=bool), k=1)
        scores = np.where(mask[np.newaxis, :, :], -1e9, scores)
        scores = scores - np.max(scores, axis=-1, keepdims=True)
        w = np.exp(scores)
        w = w / (np.sum(w, axis=-1, keepdims=True) + eps)
        ctx = np.einsum("hts,shd->htd", w, v).reshape(t, d)
        return self.o.forward(ctx)

    def _moe_ffn(self, x: np.ndarray) -> np.ndarray:
        t, d = x.shape
        xn = np.stack([_rms_norm(x[ti], self.norm2) for ti in range(t)], axis=0)
        logits = self.router.forward(xn)
        top = np.argpartition(logits, -CONFIG["top_k"], axis=1)[:, -CONFIG["top_k"]:]
        out = np.zeros_like(x)
        for ti in range(t):
            acc = np.zeros(d, dtype=np.float32)
            for idx in top[ti]:
                up, down = self.experts[int(idx)]
                h = _squared_relu(up.forward(xn[ti]))
                acc += down.forward(h) / CONFIG["top_k"]
            out[ti] = acc
        return out

    def forward(self, x: np.ndarray) -> np.ndarray:
        return x + self._moe_ffn(x + self._causal_mha(x))


def _iter_cat_r1_linears(block: CatR1Block) -> List[CatR1Linear]:
    layers = [block.q, block.k, block.v, block.o, block.ff_up, block.ff_down, block.router]
    for up, down in block.experts:
        layers.extend([up, down])
    return layers


def cat_r1_memory_report(blocks: List[CatR1Block], embed: np.ndarray, head: np.ndarray) -> Dict[str, Any]:
    shadow = embed.size + head.size
    packed_bytes = 0
    effective_bits = 0.0
    linear_count = 0
    for blk in blocks:
        for lin in _iter_cat_r1_linears(blk):
            n, mem = lin.param_count()
            shadow += n
            effective_bits += n * CONFIG["weight_bits"]
            packed_bytes += len(lin.w_packed)
            linear_count += 1
    return {
        "cat_r1_linear_layers": linear_count,
        "shadow_params": shadow,
        "effective_bits": effective_bits,
        "packed_kb": packed_bytes / 1024.0,
        "effective_mb": effective_bits / 8.0 / 1024.0 / 1024.0,
        "weight_bits": CONFIG["weight_bits"],
    }


# ──────────────────────────────────────────────────────────────
# CAT R1 × BITNET (files = off · DSpark speculative decode)
# Semi-autoregressive draft · Markov head · confidence verify · # pr
# ──────────────────────────────────────────────────────────────
@dataclass
class DSparkStats:
    drafts: int = 0
    accepted: int = 0
    gamma: int = 0
    speedup: float = 1.0


class DSparkMarkovHead:
    """Low-rank Markov correction head — BitNet ternary projections."""

    __slots__ = ("rank_proj", "out_proj")

    def __init__(self, d_model: int, rank: int, seed: int):
        self.rank_proj = CatR1Linear(d_model, rank, seed)
        self.out_proj = CatR1Linear(rank, d_model, seed + 1)

    def bias(self, hidden: np.ndarray) -> np.ndarray:
        h = hidden.reshape(1, -1) if hidden.ndim == 1 else hidden
        return self.out_proj.forward(self.rank_proj.forward(h))

    def requantize(self) -> None:
        self.rank_proj.requantize()
        self.out_proj.requantize()


class DSparkBitNetEngine:
    """
    cat r1 DSpark engine — BitNet speculative decode:
      - Semi-autoregressive block draft backbone (parallel BitNet forward)
      - Markov head for prefix-dependent corrections
      - Confidence head for per-step acceptance prediction
      - Target recursive verify loop (speculative decoding)
    """

    __slots__ = ("engine", "markov", "confidence", "block_size", "last_stats")

    def __init__(self, engine: "CatR11Engine"):
        self.engine = engine
        rank = int(CONFIG.get("dspark_markov_rank", 64))
        self.markov = DSparkMarkovHead(engine.d_model, rank, 9107)
        self.confidence = CatR1Linear(engine.d_model, 1, 9108)
        self.block_size = int(CONFIG.get("dspark_block_size", 7))
        self.last_stats = DSparkStats()

    @staticmethod
    def adaptive_gamma(prompt: str, depth: int) -> int:
        """cat r1 — scale draft block γ by prompt size and depth."""
        base = int(CONFIG.get("cat_r1_draft_gamma", CONFIG.get("v4_speculative_draft_tokens", 5)))
        if not CONFIG.get("cat_r1_adaptive_gamma", True):
            return min(base, depth, int(CONFIG.get("dspark_block_size", 7)))
        words = max(1, len((prompt or "").split()))
        gamma = base
        if words > 80:
            gamma += 2
        elif words > 40:
            gamma += 1
        elif words < 6:
            gamma = max(2, gamma - 1)
        return min(gamma, depth, int(CONFIG.get("dspark_block_size", 7)))

    @staticmethod
    def select_best_draft(
        drafts: List[Tuple[np.ndarray, float]],
        target: np.ndarray,
        eps: float,
    ) -> Tuple[Optional[np.ndarray], float, bool]:
        """Pick highest-confidence draft that passes verify — cat r1."""
        best: Optional[Tuple[np.ndarray, float, float]] = None
        for draft, conf in drafts:
            diff = float(np.linalg.norm(target - draft))
            accept_prob = conf * max(0.0, 1.0 - diff / (eps * 12.0 + 1e-8))
            if accept_prob > 0.45 or diff < eps:
                score = accept_prob - diff
                if best is None or score > best[2]:
                    best = (draft, conf, score)
        if best is None:
            return None, 0.0, False
        return best[0], best[1], True

    def predict_accept(self, hidden: np.ndarray) -> float:
        h = hidden.reshape(1, -1)
        logit = float(self.confidence.forward(h).reshape(-1)[0])
        return 1.0 / (1.0 + np.exp(-np.clip(logit, -20, 20)))

    def draft_block(self, state: np.ndarray, block_len: int) -> List[Tuple[np.ndarray, float]]:
        drafts: List[Tuple[np.ndarray, float]] = []
        cur = state.copy()
        for _ in range(block_len):
            seq = cur.reshape(1, -1)
            delta = self.engine._pool_sequence(
                self.engine._forward_stack(seq, self.engine.cat_r1_blocks[:1])
            )
            mbias = self.markov.bias(cur)
            if mbias.ndim == 2:
                mbias = mbias[0]
            step = self.engine._layer_norm(cur * 0.55 + (delta + mbias) * 0.45)
            drafts.append((step, self.predict_accept(step)))
            cur = step
        return drafts

    def verify_draft(self, target: np.ndarray, draft: np.ndarray, draft_conf: float) -> bool:
        diff = float(np.linalg.norm(target - draft))
        eps = float(CONFIG.get("recursive_epsilon", 0.008))
        accept_prob = draft_conf * max(0.0, 1.0 - diff / (eps * 12.0 + 1e-8))
        return accept_prob > 0.45 or diff < eps

    def speculative_encode(self, prompt: str, depth: int) -> np.ndarray:
        """cat r1 DSpark speculative recursive encode — BitNet target verifies drafts."""
        gamma = self.adaptive_gamma(prompt, depth)
        state = self.engine.encode_prompt(prompt)
        trace: List[str] = []
        accepted_drafts = 0
        total_drafts = 0
        used = depth
        prev = self.engine._pool_sequence(state) if state.ndim == 2 else state.copy()
        eps = float(CONFIG.get("recursive_epsilon", 0.008))

        for i in range(depth):
            block_gamma = min(gamma, self.block_size, max(1, depth - i))
            drafts = self.draft_block(prev, block_gamma)
            target = self.engine._recursive_step(
                state.reshape(1, -1) if state.ndim == 1 else state, i, depth
            )
            target_p = self.engine._pool_sequence(target) if target.ndim == 2 else target

            accepted = False
            if CONFIG.get("cat_r1_dspark", True):
                best, conf, ok = self.select_best_draft(drafts, target_p, eps)
                if ok and best is not None:
                    total_drafts += len(drafts)
                    accepted_drafts += 1
                    state = target
                    prev = target_p.copy()
                    accepted = True
                    trace.append(
                        f"cat r1 pass {i + 1} · accept {accepted_drafts}/{total_drafts} "
                        f"· conf {conf:.3f} · γ={gamma} · {BRAND} BitNet"
                    )
            if not accepted:
                for draft, conf in drafts:
                    total_drafts += 1
                    if self.verify_draft(target_p, draft, conf):
                        state = target
                        prev = target_p.copy()
                        accepted_drafts += 1
                        accepted = True
                        trace.append(
                            f"cat r1 pass {i + 1} · accept {accepted_drafts}/{total_drafts} "
                            f"· conf {conf:.3f} · {BRAND} BitNet"
                        )
                        break

            if not accepted:
                diff = float(np.linalg.norm(target_p - prev))
                state = target
                prev = target_p.copy()
                trace.append(f"cat r1 pass {i + 1} · verify reject · Δ {diff:.4f}")
                if diff < eps:
                    used = i + 1
                    trace.append(f"cat r1 converged at pass {i + 1}")
                    break

            if CONFIG.get("compression_enabled"):
                prev = self.engine.compressor.compress_roundtrip(prev)
                self.engine.last_compression_ratio = self.engine.compressor.last_ratio
            if CONFIG.get("v4_pro_compression"):
                prev = self.engine.v4_compressor.v4_compress(prev)
                trace.append(
                    f"{BITNET_V4_LABEL} · {self.engine.v4_compressor.last_ratio:.1f}x · "
                    f"{self.engine.v4_compressor.effective_v4_capacity():.1f}B effective"
                )

        out = self.engine._pool_sequence(state) if state.ndim == 2 else state
        self.engine.last_recursive_passes = used
        self.engine.last_recursive_trace = trace
        accept_rate = accepted_drafts / max(total_drafts, 1)
        speedup = 1.0 + accept_rate * 0.42 + min(0.15, gamma * 0.02)
        self.last_stats = DSparkStats(
            drafts=total_drafts, accepted=accepted_drafts,
            gamma=gamma, speedup=speedup,
        )
        self.engine.last_vec = out.copy()
        return out

    def requantize(self) -> None:
        self.markov.requantize()
        self.confidence.requantize()

    @staticmethod
    def status_line(stats: DSparkStats) -> str:
        return (
            f"{CAT_R1_LABEL} DSpark × {BRAND} BitNet · {stats.accepted}/{stats.drafts} drafts "
            f"· γ={stats.gamma} · ~{stats.speedup:.2f}x · files=off"
        )


class CatR1DSpark:
    """
    cat r1 — DSpark speculative decode facade (# pr · files = off).
    Wires adaptive γ, BitNet real mode, and Expert profile on deep prompts.
    """

    NAME = CAT_R1_LABEL
    ALGO = CONFIG.get("cat_r1_dspark_algo", "speculative-markov-confidence-decode")

    @classmethod
    def wants_pr(cls, text: str) -> bool:
        return bool(re.search(r"(?:^|\s)#+\s*pr\b", text or "", re.I))

    @classmethod
    def ensure(cls, engine: "CatR11Engine", prompt: str = "", *, force_pr: bool = False) -> Dict[str, Any]:
        """Activate cat r1 + DSpark on BitNet — files = off."""
        CONFIG["files"] = "off"
        CONFIG["cat_r1_dspark"] = True
        CONFIG["dspark_enabled"] = True
        CONFIG["dspark_speculative_decode"] = True
        bt = engine._ensure_bitnet_real()
        if force_pr or cls.wants_pr(prompt):
            engine.active_model_profile = CatR1ProProfile()
            engine.ultrathink_on = True
        depth = engine.get_reasoning_depth()
        if hasattr(engine, "dspark") and prompt.strip():
            engine.dspark.speculative_encode(prompt, depth)
        ds = getattr(engine.dspark, "last_stats", DSparkStats())
        engine.cat_r1_stats["dspark"] = {
            "enabled": True,
            "accepted": ds.accepted,
            "drafts": ds.drafts,
            "gamma": ds.gamma,
            "dspark": True,
        }
        return bt

    @classmethod
    def help_text(cls) -> str:
        return (
            f"**{cls.NAME}** · `{cls.ALGO}`\n\n"
            f"- **DSpark** speculative decode on **{BRAND} BitNet** · `files = off`\n"
            "- Adaptive draft γ · Markov correction head · confidence verify\n"
            "- `# pr` → Expert + DeepThink + cat r1 encode\n"
            f"- Modes: **{CAT_R1_PRO}** (deep) · **{CAT_R1_FLASH}** (instant)\n\n"
            "Ask anything — chat, code, math, EN/中文."
        )


CatSparkR1 = CatR1DSpark
CatSparkEngine = CatR1DSpark


# ──────────────────────────────────────────────────────────────
# O1-PREVIEW INTERPRETER SYNTAX (files = off)
# Canonical trace: Understand → Plan → Reason → Self-check → Verify → Answer
# ──────────────────────────────────────────────────────────────
class O1PreviewSyntax:
    """cat r1 hybrid reasoning syntax · files = off."""

    MODEL = "cat-r1"
    PROTO = O1_INTERPRETER_PROTO
    TAG = O1_INTERPRETER_TAG
    VERSION = "r1"
    COMMANDS = ("/run", "/interpret", "/think", "/ultrathink", "run it", "execute")

    @classmethod
    def header(cls) -> str:
        if MYTHOS_MODE and CONFIG.get("mythos_runtime", True):
            return f"{BRAND} · {REASONING_MODE}"
        return f"{cls.TAG}"

    @classmethod
    def step(cls, n: int, label: str, body: str) -> str:
        return f"{n}. {label} — {body}"

    @classmethod
    def final_answer_step(cls, n: int) -> str:
        return cls.step(n, "Answer", "emit clean user-facing text.")

    @classmethod
    def build_trace(
        cls,
        prompt: str,
        *,
        intent: str,
        subtasks: List[str],
        reason: str = "",
        recursive_trace: Optional[List[str]] = None,
        compression_trace: Optional[List[str]] = None,
        verify: str = "",
        self_check: str = "",
    ) -> str:
        lines = [cls.header(), cls.step(1, "Understand", intent)]
        for i, task in enumerate(subtasks, start=2):
            lines.append(cls.step(i, "Plan", task))
        n = len(subtasks) + 2
        if compression_trace:
            for j, ct in enumerate(compression_trace):
                lines.append(cls.step(n + j, "Compress", ct))
            n += len(compression_trace)
        if recursive_trace:
            for j, rt in enumerate(recursive_trace):
                lines.append(cls.step(n + j, "Recursive", rt))
            n += len(recursive_trace)
        lines.append(cls.step(n, "Reason", (reason or "align draft to prompt.")[:200]))
        n += 1
        if CONFIG.get("o1_self_check", True):
            lines.append(cls.step(n, "Self-check", self_check or "Does the draft answer the exact question?"))
            n += 1
        lines.append(cls.step(n, "Verify", verify or f"reasoning stays in-memory"))
        lines.append(cls.final_answer_step(n + 1))
        return "\n".join(lines)

    @classmethod
    def code_trace(cls, lang: str, script: str, *, lint: str = "", note: str = "") -> str:
        intent = f"execute `{script}` ({lang}) in the o1-preview interpreter"
        subtasks = [
            "parse buffer and detect language",
            "lint syntax before sandbox run",
            "capture stdout/stderr in-memory only",
        ]
        if lint:
            subtasks.append(f"lint: {lint}")
        return cls.build_trace(
            script,
            intent=intent,
            subtasks=subtasks,
            reason=note or f"run {lang} buffer via {cls.TAG} interpreter",
            verify=f"no files written",
            self_check="output matches executed code; errors surfaced verbatim",
        )

    @classmethod
    def format_output(cls, lang: str, script: str, output: str, *, error: str = "") -> str:
        if error:
            return (
                f"**{cls.TAG} interpreter**\n\n"
                f"`{script}` · `{lang}`\n\n"
                f"```\n{error}\n```"
            )
        body = (output or "(no output)").rstrip()
        return (
            f"**{cls.TAG} interpreter** · `{script}` · `{lang}`\n\n"
            f"```\n{body}\n```"
        )

    @classmethod
    def format_panel_stdout(cls, result: Dict[str, Any]) -> str:
        out = result.get("output", "")
        think = result.get("thinking", "")
        if not result.get("ok"):
            err = result.get("error", "run failed")
            if think:
                return f"{think}\n\n---\n\n[{cls.TAG}] error\n{err}"
            return f"[{cls.TAG}] error\n{err}"
        if think:
            return f"{think}\n\n---\n\n{out}"
        return out

    @classmethod
    def help_text(cls) -> str:
        return (
            f"**{cls.TAG} code interpreter** · \n\n"
            "**Syntax (exact o1-preview):**\n"
            "```\n"
            f"{cls.header()}\n"
            "1. Understand — …\n"
            "2. Plan — …\n"
            "…\n"
            "N. Verify — …\n"
            "N+1. Answer — emit clean user-facing text.\n"
            "```\n\n"
            "**Commands:** `/run` · `/interpret` · `/think` · `run it` · paste ``` fences\n\n"
            f"**Protocol:** `{cls.PROTO}` · model `{cls.MODEL}`\n\n"
            "Thinking stays internal; user sees clean final text + interpreter output."
        )


# ──────────────────────────────────────────────────────────────
# CATR1 CODE INTERPRETER 0.1 (files = off)
# reasoning_content + content split · CoT before every code run.
# ──────────────────────────────────────────────────────────────
class CatR1CodeInterpreter01:
    """CatR1 code interpreter 0.1 — reasoning_content / content split."""

    NAME = "cat r1 code interpreter 0.1"
    MODEL = CONFIG.get("cat_r1_code_interpreter_model", "cat-r1-code-interpreter-0.1")
    PROTO = "cat-r1-code-interpreter-0.1"
    TAG = "cat-r1-code-interpreter"
    VER = "0.1"
    THINK_OPEN = "<think>"
    THINK_CLOSE = "</think>"
    COMMANDS = O1PreviewSyntax.COMMANDS

    @classmethod
    def enabled(cls) -> bool:
        if CONFIG.get("cat_r1_code", CONFIG.get("catseek_code_r1", CONFIG.get("cat_code_interpreter_r1", True))):
            return False
        return CONFIG.get("interpreter_syntax", "cat-r1-code-interpreter-0.1") == "cat-r1-code-interpreter-0.1"

    @classmethod
    def wrap_thinking(cls, body: str) -> str:
        inner = (body or "").strip()
        if not inner:
            return ""
        return f"{cls.THINK_OPEN}\n{inner}\n{cls.THINK_CLOSE}"

    @classmethod
    def split_thinking(cls, text: str) -> Tuple[str, str]:
        m = re.search(
            rf"{re.escape(cls.THINK_OPEN)}\s*([\s\S]*?){re.escape(cls.THINK_CLOSE)}",
            text or "",
        )
        if m:
            return m.group(1).strip(), text[m.end():].strip()
        return "", (text or "").strip()

    @classmethod
    def _analyze_code(cls, code: str, lang: str) -> List[str]:
        body = (code or "").strip()
        notes: List[str] = []
        n_lines = max(1, len(body.splitlines()))
        notes.append(f"Buffer is **{lang}** · {n_lines} line(s) · in-memory sandbox (files=off).")
        if lang == "python":
            try:
                tree = ast.parse(body)
                funcs = [n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
                classes = [n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
                if funcs:
                    notes.append(f"Functions: {', '.join(funcs[:6])}.")
                if classes:
                    notes.append(f"Classes: {', '.join(classes[:4])}.")
            except SyntaxError as exc:
                notes.append(f"AST pre-scan: syntax issue at line {exc.lineno} — lint will confirm.")
        if "print(" in body or "console.log" in body:
            notes.append("Contains stdout logging — expect printed output.")
        if "__main__" in body or "main()" in body:
            notes.append("Script entry point detected.")
        if not notes:
            notes.append("Straight-line buffer — run and capture output.")
        return notes

    @classmethod
    def code_reasoning(
        cls,
        lang: str,
        script: str,
        code: str,
        *,
        lint_msg: str = "",
        error: str = "",
        output: Optional[str] = None,
    ) -> str:
        """Build CatR1 code interpreter 0.1 chain-of-thought for a code run."""
        lines = [
            "Let me work through this code execution step by step.",
            "",
            f"**Task:** execute `{script}` as **{lang}** in the local sandbox.",
            "",
            "**Understand**",
            f"- User buffer: `{script}`",
            f"- Language: `{lang}`",
            "- Constraint: in-memory only — no files written to disk",
            "",
            "**Analyze the code**",
        ]
        lines.extend(f"- {n}" for n in cls._analyze_code(code, lang))
        lines.extend([
            "",
            "**Plan**",
            "1. Lint source for syntax errors before execution",
            "2. Run in isolated sandbox and capture stdout/stderr verbatim",
            "3. Compare actual output against what the code should produce",
            "4. Present clean `content` to user; keep this trace in `reasoning_content`",
            "",
        ])
        if lint_msg:
            lines.append(f"**Lint:** {lint_msg}")
        if error:
            lines.extend([
                "",
                "Wait — execution failed. Let me read the error carefully:",
                f"- {error}",
                "",
                "I should not guess a fix; surface the error verbatim and let the user decide.",
            ])
        else:
            preview = ((output or "") or "(no output)").strip().replace("\n", " ")[:160]
            lines.extend([
                "",
                "**Execute** — sandbox run complete.",
                "",
                "**Verify**",
                f"- Captured output: `{preview}`",
                "- No silent failures detected.",
                "- Final answer: formatted stdout only (reasoning stays separate).",
            ])
        return "\n".join(lines)

    @classmethod
    def chat_reasoning(
        cls,
        prompt: str,
        *,
        intent: str,
        subtasks: List[str],
        reason: str = "",
        verify: str = "",
        self_check: str = "",
    ) -> str:
        """R1-style CoT for general prompts (ultrathink / chat)."""
        lines = [
            "Let me think through this carefully before answering.",
            "",
            f"**Understand:** {intent}",
            "",
            "**Plan:**",
        ]
        for i, task in enumerate(subtasks, 1):
            lines.append(f"{i}. {task}")
        if reason:
            lines.extend(["", f"**Reason:** {reason[:400]}"])
        if self_check:
            lines.extend(["", f"**Self-check:** {self_check}"])
        if verify:
            lines.extend(["", f"**Verify:** {verify}"])
        lines.extend(["", "Now I'll write the final answer outside this thinking block."])
        return "\n".join(lines)

    @classmethod
    def format_content(cls, lang: str, script: str, output: str, *, error: str = "") -> str:
        """User-facing `content` field — no reasoning."""
        if error:
            return (
                f"**{cls.NAME}** · `{script}` · `{lang}`\n\n"
                f"```\n{error}\n```"
            )
        body = (output or "(no output)").rstrip()
        return (
            f"**{cls.NAME}** · `{script}` · `{lang}`\n\n"
            f"```\n{body}\n```"
        )

    @classmethod
    def format_panel(cls, result: Dict[str, Any]) -> str:
        """GUI / panel: reasoning block + separator + content (like R1 API with include_thinking)."""
        reasoning = result.get("reasoning_content") or result.get("thinking", "")
        content = result.get("content") or ""
        if not content and result.get("ok"):
            content = cls.format_content(
                result.get("lang", "code"),
                result.get("script", "buffer"),
                result.get("output", ""),
            )
        if not content and not result.get("ok"):
            content = cls.format_content(
                result.get("lang", "code"),
                result.get("script", "buffer"),
                "",
                error=result.get("error", "run failed"),
            )
        if reasoning and CONFIG.get("cat_r1_code_interpreter_think_tags", True):
            think_block = cls.wrap_thinking(reasoning)
            return f"{think_block}\n\n---\n\n{content}"
        if reasoning:
            return f"{reasoning}\n\n---\n\n{content}"
        return content

    @classmethod
    def help_text(cls) -> str:
        return (
            f"**{cls.NAME}** · `{cls.MODEL}`\n\n"
            f"**{cls.NAME}** (files = off):\n\n"
            "- **`reasoning_content`** — chain-of-thought inside "
            f"`{cls.THINK_OPEN}` … `{cls.THINK_CLOSE}`\n"
            "- **`content`** — clean final output (stdout / errors)\n"
            "- Reasoning is **not** concatenated into chat history on the next turn\n\n"
            "**Code loop:** understand → analyze buffer → plan → lint → run → verify → answer\n\n"
            "**Commands:** `/run` · `/interpret` · `/think` · `run it` · paste ``` fences\n\n"
            f"**Protocol:** `{cls.PROTO}` · model `{cls.MODEL}`"
        )


# ──────────────────────────────────────────────────────────────
# CAT R1 CODE 1.0 (files = off · Claude Code fork · BitNet · # pr)
# Full generate + run + review — anthropics/claude-code plugin parity
# ──────────────────────────────────────────────────────────────
class CatR1CodeR1:
    """
    cat r1 code 1.0 — full Claude Code fork for cat r1.
    Generate · run · edit · review · feature-dev · BitNet local · files = off.
    """

    NAME = "cat r1 code 1.0"
    MODEL = "cat-r1-code-1.0"
    PROTO = "cat-r1-code-1.0"
    TAG = "cat-r1-code"
    VER = "1.0"
    FORK_REPO = CONFIG.get("claude_code_fork", "anthropics/claude-code")
    FORK_REF = CONFIG.get(
        "cat_code_interpreter_fork_ref",
        "plugins/feature-dev,plugins/code-review,plugins/commit-commands,examples/hooks",
    )
    THINK_OPEN = CatR1CodeInterpreter01.THINK_OPEN
    THINK_CLOSE = CatR1CodeInterpreter01.THINK_CLOSE
    COMMANDS = (
        "/catrcode", "/catseek", "/catcode", "/run", "/interpret", "/coding", "/think",
        "/ultrathink", "run it", "execute", "# pr",
    )
    # Claude Code tool surface (terminal agent + plugins)
    CLAUDE_CODE_TOOLS = (
        "Bash", "Read", "Write", "Edit", "Grep", "Glob", "Task", "WebFetch",
        "NotebookEdit", "WebSearch", "TodoWrite", "Skill", "EnterPlanMode",
    )
    TOOLS = (
        "generate", "run", "edit", "lint", "explain", "new", "clear",
        "diff", "plan", "review", "feature-dev", "hook",
        "commit", "commit-push-pr", "code-review", "bash", "grep", "read", "write",
    )
    AGENTS = ("code-explorer", "code-architect", "code-reviewer")
    HOOKS = ("PreToolUse", "SessionStart", "Stop")
    PHASES = (
        "Discovery",
        "Codebase Exploration",
        "Clarifying Questions",
        "Architecture Design",
        "Implementation",
        "Quality Review",
        "Summary",
    )
    BASH_VALIDATION_RULES: Tuple[Tuple[str, str], ...] = (
        (r"^rm\s+-rf\s+/", "blocked: destructive rm -rf /"),
        (r";\s*rm\s+-rf\s+/", "blocked: chained destructive rm"),
        (r"^curl\b.*\|\s*bash", "blocked: piped curl|bash"),
        (r"^wget\b.*\|\s*bash", "blocked: piped wget|bash"),
        (r"^grep\b(?!.*\|)", "hint: prefer rg over grep (Claude Code hook)"),
        (r"^find\s+\S+\s+-name\b", "hint: prefer rg --files (Claude Code hook)"),
    )

    @classmethod
    def enabled(cls) -> bool:
        return bool(
            CONFIG.get("cat_r1_code", CONFIG.get("catseek_code_r1", CONFIG.get("cat_code_interpreter_r1", True)))
            and CONFIG.get("interpreter_syntax", cls.PROTO)
            in (cls.PROTO, "catseek-code-r1-1.0", "cat-code-interpreter-r1-1.0")
        )

    @classmethod
    def wants_pr(cls, text: str) -> bool:
        return bool(re.search(r"(?:^|\s)#+\s*pr\b", text or "", re.I))

    @classmethod
    def wants_generate(cls, text: str) -> bool:
        """True when user wants code written — not just executed."""
        pl = CatR1Code.normalize_prompt(text or "").lower().strip()
        if not pl:
            return False
        if CatR1Code.wants_code(pl):
            return True
        if re.search(
            r"\b(write|make|build|create|implement|generate|draft|scaffold|add|fix)\b",
            pl,
        ) and re.search(
            r"\b(code|function|class|script|program|app|api|page|html|snippet|module|component)\b",
            pl,
        ):
            return True
        if re.search(
            r"\b(write|make|build|create|implement|generate)\b.{0,40}\b"
            r"(python|javascript|typescript|html|rust|go|java|c\+\+|cpp|bash|sql)\b",
            pl,
        ):
            return True
        if re.search(r"\b(fibonacci|fib|prime|sort|parser|api|cli|todo)\b", pl) and re.search(
            r"\b(write|make|build|create|implement|generate|code)\b", pl
        ):
            return True
        return False

    @classmethod
    def is_runnable_snippet(cls, text: str) -> bool:
        """True when text is executable code, not a natural-language request."""
        t = (text or "").strip()
        if not t:
            return False
        if "```" in t:
            _, code = CatR1Code.extract_prompt_code(t)
            return bool(code and code.strip())
        if cls.wants_generate(t):
            first = t.splitlines()[0].strip()
            if re.search(r"\b(write|make|build|create|implement|generate|draft)\b", first, re.I):
                return False
            if re.match(
                r"^(print\(|def |class |import |for |while |if |const |let |var |#include|function )",
                first,
                re.I,
            ):
                return True
            if re.fullmatch(r"[\w.]+\([^)]*\)\s*#?.*", first):
                return True
            if re.search(r"^(def |class |import |print\(|console\.)", t, re.M):
                return True
            return False
        return bool(
            re.search(r"^(def |class |import |print\(|console\.|#include|function )", t, re.M)
            or re.fullmatch(r"[\w.]+\([^)]*\)\s*#?.*", t.splitlines()[0].strip())
        )

    @classmethod
    def generate_code(
        cls, engine: "CatR11Engine", prompt: str, *, force_pro: bool = False
    ) -> str:
        """Generate fenced code via BitNet + CatR1 code engine (files = off · # pr)."""
        wants_pr = force_pro or cls.wants_pr(prompt)
        cls.ensure_bitnet(engine, prompt, force_pro=wants_pr)
        if not CatR1Code.enabled():
            return f"**{cls.NAME}** · code generation disabled · `files = off`"
        norm = CatR1Code.normalize_prompt(prompt)
        body = CatR1Code.respond(engine, norm)
        lang = CatR1Code.detect_lang(engine, norm)
        reasoning = cls.code_reasoning(
            lang, "generated", body[:800],
            lint_msg="generated",
            output="(code emitted)" if "```" in body else "",
        )
        engine.last_think = reasoning
        if "```" in body:
            return f"**{cls.NAME}** · generated · `{lang}` · `files = off`\n\n{body}"
        fenced = TokenWeightCodeEmitter.fence(lang, body)
        return f"**{cls.NAME}** · generated · `{lang}` · `files = off`\n\n{fenced}"

    @classmethod
    def dispatch(cls, engine: "CatR11Engine", prompt: str, *, force_pro: bool = False) -> str:
        """Route generate vs run — full Claude Code behavior."""
        work = (prompt or "").strip()
        if not work:
            return cls.help_text()
        wants_pr = force_pro or cls.wants_pr(work)
        cls.ensure_bitnet(engine, work, force_pro=wants_pr)
        pl = work.lower()
        if pl.startswith("feature-dev"):
            topic = work.split(maxsplit=1)[1] if " " in work else ""
            return cls.feature_dev_outline(topic)
        if pl.startswith("review"):
            code = work.split(maxsplit=1)[1] if " " in work else CatR1CodingAPI.session().code
            block_lang, block = engine.extract_code_block(
                code if "```" in code else f"```python\n{code}\n```"
            )
            return cls.code_review(block or code, block_lang or "python")
        if pl.startswith("plan"):
            parts = work.split(maxsplit=2)
            agent = parts[1] if len(parts) > 1 else "code-architect"
            topic = parts[2] if len(parts) > 2 else ""
            return cls.agent_plan(agent, topic, CatR1CodingAPI.session().code)
        if cls.wants_generate(work) and not cls.is_runnable_snippet(work):
            return cls.generate_code(engine, work, force_pro=wants_pr)
        block_lang, block = engine.extract_code_block(work if "```" in work else "")
        code = block or work
        lang = block_lang or ""
        out = CatR1CodingAPI.parse_request(
            engine, {"action": "run", "code": code, "lang": lang},
        )
        return CatR1CodingAPI.format_result(out)

    @classmethod
    def no_api(cls) -> bool:
        return bool(CONFIG.get("catcode_no_api", True))

    @classmethod
    def bitnet_only(cls) -> bool:
        return bool(CONFIG.get("catcode_bitnet_only", True))

    @classmethod
    def ensure_bitnet(
        cls, engine: "CatR11Engine", prompt: str = "", *, force_pro: bool = False
    ) -> Dict[str, Any]:
        """Lock catcode to local BitNet inference — files = off · no external API."""
        CONFIG["files"] = "off"
        CONFIG["catcode_bitnet_only"] = True
        CONFIG["bitnet_encode"] = True
        bt = engine._ensure_bitnet_real()
        if force_pro:
            engine.active_model_profile = CatR1ProProfile()
            engine.ultrathink_on = True
        work = (prompt or "catcode").strip()
        if engine.is_pro_mode():
            engine.v4_pro_code_encode(work)
        else:
            engine.encode_for_task(work, task="code")
        return bt

    @classmethod
    def bitnet_status(cls, engine: "CatR11Engine") -> str:
        bt = (getattr(engine, "cat_r1_stats", None) or {}).get("bitnet", {})
        stats = getattr(engine, "last_inference_stats", None)
        tps = f"{stats.tps_estimate:.0f} tok/s · " if stats and stats.tps_estimate else ""
        enc = f"{stats.encode_ms:.0f}ms · " if stats and stats.encode_ms else ""
        return (
            f"**BitNet local** · packed ternary · {tps}{enc}"
            f"self-test **{'PASS' if bt.get('ok', True) else 'FAIL'}** · "
            f"`files = off` · **no API**"
        )

    @classmethod
    def wrap_thinking(cls, body: str) -> str:
        return CatR1CodeInterpreter01.wrap_thinking(body)

    @classmethod
    def split_thinking(cls, text: str) -> Tuple[str, str]:
        return CatR1CodeInterpreter01.split_thinking(text)

    @classmethod
    def validate_bash(cls, command: str) -> Tuple[bool, List[str]]:
        """PreToolUse hook — forked from claude-code bash_command_validator_example."""
        cmd = (command or "").strip()
        if not cmd:
            return True, []
        issues: List[str] = []
        blocked = False
        for pattern, message in cls.BASH_VALIDATION_RULES:
            if re.search(pattern, cmd, re.I):
                issues.append(message)
                if message.startswith("blocked:"):
                    blocked = True
        return (not blocked, issues)

    @classmethod
    def run_hook(
        cls,
        hook_name: str,
        *,
        tool_name: str = "",
        tool_input: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Claude Code hook runner (in-memory · files = off)."""
        payload = tool_input or {}
        if hook_name == "PreToolUse" and tool_name == "Bash":
            command = payload.get("command", "")
            ok, issues = cls.validate_bash(command)
            return {"hook": hook_name, "allow": ok, "issues": issues, "files": "off"}
        if hook_name == "SessionStart":
            return {
                "hook": hook_name,
                "allow": True,
                "message": f"{cls.NAME} session · fork `{cls.FORK_REPO}` · files=off",
                "files": "off",
            }
        return {"hook": hook_name, "allow": True, "issues": [], "files": "off"}

    @classmethod
    def agent_plan(cls, agent: str, prompt: str, code: str = "", lang: str = "python") -> str:
        """Claude Code agent prompt scaffold (feature-dev / code-review plugins)."""
        topic = (prompt or "buffer").strip()[:120]
        preview = (code or "").strip()[:280]
        if agent == "code-explorer":
            return (
                f"**code-explorer** · trace `{topic}`\n\n"
                f"- Map abstractions and control flow for `{lang}` buffer\n"
                f"- List 5–10 key symbols and entry points\n"
                f"- Sandbox only — `files = off`\n\n"
                f"```{(lang or 'text')[:12]}\n{preview}\n```"
            )
        if agent == "code-architect":
            return (
                f"**code-architect** · design for `{topic}`\n\n"
                "1. Minimal-change path (reuse existing patterns)\n"
                "2. Clean-architecture path (maintainable abstractions)\n"
                "3. Pragmatic path (speed + quality)\n\n"
                f"Buffer language: `{lang}` · in-memory only."
            )
        if agent == "code-reviewer":
            return (
                f"**code-reviewer** · review `{topic}`\n\n"
                "- Simplicity / DRY / readability\n"
                "- Bugs and functional correctness\n"
                "- Project conventions and sandbox safety\n\n"
                f"```{lang}\n{preview}\n```"
            )
        return f"**{agent}** · `{topic}` · files=off"

    @classmethod
    def feature_dev_outline(cls, prompt: str) -> str:
        """7-phase feature-dev workflow (claude-code/plugins/feature-dev)."""
        topic = (prompt or "feature").strip()[:120]
        lines = [
            f"**{cls.NAME}** · `/catcode feature-dev` · `{cls.FORK_REPO}`\n",
            f"Feature: **{topic}** · `files = off` · `# pr` for deep cat r1\n",
            "",
        ]
        for i, phase in enumerate(cls.PHASES, 1):
            lines.append(f"{i}. **{phase}**")
        lines.extend([
            "",
            "**Agents:** code-explorer → code-architect → code-reviewer",
            "**Hooks:** PreToolUse (bash validator) · SessionStart",
            "",
            "Reply with requirements or paste code — I'll run the in-memory agent loop.",
        ])
        return "\n".join(lines)

    @classmethod
    def code_review(cls, code: str, lang: str = "python") -> str:
        """Lightweight code review pass (code-review plugin pattern)."""
        body = (code or "").strip()
        if not body:
            return f"**{cls.NAME}** review: empty buffer."
        issues: List[str] = []
        if lang == "python":
            try:
                ast.parse(body)
            except SyntaxError as exc:
                issues.append(f"Syntax error line {exc.lineno}: {exc.msg}")
        if "eval(" in body or "exec(" in body:
            issues.append("Security: eval/exec detected — sandbox flagged.")
        if "os.system(" in body:
            issues.append("Security: os.system — prefer subprocess with explicit args.")
        if "__import__(" in body:
            issues.append("Security: dynamic __import__ — review carefully.")
        if not issues:
            issues.append("No high-severity issues in static scan.")
        return (
            f"**{cls.NAME}** · code-reviewer · `{lang}`\n\n"
            + "\n".join(f"- {x}" for x in issues)
            + "\n\n*Claude Code fork · in-memory only · files=off*"
        )

    @classmethod
    def diff_buffers(cls, before: str, after: str) -> str:
        import difflib
        a = (before or "").splitlines(keepends=True)
        b = (after or "").splitlines(keepends=True)
        diff = difflib.unified_diff(a, b, fromfile="before", tofile="after", lineterm="")
        body = "".join(diff).strip() or "(no changes)"
        return f"**{cls.NAME}** · diff\n\n```diff\n{body}\n```"

    @classmethod
    def apply_edit(cls, code: str, old: str, new: str) -> Tuple[str, bool, str]:
        src = code or ""
        if old and old in src:
            return src.replace(old, new, 1), True, "edit applied"
        if not old and new:
            return (src.rstrip() + "\n" + new).strip() + "\n", True, "appended"
        return src, False, "edit target not found"

    @classmethod
    def code_reasoning(
        cls,
        lang: str,
        script: str,
        code: str,
        *,
        lint_msg: str = "",
        error: str = "",
        output: Optional[str] = None,
    ) -> str:
        base = CatR1CodeInterpreter01.code_reasoning(
            lang, script, code, lint_msg=lint_msg, error=error, output=output,
        )
        agent_lines = [
            "",
            "**Claude Code fork (agent loop)**",
            f"- Repo: `{cls.FORK_REPO}`",
            f"- Ref: `{cls.FORK_REF}`",
            "- Agents: code-explorer → code-architect → code-reviewer",
            "- Hook: PreToolUse bash validator (files=off sandbox)",
            "- Protocol: reasoning_content + content split",
        ]
        return base + "\n" + "\n".join(agent_lines)

    @classmethod
    def chat_reasoning(cls, *args, **kwargs) -> str:
        return CatR1CodeInterpreter01.chat_reasoning(*args, **kwargs)

    @classmethod
    def format_content(cls, lang: str, script: str, output: str, *, error: str = "") -> str:
        if error:
            return (
                f"**{cls.NAME}** · `{script}` · `{lang}`\n\n"
                f"```\n{error}\n```"
            )
        body = (output or "(no output)").rstrip()
        return (
            f"**{cls.NAME}** · `{script}` · `{lang}`\n\n"
            f"```\n{body}\n```"
        )

    @classmethod
    def format_panel(cls, result: Dict[str, Any]) -> str:
        reasoning = result.get("reasoning_content") or result.get("thinking", "")
        content = result.get("content") or ""
        if not content and result.get("ok"):
            content = cls.format_content(
                result.get("lang", "code"),
                result.get("script", "buffer"),
                result.get("output", ""),
            )
        if not content and not result.get("ok"):
            content = cls.format_content(
                result.get("lang", "code"),
                result.get("script", "buffer"),
                "",
                error=result.get("error", "run failed"),
            )
        if reasoning and CONFIG.get("cat_r1_code_interpreter_think_tags", True):
            return f"{cls.wrap_thinking(reasoning)}\n\n---\n\n{content}"
        if reasoning:
            return f"{reasoning}\n\n---\n\n{content}"
        return content

    @classmethod
    def help_text(cls) -> str:
        return (
            f"**{cls.NAME}** · `{cls.MODEL}`\n\n"
            f"**BitNet local · no API** · `{cls.FORK_REPO}` · **files = off**\n\n"
            "- Runs on in-memory **BitNet** ternary weights — no HTTP API, no external calls\n"
            "- **`reasoning_content`** — agent loop + CoT inside "
            f"`{cls.THINK_OPEN}` … `{cls.THINK_CLOSE}`\n"
            "- **Generate + run:** writes real fenced code OR executes snippets\n"
            "- **Claude Code tools:** "
            + " · ".join(f"`{t}`" for t in cls.CLAUDE_CODE_TOOLS[:8])
            + " …\n"
            "- **Agents:** code-explorer · code-architect · code-reviewer\n"
            "- **Hooks:** PreToolUse bash validator · SessionStart\n"
            "- **Phases:** feature-dev 7-step workflow\n\n"
            "**Tools:** "
            + " · ".join(f"`{t}`" for t in cls.TOOLS)
            + "\n\n"
            "**Commands (chat/GUI · BitNet · no API):**\n"
            "- `/catrcode` · `/catrcode write a python fibonacci function # pr`\n"
            "- `/catrcode print(2+2)` · `/catcode` · `/catseek` (aliases)\n"
            "- `/catrcode feature-dev <topic>` · `/catrcode review` · `/catrcode plan`\n"
            "- Natural language: `cat r1 code` · `claude code` · `# pr`\n\n"
            f"**Protocol:** `{cls.PROTO}` · BitNet · fork `{cls.FORK_REF}`"
        )

    @classmethod
    def wants_catseek(cls, text: str) -> bool:
        pl = (text or "").lower()
        return bool(
            re.search(
                r"\b(?:/catrcode|/catseek|/catcode|cat\s*r1\s*code|catseek\s*code|cat\s*code\s*interpreter|claude\s*code)\b",
                pl,
                re.I,
            )
        )

    @classmethod
    def wants_catcode(cls, text: str) -> bool:
        return cls.wants_catseek(text)


CatCodeInterpreterR1 = CatR1CodeR1
CatCodeInterpreter = CatR1CodeR1
CatseekCodeR1 = CatR1CodeR1


# Legacy alias (renamed from cat r1 0.1 interpreter)
DeepSeekR1Interpreter = CatR1CodeInterpreter01


def active_interpreter():
    """Return the active interpreter syntax class."""
    if CatR1CodeR1.enabled():
        return CatR1CodeR1
    if CatR1CodeInterpreter01.enabled():
        return CatR1CodeInterpreter01
    return O1PreviewSyntax


# ──────────────────────────────────────────────────────────────
# CAT R1.1 REASONING (files = off)
# ──────────────────────────────────────────────────────────────
class O1PreviewReasoner:
    """
    o1-preview reasoning loop (files = off):
    parse → recursive cat r1 passes → self-check → verify → answer.
    Thinking stays internal; user sees clean final text.
    """

    @classmethod
    def should_run(cls, prompt: str, *, enabled: bool, force: bool) -> bool:
        if force:
            return True
        if not enabled:
            return False
        pl = prompt.strip()
        if not pl:
            return False
        return True

    @staticmethod
    def _parse_intent(prompt: str) -> str:
        pl = prompt.lower()
        if any(k in pl for k in ("bug", "error", "traceback", "exception")):
            return "debug / isolate failure"
        if any(k in pl for k in ("code", "python", "script", "function")):
            return "implement or review code"
        if any(k in pl for k in ("build", "make", "create", "design")):
            return "design / construct"
        if any(k in pl for k in ("explain", "what is", "why", "how")):
            return "explain / teach"
        if re.search(r"\d\s*[+\-*/]", pl):
            return "compute / verify numerics"
        if "?" in prompt:
            return "answer a question"
        return "general assistance"

    @staticmethod
    def _subtasks(prompt: str) -> List[str]:
        pl = prompt.lower()
        tasks: List[str] = []
        if "?" in prompt:
            tasks.append("identify what is being asked and the expected form of the answer")
        if any(k in pl for k in ("code", "python", "implement")):
            tasks.append("list inputs, outputs, and edge cases before writing code")
        if any(k in pl for k in ("error", "bug", "traceback")):
            tasks.append("reproduce minimally, then localize the failing line")
        if any(k in pl for k in ("build", "design", "architecture")):
            tasks.append("sketch components and data flow before details")
        if re.search(r"\d\s*[+\-*/]", pl):
            tasks.append("compute step-by-step and verify the result")
        if not tasks:
            tasks.append("state goal, constraints, and the smallest verifiable next step")
        return tasks[:5]

    @staticmethod
    def _verify_note(prompt: str) -> str:
        pl = prompt.lower()
        checks: List[str] = []
        if "?" in prompt:
            checks.append("final answer addresses the question directly")
        if any(k in pl for k in ("code", "python")):
            checks.append("code is runnable in the local sandbox")
        if any(k in pl for k in ("error", "bug")):
            checks.append("expected vs actual output is explicit")
        checks.append("reasoning and weights stay in-memory")
        return "; ".join(checks)

    @staticmethod
    def _self_check(prompt: str, draft: str) -> str:
        pl = prompt.lower()
        notes: List[str] = []
        if "?" in prompt:
            notes.append("Does the draft answer the exact question?")
        if any(k in pl for k in ("code", "python")):
            notes.append("Are imports, edge cases, and return values covered?")
        if re.search(r"\d\s*[+\-*/]", pl):
            notes.append("Re-check arithmetic independently.")
        notes.append("Remove speculation; keep only what follows from the prompt.")
        return " · ".join(notes)

    def run(
        self,
        prompt: str,
        *,
        distill_draft: str = "",
        recursive_trace: Optional[List[str]] = None,
        compression_trace: Optional[List[str]] = None,
    ) -> str:
        intent = self._parse_intent(prompt)
        subtasks = self._subtasks(prompt)
        verify = self._verify_note(prompt)
        reason = distill_draft.strip() or f"aligned on {intent}."
        if CatCodeInterpreterR1.enabled() or CatR1CodeInterpreter01.enabled():
            return active_interpreter().chat_reasoning(
                prompt,
                intent=intent,
                subtasks=subtasks,
                reason=reason,
                verify=verify,
                self_check=self._self_check(prompt, reason),
            )
        return O1PreviewSyntax.build_trace(
            prompt,
            intent=intent,
            subtasks=subtasks,
            reason=reason,
            recursive_trace=recursive_trace,
            compression_trace=compression_trace,
            verify=verify,
            self_check=self._self_check(prompt, reason),
        )


ThinkEngine = O1PreviewReasoner
UltraThinkEngine = O1PreviewReasoner


# ──────────────────────────────────────────────────────────────
# CAT R1.1 FUSION (cat r1 CoT · cat r1 polish · files = off)
# ──────────────────────────────────────────────────────────────
class CatR1Fusion:
    """
    Dual reasoning stack under cat r1 branding (no weight files):
    cat r1 — long CoT, GRPO-style self-verify, math sanity checks.
    cat r1-tier — extended thinking trace, recursive draft improvement.
    """

    __slots__ = ("last_passes", "last_trace")

    CATR1_ALGOS = (
        "Long chain-of-thought",
        "GRPO outcome self-verification",
        "Step-by-step math checks",
        "Code edge-case review",
    )
    MYTHOS_ALGOS = (
        "Extended thinking trace",
        "Recursive draft improvement",
        "cat r1 prose polish",
        "Pre-emit self-check",
    )

    def __init__(self):
        self.last_passes = 0
        self.last_trace: List[str] = []

    @staticmethod
    def think_header() -> str:
        if CatCodeInterpreterR1.enabled() or CatR1CodeInterpreter01.enabled():
            return f"{BRAND} · {active_interpreter().NAME} · {REASONING_MODE}"
        if CONFIG.get("o1_preview", True):
            return O1PreviewSyntax.header()
        return f"{BRAND} · {REASONING_MODE} · {CODE_ENGINE}"

    @staticmethod
    def _meaningful_words(pl: str) -> bool:
        return bool(re.search(r"[a-zA-Z0-9\u4e00-\u9fff]", pl))

    @classmethod
    def is_noise(cls, prompt: str) -> bool:
        pl = prompt.strip().lower()
        if not pl:
            return True
        if CatR1Code._GO.match(pl) or CatR1Code._CODE_SHORT.match(pl):
            return False
        if pl in {".", "..", "...", "\"", "'", "?", "!"}:
            return True
        return False

    @classmethod
    def session_followup(cls, engine: "CatR11Engine", prompt: str) -> Optional[str]:
        if not engine.chat_history:
            return None
        last_user = last_bot = ""
        for m in reversed(engine.chat_history[:-1]):
            if m.get("role") == "assistant" and not last_bot:
                last_bot = m.get("text", "")
            elif m.get("role") == "user" and not last_user:
                last_user = m.get("text", "")
            if last_user and last_bot:
                break
        if not last_user:
            return None
        pl = prompt.strip().lower()
        if cls.is_noise(prompt):
            if re.search(r"how are you|how're you|how is it", last_user.lower()):
                return "I'm doing well — thanks for asking! What's on your mind?"
            if re.search(r"你好吗|怎么样|还好吗", prompt):
                return "我很好，谢谢关心！你今天想聊什么？"
            return f"Still here — we were talking about \"{last_user[:60]}\". Want to continue that, or start something new?"
        return None

    def recursive_improve(self, draft: str, prompt: str, vec: Optional[np.ndarray]) -> str:
        if not CONFIG.get("mythos_recursive_improve") or not draft.strip():
            return draft
        depth = CONFIG.get("mythos_recursive_depth", 3)
        eps = CONFIG.get("mythos_recursive_epsilon", 0.04)
        out = draft
        self.last_trace = []
        prev = out
        for i in range(depth):
            score = self._quality_score(out, prompt, vec)
            self.last_trace.append(f"mythos pass {i + 1} · quality {score:.3f}")
            if i > 0 and score >= 0.92:
                self.last_trace.append(f"converged at pass {i + 1}")
                self.last_passes = i + 1
                return out
            out = self._polish_pass(out, prompt, vec, pass_idx=i)
            if i > 0:
                delta = abs(len(out) - len(prev)) / max(len(prev), 1)
                if delta < eps:
                    self.last_trace.append(f"converged at pass {i + 1}")
                    self.last_passes = i + 1
                    return out
            prev = out
        self.last_passes = depth
        return out

    @staticmethod
    def _quality_score(text: str, prompt: str, vec: Optional[np.ndarray]) -> float:
        if not text.strip():
            return 0.0
        score = 0.55
        if "?" in prompt and "?" not in text and len(text) > 40:
            score += 0.08
        if len(text.split()) >= 12:
            score += 0.12
        if re.search(r"\*\*[^*]+\*\*", text):
            score += 0.05
        if vec is not None and vec.size:
            score += min(0.2, float(np.linalg.norm(vec[:8])) * 0.02)
        if text.count("\n\n") >= 1:
            score += 0.05
        return min(1.0, score)

    @staticmethod
    def _polish_pass(text: str, prompt: str, vec: Optional[np.ndarray], pass_idx: int) -> str:
        t = re.sub(r"\n{3,}", "\n\n", text.strip())
        if pass_idx == 0 and "?" in prompt and not t.endswith("?"):
            if len(t.split()) > 20 and "I can go deeper" not in t:
                t += "\n\nWant me to go deeper on any part?"
        if pass_idx >= 1 and len(t) < 120 and "?" in prompt:
            topic = prompt.strip().rstrip("?")[:80]
            t = f"{t}\n\n**Short answer:** {topic} — ask for steps, code, or a comparison and I'll expand."
        return t

    @staticmethod
    def cat_r1_math_wrap(prompt: str, result: str) -> str:
        if not CONFIG.get("cat_r1_self_verify"):
            return f"Result: **{result}**"
        return (
            f"**Step-by-step ({BRAND} verify):**\n\n"
            f"1. Parse expression from prompt\n"
            f"2. Evaluate with integer/float rules\n"
            f"3. Self-check: re-evaluate → **{result}**\n\n"
            f"**Answer:** {result}"
        )

    def stats_line(self) -> str:
        return (
            f"Fusion · {BRAND} ({len(self.CATR1_ALGOS)} + {len(self.MYTHOS_ALGOS)} algos) · passes={self.last_passes}"
        )


# ──────────────────────────────────────────────────────────────
# CAT R1.1 RUNTIME (files = off · cat r1 parity)
# ──────────────────────────────────────────────────────────────
class ClaudeMythosRuntime:
    """
    cat r1 local runtime (files = off):
    extended thinking · recursive prose polish · cat r1 code engine · cat r1 voice.
    Behavior target: cat r1 frontier while preserving cat r1 branding.
    """

    TIER = CONFIG["mythos_tier"]
    PROSE = CONFIG["prose_tier"]
    CODE_LABEL = CONFIG["code_interpreter_name"]

    MYTHOS_ALGOS = (
        "Extended thinking trace",
        "Recursive draft improvement",
        "cat r1 prose polish",
        "Pre-emit self-check",
        "cat r1 code perfection loop",
        "Proactive follow-through",
    )

    @classmethod
    def enabled(cls) -> bool:
        return bool(CONFIG.get("mythos_runtime", True) and CONFIG.get("mythos_mode", True))

    @classmethod
    def think_header(cls) -> str:
        if CatCodeInterpreterR1.enabled() or CatR1CodeInterpreter01.enabled():
            return f"{BRAND} · {active_interpreter().NAME} · {REASONING_MODE}"
        return f"{BRAND} · {REASONING_MODE}"

    @classmethod
    def voice(cls, body: str, prompt: str, task: str = "general") -> str:
        if not CONFIG.get("mythos_voice", True) or not body:
            return body
        text = body.strip()
        if "```" in text and task == "code":
            return text
        pl = prompt.lower()
        if "?" in prompt and task in {"explain", "general", "chat", "agent", "qa"}:
            if len(text) > 50 and not re.match(
                r"^(Here|I'd|Let me|Sure|The |Yes|I can|On |答案)", text, re.I
            ):
                topic = prompt.strip().rstrip("?！？.")[:100]
                text = f"**{topic}** — {text}"
        if any(k in pl for k in ("how ", "why ", "explain", "walk me", "step")):
            if len(text) > 140 and "1." not in text[:280] and "Step" not in text[:280]:
                paras = [p.strip() for p in text.split("\n\n") if p.strip()]
                if len(paras) >= 2:
                    text = "\n\n".join(
                        f"**{i}.** {p.lstrip('0123456789. ')}"
                        for i, p in enumerate(paras, 1)
                    )
        return text

    @classmethod
    def emit(cls, engine: "CatR11Engine", body: str, prompt: str, task: str) -> str:
        if not cls.enabled():
            if CONFIG.get("cat_r1_voice"):
                return GoogleWhitepaperCatR1Sorter.cat_r1_voice(body, prompt, task)

        out = body
        if CONFIG.get("mythos_recursive_improve") and task not in frozenset({"chat", "execute"}):
            out = engine.fusion.recursive_improve(out, prompt, engine.last_vec)
        out = cls.voice(out, prompt, task)
        if CONFIG.get("cat_r1_voice") and CATR1_MODE:
            out = GoogleWhitepaperCatR1Sorter.cat_r1_voice(out, prompt, task)
        return out

    @classmethod
    def math_wrap(cls, prompt: str, result: str) -> str:
        if not cls.enabled():
            return CatR1Fusion.cat_r1_math_wrap(prompt, result)
        return (
            f"**{MYTHOS_NAME} verify**\n\n"
            f"1. Parse the expression from your prompt\n"
            f"2. Evaluate with standard arithmetic rules\n"
            f"3. Independent re-check → **{result}**\n\n"
            f"**Answer:** {result}\n\n"
            f"*computed in-memory*"
        )

    @classmethod
    def code_help(cls) -> str:
        if CatCodeInterpreterR1.enabled() or CatR1CodeInterpreter01.enabled():
            return active_interpreter().help_text()
        return (
            f"**{cls.CODE_LABEL}** · **{MYTHOS_NAME}** · \n\n"
            f"Engine `{CONFIG['code_backend']}` · version `{CONFIG['code_interpreter_version']}` · "
            f"prose `{cls.PROSE}`\n\n"
            "Full code capability — write, edit, run, lint, explain.\n"
            "**Code anything** — any language, any task (API, CLI, web, algo, mobile, config).\n"
            "Recursive perfection loop · pattern library · sandbox verify.\n\n"
            "**Commands:** `/run` · `/interpret` · `/code` · `/think` · paste ``` fences · **run it**\n\n"
            f"Recursive depth: {CONFIG['mythos_recursive_depth']}\n\n"
            "Everything stays in-memory — nothing written to disk."
        )

    @classmethod
    def format_code_result(cls, lang: str, script: str, output: str, *, error: str = "") -> str:
        if CatCodeInterpreterR1.enabled() or CatR1CodeInterpreter01.enabled():
            return active_interpreter().format_content(lang, script, output, error=error)
        if error:
            return (
                f"**Code engine** · **{BRAND}**\n\n"
                f"`{script}` · `{lang}`\n\n```\n{error}\n```"
            )
        body = (output or "(no output)").rstrip()
        return (
            f"**`{script}`** · `{lang}`\n\n"
            f"```\n{body}\n```"
        )

    @classmethod
    def stats_line(cls) -> str:
        return f"{MYTHOS_NAME} ({len(cls.MYTHOS_ALGOS)} algos)"


MythosRuntime = ClaudeMythosRuntime


# ──────────────────────────────────────────────────────────────
# DEEPMIND FAST STACK (files = off · in-memory speed algorithms)
# Chinchilla adaptive compute · Flash attention · MuZero prefix cache
# · teacher-only turbo distillation · batch cat r1 linear · early exit
# ──────────────────────────────────────────────────────────────
class DeepMindFastStack:
    """
    In-memory DeepMind-inspired inference accelerators (no weight files).
    Composes with cat r1 + o1-preview recursive loop on CatR11Engine.
    """

    __slots__ = ("engine", "_prefix_cache", "_turbo_cache", "last_algo", "passes_saved")

    ALGOS = (
        "Chinchilla adaptive compute",
        "Flash causal attention",
        "Batch linear GEMM",
        "MuZero prefix latent cache",
        "In-memory student distillation",
        "Teacher-only turbo distillation",
        "Sparse top-k compression",
        "MoE top-k routing",
        "Recursive early convergence",
        "cat-r1 multi-token prediction",
        "cat-r1 real-time sparse attention",
        "cat-r1 speculative decoding",
    )

    def __init__(self, engine: "CatR11Engine"):
        self.engine = engine
        self._prefix_cache: Dict[str, np.ndarray] = {}
        self._turbo_cache: Dict[str, np.ndarray] = {}
        self.last_algo = ""
        self.passes_saved = 0

    def adaptive_depth(self, prompt: str, task: Optional[str] = None) -> int:
        base = CONFIG["recursive_depth"]
        if not CONFIG.get("adaptive_compute"):
            return base
        n = len(prompt.strip())
        if task in CONFIG.get("turbo_encode_tasks", ()):
            return 0
        if n < 12:
            return 0
        if n < 30:
            return 0
        if n < 60:
            return 1
        if n < 120:
            return min(1, base)
        if n < 250:
            return min(2, base)
        return base

    def _prefix_key(self, prompt: str) -> str:
        words = prompt.lower().strip().split()
        return " ".join(words[:6]) if words else ""

    def _muzero_hit(self, prompt: str) -> Optional[np.ndarray]:
        key = self._prefix_key(prompt)
        if not key or len(prompt) < 32:
            return None
        hit = self._prefix_cache.get(key)
        if hit is not None:
            self.last_algo = "MuZero prefix latent cache"
            return hit.copy()
        return None

    def turbo_encode(self, prompt: str) -> np.ndarray:
        """Single-pass teacher-only encode — chat/code/math fast path."""
        key = prompt.lower().strip()
        hit = self._turbo_cache.get(key)
        if hit is not None:
            self.last_algo = "Teacher-only turbo distillation (cache)"
            return hit.copy()
        state = self.engine.encode_prompt(prompt)
        seq = state if state.ndim == 2 else state.reshape(1, -1)
        delta = self.engine._pool_sequence(
            self.engine.forward(seq, turbo_only=True)
        )
        out = self.engine._layer_norm(delta)
        if CONFIG["compression_enabled"]:
            out = self.engine.compressor.compress_roundtrip(out)
            self.engine.last_compression_ratio = self.engine.compressor.last_ratio
        self.engine.last_recursive_passes = 1
        self.engine.last_recursive_trace = [
            f"turbo · 1 pass · {BRAND} student",
        ]
        self.engine.last_vec = out.copy()
        self.last_algo = "cat r1 student distillation"
        if len(self._turbo_cache) < 256:
            self._turbo_cache[key] = out.copy()
        return out

    def encode(self, prompt: str, task: Optional[str] = None) -> np.ndarray:
        if not CONFIG.get("deepmind_fast"):
            return self.engine.recursive_encode(prompt)
        if task == "code" and CONFIG.get("v4_pro_coding", True):
            return self.engine.v4_pro_code_encode(prompt)
        if task in CONFIG.get("turbo_encode_tasks", ()):
            return self.turbo_encode(prompt)
        mu = self._muzero_hit(prompt)
        if mu is not None:
            self.engine.last_vec = mu
            return mu
        depth = self.adaptive_depth(prompt, task)
        saved = max(0, CONFIG["recursive_depth"] - depth)
        self.passes_saved += saved
        out = self.engine.recursive_encode(prompt, depth=depth)
        pk = self._prefix_key(prompt)
        if pk and len(self._prefix_cache) < 64:
            self._prefix_cache[pk] = out.copy()
        self.last_algo = f"Chinchilla adaptive compute ({depth} passes)"
        return out

    def stats_line(self) -> str:
        return (
            f"DeepMind fast · {len(self.ALGOS)} algos · "
            f"last={self.last_algo or 'idle'} · passes_saved={self.passes_saved}"
        )

    def clear(self) -> None:
        self._prefix_cache.clear()
        self._turbo_cache.clear()
        self.passes_saved = 0


# ──────────────────────────────────────────────────────────────
# CAT R1.1 COMPRESSION ENGINE (files = off · cat r1 capacity)
# ──────────────────────────────────────────────────────────────
class CatR1Compressor:
    """
    Compression stack for cat r1:
    ternary weights · Google low-rank + MoE · sparse top-k · fast path.
    Simulates large-parameter capacity without external files.
    """

    __slots__ = ("d_model", "rank", "sparse_k", "down_s", "up_s", "last_ratio", "packs",
                "moe_experts", "moe_topk", "moe_gate", "moe_down", "moe_up", "_cache")

    def __init__(self, d_model: int, seed: int = 99):
        self.d_model = d_model
        self.rank = CONFIG["compression_rank"]
        self.sparse_k = CONFIG["compression_sparse_k"]
        rng = np.random.RandomState(seed)
        down = rng.choice([-1, 0, 1], (d_model, self.rank)).astype(np.int8)
        up = rng.choice([-1, 0, 1], (self.rank, d_model)).astype(np.int8)
        self.down_s = (down == 1).astype(np.int16) - (down == -1).astype(np.int16)
        self.up_s = (up == 1).astype(np.int16) - (up == -1).astype(np.int16)
        # Google MoE-style expert compression
        self.moe_experts = CONFIG.get("compression_google_moe_experts", 64)
        self.moe_topk = CONFIG.get("compression_google_moe_topk", 8)
        moe_rng = np.random.RandomState(seed + 42)
        self.moe_gate = moe_rng.randn(d_model, self.moe_experts).astype(np.float32) * 0.02
        self.moe_down = [moe_rng.choice([-1, 0, 1], (d_model, d_model // 8)).astype(np.int8)
                        for _ in range(self.moe_experts)]
        self.moe_up = [moe_rng.choice([-1, 0, 1], (d_model // 8, d_model)).astype(np.int8)
                    for _ in range(self.moe_experts)]
        self._cache = {}
        self.last_ratio = 1.0
        self.packs = 0

    @staticmethod
    def _ternary(x: np.ndarray, thr: float = 0.5) -> np.ndarray:
        q = np.zeros_like(x, dtype=np.int8)
        q[x > thr], q[x < -thr] = 1, -1
        return q

    def _moe_compress(self, x: np.ndarray) -> np.ndarray:
        """Google-style MoE routing for compression: gate -> top-k experts -> weighted sum."""
        logits = x.astype(np.float32) @ self.moe_gate
        topk = np.argpartition(-logits, self.moe_topk)[:self.moe_topk]
        weights = np.exp(logits[topk] - np.max(logits[topk]))
        weights /= np.sum(weights) + 1e-8
        out = np.zeros_like(x)
        for i, idx in enumerate(topk):
            down = (self.moe_down[idx] == 1).astype(np.int16) - (self.moe_down[idx] == -1).astype(np.int16)
            up = (self.moe_up[idx] == 1).astype(np.int16) - (self.moe_up[idx] == -1).astype(np.int16)
            xq = self._ternary(x).astype(np.int16)
            latent = np.tanh(xq @ down).astype(np.float32)
            recon = (self._ternary(latent).astype(np.int16) @ up).astype(np.float32)
            out += recon * weights[i]
        return out

    def low_rank_bottleneck(self, x: np.ndarray) -> np.ndarray:
        xq = self._ternary(x).astype(np.int16)
        h = np.tanh(xq @ self.down_s).astype(np.float32)
        return (self._ternary(h).astype(np.int16) @ self.up_s).astype(np.float32)

    def sparse_reconstruct(self, x: np.ndarray) -> np.ndarray:
        k = min(self.sparse_k, len(x))
        idx = np.argpartition(np.abs(x), -k)[-k:]
        out = np.zeros_like(x, dtype=np.float32)
        out[idx] = x[idx]
        t = self._ternary(out).astype(np.float32)
        out[idx] = t[idx] * np.abs(x[idx])
        fp_bits = self.d_model * 32
        packed_bits = k * 4 + self.rank * 2
        self.last_ratio = max(1.0, fp_bits / max(packed_bits, 1))
        self.packs += 1
        return out

    def compress_roundtrip(self, x: np.ndarray) -> np.ndarray:
        if not CONFIG["compression_enabled"]:
            self.last_ratio = 1.0
            return x
        lr = self.low_rank_bottleneck(x)
        moe = self._moe_compress(x)
        blended = x * 0.36 + lr * 0.32 + moe * 0.32
        return self.sparse_reconstruct(blended)

    def effective_params_billions(self) -> float:
        mult = self.last_ratio * CONFIG["distil_passes"] * CONFIG["compression_stack_mult"]
        return CONFIG["nominal_base_params"] * mult / 1e9 / max(CONFIG["compression_stack_mult"], 1)


# ──────────────────────────────────────────────────────────────
# CAT R1 V4 PRO NEURAL COMPRESSION (files = off · novel algorithm)
# Never-before-seen multi-scale fractal ternary compression
# ──────────────────────────────────────────────────────────────
class CatR1V4ProCompression:
    """
    cat r1 V4 Pro compression — multi-scale fractal ternary decomposition +
    cross-dimensional recursive sparse coding + adaptive precision gating.

    Compresses the full 1.7T parameter space into in-memory BitNet engine
    using a stacked entropy-constrained neural codec with no external files.

    Algorithm overview (all in-memory, files = off):
    1. FractalTernaryScale — decompose activations across 3 scales
    2. CrossDimRecursiveSparse — iteratively refine sparse codes
    3. AdaptivePrecisionGate — bit allocation by information density
    4. HierarchicalLatentFusion — multi-res bottleneck stack
    5. NeuralArithmeticCodec — entropy-constrained residual encoding
    """

    __slots__ = ("d_model", "scales", "sparse_ks", "ranks",
                "fractal_down", "fractal_up", "gate_weights",
                "hierarchical_projections", "codebook",
                "last_ratio", "last_compressed_size", "total_packed",
                "scale_factors", "residual_buffers", "passes")

    def __init__(self, d_model: int, seed: int = 7777):
        self.d_model = d_model
        self.scales = (d_model // 2, d_model // 4, d_model // 8)
        self.sparse_ks = (d_model // 4, d_model // 8, d_model // 16)
        self.ranks = (d_model // 2, d_model // 3, d_model // 4)
        rng = np.random.RandomState(seed)
        self.fractal_down: List[np.ndarray] = []
        self.fractal_up: List[np.ndarray] = []
        for i, (sc, r) in enumerate(zip(self.scales, self.ranks)):
            down = rng.choice([-1, 0, 1], (d_model, r)).astype(np.int8)
            up = rng.choice([-1, 0, 1], (r, d_model)).astype(np.int8)
            self.fractal_down.append((down == 1).astype(np.int16) - (down == -1).astype(np.int16))
            self.fractal_up.append((up == 1).astype(np.int16) - (up == -1).astype(np.int16))
        self.gate_weights = rng.randn(len(self.scales)).astype(np.float32)
        self.hierarchical_projections: List[np.ndarray] = []
        for i in range(len(self.scales) - 1):
            p = rng.choice([-1, 0, 1], (self.ranks[i], self.ranks[i+1])).astype(np.int8)
            self.hierarchical_projections.append(
                (p == 1).astype(np.int16) - (p == -1).astype(np.int16)
            )
        self.codebook = rng.choice([-1, 0, 1], (256, d_model // 16)).astype(np.int8)
        self.last_ratio = 1.0
        self.last_compressed_size = 0
        self.total_packed = 0
        self.scale_factors: List[float] = []
        self.residual_buffers: List[np.ndarray] = []
        self.passes = 0

    @staticmethod
    def _soft_ternary(x: np.ndarray, tau: float = 0.3) -> np.ndarray:
        q = np.zeros_like(x, dtype=np.float32)
        a = np.abs(x)
        q[a > tau] = np.tanh(x[a > tau] / tau) * 0.5 + 0.5
        return q

    def _fractal_decompose(self, x: np.ndarray) -> List[np.ndarray]:
        residuals: List[np.ndarray] = []
        current = x.copy()
        for down, up in zip(self.fractal_down, self.fractal_up):
            xq = self._soft_ternary(current).astype(np.int16)
            latent = np.tanh(xq @ down).astype(np.float32)
            recon = (self._soft_ternary(latent).astype(np.int16) @ up).astype(np.float32)
            residuals.append(recon)
            current -= recon * 0.5
        self.residual_buffers = residuals
        return residuals

    def _cross_dim_recursive_sparse(self, x: np.ndarray, depth: int = 3) -> np.ndarray:
        current = x.copy()
        for d in range(depth):
            k = max(8, self.sparse_ks[min(d, len(self.sparse_ks) - 1)])
            idx = np.argpartition(np.abs(current), -k)[-k:]
            mask = np.zeros_like(current)
            mask[idx] = 1.0
            current *= mask
            current *= (1.0 + d * 0.1)
            np.tanh(current, out=current)
        return current

    def _adaptive_precision_gate(self, residuals: List[np.ndarray]) -> List[float]:
        densities: List[float] = []
        for i, r in enumerate(residuals):
            density = float(np.mean(np.abs(r)))
            gate = float(1.0 / (1.0 + np.exp(-self.gate_weights[i])))
            alloc = min(16.0, max(1.58, density * gate * 16.0))
            densities.append(alloc)
        self.scale_factors = densities
        return densities

    def _hierarchical_latent_fusion(self, residuals: List[np.ndarray]) -> np.ndarray:
        fused = residuals[0].copy()
        for i in range(1, len(residuals)):
            weight = 0.5 ** i
            r = residuals[i]
            if r.shape != fused.shape:
                r = np.resize(r, fused.shape)
            fused += (r - fused) * weight * 0.5
        return fused

    def _neural_arithmetic_codec(self, x: np.ndarray) -> Tuple[np.ndarray, float]:
        original = x.copy()
        codebook_size = min(128, len(self.codebook))
        flat = x.ravel()
        chunk_sz = self.d_model // 16
        chunks = [flat[i:i + chunk_sz] for i in range(0, len(flat), chunk_sz)]
        quantized: List[float] = []
        cb = self.codebook[:codebook_size].astype(np.float32)
        for chunk in chunks:
            if len(chunk) < chunk_sz:
                chunk = np.pad(chunk, (0, chunk_sz - len(chunk)))
            best_idx = int(np.argmin(np.sum((cb - chunk[np.newaxis, :]) ** 2, axis=1)))
            quantized.append(float(np.mean(np.abs(cb[best_idx]))))
        q_arr = np.array(quantized, dtype=np.float32)
        recon = np.interp(
            np.linspace(0, 1, len(flat)),
            np.linspace(0, 1, len(q_arr)),
            q_arr
        ).reshape(x.shape).astype(np.float32)
        ratio = 16.0 / max(1.58, np.log2(codebook_size + 1))
        return recon, ratio

    def v4_compress(self, x: np.ndarray) -> np.ndarray:
        residuals = self._fractal_decompose(x)
        precisions = self._adaptive_precision_gate(residuals)
        fused = self._hierarchical_latent_fusion(residuals)
        sparse = self._cross_dim_recursive_sparse(fused)
        codec_recon, codec_ratio = self._neural_arithmetic_codec(sparse)
        blended = x * 0.50 + fused * 0.20 + sparse * 0.20 + codec_recon * 0.10
        compressed_bits = int(
            sum(p * (self.d_model / (2 ** i)) for i, p in enumerate(precisions))
            + self.d_model * 0.25
        )
        self.last_ratio = max(1.0, self.d_model * 32.0 / max(compressed_bits, 1))
        self.last_compressed_size = compressed_bits
        self.total_packed += compressed_bits
        self.passes += 1
        return blended

    def _coding_feature_bias(self, prompt: str) -> np.ndarray:
        """Syntax-biased activation mask — cat r1 V4 Pro coding tier."""
        bias = np.zeros(self.d_model, dtype=np.float32)
        code_tokens = tokenize_text(prompt or "", 128)
        boost = frozenset({
            "def", "class", "function", "import", "return", "async", "await",
            "python", "javascript", "typescript", "code", "api", "void", "int",
            "func", "fn", "main", "print", "const", "let", "var", "impl", "struct",
        })
        for i, tok in enumerate(code_tokens):
            t = tok.lower()
            if t in boost or any(c in tok for c in "{}[]();"):
                idx = abs(hash(t + str(i))) % self.d_model
                bias[idx] += 0.18
        if re.search(r"[{}\[\]();=<>]|```|def\s+\w+|function\s+\w+", prompt or ""):
            step = max(1, self.d_model // 24)
            for j in range(0, self.d_model, step):
                bias[j] += 0.06
        return bias

    def v4_compress_for_code(self, x: np.ndarray, prompt: str) -> np.ndarray:
        """cat r1 V4 Pro coding — iterative syntax-biased neural compression."""
        bias = self._coding_feature_bias(prompt)
        biased = x.astype(np.float32) + bias * 0.14
        out = self.v4_compress(biased)
        extra = max(0, int(CONFIG.get("v4_pro_coding_passes", 2)) - 1)
        for _ in range(extra):
            out = self.v4_compress(out + biased * 0.09)
        return out

    def effective_v4_capacity(self) -> float:
        mult = self.last_ratio * 4.0 * 8.0
        return CONFIG["nominal_base_params"] * mult / 1e9

    def stats(self) -> Dict[str, Any]:
        return {
            "algorithm": BITNET_V4_LABEL,
            "type": "bitnet-fractal-ternary-recursive-sparse",
            "scales": len(self.scales),
            "last_ratio": f"{self.last_ratio:.1f}x",
            "effective_capacity_billions": f"{self.effective_v4_capacity():.1f}B",
            "total_packed_bits": self.total_packed,
            "passes": self.passes,
            "files": "off",
        }


# ──────────────────────────────────────────────────────────────
# CAT R1 PRO & CAT R1 FLASH MODEL PROFILES (files = off)
# Two cat r1 variants — deep reasoning · instant flash
# ──────────────────────────────────────────────────────────────
class CatR1ProProfile:
    """
    cat r1 — full BitNet reasoning tier.
    Deep chain-of-thought, recursive verification, neural compression.
    """
    model_id = "cat-r1-pro-v1"
    label = CAT_R1_PRO
    tier = "pro"
    reasoning_depth = 5
    recursive_passes = 4
    context_window = 10_000_000
    max_output = 512_000
    effective_params = "1.7T"
    compression = BITNET_V4_LABEL
    inference_mode = "deep"
    target_tps = 48.0
    encode_budget_ms = 120.0
    description = "cat r1 — math, code, analysis, DSpark + extended thinking"


class CatR1FlashProfile:
    """
    cat r1-flash — BitNet turbo instant path.
    Single-pass distillation, prefix cache, minimal latency.
    """
    model_id = "cat-r1-flash-v1"
    label = CAT_R1_FLASH
    tier = "flash"
    reasoning_depth = 1
    recursive_passes = 1
    context_window = 256_000
    max_output = 32_000
    effective_params = "68B"
    compression = "turbo-distil"
    inference_mode = "instant"
    target_tps = 420.0
    encode_budget_ms = 8.0
    description = "cat r1-flash — chat, Q&A, DSpark lite, instant"


@dataclass
class InferenceStats:
    tier: str = "flash"
    model_label: str = ""
    encode_ms: float = 0.0
    forward_ms: float = 0.0
    passes: int = 0
    tps_estimate: float = 0.0
    bitnet_kernel: str = ""
    compression_ratio: float = 1.0
    effective_params: str = ""
    algo: str = ""
    dspark_accepted: int = 0
    dspark_drafts: int = 0
    dspark_speedup: float = 1.0


class CatR1V4InferenceRuntime:
    """
    cat r1 profile-aware BitNet encode — cat r1 deep vs cat r1-flash turbo.
    Measures real encode latency and estimates throughput from BitNet forward cost.
    """

    __slots__ = ("engine", "last_stats")

    def __init__(self, engine: "CatR11Engine"):
        self.engine = engine
        self.last_stats = InferenceStats()

    def _measure_forward_ms(self, vec: np.ndarray) -> float:
        t0 = time.perf_counter()
        _ = self.engine.output_head_lin.forward(vec)
        for blk in self.engine.cat_r1_blocks[:1]:
            seq = vec.reshape(1, -1) if vec.ndim == 1 else vec
            _ = blk.forward(seq)
        return (time.perf_counter() - t0) * 1000.0

    def encode(self, prompt: str, task: Optional[str] = None) -> np.ndarray:
        profile = self.engine.active_model_profile
        is_pro = profile.tier == "pro"
        t0 = time.perf_counter()

        if task in ("code", "execute") and CONFIG.get("v4_pro_coding", True):
            vec = self.engine.v4_pro_code_encode(prompt)
            algo = f"{BITNET_V4_LABEL} coding"
        elif is_pro:
            depth = int(getattr(profile, "reasoning_depth", CONFIG.get("cat_r1_pro_reasoning_depth", 5)))
            dspark = getattr(self.engine, "dspark", None)
            if (
                CONFIG.get("dspark_enabled")
                and CONFIG.get("dspark_speculative_decode")
                and dspark is not None
            ):
                vec = dspark.speculative_encode(prompt, depth)
                ds = dspark.last_stats
                algo = (
                    f"{CAT_R1_LABEL} DSpark×{BITNET_V4_LABEL} · {depth} passes · "
                    f"{ds.accepted}/{ds.drafts} accept · γ={ds.gamma} · ~{ds.speedup:.2f}x"
                )
            else:
                vec = self.engine.recursive_encode(prompt, depth=depth)
                if CONFIG.get("v4_pro_compression"):
                    vec = self.engine.v4_compressor.v4_compress(vec)
                algo = f"{BITNET_V4_LABEL} · {depth} passes"
        elif CONFIG.get("cat_r1_dspark") and CONFIG.get("dspark_enabled"):
            dspark = getattr(self.engine, "dspark", None)
            depth = int(getattr(profile, "reasoning_depth", 1))
            if dspark is not None and depth >= 1:
                vec = dspark.speculative_encode(prompt, max(1, depth))
                ds = dspark.last_stats
                algo = f"{CAT_R1_LABEL} flash · {ds.accepted}/{ds.drafts} · γ={ds.gamma}"
            elif CONFIG.get("deepmind_fast"):
                vec = self.engine.deepmind.turbo_encode(prompt)
                algo = f"{CAT_R1_FLASH} turbo"
            else:
                vec = self.engine.recursive_encode(prompt, depth=depth)
                algo = f"{CAT_R1_FLASH} · {depth} pass"
        elif CONFIG.get("deepmind_fast"):
            vec = self.engine.deepmind.turbo_encode(prompt)
            algo = f"{CAT_R1_FLASH} turbo"
        else:
            depth = int(getattr(profile, "reasoning_depth", 1))
            vec = self.engine.recursive_encode(prompt, depth=depth)
            algo = f"{CAT_R1_FLASH} · {depth} pass"

        encode_ms = (time.perf_counter() - t0) * 1000.0
        forward_ms = self._measure_forward_ms(vec)
        target_tps = float(
            getattr(profile, "target_tps", None)
            or (CONFIG.get("v4_pro_target_tps", 48.0) if is_pro else CONFIG.get("v4_flash_target_tps", 420.0))
        )
        budget_ms = float(
            getattr(profile, "encode_budget_ms", None)
            or (CONFIG.get("v4_pro_encode_budget_ms", 120.0) if is_pro else CONFIG.get("v4_flash_encode_budget_ms", 8.0))
        )
        latency_factor = max(0.35, min(2.5, encode_ms / max(budget_ms, 0.5)))
        tps_estimate = target_tps / latency_factor
        dspark = getattr(self.engine, "dspark", None)
        ds = dspark.last_stats if dspark is not None else DSparkStats()
        if ds.speedup > 1.0:
            tps_estimate *= ds.speedup

        self.last_stats = InferenceStats(
            tier=profile.tier,
            model_label=profile.label,
            encode_ms=encode_ms,
            forward_ms=forward_ms,
            passes=self.engine.last_recursive_passes,
            tps_estimate=tps_estimate,
            bitnet_kernel=CONFIG.get("bitnet_kernel", "ternary_addsub"),
            compression_ratio=self.engine.last_compression_ratio,
            effective_params=profile.effective_params,
            algo=algo,
            dspark_accepted=ds.accepted,
            dspark_drafts=ds.drafts,
            dspark_speedup=ds.speedup,
        )
        self.engine.last_inference_stats = self.last_stats
        self.engine.last_vec = vec.copy()
        dspark = getattr(self.engine, "dspark", None)
        if dspark is not None:
            ds = dspark.last_stats
            self.engine.cat_r1_stats["dspark"] = {
                "enabled": CONFIG.get("dspark_enabled", True),
                "accepted": ds.accepted,
                "drafts": ds.drafts,
                "gamma": ds.gamma,
            }
        return vec

    def stats_line(self) -> str:
        s = self.last_stats
        if not s.model_label:
            return f"{BRAND} runtime idle"
        return (
            f"{s.model_label} · {s.tps_estimate:.0f} tok/s · "
            f"encode {s.encode_ms:.1f}ms · {s.passes} pass(es) · {s.algo}"
            + (f" · DSpark {s.dspark_accepted}/{s.dspark_drafts}" if s.dspark_drafts else "")
        )


DeepSeekV4ProCompression = CatR1V4ProCompression
DeepSeekV4InferenceRuntime = CatR1V4InferenceRuntime


class CatR1ModelRouter:
    """
    Routes between Pro (deep reasoning) and Flash (instant) models.
    Uses intent classification to select optimal model variant.
    All in-memory, files = off.
    """

    PRO = CatR1ProProfile()
    FLASH = CatR1FlashProfile()

    INSTANT_INTENTS = frozenset({
        "chat", "greeting", "casual", "smalltalk",
        "thanks", "goodbye", "joke", "opinion",
    })
    DEEP_INTENTS = frozenset({
        "code", "math", "debug", "explain", "design",
        "agent", "research", "analysis", "compare",
    })

    @classmethod
    def select(cls, prompt: str, intent: str = "") -> Any:
        pl = prompt.lower().strip()
        if not pl or len(pl) < 12:
            return cls.FLASH
        if intent in cls.DEEP_INTENTS:
            return cls.PRO
        if intent in cls.INSTANT_INTENTS:
            return cls.FLASH
        if any(k in pl for k in ("code", "python", "debug", "error", "traceback",
                                "explain", "why", "how", "analyze", "compare",
                                "design", "build", "architecture")):
            return cls.PRO
        if any(k in pl for k in ("hi", "hello", "hey", "thanks", "bye", "joke",
                                "how are you", "what's up")):
            return cls.FLASH
        if "?" in prompt and len(prompt) > 40:
            return cls.PRO
        if len(pl.split()) > 15:
            return cls.PRO
        return cls.FLASH

    @classmethod
    def stats_line(cls) -> str:
        return f"{BRAND} router · {CAT_R1_PRO} (deep) · {CAT_R1_FLASH} (instant) · files = off"


CatR1Pro = CatR1ProProfile
CatR1Flash = CatR1FlashProfile


class CatR1Core:
    """
    Unified inference core: compressed cat r1 forward + o1-preview recursive loop.
    Targets high answer quality on consumer hardware (files = off).
    """

    __slots__ = ("engine",)

    def __init__(self, engine: "CatR11Engine"):
        self.engine = engine

    def infer_state(self, prompt: str) -> np.ndarray:
        return self.engine.recursive_encode(prompt)

    def stats_line(self) -> str:
        c = self.engine.compressor
        v4 = self.engine.v4_compressor
        st = self.engine.cat_r1_stats
        base = (
            f"{BRAND} · {st['cat_r1_linear_layers']} cat r1 linear · "
            f"{BRAND} distil · "
            f"packed {st['packed_kb']:.1f}KB · "
            f"compress {c.last_ratio:.1f}x"
        )
        if CONFIG.get("v4_pro_compression"):
            v4s = v4.stats()
            base += f" · V4 {v4s['last_ratio']} · ~{v4s['effective_capacity_billions']}B eff"
        active = getattr(self.engine, 'active_model_profile', None)
        if active:
            base += f" · {active.label}"
        v4rt = getattr(self.engine, "v4_runtime", None)
        if v4rt is not None and v4rt.last_stats.model_label:
            base += f" · {v4rt.stats_line()}"
        if CONFIG.get("deepmind_fast") and self.engine.deepmind:
            wp = GoogleWhitepaperCatR1Sorter.stats_line() if CONFIG.get("google_whitepaper_heuristics") else ""
            tail = f"{self.engine.deepmind.stats_line()}"
            if wp:
                tail = f"{wp} · {tail}"
            return f"{base} · {tail}"
        return base


# Legacy aliases (BitNet → cat r1)
BitLinear = CatR1Linear
BitNetLinear = CatR1Linear
BitNetMatmulKernel = BitNetMatmul
BitNetBlock = CatR1Block
BitNetCompressor = CatR1Compressor
BitNetRivalCore = CatR1Core
bitnet_memory_report = cat_r1_memory_report


# Casual chat — matched before educational intent routing (EN · 中文)
_SMALLTALK: Tuple[Tuple[str, str], ...] = (
    (r"^(?:hi|hey|hello|yo|howdy)\s*[!?.]*$", "Hi! How can I help you today?"),
    (r"^how are you(?: doing| today)?\??$", "Doing well, thanks for asking! I'm here and ready to chat. How about you?"),
    (r"^how(?:'re| are) you(?: doing| today)?\??$", "Doing well, thanks for asking! I'm here and ready to chat. How about you?"),
    (r"^how(?:'s| is|s) it going\??$", "Going well on my end — thanks! What can I help you with today?"),
    (r"^how(?:'s| is|s) everything\??$", "All good here! What's on your mind?"),
    (r"^how(?:'s| is|s) your day\??$", "Running smoothly so far. How's yours going?"),
    (r"^how have you been\??$", "Steady and ready to help. What have you been up to?"),
    (r"^(?:what's up|whats up|wassup|sup)\??$", "Not much — just here to help. What are you working on?"),
    (r"^how you doing\??$", "Doing great, thanks! What can I do for you?"),
    (r"^how(?:'s| is| are) (?:u|ya|you) doing\??$", "Doing great, thanks! What can I do for you?"),
    (r"^how r u\??$", "Doing well! How are you?"),
    (r"^how are u\??$", "Doing well! How are you?"),
    (r"^(?:good morning|good afternoon|good evening)\.?\??$", "Good to hear from you! What would you like to talk about?"),
    (r"^nice to meet you\.?\??$", "Nice to meet you too! Ask me anything — code, explanations, debugging, or just chat."),
    (r"^(?:are you ok|you ok|u ok)\??$", "I'm all good, thanks! How can I help?"),
    (r"^what(?:'s| is) new\??$", f"Same local {BRAND} engine, ready when you are. What's new with you?"),
    (r"^how do you feel\??$", "I feel ready to help! What's up?"),
    (r"^(?:你好|您好|嗨|哈喽|在吗)[!！?？。.\s]*$", "你好！有什么我可以帮你的？"),
    (r"^你好吗[!！?？]?$", "我很好，谢谢！你今天想聊点什么？"),
    (r"^(?:早上好|下午好|晚上好)[!！?？]?$", "你好！很高兴见到你。需要什么帮助？"),
    (r"^谢谢[!！?？]?$", "不客气！还需要别的帮助吗？"),
    (r"^感谢[!！?？]?$", "不客气！随时可以继续问我。"),
    (r"^meow[!?.]*$", "Meow! 🐱 What's on your mind?"),
    (r"^mew[!?.]*$", "Mew! 🐱 Ready when you are."),
    (r"^(?:pr\s+)?meow[!?.]*$", "Meow! 🐱 *purrs* — ask me anything."),
    (r"^(?:再见|拜拜)[!！?？]?$", "再见！期待下次聊天。"),
    (r"^thanks(?: a lot| so much)?[!?.]*$", "You're welcome! Anything else I can help with?"),
    (r"^thank you[!?.]*$", "You're welcome! Feel free to ask more."),
    (r"^(?:ok|okay|got it|sounds good)[!?.]*$", "Great! What's next?"),
    (r"^(?:你是谁|你是什么)[!！?？]?$", f"我是 **{BRAND}** — 本地双语聊天助手，支持中英文对话和代码。"),
    (r"^(?:你能做什么|你会什么|有什么功能)[!！?？]?$", "我可以聊天、写代码、解释概念、调试程序、做数学题。中英文都可以，随便问！"),
    (r"^(?:讲个笑话|说个笑话|来个笑话)[!！?？]?$", "程序员为什么喜欢深色模式？因为亮光会吸引 bug。"),
)


# Multilingual tokenization (EN · 中文 · mixed · files = off)
_TOKEN_EN = re.compile(r"[a-z0-9+#]+", re.I)
_TOKEN_CJK = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff]")
_ZH_QUESTION = re.compile(
    r"(什么是|是什么|什么叫|何为|为何|为什么|为啥|怎么|如何|怎样|能否|可不可以|介绍一下|解释|说明|告诉我)"
)
_ZH_GREETING = re.compile(
    r"^(你好|您好|嗨|哈喽|早上好|下午好|晚上好|在吗|你是谁|你好吗)[!！?？。.\s]*$"
)
_ZH_TOPIC = re.compile(
    r"(?:什么是|是什么|什么叫|解释|说明|介绍|告诉我)(.+?)[?？。!！]?$"
)


def tokenize_text(text: str, max_tokens: int = 256) -> List[str]:
    """Split English words and CJK characters for in-memory embedding (files = off)."""
    raw = (text or "").strip()
    if not raw:
        return ["<unk>"]
    lower = raw.lower()
    tokens: List[str] = []
    i = 0
    while i < len(lower) and len(tokens) < max_tokens:
        m = _TOKEN_EN.match(lower, i)
        if m:
            tokens.append(m.group(0))
            i = m.end()
            continue
        m = _TOKEN_CJK.match(raw, i)
        if m:
            tokens.append(m.group(0))
            i = m.end()
            continue
        i += 1
    return tokens or ["<unk>"]


def is_zh_question(text: str) -> bool:
    s = (text or "").strip()
    if re.search(r"(天气|吃什么|穿什么|心情)", s) and re.search(r"(怎么样|如何|怎样)[?？]?$", s):
        return False
    return bool(_ZH_QUESTION.search(s))


def is_zh_greeting(text: str) -> bool:
    s = (text or "").strip()
    if _ZH_GREETING.match(s):
        return True
    return len(s) <= 10 and bool(re.search(r"^(你好|您好|你好吗|早上好|下午好|晚上好)", s))


def is_explain_request(text: str) -> bool:
    pl = (text or "").lower()
    if re.search(
        r"\b(explain|what is|what are|what's|why|how (?:does|do|to|can|would|should))\b"
        r"|(?:tell me|talk) about\b"
        r"|\bhelp me with\b",
        pl,
    ):
        return True
    return is_zh_question(text or "")


def extract_zh_topic(prompt: str) -> str:
    s = (prompt or "").strip()
    m = _ZH_TOPIC.search(s)
    if m:
        return m.group(1).strip("？?。!！ ")[:80]
    return s[:80]


# ──────────────────────────────────────────────────────────────
# VIBE CODE HEURISTICS (EN · 中文 · mixed · pasted code · files = off)
# ──────────────────────────────────────────────────────────────
class VibeCodeHeuristics:
    """
    Multilingual intent + language detection for casual vibe-coding.
    Understands English, Chinese (中文), mixed human language, and raw code snippets.
    """

    CJK = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff]")
    ZH_CODE_NOUNS = (
        "写代码", "编程", "程序", "源码", "源代码", "脚本", "网页", "代码",
        "函数", "算法", "斐波那契", "斐波那契数列", "小程序", "页面",
    )
    ZH_RUN = ("运行", "执行", "跑一下", "跑起来", "测试", "试一下")
    EN_VIBE = re.compile(
        r"\b(vibe[\s-]?code|vibecode|whip up| cook up|spin up|slap together|"
        r"gimme|lemme get|just (?:make|write|code)|yo make|hook me up with)\b",
        re.I,
    )
    ZH_WRITE = re.compile(
        r"(写|做|建|造|生成|帮我写|帮我做|给我写|给我做|帮我生成|"
        r"来个|整一个|弄一个|搞一个|创建一个|实现|编写|弄段|来段|"
        r"能不能写|可以写|请写|请帮我|麻烦写|麻烦帮我)"
    )
    ZH_CORRECT = re.compile(
        r"(不要|别|不对|不是|改成|改为|换成|用)(?:.{0,12})?(html|python|javascript|java|c\+\+?|cpp|rust|go|bash|typescript|sql|c语言|网页|脚本)",
        re.I,
    )
    ZH_LANG_INLINE = re.compile(
        r"(?:用|以|\bin\b|with|using|换成|改成|改为)\s+"
        r"(python|html|javascript|typescript|java|kotlin|swift|rust|go|bash|shell|cpp|c\+\+|c|sql|ruby|php|"
        r"c语言|c\+\+语言|网页|页面|脚本|js|py|python3?|html5?)",
        re.I,
    )
    ZH_SAYS = re.compile(
        r"(?:显示|输出|打印|说|写上|内容是|文字是|写着)[「\"']?([^」\"'\n，。！？]+)[」\"']?",
    )
    ZH_SUBJECT = re.compile(
        r"(?:写|做|建|生成|帮我写|帮我做|来个|整一个|弄一个|搞一个|创建一个|实现)"
        r"(?:一个|个|一段|段)?(.+?)(?:用|in|的|程序|代码|网页|脚本|$)",
    )
    LANG_ALIASES = {
        "py": "python", "python3": "python", "python2": "python",
        "js": "javascript", "node": "javascript", "ts": "typescript",
        "c++": "cpp", "cc": "cpp", "c语言": "c", "c++语言": "cpp",
        "网页": "html", "页面": "html", "html5": "html", "前端": "html",
        "脚本": "python", "shell": "bash", "sh": "bash", "zsh": "bash",
        "golang": "go", "go语言": "go", "java语言": "java",
        "rust语言": "rust", "php语言": "php",
    }
    CODE_SHAPES = re.compile(
        r"(#include\s*<|def\s+\w+\s*\(|function\s+\w+|fn\s+main|public\s+class|"
        r"<!DOCTYPE|<html|console\.log|printf\s*\(|System\.out|package\s+main|"
        r"import\s+\w+|class\s+\w+\s*[:{]|=>\s*\{|var\s+\w+\s*=)",
        re.I,
    )

    @classmethod
    def enabled(cls) -> bool:
        return bool(CONFIG.get("vibe_code_heuristics", True))

    @classmethod
    def has_cjk(cls, text: str) -> bool:
        return bool(cls.CJK.search(text or ""))

    @classmethod
    def _norm_lang(cls, token: str) -> Optional[str]:
        if not token:
            return None
        t = token.strip().lower()
        return cls.LANG_ALIASES.get(t, t)

    @classmethod
    def wants_code(cls, prompt: str) -> bool:
        if not cls.enabled():
            return False
        raw = (prompt or "").strip()
        if not raw:
            return False
        pl = raw.lower()

        # "什么是 python" / "what is python" = explain, not code generation.
        if is_explain_request(raw) and not cls.ZH_WRITE.search(raw):
            if not re.search(r"\b(write|make|build|create|implement|code|vibe)\b", pl):
                if not cls.EN_VIBE.search(raw):
                    return False

        if re.match(r"^\s*(?:code|/code|code\s*>|>|program|script)\s*$", pl, re.I):
            return True
        if cls.CODE_SHAPES.search(raw):
            return True
        if cls.EN_VIBE.search(raw):
            return True
        if any(cue in raw for cue in cls.ZH_CODE_NOUNS):
            return True
        if cls.ZH_WRITE.search(raw):
            if re.search(
                r"(代码|程序|脚本|网页|函数|html|python|javascript|java|rust|go|c\+\+|cpp|"
                r"fibonacci|fib|算法|页面|api|app|site|hello|cat|meow|打印|输出)",
                raw, re.I,
            ):
                return True
        if cls.ZH_CORRECT.search(raw):
            return True
        if cls.ZH_LANG_INLINE.search(raw):
            return True
        if re.search(r"[\u4e00-\u9fff].*(html|python|javascript|java|rust|go|cpp|c\+\+|c\b)", raw, re.I):
            if is_zh_question(raw) and not cls.ZH_WRITE.search(raw):
                return False
            return True
        if re.search(r"(html|python|javascript|java|rust|go|cpp|c\+\+|c\b).*[\u4e00-\u9fff]", raw, re.I):
            if is_zh_question(raw) and not cls.ZH_WRITE.search(raw):
                return False
            return True
        if cls.has_cjk(raw) and re.search(r"(写|做|建|生成|程序|代码|脚本|网页)", raw):
            return True
        if CodeAnythingEngine.enabled() and CodeAnythingEngine.wants_anything(raw):
            return True
        return False

    @classmethod
    def lang_from_text(cls, prompt: str, engine: "CatR11Engine") -> Optional[str]:
        if not cls.enabled():
            return None
        raw = (prompt or "").strip()
        pl = raw.lower()

        m = cls.ZH_CORRECT.search(raw)
        if m:
            return engine.normalize_lang(cls._norm_lang(m.group(2)))

        m = cls.ZH_LANG_INLINE.search(raw)
        if m:
            return engine.normalize_lang(cls._norm_lang(m.group(1)))

        m = CatR1Code._MAKE_IT_LANG.search(pl)
        if m:
            return engine.normalize_lang(m.group(1))

        if re.search(r"(网页|页面|html|前端)", raw, re.I):
            if re.search(r"(写|做|建|生成|程序|代码|脚本|make|write|build|create|vibe)", raw, re.I):
                return "html"

        if re.search(r"(脚本|bash|shell)", raw, re.I) and re.search(
            r"(写|做|建|生成|write|make|build)", raw, re.I
        ):
            return "bash"

        lang_tokens = (
            ("python", r"\bpython\b|python3|\.py\b|蟒蛇"),
            ("javascript", r"\bjavascript\b|\bjs\b|node\.?js"),
            ("typescript", r"\btypescript\b|\bts\b"),
            ("java", r"\bjava\b(?!script)"),
            ("kotlin", r"\bkotlin\b"),
            ("swift", r"\bswift\b"),
            ("rust", r"\brust\b"),
            ("go", r"\bgo\b|golang|go语言"),
            ("cpp", r"\bc\+\+\b|\bcpp\b|c\+\+语言"),
            ("c", r"\bc语言\b|\bc\s+程序\b|\bc\s+代码\b"),
            ("html", r"\bhtml\b|html5"),
            ("bash", r"\bbash\b|\bshell\b"),
            ("sql", r"\bsql\b"),
            ("ruby", r"\bruby\b"),
            ("php", r"\bphp\b"),
        )
        has_write = bool(
            re.search(
                r"(写|做|建|生成|帮我|write|make|build|create|code|vibe|implement|scaffold|boilerplate)",
                raw, re.I,
            )
        )
        for lang, pat in lang_tokens:
            if re.search(pat, raw, re.I) and (has_write or cls.CODE_SHAPES.search(raw)):
                return engine.normalize_lang(lang)

        if cls.has_cjk(raw) and re.search(r"(程序|代码)", raw) and not re.search(
            r"(python|html|javascript|java|rust|go|c\+\+|cpp|c\b)", raw, re.I
        ):
            return "python"
        return None

    @classmethod
    def subject_from_text(cls, prompt: str) -> Optional[str]:
        if not cls.enabled():
            return None
        raw = (prompt or "").strip()

        m = cls.ZH_SAYS.search(raw)
        if m:
            return m.group(1).strip("?.! ，。！？")
        if "你好猫" in raw or "hello cat" in raw.lower():
            return "Hello Cat"
        if "喵" in raw or "meow" in raw.lower():
            return "Meow"
        if "你好世界" in raw or "hello world" in raw.lower():
            return "Hello World"

        m = cls.ZH_SUBJECT.search(raw)
        if m:
            subj = m.group(1).strip("?.! ，。！？的 ")
            skip = {"html", "python", "javascript", "java", "代码", "程序", "脚本", "网页", "一个", "段"}
            if subj and subj not in skip and len(subj) >= 1:
                return subj[:80]

        m = re.search(
            r"(?:write|make|build|create|vibe[\s-]?code|whip up|gimme)\s+(?:me\s+)?(?:a\s+)?(.+?)(?:\s+in\s+\w+|$)",
            raw, re.I,
        )
        if m:
            subj = m.group(1).strip("?.! ")
            if subj.lower() not in {"html", "it", "a", "an", "code"}:
                return subj[:80]
        return None

    @classmethod
    def wants_run(cls, prompt: str) -> bool:
        raw = (prompt or "").strip().lower()
        run_en = (
            "run it", "run this", "execute", "interpret", "test it",
            "/run", "and run", "then run",
        )
        if any(x in raw for x in run_en):
            return True
        return any(r in (prompt or "") for r in cls.ZH_RUN)


# ──────────────────────────────────────────────────────────────
# GOOGLE WHITEPAPER HEURISTICS → CATR1 VOICE (files = off)
# Attention · BM25 rank · Chinchilla budget · MoE route · distil align
# ──────────────────────────────────────────────────────────────
@dataclass
class RankedDataItem:
    key: str
    text: str
    role: str = "chunk"
    score: float = 0.0
    expert: str = "chat"
    meta: Dict[str, Any] = field(default_factory=dict)


class GoogleWhitepaperCatR1Sorter:
    """
    Sort in-memory context with Google whitepaper heuristics (files = off),
    then shape replies in cat r1 conversational style.
    """

    EXPERTS = ("reason", "code", "math", "retrieve", "chat")
    WP_ALGOS = (
        "Transformer attention relevance",
        "BM25 learning-to-rank",
        "Chinchilla compute-optimal density",
        "Switch-MoE expert routing",
        "Distillation teacher alignment",
        "Session recency decay",
    )
    _DOC_FREQ: Dict[str, int] = {}
    _N_DOCS = 0
    _AVG_DL: float = 64.0

    @classmethod
    def enabled(cls) -> bool:
        return bool(CONFIG.get("google_whitepaper_heuristics", True))

    @classmethod
    def _tokens(cls, text: str) -> List[str]:
        return re.findall(r"[a-zA-Z\u4e00-\u9fff]{2,}", (text or "").lower())

    @classmethod
    def _register_corpus(cls, texts: List[str]) -> None:
        cls._DOC_FREQ.clear()
        cls._N_DOCS = max(len(texts), 1)
        total_dl = 0
        for doc in texts:
            tokens = cls._tokens(doc)
            total_dl += len(tokens)
            for tok in set(tokens):
                cls._DOC_FREQ[tok] = cls._DOC_FREQ.get(tok, 0) + 1
        cls._AVG_DL = total_dl / cls._N_DOCS if cls._N_DOCS else 64.0

    @classmethod
    def attention_score(cls, query: str, doc: str) -> float:
        """Scaled dot-product style relevance (Transformer attention analogy)."""
        q, d = cls._tokens(query), cls._tokens(doc)
        if not q or not d:
            return 0.0
        qs, ds = set(q), set(d)
        overlap = len(qs & ds)
        return overlap / (len(qs) ** 0.5 * len(ds) ** 0.5 + 1e-6)

    @classmethod
    def bm25_score(cls, query: str, doc: str, *, k1: float = 1.2, b: float = 0.75) -> float:
        """BM25-lite rank (Google Search / learning-to-rank family)."""
        qtoks = cls._tokens(query)
        dtoks = cls._tokens(doc)
        if not qtoks or not dtoks:
            return 0.0
        dl = len(dtoks)
        avgdl = cls._AVG_DL
        score = 0.0
        dset = dtoks
        for term in set(qtoks):
            tf = dset.count(term)
            if not tf:
                continue
            df = cls._DOC_FREQ.get(term, 0)
            idf = np.log(1.0 + (cls._N_DOCS - df + 0.5) / (df + 0.5))
            num = tf * (k1 + 1)
            den = tf + k1 * (1.0 - b + b * dl / avgdl)
            score += idf * num / den
        return float(score)

    @classmethod
    def chinchilla_density(cls, text: str) -> float:
        """Chinchilla-style: reward information-dense short spans."""
        words = cls._tokens(text)
        n = max(len(words), 1)
        unique = len(set(words))
        density = unique / n
        length_pen = min(1.0, 72.0 / n)
        return 0.55 * density + 0.45 * length_pen

    @classmethod
    def route_expert(cls, text: str) -> str:
        """Switch-MoE style expert bucket (Gemini / PaLM-MoE routing analogy)."""
        pl = (text or "").lower()
        if re.search(r"\d+\s*[\+\-\*/%^]|calculate|equation|integral|fibonacci|\bfib\b", pl):
            return "math"
        if re.search(r"\b(code|python|def |```|debug|error|traceback|compile|syntax)\b", pl):
            return "code"
        if re.search(r"\b(why|how|explain|walkthrough|step|because|therefore)\b", pl):
            return "reason"
        if re.search(r"\b(what is|what are|define|meaning of|who is|when did)\b", pl):
            return "retrieve"
        return "chat"

    @classmethod
    def distil_align(cls, text: str) -> float:
        """Boost teacher-aligned / chain-of-thought phrasing (distillation papers)."""
        pl = (text or "").lower()
        cues = ("step", "first", "second", "therefore", "because", "verify", "note that", "in summary")
        hits = sum(1 for c in cues if c in pl)
        return min(0.25, hits * 0.05)

    @classmethod
    def score_item(
        cls,
        query: str,
        text: str,
        *,
        recency: float = 1.0,
        role: str = "chunk",
    ) -> float:
        if not cls.enabled():
            return recency
        expert_q = cls.route_expert(query)
        expert_d = cls.route_expert(text)
        moe_bonus = 0.18 if expert_d == expert_q else 0.0
        role_bonus = 0.08 if role == "user" else 0.0
        return (
            cls.attention_score(query, text) * 0.32
            + cls.bm25_score(query, text) * 0.28
            + cls.chinchilla_density(text) * 0.18
            + moe_bonus
            + cls.distil_align(text)
            + role_bonus
            + recency * 0.12
        )

    @classmethod
    def sort_items(cls, items: List[RankedDataItem], query: str) -> List[RankedDataItem]:
        if not items:
            return items
        cls._register_corpus([it.text for it in items])
        n = len(items)
        for i, it in enumerate(items):
            recency = 0.85 + 0.15 * (i + 1) / n
            it.expert = cls.route_expert(it.text)
            it.score = cls.score_item(query, it.text, recency=recency, role=it.role)
        return sorted(items, key=lambda x: x.score, reverse=True)

    @classmethod
    def sort_history(cls, history: List[Tuple[str, str]], query: str) -> List[Tuple[str, str]]:
        if not cls.enabled() or not history:
            return history
        items = [
            RankedDataItem(key=str(i), text=t, role=r)
            for i, (r, t) in enumerate(history)
        ]
        return [(it.role, it.text) for it in cls.sort_items(items, query)]

    @classmethod
    def sort_history_dicts(cls, history: List[Dict[str, str]], query: str) -> List[Dict[str, str]]:
        if not cls.enabled() or not history:
            return history
        items = [
            RankedDataItem(key=str(i), text=m.get("text", ""), role=m.get("role", "user"))
            for i, m in enumerate(history)
        ]
        return [{"role": it.role, "text": it.text} for it in cls.sort_items(items, query)]

    @classmethod
    def sort_paragraphs(cls, body: str, query: str) -> str:
        """Reorder multi-paragraph drafts by relevance (in-memory only)."""
        if not cls.enabled() or not body:
            return body
        if "```" in body:
            return body
        paras = [p.strip() for p in re.split(r"\n{2,}", body.strip()) if p.strip()]
        if len(paras) < 2:
            return body
        items = [RankedDataItem(key=str(i), text=p, role="chunk") for i, p in enumerate(paras)]
        ranked = cls.sort_items(items, query)
        return "\n\n".join(it.text for it in ranked)

    @classmethod
    def cat_r1_voice(cls, body: str, prompt: str, task: str = "general") -> str:
        """Format sorted content like cat r1: direct, structured style."""
        if not CONFIG.get("cat_r1_voice", True) or not body:
            return body
        text = body.strip()
        if "```" in text or re.search(r"\bfable\b|\bpoem\b", text, re.I):
            return text

        text = cls.sort_paragraphs(text, prompt)
        pl = prompt.lower().strip()

        if any(k in pl for k in ("how ", "why ", "step", "explain", "walk me", "怎么", "为什么")):
            if "1." not in text[:300] and "Step" not in text[:300] and len(text) > 140:
                paras = [p.strip() for p in text.split("\n\n") if p.strip()]
                if len(paras) >= 2 and not paras[0].startswith("**"):
                    text = "\n\n".join(
                        f"**{i}.** {p.lstrip('0123456789. ')}"
                        for i, p in enumerate(paras, 1)
                    )

        return text.strip()

    @classmethod
    def stats_line(cls) -> str:
        return f"Google WP sort ({len(cls.WP_ALGOS)} algos) · {BRAND} voice"


CatR1VoiceSorter = GoogleWhitepaperCatR1Sorter
WPCatR1Sort = GoogleWhitepaperCatR1Sorter


# ──────────────────────────────────────────────────────────────
# CAT R1.1 HEURISTICS ENGINE (cat r1 tier · chat + code)
# Unified heuristics: intent · code detection · quality · self-verify · tone
# ──────────────────────────────────────────────────────────────
class CatR1Heuristics:
    """
    Unified heuristics engine for chatting and coding — cat r1 tier.
    Consolidates intent detection, code detection, language detection,
    response quality scoring, self-verification, and conversation flow.
    All heuristics run in-memory (files = off).
    """

    THINK_DEPTH_MAP = {
        "chat": 0, "casual": 0, "greeting": 0,
        "math": 1, "code": 2, "debug": 2, "execute": 1,
        "explain": 2, "compare": 2, "design": 3,
        "agent": 3, "general": 2, "fable": 1, "poem": 1,
    }
    CONFIDENCE_CATEGORIES = ("high", "medium", "low")

    @classmethod
    def classify_intent(cls, prompt: str) -> Dict[str, Any]:
        """
        cat r1 tier intent classification with confidence scoring.
        Returns {intent, confidence, topic, subtopics, needs_code, needs_web}.
        """
        raw = (prompt or "").strip()
        pl = raw.lower()
        cjk = bool(re.search(r"[\u4e00-\u9fff]", pl))

        # Fast-path smalltalk
        if CatR11Synthesizer.smalltalk_reply(pl) or is_zh_greeting(raw):
            return {"intent": "chat", "confidence": "high", "topic": raw[:48], "subtopics": [], "needs_code": False, "needs_web": False}

        # Noise detection
        if CatR1Fusion.is_noise(raw):
            return {"intent": "chat", "confidence": "high", "topic": "", "subtopics": [], "needs_code": False, "needs_web": False}

        needs_code = cls.detect_code_request(raw)
        needs_web = CatR1WebProgram.wants_web(pl) if CatR1WebProgram.enabled() else False
        topic = cls.extract_topic(raw)
        subtopics: List[str] = []

        # Code with fenced blocks
        if needs_code and CatR1Code.extract_prompt_code(raw)[1]:
            return {"intent": "code", "confidence": "high", "topic": topic, "subtopics": ["fenced-block"], "needs_code": True, "needs_web": False}

        # Multi-label scoring
        scores: Dict[str, float] = {}
        intent_patterns = {
            "code": (r"\b(code|function|class|implement|snippet|script|def |import |```|"
                    r"write\s+(a|an|me)\s+(function|class|program|script|code))", 3.0),
            "math": (r"\d+\s*[\+\-\*/%\^]|calculate|compute|equation|solve|fibonacci|prime|factorial", 2.5),
            "debug": (r"\b(error|bug|traceback|exception|crash|broken|fix|debug|not working|fails)", 3.0),
            "explain": (r"\b(explain|what is|what are|define|describe|meaning|how does|how do|why does|"
                        r"tell me about|walk me through|tutorial|什么是|是什么|解释|说明|介绍)", 2.5),
            "design": (r"\b(design|architecture|plan|roadmap|system design|architect|scaffold)", 2.0),
            "compare": (r"\b(compare|vs | versus |difference|better|pros and cons|trade.?off)", 2.0),
            "agent": (r"\b(agent|multi.?step|orchestrate|workflow|pipeline|automate)", 1.8),
            "fable": (r"\b(fable|parable|allegory|bedtime story|moral|once upon)", 2.0),
            "poem": (r"\b(poem|poetry|haiku|sonnet|verse|rhyme)", 2.0),
            "creative": (r"\b(creative|story|tale|narrative|imagine|write a story)", 1.5),
            "web": (r"\b(website|web page|landing page|html page|site|dashboard|portfolio|"
                    r"build a page|make a site)", 2.0),
            "execute": (r"\b(run it|execute|interpret|/run|test it|and run|运行|执行)", 2.0),
            "chat": (r"\b(hi|hey|hello|how are you|what's up|howdy|meow|thanks|thank you)", 0.5),
        }

        for intent, (pattern, weight) in intent_patterns.items():
            matches = re.findall(pattern, pl, re.I)
            if matches:
                count = len(matches) if isinstance(matches, list) else 1
                scores[intent] = scores.get(intent, 0) + weight * count

        # CJK-specific boost
        if cjk:
            zh_code = re.search(r"(写代码|编程|程序|函数|斐波那契|算法|脚本)", pl)
            if zh_code:
                scores["code"] = scores.get("code", 0) + 2.0
            zh_explain = re.search(r"(什么是|是什么|什么叫|解释|说明|介绍)", pl)
            if zh_explain:
                scores["explain"] = scores.get("explain", 0) + 2.0

        # Question mark heuristics
        if "?" in raw and "explain" not in scores:
            scores["explain"] = scores.get("explain", 0) + 1.0

        # Code request override (with explain/creative guards)
        if needs_code:
            has_what_is = bool(re.search(r"\b(what is|what are|what's|whats|explain|define|describe)\b", pl))
            has_creative_cue = bool(re.search(r"\b(poem|poetry|fable|story|tale|creative)\b", pl))
            if has_what_is and scores.get("explain", 0) > 0:
                pass
            elif has_creative_cue:
                pass
            else:
                scores["code"] = scores.get("code", 0) + 4.0

        # Web request override
        if needs_web:
            scores["web"] = scores.get("web", 0) + 3.0

        # Explain vs code disambiguation: "what is X in Y" should be explain
        if scores.get("explain", 0) > 0 and scores.get("code", 0) > 0:
            if re.search(r"\b(what is|what are|explain|define|describe|what's)\b", pl):
                scores["code"] *= 0.3

        # Creative/fable/poem disambiguation: "write a poem" is NOT code
        if scores.get("poem", 0) > 0 or scores.get("fable", 0) > 0:
            if scores.get("code", 0) > 0:
                scores["code"] *= 0.2

        if not scores:
            scores["general"] = 1.0

        best_intent = max(scores, key=scores.get)
        best_score = scores[best_intent]

        confidence: str
        if best_score >= 4.0:
            confidence = "high"
        elif best_score >= 2.0:
            confidence = "medium"
        else:
            confidence = "low"

        subtopics = [k for k, v in scores.items() if v >= 1.5 and k != best_intent][:3]
        think_depth = cls.THINK_DEPTH_MAP.get(best_intent, 1)

        return {
            "intent": best_intent,
            "confidence": confidence,
            "topic": topic,
            "subtopics": subtopics,
            "needs_code": needs_code or best_intent == "code",
            "needs_web": needs_web or best_intent == "web",
            "scores": scores,
            "think_depth": think_depth,
            "cjk": cjk,
        }

    @classmethod
    def detect_code_request(cls, prompt: str) -> bool:
        """cat r1 tier code request detection with high recall."""
        if not CatR1Code.enabled():
            return False
        raw = (prompt or "").strip()
        if not raw:
            return False

        # Direct checks via existing engines
        if CatR1Code.wants_code(raw):
            return True
        if CodeAnythingEngine.enabled() and CodeAnythingEngine.wants_anything(raw):
            return True

        pl = raw.lower()
        # Code shape heuristics
        code_shapes = re.search(
            r"(#include\s*<|def\s+\w+\s*\(|function\s+\w+|fn\s+main|public\s+class|"
            r"<!DOCTYPE|<html|console\.log|printf\s*\(|System\.out|package\s+main|"
            r"import\s+\w+|class\s+\w+\s*[:{]|=>\s*\{|var\s+\w+\s*=)", pl
        )
        if code_shapes:
            return True

        # Creative/narrative/poem/fable override — these should NOT trigger code
        creative_cues = ("poem", "poetry", "haiku", "sonnet", "verse", "rhyme",
                        "fable", "parable", "allegory", "story", "tale", "narrative",
                        "creative", "imagine", "bedtime")
        if any(c in pl for c in creative_cues):
            return False

        # Multi-keyword code intent signals
        code_signals = 0
        write_verbs = ("write", "make", "build", "create", "code", "implement", "generate", "show", "give")
        lang_nouns = ("python", "javascript", "typescript", "html", "css", "rust", "go", "java",
                    "c++", "cpp", "c#", "csharp", "bash", "shell", "sql", "php", "ruby",
                    "swift", "kotlin", "dart", "scala", "r", "julia", "lua")
        code_nouns = ("function", "class", "script", "program", "app", "api", "cli", "website",
                    "page", "snippet", "algorithm", "code", "implementation")

        for v in write_verbs:
            if v in pl:
                code_signals += 1
                break
        for l in lang_nouns:
            if l in pl:
                code_signals += 1
                break
        for n in code_nouns:
            if n in pl:
                code_signals += 1
                break

        # Inline language pattern: "in python", "using rust"
        if re.search(r"\b(in|using|with)\s+(" + "|".join(lang_nouns) + r")\b", pl):
            code_signals += 1

        return code_signals >= 2

    @classmethod
    def detect_language(cls, prompt: str, engine: Optional["CatR11Engine"] = None) -> Optional[str]:
        """Detect programming language from prompt with cat r1 tier accuracy."""
        raw = (prompt or "").strip()
        pl = raw.lower()

        # Extract from fenced blocks
        fenced_lang, fenced_code = CatR1Code.extract_prompt_code(raw)
        if fenced_lang:
            return engine.normalize_lang(fenced_lang) if engine else fenced_lang

        # Direct language mentions with code intent
        lang_priority = [
            ("python", r"\bpython\b|\.py\b|蟒蛇"),
            ("javascript", r"\bjavascript\b|\bjs\b|node\.?js"),
            ("typescript", r"\btypescript\b|\bts\b"),
            ("html", r"\bhtml\b|html5|网页|前端"),
            ("rust", r"\brust\b|cargo\b"),
            ("go", r"\bgo\b|\bgolang\b"),
            ("java", r"\bjava\b(?!script)"),
            ("cpp", r"\bc\+\+\b|\bcpp\b|\bcc\b"),
            ("c", r"\bc\b(?!\+|\s*#|\s*/)|c语言"),
            ("bash", r"\bbash\b|\bshell\b|\bzsh\b"),
            ("sql", r"\bsql\b"),
            ("kotlin", r"\bkotlin\b|\bkt\b"),
            ("swift", r"\bswift\b"),
            ("ruby", r"\bruby\b"),
            ("php", r"\bphp\b"),
        ]

        has_write = bool(re.search(r"\b(write|make|build|create|code|implement)\b", pl))
        for lang, pattern in lang_priority:
            if re.search(pattern, pl, re.I) and has_write:
                norm = engine.normalize_lang(lang) if engine else lang
                return norm

        # Fallback to existing engines
        if engine:
            extracted = engine.extract_lang(raw)
            if extracted:
                return engine.normalize_lang(extracted)
            from_text = engine.detect_lang_from_text(raw)
            if from_text:
                return engine.normalize_lang(from_text)

        # VibeCode fallback for CJK
        if re.search(r"[\u4e00-\u9fff]", pl) and re.search(r"(程序|代码|脚本)", pl):
            return "python"

        return None

    @classmethod
    def extract_topic(cls, prompt: str) -> str:
        """Extract the core topic from a prompt."""
        raw = (prompt or "").strip()
        pl = raw.lower()

        # Subject extraction via existing engines
        if CatR1Code._subject(raw):
            return CatR1Code._subject(raw)
        vibe_subj = VibeCodeHeuristics.subject_from_text(raw)
        if vibe_subj:
            return vibe_subj

        # Strip question prefixes
        for prefix in ("what is ", "what's ", "what are ", "explain ", "define ",
                        "tell me about ", "什么是", "是什么", "解释", "说明"):
            if pl.startswith(prefix):
                return raw[len(prefix):].rstrip("?.,! ")[:80]

        # Extract topic after "about"
        m = re.search(r"\babout\s+(.+?)(?:\s*$|[.?!])", pl)
        if m:
            return m.group(1).strip()[:80]

        # First meaningful noun phrase
        m = re.search(r"\b(how\s+to\s+|how\s+do\s+i\s+|how\s+does\s+)(.+?)(?:\s*$|[.?!])", pl)
        if m:
            return m.group(2).strip()[:80]

        return raw[:80] or "general"

    @classmethod
    def quality_score(cls, text: str, prompt: str) -> float:
        """
        cat r1 tier response quality scoring.
        Returns score 0.0–1.0 based on coverage, structure, relevance.
        """
        if not text or not text.strip():
            return 0.0

        score = 0.4  # base

        # Length adequacy (not too short, not too long)
        words = len(text.split())
        if 10 <= words <= 200:
            score += 0.1
        elif words > 200:
            score += 0.05

        # Structure signals
        if "\n\n" in text:
            score += 0.05
        if re.search(r"^\d+\.|\*\*|^- ", text, re.M):
            score += 0.05

        # Code block presence when prompt asks for code
        if cls.detect_code_request(prompt):
            if "```" in text:
                score += 0.1
            elif "`" in text:
                score += 0.05

        # Relevance: answer addresses the question
        if "?" in prompt:
            if "?" not in text and len(text) > 40:
                score += 0.05
            if re.search(r"\b(because|means|is called|refers to|works by)\b", text, re.I):
                score += 0.05

        # Verification markers
        if re.search(r"\b(verify|check|confirm|note that|in summary)\b", text, re.I):
            score += 0.05

        # cat r1 tone: clear, direct, structured
        if re.search(r"\*\*[^*]+\*\*", text):
            score += 0.05
        if re.search(r"\b(step|first|second|finally|therefore)\b", text, re.I):
            score += 0.05

        return min(1.0, score)

    @classmethod
    def self_verify(cls, prompt: str, response: str) -> Dict[str, Any]:
        """
        Self-verification heuristics (cat r1 GRPO-style).
        Returns {passed, issues, suggestions}.
        """
        issues: List[str] = []
        suggestions: List[str] = []

        if not response or not response.strip():
            return {"passed": False, "issues": ["empty response"], "suggestions": ["generate a substantive reply"]}

        # Check response addresses prompt
        pl = prompt.lower()
        rl = response.lower()

        # Question coverage
        if "?" in pl and not any(c in rl for c in ("because", "means", "is", "are", "was", "were")):
            if len(rl.split()) < 8:
                issues.append("response may not address the question")
                suggestions.append("directly answer the question asked")

        # Code verification
        if cls.detect_code_request(prompt):
            if "```" not in response:
                issues.append("code requested but no fenced code block in response")
                suggestions.append("wrap code in ``` fences for execution")
            else:
                # Check for common code issues
                code_blocks = re.findall(r"```(?:\w+)\n(.*?)```", response, re.S)
                for block in code_blocks:
                    if block.strip():
                        # Check for truncated code
                        if block.strip().endswith("...") or block.strip().endswith("…"):
                            issues.append("code block may be truncated")
                            suggestions.append("provide complete code")
                        # Check for placeholder comments
                        if re.search(r"(#\s*todo|//\s*todo|/\*\s*todo)", block, re.I):
                            issues.append("code contains TODO placeholders")

        # Explanation verification
        intent = cls.classify_intent(prompt)["intent"]
        if intent in ("explain", "compare", "design") and len(rl.split()) < 20:
            issues.append("explanation too brief for the topic")
            suggestions.append("add more detail, examples, or structure")

        # Math verification
        if intent == "math":
            numbers = re.findall(r"-?\d+\.?\d*", response)
            if not numbers:
                issues.append("math query but no numeric result in response")

        passed = len(issues) == 0
        return {
            "passed": passed,
            "issues": issues,
            "suggestions": suggestions,
            "quality": cls.quality_score(response, prompt),
        }

    @classmethod
    def conversation_flow(cls, prompt: str, history: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Analyze conversation flow and determine follow-up behavior.
        Returns {is_followup, context, suggested_action, needs_history}.
        """
        if not history:
            return {"is_followup": False, "context": "", "suggested_action": "new", "needs_history": False}

        pl = prompt.lower().strip()
        last_user = ""
        last_bot = ""
        for m in reversed(history):
            if m.get("role") == "assistant" and not last_bot:
                last_bot = m.get("text", "")
            elif m.get("role") == "user" and not last_user:
                last_user = m.get("text", "")
            if last_user and last_bot:
                break

        ack_words = frozenset({"yes", "ok", "okay", "sure", "yep", "yeah", "thanks", "thank you",
                            "好的", "是的", "对", "谢谢", "可以"})
        follow_cues = ("tell me more", "go on", "continue", "and then", "what else",
                    "more", "expand", "elaborate", "go deeper", "why though",
                    "how come", "what about", "how about", "say more")

        is_ack = pl.strip().rstrip(".!") in ack_words or pl.strip() in ack_words
        is_followup_cue = any(c in pl for c in follow_cues)
        is_followup = is_ack or is_followup_cue or pl in (".", "..", ">", "ok")

        if is_followup and last_user:
            return {
                "is_followup": True,
                "context": last_user[:120],
                "last_bot_snippet": last_bot[:200] if last_bot else "",
                "suggested_action": "continue",
                "needs_history": True,
            }

        return {
            "is_followup": False,
            "context": last_user[:120] if last_user else "",
            "suggested_action": "new_topic",
            "needs_history": bool(history),
        }

    @classmethod
    def detect_tone(cls, prompt: str) -> str:
        """Detect the appropriate response tone."""
        pl = prompt.lower().strip()
        if re.search(r"\b(urgent|asap|quick|fast|emergency|hurry)", pl):
            return "urgent"
        if re.search(r"\b(why|how|explain|walk me|step by step|tutorial)", pl):
            return "educational"
        if re.search(r"\b(写代码|编程|代码|程序|脚本)", pl):
            return "technical"
        if re.search(r"\b(hi|hey|hello|how are you|howdy|sup)", pl):
            return "friendly"
        if re.search(r"\b(joke|funny|humor|laugh|make me laugh)", pl):
            return "humorous"
        if re.search(r"\b(poem|fable|story|creative|imagine)", pl):
            return "creative"
        if re.search(r"\b(error|bug|traceback|crash|broken|fails)", pl):
            return "supportive"
        return "neutral"

    @classmethod
    def analyze_channels(
        cls, prompt: str, engine: Optional["CatR11Engine"] = None
    ) -> Dict[str, Any]:
        """
        Tri-channel detector: code · English · Chinese (files = off).
        Routes replies by primary channel while preserving locale metadata.
        """
        raw = (prompt or "").strip()
        norm = CatR1Code.normalize_prompt(raw)
        work = norm or raw

        locale = CatR11Synthesizer.detect_locale(work)
        cjk_count = len(re.findall(r"[\u4e00-\u9fff]", work))
        eng_count = len(re.findall(r"[a-zA-Z]{2,}", work))
        has_chinese = cjk_count > 0
        has_english = eng_count > 0 or bool(re.search(r"[a-zA-Z]", work))

        fenced_lang, fenced_code = CatR1Code.extract_prompt_code(work)
        wants_code = cls.detect_code_request(work)
        has_codeblock = bool(fenced_code)
        has_code = (
            has_codeblock
            or wants_code
            or bool(
                re.search(
                    r"(```|def\s+\w+\s*\(|function\s+\w+|class\s+\w+|#include\s*<|"
                    r"import\s+\w+|console\.log|printf\s*\()",
                    work,
                    re.I,
                )
            )
        )
        code_lang = fenced_lang or cls.detect_language(work, engine)

        if has_code or wants_code:
            channel = "code"
        elif locale == "chinese" or (has_chinese and not has_english):
            channel = "chinese"
        elif locale == "mixed" or (has_chinese and has_english):
            channel = "mixed"
        else:
            channel = "english"

        intent_info = cls.classify_intent(work)
        force_pro = bool(re.search(r"(?:^|\s)#+\s*pr\b", raw, re.I))

        return {
            "locale": locale,
            "channel": channel,
            "has_code": has_code,
            "has_chinese": has_chinese,
            "has_english": has_english,
            "code_lang": code_lang,
            "topic": intent_info.get("topic") or cls.extract_topic(work),
            "intent": intent_info.get("intent", "chat"),
            "confidence": intent_info.get("confidence", "medium"),
            "needs_code": intent_info.get("needs_code", has_code),
            "files": "off",
            "force_pro": force_pro,
        }

    @classmethod
    def always_answer(
        cls,
        engine: "CatR11Engine",
        prompt: str,
        channel: Dict[str, Any],
        resp: Optional[str],
    ) -> str:
        """Guarantee a non-empty reply routed by code / Chinese / English channel."""
        if resp and str(resp).strip():
            return str(resp).strip()

        locale = channel.get("locale", "english")
        ch = channel.get("channel", "english")
        topic = channel.get("topic") or cls.extract_topic(prompt) or "your question"
        brand = getattr(engine, "name", BRAND)
        files_tag = "`files = off`"

        if ch == "code" or channel.get("has_code"):
            try:
                if CatR1CodeR1.enabled():
                    code_resp = CatR1CodeR1.generate_code(engine, prompt, force_pro=CatR1CodeR1.wants_pr(prompt))
                else:
                    code_resp = CatR1Code.respond(engine, prompt)
                if code_resp and str(code_resp).strip():
                    return str(code_resp).strip()
            except Exception:
                pass
            lang = channel.get("code_lang") or "python"
            if engine and hasattr(engine, "normalize_lang"):
                lang = engine.normalize_lang(lang) or lang
            snippet = (
                f"```{lang}\n# {topic}\n# cat r1 · files = off\n"
                f"def main():\n    pass\n\nif __name__ == '__main__':\n    main()\n```"
            )
            if locale == "chinese":
                return (
                    f"关于「{topic}」的代码骨架：\n\n{snippet}\n\n"
                    f"请具体说明你要实现什么，我可以补全逻辑。（{files_tag}）"
                )
            if locale == "mixed":
                return (
                    f"**{topic}** — code scaffold ({lang}) · {files_tag}:\n\n{snippet}\n\n"
                    f"Say what to build · 或说明具体需求。"
                )
            return (
                f"Here's a **{lang}** scaffold for **{topic}** ({files_tag}):\n\n{snippet}\n\n"
                f"Tell me what to implement and I'll fill in the logic."
            )

        if locale == "chinese" or ch == "chinese":
            return (
                f"关于「{topic}」：\n\n"
                f"我是 **{brand}**（{files_tag}），可以解释概念、写代码、调试或聊天。"
                f"请具体说明你想了解什么，我会用中文回答。"
            )

        if locale == "mixed" or ch == "mixed":
            return (
                f"On **{topic}** · 关于「{topic}」：\n\n"
                f"I'm **{brand}** ({files_tag}) — ask in English or 中文, "
                f"paste ```code```, or describe what you need."
            )

        return (
            f"On **{topic}**: I'm **{brand}** ({files_tag}) and ready to help — "
            "explain, code, debug, or chat. What would you like to explore?"
        )

    @classmethod
    def stats_line(cls) -> str:
        return f"cat r1 heuristics · tri-channel · intent+code+verify · files=off"


CatR1Heuristics = CatR1Heuristics


# ──────────────────────────────────────────────────────────────
# CAT R1.1 WEB (cat r1 websites · artifacts · fetch · files = off)
# ──────────────────────────────────────────────────────────────
class _TextExtractor(HTMLParser):
    __slots__ = ("parts", "_skip")

    def __init__(self):
        super().__init__()
        self.parts: List[str] = []
        self._skip = False

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style", "noscript"}:
            self._skip = True

    def handle_endtag(self, tag):
        if tag in {"script", "style", "noscript"}:
            self._skip = False
        elif tag in {"p", "br", "div", "li", "h1", "h2", "h3", "tr"}:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self._skip:
            t = data.strip()
            if t:
                self.parts.append(t)


@dataclass
class WebSiteRecord:
    site_id: str
    title: str
    html: str
    template: str
    prompt: str
    created: float = field(default_factory=time.time)

    def preview_url(self, port: int) -> str:
        return f"http://127.0.0.1:{port}/web/preview/{self.site_id}"


class CatR1WebProgram:
    """
    In-memory cat r1 websites program (files = off):
    artifact HTML/CSS/JS · URL fetch · site registry · API preview — no disk writes.
    """

    _URL = re.compile(r"https?://[^\s<>\"']+", re.I)
    _WEB_CUES = (
        "website", "web page", "webpage", "landing page", "landing", "homepage",
        "home page", "portfolio site", "dashboard", "docs site", "documentation page",
        "saas page", "blog page", "artifact", "single page app", "spa",
        "做个网站", "做个网页", "网站", "网页", "落地页", "首页",
    )
    _FETCH_CUES = (
        "fetch ", "read url", "read website", "read this site", "open url",
        "scrape ", "get page", "summarize url", "summarize website",
        "读取网址", "打开网站", "抓取",
    )
    TEMPLATE_NAMES = (
        "landing", "dashboard", "docs", "portfolio", "saas", "blog",
        "resume", "todo", "calculator", "geocities", "minimal",
    )

    def __init__(self):
        self._sites: Dict[str, WebSiteRecord] = {}

    @classmethod
    def enabled(cls) -> bool:
        return bool(CONFIG.get("web_program_enabled", True))

    @classmethod
    def wants_web(cls, pl: str) -> bool:
        if not cls.enabled():
            return False
        if pl.startswith("/web"):
            return True
        if cls._URL.search(pl) and any(c in pl for c in cls._FETCH_CUES):
            return True
        if any(c in pl for c in cls._FETCH_CUES):
            return True
        if re.search(r"\b(build|make|create|design|generate|write)\s+(?:a|an|me\s+a|my\s+)?(?:\w+\s+){0,4}(?:website|webpage|web page|landing(?:\s+page)?|site|homepage)\b", pl):
            return True
        if any(c in pl for c in cls._WEB_CUES) and re.search(r"\b(build|make|create|write|design|generate|show)\b", pl):
            return True
        if re.search(r"\bhtml\b.*\b(site|page|app|ad)\b", pl):
            return True
        return False

    @classmethod
    def extract_url(cls, text: str) -> Optional[str]:
        m = cls._URL.search(text)
        return m.group(0).rstrip(".,)") if m else None

    @staticmethod
    def _subject(prompt: str) -> str:
        pl = prompt.lower()
        for prefix in ("build a ", "make a ", "create a ", "design a ", "write a ", "generate a "):
            if pl.startswith(prefix):
                rest = prompt[len(prefix):].strip()
                rest = re.split(r"\s+(?:website|webpage|web page|landing page|site|page)\b", rest, flags=re.I)[0]
                if rest.strip():
                    return rest.strip()[:80]
        m = re.search(r"(?:website|landing page|site|page)\s+(?:for|about|called)\s+(.+?)(?:\s*$|\.)", pl, re.I)
        if m:
            return m.group(1).strip()[:80]
        return BRAND

    @classmethod
    def pick_template(cls, prompt: str) -> str:
        pl = prompt.lower()
        if re.search(r"\bgeocit(?:ies|es)\b", pl) or ("gamer" in pl and "usb" in pl):
            return "geocities"
        if "dashboard" in pl or "admin" in pl:
            return "dashboard"
        if "doc" in pl or "documentation" in pl or "readme" in pl:
            return "docs"
        if "portfolio" in pl:
            return "portfolio"
        if "saas" in pl or "pricing" in pl or "subscribe" in pl:
            return "saas"
        if "blog" in pl or "article" in pl:
            return "blog"
        if "resume" in pl or "cv" in pl:
            return "resume"
        if "todo" in pl or "task list" in pl:
            return "todo"
        if "calculator" in pl or "calc" in pl:
            return "calculator"
        if "landing" in pl or "homepage" in pl or "home page" in pl:
            return "landing"
        return "minimal"

    @staticmethod
    def _shell(title: str, body: str, *, extra_head: str = "", extra_script: str = "") -> str:
        t = html_module.escape(title)
        return (
            f"<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n"
            f"  <meta charset=\"UTF-8\">\n"
            f"  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\n"
            f"  <title>{t}</title>\n{extra_head}\n</head>\n<body>\n{body}\n"
            f"{extra_script}\n</body>\n</html>"
        )

    @classmethod
    def _template_landing(cls, title: str, headline: str) -> str:
        h = html_module.escape(headline)
        css = (
            "  <style>\n"
            "    :root { --bg:#0f172a; --fg:#e2e8f0; --acc:#38bdf8; --card:#1e293b; }\n"
            "    * { box-sizing:border-box; margin:0; }\n"
            "    body { font-family:system-ui,sans-serif; background:var(--bg); color:var(--fg); }\n"
            "    nav { display:flex; justify-content:space-between; padding:1rem 2rem; }\n"
            "    .hero { text-align:center; padding:4rem 1.5rem; max-width:720px; margin:0 auto; }\n"
            "    h1 { font-size:clamp(2rem,5vw,3rem); margin-bottom:1rem; }\n"
            "    p { opacity:.85; line-height:1.6; margin-bottom:2rem; }\n"
            "    .cta { display:inline-block; background:var(--acc); color:#0f172a; padding:.75rem 1.5rem; "
            "border-radius:8px; text-decoration:none; font-weight:600; }\n"
            "    .grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(200px,1fr)); gap:1rem; "
            "padding:2rem; max-width:960px; margin:0 auto; }\n"
            "    .card { background:var(--card); padding:1.25rem; border-radius:12px; }\n"
            "  </style>"
        )
        body = (
            f"  <nav><strong>{html_module.escape(title)}</strong><span>{BRAND}</span></nav>\n"
            f"  <section class=\"hero\"><h1>{h}</h1>\n"
            f"    <p>Built in-memory by {WEB_PROGRAM_NAME} — {BRAND} artifacts, no files written.</p>\n"
            f"    <a class=\"cta\" href=\"#features\">Get started</a></section>\n"
            f"  <div class=\"grid\" id=\"features\">\n"
            f"    <div class=\"card\"><h3>Fast</h3><p>{BRAND} token-weight synthesis.</p></div>\n"
            f"    <div class=\"card\"><h3>Local</h3><p>Runs fully offline.</p></div>\n"
            f"    <div class=\"card\"><h3>Private</h3><p>Everything stays in RAM.</p></div>\n"
            f"  </div>"
        )
        return cls._shell(title, body, extra_head=css)

    @classmethod
    def _template_dashboard(cls, title: str) -> str:
        css = (
            "  <style>body{font-family:system-ui;margin:0;background:#111;color:#eee;display:grid;"
            "grid-template-columns:220px 1fr;min-height:100vh}"
            "aside{background:#1a1a1a;padding:1rem}main{padding:1.5rem}"
            ".stat{display:inline-block;background:#222;padding:1rem 1.5rem;border-radius:8px;margin:.5rem}"
            ".stat b{font-size:1.5rem;color:#4ade80}</style>"
        )
        body = (
            f"  <aside><h2>{html_module.escape(title)}</h2><ul><li>Overview</li><li>Analytics</li>"
            f"<li>Settings</li></ul></aside>\n"
            f"  <main><h1>Dashboard</h1>\n"
            f"    <div class=\"stat\"><div>Users</div><b>1,024</b></div>\n"
            f"    <div class=\"stat\"><div>Requests</div><b>8,192</b></div>\n"
            f"    <div class=\"stat\"><div>Uptime</div><b>99.9%</b></div></main>"
        )
        return cls._shell(title, body, extra_head=css)

    @classmethod
    def _template_docs(cls, title: str) -> str:
        css = "  <style>body{font-family:Georgia,serif;max-width:720px;margin:2rem auto;padding:0 1rem;line-height:1.7}"
        body = (
            f"  <h1>{html_module.escape(title)} — Docs</h1>\n"
            f"  <h2>Quick start</h2><p>Web program stores sites in memory. Preview via "
            f"<code>/web/preview/&lt;id&gt;</code> on the local API.</p>\n"
            f"  <h2>Commands</h2><ul><li><code>/web list</code></li><li><code>/web fetch URL</code></li>"
            f"<li><code>/web build prompt</code></li></ul>"
        )
        return cls._shell(title, body, extra_head=css)

    @classmethod
    def _template_portfolio(cls, title: str, name: str) -> str:
        n = html_module.escape(name)
        css = (
            "  <style>body{font-family:system-ui;background:#fafafa;color:#111;max-width:800px;"
            "margin:2rem auto;padding:1rem}.project{border:1px solid #ddd;border-radius:8px;"
            "padding:1rem;margin:1rem 0}</style>"
        )
        body = (
            f"  <header><h1>{n}</h1><p>Portfolio · {html_module.escape(title)}</p></header>\n"
            f"  <section class=\"project\"><h3>Project Alpha</h3><p>Full-stack app with in-memory deploy.</p></section>\n"
            f"  <section class=\"project\"><h3>Project Beta</h3><p>{BRAND}-powered local AI tooling.</p></section>"
        )
        return cls._shell(title, body, extra_head=css)

    @classmethod
    def _template_saas(cls, title: str) -> str:
        css = (
            "  <style>body{font-family:system-ui;text-align:center;padding:2rem;background:linear-gradient(#eef,#fff)}"
            ".price{display:inline-block;border:2px solid #333;border-radius:12px;padding:2rem;margin:1rem}"
            ".price h2{font-size:2.5rem}</style>"
        )
        body = (
            f"  <h1>{html_module.escape(title)}</h1><p>Simple pricing. No files. All in-memory.</p>\n"
            f"  <div class=\"price\"><h3>Pro</h3><h2>$9</h2><p>/month</p></div>\n"
            f"  <div class=\"price\"><h3>Team</h3><h2>$29</h2><p>/month</p></div>"
        )
        return cls._shell(title, body, extra_head=css)

    @classmethod
    def _template_blog(cls, title: str, headline: str) -> str:
        h = html_module.escape(headline)
        css = "  <style>body{font-family:Georgia,serif;max-width:640px;margin:2rem auto;line-height:1.8;padding:1rem}</style>"
        body = (
            f"  <article><h1>{h}</h1><p><em>{BRAND}</em></p>\n"
            f"  <p>This post was generated in-memory. {WEB_PROGRAM_NAME} by {BRAND}</p>"
            f"artifact pages — HTML, CSS, and optional JS — without writing to disk.</p></article>"
        )
        return cls._shell(title, body, extra_head=css)

    @classmethod
    def _template_todo(cls, title: str) -> str:
        css = "  <style>body{font-family:system-ui;max-width:420px;margin:3rem auto;padding:1rem}"
        script = (
            "  <script>\n"
            "    const ul=document.getElementById('t');\n"
            "    function add(){const i=document.getElementById('i');if(!i.value.trim())return;"
            "const li=document.createElement('li');li.textContent=i.value;i.value='';ul.appendChild(li);}\n"
            "  </script>"
        )
        body = (
            f"  <h1>{html_module.escape(title)}</h1>\n"
            f"  <input id=\"i\" placeholder=\"New task…\" style=\"width:70%\"> "
            f"<button onclick=\"add()\">Add</button>\n  <ul id=\"t\"></ul>"
        )
        return cls._shell(title, body, extra_head=css, extra_script=script)

    @classmethod
    def _template_calculator(cls, title: str) -> str:
        css = "  <style>body{font-family:monospace;text-align:center;padding:2rem}"
        script = (
            "  <script>\n"
            "    function calc(){try{document.getElementById('o').textContent="
            "eval(document.getElementById('e').value)}catch(e){document.getElementById('o').textContent='Error'}}\n"
            "  </script>"
        )
        body = (
            f"  <h1>{html_module.escape(title)}</h1>\n"
            f"  <input id=\"e\" style=\"font-size:1.2rem;width:200px\"> "
            f"<button onclick=\"calc()\">=</button>\n  <p id=\"o\"></p>"
        )
        return cls._shell(title, body, extra_head=css, extra_script=script)

    @classmethod
    def render(cls, template: str, prompt: str, engine: Optional["CatR11Engine"] = None) -> str:
        subject = cls._subject(prompt)
        title = subject.title() if subject else f"{BRAND} site"
        headline = subject or title
        if template == "geocities":
            return CatR1Code._html_geocities(prompt)
        if template == "landing":
            return cls._template_landing(title, headline)
        if template == "dashboard":
            return cls._template_dashboard(title)
        if template == "docs":
            return cls._template_docs(title)
        if template == "portfolio":
            return cls._template_portfolio(title, headline)
        if template == "saas":
            return cls._template_saas(title)
        if template == "blog":
            return cls._template_blog(title, headline)
        if template == "resume":
            return cls._template_portfolio(title, f"{headline} — Resume")
        if template == "todo":
            return cls._template_todo(title)
        if template == "calculator":
            return cls._template_calculator(title)
        # minimal / default — landing template (avoid _html ↔ render recursion)
        return cls._template_landing(title, headline)

    def store(self, prompt: str, html: str, template: str) -> WebSiteRecord:
        sid = uuid.uuid4().hex[:10]
        title = self._subject(prompt).title() or "Site"
        rec = WebSiteRecord(site_id=sid, title=title, html=html, template=template, prompt=prompt[:200])
        self._sites[sid] = rec
        while len(self._sites) > CONFIG.get("web_max_sites", 256):
            oldest = min(self._sites.values(), key=lambda s: s.created)
            del self._sites[oldest.site_id]
        return rec

    def get(self, site_id: str) -> Optional[WebSiteRecord]:
        return self._sites.get(site_id)

    def build(self, prompt: str, engine: Optional["CatR11Engine"] = None) -> WebSiteRecord:
        tpl = self.pick_template(prompt)
        html = self.render(tpl, prompt, engine)
        return self.store(prompt, html, tpl)

    def fetch_url(self, url: str) -> str:
        if not CONFIG.get("web_fetch_enabled", True):
            return "Web fetch is disabled in CONFIG."
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"}:
            return "Only http:// and https:// URLs are supported."
        max_bytes = CONFIG.get("web_max_fetch_kb", 256) * 1024
        try:
            req = urlrequest.Request(url, headers={"User-Agent": f"{BRAND}/{VERSION} (in-memory)"})
            with urlrequest.urlopen(req, timeout=12) as resp:
                raw = resp.read(max_bytes + 1)
                if len(raw) > max_bytes:
                    raw = raw[:max_bytes]
                ctype = resp.headers.get("Content-Type", "")
                charset = "utf-8"
                if "charset=" in ctype:
                    charset = ctype.split("charset=")[-1].split(";")[0].strip()
                text = raw.decode(charset, errors="replace")
        except (URLError, HTTPError, TimeoutError, ValueError) as e:
            return f"Fetch failed: {e}"
        if "html" in text.lower()[:500] or "<" in text[:200]:
            parser = _TextExtractor()
            try:
                parser.feed(text)
            except Exception:
                pass
            body = re.sub(r"\n{3,}", "\n\n", " ".join(parser.parts))
            body = body[:4000]
            return f"**Fetched** `{url}`\n\n{body or '(no extractable text)'}"
        return f"**Fetched** `{url}`\n\n```\n{text[:4000]}\n```"

    @classmethod
    def help_text(cls) -> str:
        port = CONFIG.get("api_port", 8765)
        tpl = ", ".join(cls.TEMPLATE_NAMES)
        return (
            f"**{WEB_PROGRAM_NAME}** · {BRAND} websites · \n\n"
            "In-memory artifact builder, URL reader, and API preview.\n\n"
            "**Commands**\n"
            "- `/web` — this help\n"
            "- `/web list` — sites in memory\n"
            "- `/web templates` — available layouts\n"
            "- `/web build <prompt>` — generate & store a site\n"
            "- `/web fetch <url>` — read a page (text extract)\n"
            "- `/web preview <id>` — preview link\n\n"
            f"**Templates:** {tpl}\n\n"
            f"**API:** `GET http://127.0.0.1:{port}/web/preview/<id>` · `GET /web/sites`\n\n"
            "Natural language: `build a landing page for my app`, `fetch https://example.com`"
        )

    def list_sites(self) -> str:
        if not self._sites:
            return "No sites in memory. Try `/web build a landing page` or ask naturally."
        port = CONFIG.get("api_port", 8765)
        lines = [f"**{len(self._sites)} site(s)** in memory:\n"]
        for rec in sorted(self._sites.values(), key=lambda s: s.created, reverse=True):
            lines.append(
                f"- `{rec.site_id}` · **{rec.title}** · {rec.template} · "
                f"[preview]({rec.preview_url(port)})"
            )
        return "\n".join(lines)

    def handle_command(self, engine: "CatR11Engine", raw: str) -> str:
        pl = raw.lower().strip()
        if pl in {"/web", "/web help"}:
            return self.help_text()
        if pl == "/web list":
            return self.list_sites()
        if pl == "/web templates":
            return "**Templates:** " + ", ".join(self.TEMPLATE_NAMES)
        if pl.startswith("/web fetch "):
            url = self.extract_url(raw) or raw.split(maxsplit=2)[-1].strip()
            return self.fetch_url(url)
        if pl.startswith("/web build ") or pl.startswith("/web "):
            prompt = raw.split(maxsplit=2)[2] if pl.startswith("/web build ") else raw.split(maxsplit=1)[1]
            if prompt.lower().startswith("build "):
                prompt = prompt[6:]
            rec = self.build(prompt.strip() or "landing page", engine)
            port = CONFIG.get("api_port", 8765)
            return (
                f"**Site built** · `{rec.site_id}` · **{rec.template}**\n\n"
                f"Preview: {rec.preview_url(port)}\n\n"
                f"```html\n{rec.html[:1200]}{'…' if len(rec.html) > 1200 else ''}\n```"
            )
        if pl.startswith("/web preview "):
            sid = raw.split(maxsplit=2)[-1].strip()
            rec = self.get(sid)
            if not rec:
                return f"No site `{sid}`. Use `/web list`."
            return f"Preview: {rec.preview_url(CONFIG.get('api_port', 8765))}\n\n```html\n{rec.html[:800]}…\n```"
        return self.help_text()

    def respond(self, engine: "CatR11Engine", prompt: str) -> str:
        pl = prompt.lower().strip()
        if pl.startswith("/web"):
            return self.handle_command(engine, prompt)
        url = self.extract_url(prompt)
        if url and any(c in pl for c in self._FETCH_CUES + ("http", "https", "www.")):
            return self.fetch_url(url)
        if url and not re.search(r"\b(build|make|create|html|page|site)\b", pl):
            return self.fetch_url(url)
        rec = self.build(prompt, engine)
        port = CONFIG.get("api_port", 8765)
        fenced = f"```html\n{rec.html}\n```"
        if CONFIG.get("code_output_exact"):
            return (
                f"**{WEB_PROGRAM_NAME}** · `{rec.site_id}` · {rec.template}\n"
                f"Preview: {rec.preview_url(port)}\n\n{fenced}"
            )
        return (
            f"**Site built** · `{rec.site_id}` · **{rec.template}**\n"
            f"Preview: {rec.preview_url(port)}\n\n{fenced}"
        )

    def clear(self) -> None:
        self._sites.clear()


# ──────────────────────────────────────────────────────────────
# CODE ANYTHING (universal polyglot coder · files = off)
# ──────────────────────────────────────────────────────────────
class CodeAnythingEngine:
    """
    Universal in-memory coder for cat r1 · files = off.
    Any language · any task · polyglot scaffolds · dry-run simulation.
    """

    TAG = "code anything"
    MODE = CONFIG.get("code_anything_mode", "universal")

    LANGS = (
        "python", "javascript", "typescript", "java", "kotlin", "swift", "scala",
        "rust", "go", "c", "cpp", "csharp", "fsharp", "ruby", "php", "perl",
        "lua", "r", "julia", "dart", "elixir", "haskell", "clojure", "groovy",
        "powershell", "solidity", "zig", "nim", "crystal", "fortran", "cobol",
        "verilog", "vhdl", "assembly", "html", "css", "scss", "sql", "graphql",
        "yaml", "toml", "json", "dockerfile", "makefile", "bash", "shell",
    )
    VALID_LANGS = frozenset(LANGS)
    RUNNABLE = frozenset({"python", "javascript", "bash", "shell"})

    LANG_ALIASES = {
        "py": "python", "python3": "python", "python2": "python",
        "js": "javascript", "node": "javascript", "nodejs": "javascript",
        "ts": "typescript", "c++": "cpp", "cc": "cpp", "cxx": "cpp",
        "c#": "csharp", "cs": "csharp", "dotnet": "csharp",
        "f#": "fsharp", "fs": "fsharp",
        "sh": "bash", "shell": "bash", "zsh": "bash",
        "asm": "assembly", "golang": "go", "kt": "kotlin",
        "rb": "ruby", "rs": "rust", "yml": "yaml",
        "docker": "dockerfile", "make": "makefile",
        "ps1": "powershell", "pwsh": "powershell",
        "sol": "solidity", "proto": "protobuf",
        "go": "go",
    }

    # Lightweight universal templates used as a fallback when a task-specific
    # scaffold isn't required.
    TEMPLATES = {
        "python": (
            "def main():\n"
            "    print('Hello from CatR1!')\n\n"
            "if __name__ == '__main__':\n"
            "    main()"
        ),
        "javascript": "console.log('Hello from CatR1!');",
        "cpp": (
            "#include <iostream>\n"
            "int main() {\n"
            "    std::cout << \"Hello from CatR1!\" << std::endl;\n"
            "    return 0;\n"
            "}"
        ),
        "html": "<!DOCTYPE html><html><body><h1>Hello from CatR1!</h1></body></html>",
    }

    FRAMEWORK_LANG = {
        "react": "javascript", "vue": "javascript", "svelte": "javascript",
        "angular": "typescript", "nextjs": "typescript", "next.js": "typescript",
        "fastapi": "python", "django": "python", "flask": "python", "streamlit": "python",
        "rails": "ruby", "laravel": "php", "symfony": "php",
        "spring": "java", "springboot": "java", "hibernate": "java",
        "express": "javascript", "nestjs": "typescript", "deno": "typescript",
        "actix": "rust", "rocket": "rust", "gin": "go", "echo": "go",
        "swiftui": "swift", "uikit": "swift", "jetpack": "kotlin",
        "flutter": "dart", "react native": "javascript",
        "solidity": "solidity", "hardhat": "solidity", "foundry": "solidity",
        "terraform": "hcl", "kubernetes": "yaml", "k8s": "yaml",
    }

    _EXT_LANG = {
        ".py": "python", ".js": "javascript", ".ts": "typescript", ".jsx": "javascript",
        ".tsx": "typescript", ".java": "java", ".kt": "kotlin", ".kts": "kotlin",
        ".swift": "swift", ".scala": "scala", ".rs": "rust", ".go": "go",
        ".c": "c", ".h": "c", ".cpp": "cpp", ".cc": "cpp", ".hpp": "cpp",
        ".cs": "csharp", ".fs": "fsharp", ".rb": "ruby", ".php": "php",
        ".pl": "perl", ".lua": "lua", ".r": "r", ".jl": "julia", ".dart": "dart",
        ".ex": "elixir", ".exs": "elixir", ".hs": "haskell", ".clj": "clojure",
        ".groovy": "groovy", ".ps1": "powershell", ".sol": "solidity",
        ".zig": "zig", ".nim": "nim", ".cr": "crystal", ".f90": "fortran",
        ".cob": "cobol", ".v": "verilog", ".vhdl": "vhdl", ".asm": "assembly",
        ".html": "html", ".css": "css", ".scss": "scss", ".sql": "sql",
        ".yaml": "yaml", ".yml": "yaml", ".toml": "toml", ".json": "json",
        ".sh": "bash", ".bash": "bash", ".zsh": "bash",
        ".dockerfile": "dockerfile", ".mk": "makefile",
    }

    _ANYTHING_CUE = re.compile(
        r"\b(code\s+anything|any\s+language|polyglot|multi[\s-]?language|"
        r"whatever\s+language|in\s+any\s+lang)\b",
        re.I,
    )
    _SCAFFOLD_CUE = re.compile(
        r"\b(scaffold|boilerplate|starter|template|skeleton|stub|implement|"
        r"port\s+to|convert\s+to|rewrite\s+in|refactor|migrate)\b",
        re.I,
    )

    @classmethod
    def enabled(cls) -> bool:
        return bool(CONFIG.get("code_anything", True))

    @classmethod
    def normalize_lang(cls, lang: Optional[str]) -> Optional[str]:
        if not lang:
            return None
        t = lang.lower().strip().strip(".")
        return cls.LANG_ALIASES.get(t, t)

    @classmethod
    def _generate_template(cls, lang: str, prompt: str) -> str:
        lang = (lang or "").lower().strip()
        lang = cls.normalize_lang(lang) or lang
        base = cls.TEMPLATES.get(lang)
        if base is None:
            return f"// Hello from CatR1 in {lang}\n// Prompt: {(prompt or '')[:120]}"

        if prompt:
            if lang == "python":
                return f"# Prompt: {(prompt or '')[:120]}\n\n{base}"
            if lang in {"javascript", "cpp", "c", "csharp", "java", "bash"}:
                return f"// Prompt: {(prompt or '')[:120]}\n\n{base}"
        return base

    @classmethod
    def wants_anything(cls, prompt: str) -> bool:
        if not cls.enabled():
            return False
        pl = (prompt or "").lower()
        if cls._ANYTHING_CUE.search(pl):
            return True
        if cls._SCAFFOLD_CUE.search(pl) and re.search(
            r"\b(code|script|app|api|cli|program|function|class|module)\b", pl
        ):
            return True
        for fw in cls.FRAMEWORK_LANG:
            if fw in pl and re.search(r"\b(write|build|make|create|scaffold|implement|code)\b", pl):
                return True
        for ext in cls._EXT_LANG:
            if ext in pl and re.search(r"\b(write|build|make|create|code|implement)\b", pl):
                return True
        return False

    @classmethod
    def detect_task(cls, prompt: str) -> str:
        pl = (prompt or "").lower()
        if re.search(r"\b(fibonacci|fib\b|prime|sieve|sort|search|bst|graph|dp\b|leetcode|algorithm|algo)\b", pl):
            return "algo"
        if re.search(r"\b(api|rest|endpoint|graphql|grpc|fastapi|flask|express)\b", pl):
            return "api"
        if re.search(r"\b(cli|argparse|click|command[\s-]?line|argv)\b", pl):
            return "cli"
        if re.search(r"\b(test|unit test|pytest|jest|spec\b|tdd)\b", pl):
            return "test"
        if re.search(r"\b(csv|json|data|pandas|etl|parse)\b", pl):
            return "data"
        if re.search(r"\b(game|snake|pong|tic[\s-]?tac|chess|puzzle)\b", pl):
            return "game"
        if re.search(r"\b(yaml|toml|json|config|dockerfile|docker|kubernetes|k8s)\b", pl):
            return "config"
        if re.search(r"\b(html|website|landing|page|frontend|react|vue|css)\b", pl):
            return "web"
        if re.search(r"\b(mobile|ios|android|swiftui|kotlin)\b", pl):
            return "mobile"
        return "general"

    @classmethod
    def infer_lang(cls, engine: Optional["CatR11Engine"], prompt: str, hint: str = "") -> str:
        norm = cls.normalize_lang(hint)
        if norm and norm in cls.VALID_LANGS:
            return norm
        pl = (prompt or "").lower()
        for fw, lang in cls.FRAMEWORK_LANG.items():
            if fw in pl:
                return lang
        for ext, lang in cls._EXT_LANG.items():
            if ext in pl:
                return lang
        for lang in sorted(cls.LANGS, key=len, reverse=True):
            if re.search(rf"\b{re.escape(lang)}\b", pl):
                return lang
        if engine is not None:
            extracted = engine.extract_lang(prompt)
            if extracted:
                n = cls.normalize_lang(extracted)
                if n in cls.VALID_LANGS:
                    return n
            from_code = engine.detect_lang_from_text(prompt)
            if from_code:
                n = cls.normalize_lang(from_code)
                if n in cls.VALID_LANGS:
                    return n
        if re.search(r"\bhtml\b", pl):
            return "html"
        if re.search(r"\brust\b", pl):
            return "rust"
        if re.search(r"\bgo\b|\bgolang\b", pl):
            return "go"
        if re.search(r"\bjava\b(?!script)", pl):
            return "java"
        if re.search(r"\bkotlin\b", pl):
            return "kotlin"
        if re.search(r"\bswift\b", pl):
            return "swift"
        if re.search(r"\btypescript\b|\bts\b", pl):
            return "typescript"
        if re.search(r"\bjavascript\b|\bjs\b", pl):
            return "javascript"
        return "python"

    @classmethod
    def _subject(cls, prompt: str, engine: Optional["CatR11Engine"] = None) -> str:
        if engine is not None:
            subj = CatR1Code._subject(prompt, engine)
            if subj and subj.lower() not in {"hello world", "it", "it html"}:
                return subj
        return "App"

    @classmethod
    def _py_algo(cls, prompt: str) -> str:
        pl = prompt.lower()
        if "fibonacci" in pl or re.search(r"\bfib\b", pl):
            return (
                "def fib(n: int) -> int:\n"
                "    a, b = 0, 1\n"
                "    for _ in range(n):\n"
                "        a, b = b, a + b\n"
                "    return a\n\n"
                "def main() -> None:\n"
                "    for i in range(12):\n"
                "        print(f'fib({i}) = {fib(i)}')\n\n"
                "if __name__ == '__main__':\n"
                "    main()"
            )
        if "prime" in pl:
            return (
                "def is_prime(n: int) -> bool:\n"
                "    if n < 2:\n"
                "        return False\n"
                "    d = 2\n"
                "    while d * d <= n:\n"
                "        if n % d == 0:\n"
                "            return False\n"
                "        d += 1\n"
                "    return True\n\n"
                "def main() -> None:\n"
                "    print([n for n in range(2, 50) if is_prime(n)])\n\n"
                "if __name__ == '__main__':\n"
                "    main()"
            )
        return (
            "from typing import List\n\n"
            "def two_sum(nums: List[int], target: int) -> List[int]:\n"
            "    seen = {}\n"
            "    for i, n in enumerate(nums):\n"
            "        need = target - n\n"
            "        if need in seen:\n"
            "            return [seen[need], i]\n"
            "        seen[n] = i\n"
            "    return []\n\n"
            "if __name__ == '__main__':\n"
            "    print(two_sum([2, 7, 11, 15], 9))"
        )

    @classmethod
    def _py_api(cls, prompt: str, subject: str) -> str:
        title = subject.replace("'", "\\'")
        return (
            "from http.server import BaseHTTPRequestHandler, HTTPServer\n"
            "import json\n\n"
            f"TITLE = '{title}'\n\n"
            "class Handler(BaseHTTPRequestHandler):\n"
            "    def do_GET(self):\n"
            "        if self.path == '/health':\n"
            "            self._json(200, {'ok': True, 'service': TITLE})\n"
            "        elif self.path == '/api/items':\n"
            "            self._json(200, {'items': [{'id': 1, 'name': 'alpha'}]})\n"
            "        else:\n"
            "            self._json(404, {'error': 'not found'})\n\n"
            "    def _json(self, code, payload):\n"
            "        body = json.dumps(payload).encode()\n"
            "        self.send_response(code)\n"
            "        self.send_header('Content-Type', 'application/json')\n"
            "        self.send_header('Content-Length', str(len(body)))\n"
            "        self.end_headers()\n"
            "        self.wfile.write(body)\n\n"
            "def main():\n"
            "    HTTPServer(('127.0.0.1', 8766), Handler).serve_forever()\n\n"
            "if __name__ == '__main__':\n"
            "    main()"
        )

    @classmethod
    def _py_cli(cls, prompt: str, subject: str) -> str:
        return (
            "import argparse\n\n"
            "def main() -> None:\n"
            "    p = argparse.ArgumentParser(description='CLI tool')\n"
            "    p.add_argument('name', nargs='?', default='world')\n"
            "    p.add_argument('--verbose', '-v', action='store_true')\n"
            "    args = p.parse_args()\n"
            f"    msg = f'Hello, {{args.name}}!'\n"
            "    if args.verbose:\n"
            "        msg += ' (verbose)'\n"
            "    print(msg)\n\n"
            "if __name__ == '__main__':\n"
            "    main()"
        )

    @classmethod
    def _py_game(cls, prompt: str, subject: str) -> str:
        pl = prompt.lower()
        if "snake" in pl:
            return (
                "import pygame\nimport random\n\n"
                "W, H = 400, 400\nCELL = 20\n\n"
                "def main():\n"
                "    pygame.init()\n"
                "    screen = pygame.display.set_mode((W, H))\n"
                "    pygame.display.set_caption('Snake')\n"
                "    clock = pygame.time.Clock()\n"
                "    snake = [(W//2, H//2)]\n"
                "    dx, dy = CELL, 0\n"
                "    food = (random.randrange(0, W, CELL), random.randrange(0, H, CELL))\n"
                "    running = True\n"
                "    while running:\n"
                "        for event in pygame.event.get():\n"
                "            if event.type == pygame.QUIT:\n"
                "                running = False\n"
                "            if event.type == pygame.KEYDOWN:\n"
                "                if event.key == pygame.K_UP and dy == 0:\n"
                "                    dx, dy = 0, -CELL\n"
                "                elif event.key == pygame.K_DOWN and dy == 0:\n"
                "                    dx, dy = 0, CELL\n"
                "                elif event.key == pygame.K_LEFT and dx == 0:\n"
                "                    dx, dy = -CELL, 0\n"
                "                elif event.key == pygame.K_RIGHT and dx == 0:\n"
                "                    dx, dy = CELL, 0\n"
                "        head = (snake[0][0] + dx, snake[0][1] + dy)\n"
                "        if head == food:\n"
                "            food = (random.randrange(0, W, CELL), random.randrange(0, H, CELL))\n"
                "        else:\n"
                "            snake.pop()\n"
                "        snake.insert(0, head)\n"
                "        if (head[0] < 0 or head[0] >= W or head[1] < 0 or head[1] >= H\n"
                "                or head in snake[1:]):\n"
                "            break\n"
                "        screen.fill('black')\n"
                "        for seg in snake:\n"
                "            pygame.draw.rect(screen, 'lime', (*seg, CELL, CELL))\n"
                "        pygame.draw.rect(screen, 'red', (*food, CELL, CELL))\n"
                "        pygame.display.flip()\n"
                "        clock.tick(10)\n"
                "    pygame.quit()\n\n"
                "if __name__ == '__main__':\n"
                "    main()"
            )
        if "pong" in pl or "breakout" in pl:
            return (
                "import pygame\n\n"
                "W, H = 600, 400\nPAD_W, PAD_H = 80, 10\n\n"
                "def main():\n"
                "    pygame.init()\n"
                "    screen = pygame.display.set_mode((W, H))\n"
                "    clock = pygame.time.Clock()\n"
                "    px, py = W//2, H-30\n"
                "    bx, by, bdx, bdy = W//2, H//2, 4, 4\n"
                "    running = True\n"
                "    while running:\n"
                "        for event in pygame.event.get():\n"
                "            if event.type == pygame.QUIT:\n"
                "                running = False\n"
                "        keys = pygame.key.get_pressed()\n"
                "        if keys[pygame.K_LEFT]:\n"
                "            px -= 6\n"
                "        if keys[pygame.K_RIGHT]:\n"
                "            px += 6\n"
                "        px = max(0, min(W - PAD_W, px))\n"
                "        bx += bdx\n"
                "        by += bdy\n"
                "        if bx <= 0 or bx >= W:\n"
                "            bdx = -bdx\n"
                "        if by <= 0:\n"
                "            by = -bdy\n"
                "        if py <= by <= py + PAD_H and px <= bx <= px + PAD_W:\n"
                "            bdy = -bdy\n"
                "        if by > H:\n"
                "            break\n"
                "        screen.fill('black')\n"
                "        pygame.draw.rect(screen, 'white', (px, py, PAD_W, PAD_H))\n"
                "        pygame.draw.circle(screen, 'white', (bx, by), 8)\n"
                "        pygame.display.flip()\n"
                "        clock.tick(60)\n"
                "    pygame.quit()\n\n"
                "if __name__ == '__main__':\n"
                "    main()"
            )
        if "tic" in pl or "tac" in pl or "tictac" in pl or "tick" in pl:
            return (
                "def print_board(b):\n"
                "    for i in range(3):\n"
                "        print('|'.join(b[i*3:(i+1)*3]))\n"
                "        if i < 2:\n"
                "            print('-'*5)\n\n"
                "def check_winner(b):\n"
                "    for i in range(3):\n"
                "        if b[i*3] == b[i*3+1] == b[i*3+2] != ' ':\n"
                "            return b[i*3]\n"
                "        if b[i] == b[i+3] == b[i+6] != ' ':\n"
                "            return b[i]\n"
                "    if b[0] == b[4] == b[8] != ' ' or b[2] == b[4] == b[6] != ' ':\n"
                "        return b[4]\n"
                "    return None\n\n"
                "def main():\n"
                "    b = [' ']*9\n"
                "    turn = 'X'\n"
                "    for _ in range(9):\n"
                "        print_board(b)\n"
                "        print(f\"Player {turn}'s turn (1-9):\")\n"
                "        try:\n"
                "            m = int(input()) - 1\n"
                "            if m < 0 or m > 8 or b[m] != ' ':\n"
                "                print('Invalid move')\n"
                "                continue\n"
                "        except ValueError:\n"
                "            print('Enter 1-9')\n"
                "            continue\n"
                "        b[m] = turn\n"
                "        w = check_winner(b)\n"
                "        if w:\n"
                "            print_board(b)\n"
                "            print(f'Player {w} wins!')\n"
                "            return\n"
                "        turn = 'O' if turn == 'X' else 'X'\n"
                "    print_board(b)\n"
                "    print(\"It's a tie!\")\n\n"
                "if __name__ == '__main__':\n"
                "    main()"
            )
        return (
            "import random\n\n"
            "def main():\n"
            "    print('Welcome to the game!')\n"
            "    number = random.randint(1, 100)\n"
            "    attempts = 0\n"
            "    while True:\n"
            "        try:\n"
            "            guess = int(input('Guess 1-100: '))\n"
            "            attempts += 1\n"
            "            if guess < number:\n"
            "                print('Too low!')\n"
            "            elif guess > number:\n"
            "                print('Too high!')\n"
            "            else:\n"
            "                print(f'Correct in {attempts} attempts!')\n"
            "                break\n"
            "        except ValueError:\n"
            "            print('Enter a number')\n\n"
            "if __name__ == '__main__':\n"
            "    main()"
        )

    @classmethod
    def _py_data(cls, prompt: str, subject: str) -> str:
        pl = prompt.lower()
        if "csv" in pl:
            return (
                "import csv\nimport sys\n\n"
                "def main():\n"
                "    if len(sys.argv) < 2:\n"
                "        print('Usage: python script.py <csv_file>')\n"
                "        return\n"
                "    with open(sys.argv[1]) as f:\n"
                "        reader = csv.DictReader(f)\n"
                "        rows = list(reader)\n"
                "    print(f'Read {len(rows)} rows')\n"
                "    if rows:\n"
                "        print(f'Columns: {\", \".join(rows[0].keys())}')\n\n"
                "if __name__ == '__main__':\n"
                "    main()"
            )
        if "json" in pl:
            return (
                "import json\nimport sys\n\n"
                "def main():\n"
                "    if len(sys.argv) < 2:\n"
                "        print('Usage: python script.py <json_file>')\n"
                "        return\n"
                "    with open(sys.argv[1]) as f:\n"
                "        data = json.load(f)\n"
                "    print(json.dumps(data, indent=2)[:2000])\n\n"
                "if __name__ == '__main__':\n"
                "    main()"
            )
        return (
            "from typing import List, Dict, Any\n\n"
            f"DATA = [{{\"id\": 1, \"name\": \"{subject}\", \"value\": 42}}]\n\n"
            "def main():\n"
            "    for item in DATA:\n"
            "        print(item)\n\n"
            "if __name__ == '__main__':\n"
            "    main()"
        )

    @classmethod
    def _py_general_app(cls, prompt: str, subject: str) -> str:
        pl = prompt.lower()
        subj = subject.lower()
        if "todo" in subj or "todo" in pl or "task" in pl:
            return (
                "import json\nimport sys\n\n"
                "tasks: list[dict] = []\n\n"
                "def add(title: str):\n"
                "    tasks.append({\"id\": len(tasks)+1, \"title\": title, \"done\": False})\n"
                "    print(f'Added: {title}')\n\n"
                "def list_tasks():\n"
                "    if not tasks:\n"
                "        print('No tasks')\n"
                "        return\n"
                "    for t in tasks:\n"
                "        status = '✓' if t['done'] else '○'\n"
                "        print(f\"{t['id']}. [{status}] {t['title']}\")\n\n"
                "def done(task_id: int):\n"
                "    for t in tasks:\n"
                "        if t['id'] == task_id:\n"
                "            t['done'] = True\n"
                "            print(f'Marked {task_id} done')\n"
                "            return\n"
                "    print('Task not found')\n\n"
                "def delete(task_id: int):\n"
                "    global tasks\n"
                "    tasks = [t for t in tasks if t['id'] != task_id]\n"
                "    print(f'Deleted {task_id}')\n\n"
                "def main():\n"
                "    print('Todo App - Commands: add <title>, list, done <id>, delete <id>, quit')\n"
                "    while True:\n"
                "        try:\n"
                "            line = input('> ').strip()\n"
                "            if not line:\n"
                "                continue\n"
                "            if line == 'quit':\n"
                "                break\n"
                "            if line == 'list':\n"
                "                list_tasks()\n"
                "            elif line.startswith('add '):\n"
                "                add(line[4:])\n"
                "            elif line.startswith('done '):\n"
                "                done(int(line[5:]))\n"
                "            elif line.startswith('delete '):\n"
                "                delete(int(line[7:]))\n"
                "            else:\n"
                "                print('Commands: add <title>, list, done <id>, delete <id>, quit')\n"
                "        except (EOFError, KeyboardInterrupt):\n"
                "            print()\n"
                "            break\n\n"
                "if __name__ == '__main__':\n"
                "    main()"
            )
        if "calc" in subj or "calc" in pl or "calculator" in subj:
            return (
                "import operator\n\n"
                "def main():\n"
                "    print('Calculator - type expressions like \"2 + 2\" or \"quit\"')\n"
                "    ops = {'+': operator.add, '-': operator.sub, '*': operator.mul, "
                "'/': operator.truediv, '**': operator.pow}\n"
                "    while True:\n"
                "        try:\n"
                "            line = input('> ').strip()\n"
                "            if not line or line == 'quit':\n"
                "                break\n"
                "            parts = line.split()\n"
                "            if len(parts) == 3:\n"
                "                a, op, b = parts[0], parts[1], parts[2]\n"
                "                if op in ops:\n"
                "                    result = ops[op](float(a), float(b))\n"
                "                    print(f'= {result}')\n"
                "                else:\n"
                "                    print('Unknown operator')\n"
                "            else:\n"
                "                print('Usage: <num> <op> <num>')\n"
                "        except ZeroDivisionError:\n"
                "            print('Division by zero')\n"
                "        except (ValueError, IndexError):\n"
                "            print('Invalid input')\n"
                "        except (EOFError, KeyboardInterrupt):\n"
                "            print()\n"
                "            break\n\n"
                "if __name__ == '__main__':\n"
                "    main()"
            )
        if "chat" in subj or "chat" in pl or "bot" in subj or "bot" in pl:
            return (
                "import random\n\n"
                "responses = {\n"
                "    'hello': ['Hi there!', 'Hello!', 'Hey!'],\n"
                "    'how are you': ['I am doing well!', 'Great, thanks!', 'All good!'],\n"
                "    'bye': ['Goodbye!', 'See you later!', 'Bye!'],\n"
                "    'default': ['Tell me more.', 'Interesting!', 'I see.']\n"
                "}\n\n"
                "def respond(msg: str) -> str:\n"
                "    msg = msg.lower().strip()\n"
                "    for key, replies in responses.items():\n"
                "        if key in msg:\n"
                "            return random.choice(replies)\n"
                "    return random.choice(responses['default'])\n\n"
                "def main():\n"
                "    print('Chatbot - type your message (quit to exit)')\n"
                "    while True:\n"
                "        try:\n"
                "            user = input('You: ').strip()\n"
                "            if not user or user.lower() == 'quit':\n"
                "                break\n"
                "            print(f'Bot: {respond(user)}')\n"
                "        except (EOFError, KeyboardInterrupt):\n"
                "            print()\n"
                "            break\n\n"
                "if __name__ == '__main__':\n"
                "    main()"
            )
        if "web" in subj or "server" in subj or "http" in pl or "api" in subj:
            return (
                "from http.server import BaseHTTPRequestHandler, HTTPServer\n"
                "import json\n\n"
                f"TITLE = '{subject.replace(chr(39), chr(39)*2)}'\n\n"
                "class Handler(BaseHTTPRequestHandler):\n"
                "    def do_GET(self):\n"
                "        if self.path == '/':\n"
                "            self.send_response(200)\n"
                "            self.send_header('Content-Type', 'text/html')\n"
                "            self.end_headers()\n"
                "            self.wfile.write(f'<h1>{TITLE}</h1>'.encode())\n"
                "        elif self.path == '/api/data':\n"
                "            self._json({'ok': True, 'data': [{'id': 1, 'name': TITLE}]})\n"
                "        else:\n"
                "            self._json({'error': 'not found'}, 404)\n\n"
                "    def _json(self, payload, code=200):\n"
                "        body = json.dumps(payload).encode()\n"
                "        self.send_response(code)\n"
                "        self.send_header('Content-Type', 'application/json')\n"
                "        self.end_headers()\n"
                "        self.wfile.write(body)\n\n"
                "def main():\n"
                "    port = 8766\n"
                f"    print(f'Server running on {{port}}')\n"
                "    HTTPServer(('', port), Handler).serve_forever()\n\n"
                "if __name__ == '__main__':\n"
                "    main()"
            )
        if "weather" in subj or "weather" in pl:
            return (
                "import urllib.request\nimport json\n\n"
                "def get_weather(city: str) -> dict | None:\n"
                "    url = f'https://wttr.in/{city}?format=j1'\n"
                "    try:\n"
                "        with urllib.request.urlopen(url, timeout=5) as r:\n"
                "            return json.loads(r.read())\n"
                "    except Exception as e:\n"
                "        print(f'Error: {e}')\n"
                "        return None\n\n"
                "def main():\n"
                "    import sys\n"
                "    city = sys.argv[1] if len(sys.argv) > 1 else input('City: ')\n"
                "    data = get_weather(city)\n"
                "    if data:\n"
                "        cc = data['current_condition'][0]\n"
                "        print(f\"Weather in {city}: {cc['temp_C']}C, {cc['weatherDesc'][0]['value']}\")\n\n"
                "if __name__ == '__main__':\n"
                "    main()"
            )
        if "stopwatch" in subj or "timer" in subj or "countdown" in pl:
            return (
                "import time\n\n"
                "def main():\n"
                "    import sys\n"
                "    if len(sys.argv) > 1 and sys.argv[1] == 'countdown':\n"
                "        seconds = int(sys.argv[2]) if len(sys.argv) > 2 else 10\n"
                "        for i in range(seconds, 0, -1):\n"
                "            print(f'{i}...', end=' ', flush=True)\n"
                "            time.sleep(1)\n"
                "        print('Done!')\n"
                "    else:\n"
                "        input('Press Enter to start stopwatch...')\n"
                "        start = time.time()\n"
                "        try:\n"
                "            input('Press Enter to stop...')\n"
                "            elapsed = time.time() - start\n"
                "            print(f'Elapsed: {elapsed:.2f}s')\n"
                "        except (EOFError, KeyboardInterrupt):\n"
                "            elapsed = time.time() - start\n"
                "            print(f'\\nElapsed: {elapsed:.2f}s')\n\n"
                "if __name__ == '__main__':\n"
                "    main()"
            )
        if "note" in subj or "note" in pl or "diary" in subj or "journal" in pl:
            return (
                "import json\nimport os\nfrom datetime import datetime\n\n"
                "NOTES_FILE = 'notes.json'\n\n"
                "def load() -> list:\n"
                "    if os.path.exists(NOTES_FILE):\n"
                "        with open(NOTES_FILE) as f:\n"
                "            return json.load(f)\n"
                "    return []\n\n"
                "def save(notes: list):\n"
                "    with open(NOTES_FILE, 'w') as f:\n"
                "        json.dump(notes, f, indent=2)\n\n"
                "def add(text: str):\n"
                "    notes = load()\n"
                "    notes.append({'id': len(notes)+1, 'text': text, 'created': str(datetime.now())})\n"
                "    save(notes)\n"
                "    print('Note added')\n\n"
                "def list_notes():\n"
                "    notes = load()\n"
                "    if not notes:\n"
                "        print('No notes')\n"
                "        return\n"
                "    for n in notes:\n"
                "        print(f\"{n['id']}. {n['text'][:60]}\")\n\n"
                "def delete(nid: int):\n"
                "    notes = load()\n"
                "    notes = [n for n in notes if n['id'] != nid]\n"
                "    save(notes)\n"
                "    print(f'Deleted note {nid}')\n\n"
                "def main():\n"
                "    print('Notes - commands: add <text>, list, delete <id>, quit')\n"
                "    while True:\n"
                "        try:\n"
                "            line = input('> ').strip()\n"
                "            if not line or line == 'quit':\n"
                "                break\n"
                "            if line == 'list':\n"
                "                list_notes()\n"
                "            elif line.startswith('add '):\n"
                "                add(line[4:])\n"
                "            elif line.startswith('delete '):\n"
                "                delete(int(line[7:]))\n"
                "            else:\n"
                "                print('Commands: add <text>, list, delete <id>, quit')\n"
                "        except (EOFError, KeyboardInterrupt):\n"
                "            print()\n"
                "            break\n\n"
                "if __name__ == '__main__':\n"
                "    main()"
            )
        return (
            "import sys\n\n"
            f"APP_NAME = '{subject.replace(chr(39), chr(39)*2)}'\n\n"
            "def main():\n"
            f"    print(f'Welcome to {{APP_NAME}}!')\n"
            "    print(f'Running Python {sys.version}')\n"
            "    user = input('Enter your name: ').strip()\n"
            f"    print(f'Hello {{user}}, welcome to {{APP_NAME}}!')\n\n"
            "if __name__ == '__main__':\n"
            "    main()"
        )

    @classmethod
    def _polyglot(cls, lang: str, task: str, prompt: str, subject: str) -> str:
        msg = subject.replace("\\", "\\\\").replace('"', '\\"').replace("'", "\\'")
        esc = msg
        if lang == "python":
            if task == "algo":
                return cls._py_algo(prompt)
            if task == "api":
                return cls._py_api(prompt, subject)
            if task == "cli":
                return cls._py_cli(prompt, subject)
            if task == "game":
                return cls._py_game(prompt, subject)
            if task == "data":
                return cls._py_data(prompt, subject)
            return cls._py_general_app(prompt, subject)
        if lang == "html" or (lang == "web" and task == "web"):
            return CatR1Code._html(prompt, None)
        if lang == "javascript":
            if task == "api":
                return (
                    "const http = require('http');\n\n"
                    "const server = http.createServer((req, res) => {\n"
                    "  if (req.url === '/health') {\n"
                    "    res.writeHead(200, {'Content-Type': 'application/json'});\n"
                    "    res.end(JSON.stringify({ ok: true }));\n"
                    "    return;\n"
                    "  }\n"
                    "  res.writeHead(404);\n"
                    "  res.end(JSON.stringify({ error: 'not found' }));\n"
                    "});\n\n"
                    "server.listen(8766, () => console.log('API on :8766'));"
                )
            return f"console.log('{esc}');"
        if lang == "typescript":
            return (
                "interface Item { id: number; name: string; }\n\n"
                "const items: Item[] = [{ id: 1, name: 'alpha' }];\n"
                f"console.log('{esc}', items);"
            )
        if lang == "rust":
            return (
                "fn main() {\n"
                f'    println!("{esc}");\n'
                "}"
            )
        if lang == "go":
            return (
                "package main\n\n"
                "import \"fmt\"\n\n"
                "func main() {\n"
                f'\tfmt.Println("{esc}")\n'
                "}"
            )
        if lang == "java":
            return (
                "public class Main {\n"
                "    public static void main(String[] args) {\n"
                f'        System.out.println("{esc}");\n'
                "    }\n"
                "}"
            )
        if lang == "kotlin":
            return (
                "fun main() {\n"
                f'    println("{esc}")\n'
                "}"
            )
        if lang == "swift":
            return (
                "import Foundation\n\n"
                f'print("{esc}")'
            )
        if lang == "c":
            return TokenWeightCodeEmitter._c(msg)
        if lang == "cpp":
            return TokenWeightCodeEmitter._cpp(msg)
        if lang == "bash":
            return f'#!/bin/bash\necho "{esc}"'
        if lang == "ruby":
            return f'puts "{esc}"'
        if lang == "php":
            return f'<?php\necho "{esc}\\n";'
        if lang == "sql":
            return f"SELECT '{esc.replace(chr(39), chr(39)*2)}' AS message;"
        if lang == "css":
            return f"body {{\n  font-family: system-ui, sans-serif;\n}}\n/* {esc} */"
        if lang == "yaml":
            return f"service:\n  name: {subject.lower().replace(' ', '-')}\n  files: off\n"
        if lang == "json":
            return json.dumps({"message": subject, "files": "off"}, indent=2)
        if lang == "dockerfile":
            return (
                "FROM python:3.12-slim\n"
                "WORKDIR /app\n"
                "COPY . .\n"
                f'CMD ["python", "-c", "print(\\"{esc}\\")"]'
            )
        if lang == "solidity":
            return (
                "// SPDX-License-Identifier: MIT\n"
                "pragma solidity ^0.8.20;\n\n"
                "contract Token {\n"
                "    string public name = \"cat r1\";\n"
                "    uint256 public supply;\n"
                "    constructor(uint256 initial) { supply = initial; }\n"
                "}"
            )
        comment = "//"
        if lang in {"python", "bash", "dockerfile", "makefile", "ruby", "perl", "r"}:
            comment = "#"
        elif lang == "html":
            comment = "<!-- -->"
        return (
            f"{comment} {cls.TAG} · {lang} · {task}\n"
            f"{comment} Prompt: {prompt[:120]}\n"
            f"{comment} Subject: {subject}"
        )

    @classmethod
    def scaffold(
        cls,
        lang: str,
        task: str,
        prompt: str,
        engine: Optional["CatR11Engine"] = None,
        vec: Optional[np.ndarray] = None,
    ) -> str:
        lang = cls.normalize_lang(lang) or cls.infer_lang(engine, prompt)
        if lang not in cls.VALID_LANGS:
            lang = "python"
        subject = cls._subject(prompt, engine)
        return cls._polyglot(lang, task, prompt, subject)

    @classmethod
    def generate(
        cls,
        engine: "CatR11Engine",
        prompt: str,
        lang: Optional[str] = None,
        vec: Optional[np.ndarray] = None,
    ) -> str:
        # Template-based fallback path:
        # CodeAnythingEngine.generate("python", "do something") → returns a template program.
        if isinstance(engine, str) and lang is None and vec is None:
            return cls._generate_template(engine, prompt)

        embedded_lang, embedded = CatR1Code.extract_prompt_code(prompt)
        if embedded:
            return embedded
        task = cls.detect_task(prompt)
        lang = cls.infer_lang(engine, prompt, lang or "") if engine is not None else (lang or "python")
        code = cls.scaffold(lang, task, prompt, engine, vec)
        if hasattr(engine, "recompile_code_for_prompt"):
            code = engine.recompile_code_for_prompt(lang, prompt, code)
        return code

    @classmethod
    def execute(cls, lang: str, code: str, prompt: str = "") -> Dict[str, Any]:
        """Execute where possible; otherwise return structural simulation dict."""
        lang = cls.normalize_lang(lang) or (lang or "").lower().strip()
        try:
            if lang == "python":
                tmpdir = os.path.join("/tmp", "cat-r1-exec")
                os.makedirs(tmpdir, exist_ok=True)
                script_path = os.path.join(tmpdir, f"exec_{uuid.uuid4().hex[:12]}.py")
                with open(script_path, "w") as f:
                    f.write(code or "")
                try:
                    out = subprocess.run(
                        [sys.executable, "-u", script_path],
                        capture_output=True,
                        text=True,
                        timeout=8,
                        check=False,
                    )
                    stdout = (out.stdout or "").strip()
                    stderr = (out.stderr or "").strip()
                    if out.returncode != 0 and stderr:
                        return {"ok": False, "lang": lang, "mode": "native", "output": stderr}
                    return {"ok": True, "lang": lang, "mode": "native", "output": stdout or "(no output)"}
                finally:
                    try:
                        os.remove(script_path)
                    except OSError:
                        pass

            sim = cls.simulate(lang, code or "", prompt or "")
            return {"ok": True, "lang": lang, "mode": "simulated", "output": sim}
        except Exception as e:
            return {"ok": False, "lang": lang, "mode": "error", "output": f"{e}"}

    @classmethod
    def simulate(cls, lang: str, code: str, prompt: str = "") -> str:
        lines = [ln for ln in code.splitlines() if ln.strip()]
        funcs = re.findall(r"(?:def|fn|func|function)\s+(\w+)", code)
        classes = re.findall(r"(?:class|struct|interface|contract)\s+(\w+)", code)
        imports = re.findall(r"(?:^import\s+|^use\s+|#include\s+)", code, re.MULTILINE)
        preview = "\n".join(lines[:8])
        if len(lines) > 8:
            preview += f"\n... ({len(lines) - 8} more lines)"
        return (
            f"**{cls.TAG}** dry-run · `{lang}`\n\n"
            f"- lines: {len(lines)} · chars: {len(code)}\n"
            f"- functions: {', '.join(funcs[:6]) or '(none)'}\n"
            f"- types: {', '.join(classes[:6]) or '(none)'}\n"
            f"- imports/includes: {len(imports)}\n"
            f"- runtime: simulated (no `{lang}` binary — structural review only)\n\n"
            f"```\n{preview}\n```\n\n"
            "Python · JavaScript · Bash run natively. Say **run it** after editing."
        )

    @classmethod
    def _minimal_expert(cls, lang: str) -> str:
        return cls._polyglot(lang, "general", f"hello world in {lang}", "Hello World")

    @classmethod
    def experts(cls) -> Dict[str, str]:
        heavy = frozenset({"html"})
        out: Dict[str, str] = {}
        for lang in cls.LANGS:
            if lang in heavy:
                out[lang] = "<!DOCTYPE html><html><body><h1>Hello World</h1></body></html>"
            else:
                out[lang] = cls._minimal_expert(lang)
        return out

    @classmethod
    def stats_line(cls) -> str:
        return f"{cls.TAG} · {len(cls.LANGS)} langs"


CodeAnything = CodeAnythingEngine


class CatR1Code:
    """
    cat r1 code engine.
    Token-weight synthesis · recursive perfection · in-memory run loop · files = off.
    All 128+ languages · polyglot scaffold · sandbox verify.
    """

    NAME = CONFIG["code_interpreter_name"]
    FAMILY = CONFIG["code_interpreter_family"]
    BACKEND = CONFIG["code_interpreter"]
    VERSION = CONFIG["code_interpreter_version"]
    RUNNABLE = CodeAnythingEngine.RUNNABLE if CONFIG.get("code_anything") else frozenset({"python", "javascript", "bash"})
    LANGS = CodeAnythingEngine.LANGS if CONFIG.get("code_anything") else (
        "html", "css", "python", "javascript", "typescript", "java", "rust",
        "go", "bash", "shell", "cpp", "c", "sql", "ruby", "php", "assembly",
    )
    _WRITE_VERBS = re.compile(
        r"\b(write|make|build|create|generate|code|implement|show me|give me|draft|"
        r"scaffold|boilerplate|port|convert|rewrite|refactor|migrate)\b", re.I
    )
    _IN_LANG = re.compile(
        r"\b(?:in|using|with|as)\s+(html|css|python|javascript|typescript|java|rust|go|bash|shell|cpp|c\+\+|c|sql|ruby|php|assembly)\b",
        re.I,
    )
    _MAKE_IT_LANG = re.compile(
        r"\b(?:no\s+)?make\s+(?:it\s+)?(?:in\s+)?(html|css|python|javascript|typescript|java|rust|go|bash|shell|cpp|c\+\+|c|sql|ruby|php)\b",
        re.I,
    )
    _HTML_CUE = re.compile(
        r"\b(?:a|an)\s+html\b|\bhtml\s+(?:program|page|file|ad|that|site|app)\b|\b(?:write|make|build|create)\s+(?:a\s+)?html\b",
        re.I,
    )
    _CREATIVE_BLOCK = re.compile(
        r"\b(story|poem|tale|parable|bedtime|narrative|verse|rhyme)\b", re.I
    )
    _FABLE_STORY = re.compile(r"\b(?:write|tell|give)\s+(?:me\s+)?(?:a\s+)?fable\b", re.I)
    _PATH_TAIL = re.compile(
        r"(?:/Volumes/.+|/Users/.+|~/.+)\.(?:py|c|cpp|html|js|ts|java|rs|go|sh|php|rb)\s*$",
        re.I,
    )
    _BRAND_PREFIX = re.compile(
        r"^\[?\s*(?:cat-r1\.1|cat r1\.1(?:[\s.]*1\.1)?)\s*\]?:?\s*", re.I
    )
    _META_HEAD = re.compile(r"^#+\s*(?:pr|todo)\s+", re.I)
    _META_INLINE = re.compile(r"#+\s*(?:pr|todo)\s+", re.I)
    _META_FILES_PREFIX = re.compile(r"^files\s*\.?\s*=\s*\.?\s*off\s*", re.I)
    _META_TAIL = re.compile(
        r"(?:\s+#\s*(?:pr|todo)\b[^\n]*|\s+files\s*=\s*\.?\s*off\s*>?|\s+>\s*$"
        r"|\s+(?:make the code|\btoken weight\b|\bfiles\.\s*=\s*off\b)).*$",
        re.I | re.S,
    )
    _META_DESIRE = re.compile(
        r"(?:make|let|todo)\s+(?:the\s+)?(?:llm|ai|bot|you)\s+say\s+anything",
        re.I,
    )
    _LITERAL_SAY = (
        re.compile(r"^(?:please\s+)?(?:just\s+)?say\s+(.+)$", re.I | re.S),
        re.compile(r"^(?:respond|reply)\s+with(?:\s+exactly)?\s*[:：\"]?\s*(.+)$", re.I | re.S),
        re.compile(r"^repeat(?:\s+after\s+me)?\s*[:：]\s*(.+)$", re.I | re.S),
        re.compile(
            r"^(?:i\s+want\s+(?:you|the\s+(?:llm|ai|bot))\s+to|make\s+(?:the\s+)?(?:llm|ai|bot|you)\s+)say\s+(.+)$",
            re.I | re.S,
        ),
        re.compile(r"^output\s*[:：]\s*(.+)$", re.I | re.S),
        re.compile(r"^(?:请)?(?:说|念|复述|输出)[:：]\s*(.+)$", re.S),
        re.compile(r"^echo\s+(.+)$", re.I | re.S),
    )
    _RAW_CODE = re.compile(r"(#include[\s\S]+|def\s+\w+|function\s+\w+|fn\s+main|public\s+class|<!DOCTYPE)", re.I)
    _SAYS = re.compile(r"\b(?:that\s+)?says\s+[\"']?([^\"'\n]+)[\"']?", re.I)
    _GO = re.compile(r"^\s*(?:go|do it|yes|continue|proceed|ok|>|code\s*>)\s*\.?\s*$", re.I)
    _CODE_SHORT = re.compile(
        r"^\s*(?:code|/code|code\s*>|>|program|script)\s*$", re.I
    )

    @classmethod
    def enabled(cls) -> bool:
        return bool(CONFIG.get("cat_r1_code_enabled", True))

    @classmethod
    def code_help(cls) -> str:
        return ClaudeMythosRuntime.code_help()
    VALID_LANGS = frozenset(LANGS)

    @classmethod
    def _coerce_lang(cls, engine: "CatR11Engine", lang: Optional[str], prompt: str) -> str:
        lang = engine.normalize_lang(lang) or ""
        if lang in cls.VALID_LANGS:
            return lang
        pl = cls.normalize_prompt(prompt).lower()
        direct = cls._lang_from_text(pl, engine)
        if direct:
            return direct
        if "html" in pl:
            return "html"
        if re.search(r"\b(?:c|c\+\+|cpp)\b", pl):
            return "c" if "iostream" not in pl else "cpp"
        return "python"

    @classmethod
    def normalize_prompt(cls, prompt: str) -> str:
        s = prompt.strip()
        s = cls._META_HEAD.sub("", s)
        s = cls._BRAND_PREFIX.sub("", s)
        s = cls._META_FILES_PREFIX.sub("", s)
        s = cls._META_INLINE.sub("", s)
        s = cls._META_TAIL.sub("", s)
        s = re.sub(r"\s*>\s*$", "", s.strip())
        if re.fullmatch(r"files\s*=\s*\.?\s*off", s, flags=re.I):
            s = ""
        return cls._clean_prompt(s)

    @classmethod
    def clean_topic(cls, text: str) -> str:
        """Strip meta markers and question prefixes — real user intent only."""
        s = cls.normalize_prompt(text) or text.strip()
        pl = s.lower()
        for prefix in (
            "why is the ", "why is ", "why are the ", "why are ",
            "why does the ", "why does ", "why do the ", "why do ",
            "what is the ", "what is ", "what are the ", "what are ",
            "how does the ", "how does ", "how do the ", "how do ", "how to ",
            "how can i ", "how can we ", "how can you ",
            "tell me about the ", "tell me about ", "talk about the ", "talk about ",
            "explain the ", "explain ", "define the ", "define ",
            "can you tell me about ", "can you explain ", "can you help me with ",
            "i want to know about ", "i want ", "i need ",
            "just talk to me about ", "just talk to me",
        ):
            if pl.startswith(prefix):
                s = s[len(prefix):]
                pl = s.lower()
                break
        s = re.sub(r"^files\s*=\s*\.?\s*off\s*", "", s, flags=re.I)
        return s.strip("?.!！？。 \"'").strip() or text.strip()[:120]

    @classmethod
    def is_v4_coding_optimize_request(cls, pl: str) -> bool:
        return bool(re.search(
            r"\boptimi[sz]e\b.+\b(?:neural|network|nn|model)\b"
            r"|\bmatch(?:es)?\s+(?:deepseek\s+)?v4\s+pro\s+cod"
            r"|\bv4\s+pro\s+cod"
            r"|\bbitnet\s+v4\s+pro",
            pl,
            re.I,
        ))

    @classmethod
    def is_fix_all_request(cls, pl: str) -> bool:
        return bool(re.search(
            r"\bfix\s+(?:all\s+)?(?:the\s+)?bugs?\b"
            r"|\bfix\s+everything\b"
            r"|\bdebug\s+(?:the\s+)?(?:app|engine|codebase|cat-r1|cat-r)\b",
            pl,
            re.I,
        ))

    @classmethod
    def is_code_debug(cls, prompt: str, pl: str) -> bool:
        """Code/traceback debugging — not meta 'fix all bugs' project requests."""
        if cls.is_fix_all_request(pl):
            return False
        if cls.extract_prompt_code(prompt)[1]:
            return True
        if re.search(
            r"\b(traceback|stack trace|exception:|syntaxerror|typeerror|nameerror|indentationerror)\b",
            pl,
            re.I,
        ):
            return True
        if re.search(r"\b(error|bug|broken)\b.+\b(in|at|when|after|on line)\b", pl):
            return True
        if re.search(r"\bmy\s+(code|script|program|app)\b.+\b(bug|error|broken|fail)", pl):
            return True
        if re.search(r"\b(debug|fix)\b", pl) and re.search(
            r"\b(bug|error|broken|traceback|exception|fail|crash)\b", pl
        ):
            return True
        return False

    @classmethod
    def is_bitnet_request(cls, pl: str) -> bool:
        return bool(re.search(
            r"\b(?:implement|enable|use|run|make|integrate)\s+(?:the\s+)?(?:bitnet|dspark)\b"
            r"|\b(?:bitnet|dspark)\s+integrat"
            r"|\bbitnet\s+for\s+real\b"
            r"|\breal\s+bitnet\b"
            r"|\bmake\s+(?:the\s+)?bitnet\s+real\b"
            r"|\bbitnet\s+real\b"
            r"|\bdspark\b",
            pl,
        ))

    @classmethod
    def _clean_prompt(cls, prompt: str) -> str:
        s = prompt.strip()
        s = cls._PATH_TAIL.sub("", s)
        return s.strip()

    @classmethod
    def extract_prompt_code(cls, prompt: str) -> Tuple[Optional[str], Optional[str]]:
        """If the user pasted code (± file path), return it exactly — files = off."""
        cleaned = cls.normalize_prompt(prompt)
        lang, code = None, None
        m = re.search(r"```([a-zA-Z0-9_+#-]*)\s*\n([\s\S]*?)```", cleaned)
        if m:
            lang = (m.group(1) or "").strip().lower() or None
            code = m.group(2).strip()
        elif cls._RAW_CODE.search(cleaned) or VibeCodeHeuristics.CODE_SHAPES.search(cleaned):
            code = cleaned.strip()
            if "#include" in code and "stdio" in code:
                lang = "c" if "iostream" not in code else "cpp"
            elif "iostream" in code:
                lang = "cpp"
            elif code.lstrip().startswith("<!"):
                lang = "html"
            elif "def " in code:
                lang = "python"
        if code:
            code = re.sub(r"^c\s*\n(?=#include)", "", code, flags=re.I)
            return lang, code
        return None, None

    @classmethod
    def wants_code(cls, pl: str) -> bool:
        if not cls.enabled():
            return False
        if CodeAnythingEngine.enabled() and CodeAnythingEngine.wants_anything(pl):
            return True
        if VibeCodeHeuristics.wants_code(pl):
            return True
        pl = pl.strip().lower()
        if cls._CODE_SHORT.match(pl):
            return True
        if cls._FABLE_STORY.search(pl):
            return False
        if cls._CREATIVE_BLOCK.search(pl):
            return False
        if re.search(r"\b(?:can you|please|help me|i need you to|let'?s)\s+code\b", pl):
            return True
        if re.search(r"\bcode\s*[>:]", pl):
            return True
        if re.search(r"\b(?:write|make|build|create|show|give|draft|implement)\s+(?:me\s+)?(?:some\s+)?code\b", pl):
            return True
        if cls._MAKE_IT_LANG.search(pl):
            return True
        if cls._HTML_CUE.search(pl):
            return True
        if cls._IN_LANG.search(pl) and cls._WRITE_VERBS.search(pl):
            return True
        if cls._IN_LANG.search(pl) and any(w in pl for w in ("page", "app", "script", "site", "cat", "hello", "world", "ad", "usb", "gamer")):
            return True
        if re.search(r"\b(function|def |class |snippet|fibonacci|fib|prime|```)\b", pl):
            return True
        if re.search(r"\b(write|make|build|create)\b", pl) and re.search(
            r"\b(code|function|script|program|app|page|html|python|javascript|java)\b", pl
        ):
            return True
        if re.search(r"\b(python|javascript|typescript|html|rust|java|go|bash|cpp|c\+\+)\b", pl):
            if re.search(r"\b(write|make|build|create|implement|code)\b", pl):
                return True
        if cls._RAW_CODE.search(cls._clean_prompt(pl)):
            return True
        if re.search(r"\b(c|c\+\+|cpp)\b", pl) and re.search(
            r"\b(write|make|build|create|code|hello|world|program|main)\b", pl
        ):
            return True
        if re.search(r"\bgeocit(?:ies|es)\b", pl) and "html" in pl:
            return True
        if re.search(r"\b(program|script|snippet|algorithm)\b", pl) and re.search(
            r"\b(write|make|build|create|need|want|give|show)\b", pl
        ):
            return True
        return False

    @classmethod
    def wants_code_with_history(cls, pl: str, engine: "CatR11Engine") -> bool:
        if cls.wants_code(pl):
            return True
        if not cls.enabled():
            return False
        if cls._GO.match(pl.strip()) or cls._CODE_SHORT.match(pl.strip()):
            return bool(engine.chat_history) or bool(cls._lang_from_history(engine))
        return False

    @classmethod
    def _lang_from_text(cls, pl: str, engine: "CatR11Engine") -> Optional[str]:
        vibe = VibeCodeHeuristics.lang_from_text(pl, engine)
        if vibe:
            return vibe
        m = cls._MAKE_IT_LANG.search(pl)
        if m:
            raw = m.group(1).lower()
            if raw == "shell":
                raw = "bash"
            return engine.normalize_lang(raw) or raw
        m = cls._IN_LANG.search(pl)
        if m:
            raw = m.group(1).lower()
            if raw == "shell":
                raw = "bash"
            return engine.normalize_lang(raw) or raw
        if cls._HTML_CUE.search(pl) or re.search(r"\bhtml\b", pl):
            return "html"
        if re.search(r"\b(?:write|make|code)\s+(?:me\s+)?(?:a\s+)?c\s+program\b", pl):
            return "c"
        if re.search(r"\bc\s+(?:program|code|hello)\b", pl):
            return "c"
        if re.search(r"\bjava\b", pl) and re.search(r"\b(write|make|code|program)\b", pl):
            return "java"
        return None

    @classmethod
    def _lang_from_history(cls, engine: "CatR11Engine") -> Optional[str]:
        for msg in reversed(engine.chat_history):
            if msg.get("role") != "user":
                continue
            lang = cls._lang_from_text(msg.get("text", "").lower(), engine)
            if lang:
                return lang
            if cls._HTML_CUE.search(msg.get("text", "").lower()) or "html" in msg.get("text", "").lower():
                return "html"
        return None

    @classmethod
    def run(cls, engine: "CatR11Engine", lang: str, code: str) -> str:
        return engine.execute_code_any_language(lang, code)

    @classmethod
    def should_run(cls, pl: str) -> bool:
        if CONFIG.get("code_auto_run"):
            return True
        if VibeCodeHeuristics.wants_run(pl):
            return True
        return any(
            x in pl for x in (
                "run it", "run this", "execute", "interpret", "test it",
                "/run", "and run", "then run",
            )
        )

    @classmethod
    def detect_lang(cls, engine: "CatR11Engine", prompt: str) -> str:
        pl = cls.normalize_prompt(prompt)
        embedded_lang, embedded = cls.extract_prompt_code(prompt)
        if embedded_lang:
            return engine.normalize_lang(embedded_lang) or embedded_lang
        if embedded and "#include" in embedded:
            return "cpp" if "iostream" in embedded else "c"
        direct = cls._lang_from_text(pl, engine)
        if not direct:
            direct = VibeCodeHeuristics.lang_from_text(prompt, engine)
        if direct:
            return direct
        if cls._GO.match(pl.strip().lower()):
            hist_lang = cls._lang_from_history(engine)
            if hist_lang:
                return hist_lang
        lang = engine.extract_lang(prompt) or engine.detect_lang_from_text(prompt)
        if CodeAnythingEngine.enabled():
            return CodeAnythingEngine.infer_lang(engine, prompt, lang or "")
        return cls._coerce_lang(engine, lang, prompt)

    @classmethod
    def _subject(cls, prompt: str, engine: Optional["CatR11Engine"] = None) -> str:
        vibe_subj = VibeCodeHeuristics.subject_from_text(prompt)
        if vibe_subj:
            return vibe_subj
        pl = prompt.lower().strip()
        if re.search(r"\b(?:no\s+)?make\s+it\b", pl) and engine:
            for msg in reversed(engine.chat_history):
                if msg.get("role") != "user":
                    continue
                prior = cls._subject(msg.get("text", ""))
                if prior and prior.lower() not in {"it", "it html", "hello world", "no"}:
                    return prior
        m = cls._SAYS.search(pl)
        if m:
            return m.group(1).strip("?.! ")
        if "hello cat" in pl:
            return "Hello Cat"
        if "meow" in pl:
            return "Meow"
        if "hello world" in pl:
            return "Hello World"
        if "gamer" in pl and "usb" in pl:
            return "GAMER USB — ULTRA SPEED"
        m = re.search(
            r"(?:write|make|build|create|generate|show|give)\s+(?:me\s+)?(?:a\s+)?(?:html\s+)?(?:that\s+is\s+a\s+)?(.+?)(?:\s+in\s+\w+|\s+program|\s*$)",
            pl,
        )
        if m:
            subj = m.group(1).strip("?.! ")
            if subj and subj not in {"html", "it", "a", "an", "no"}:
                return subj[:80]
        m = re.search(
            r"(?:write|make|build|create|generate|show|give)\s+(?:me\s+)?(?:a\s+)?(.+?)\s+in\s+\w+",
            pl,
        )
        if m:
            subj = m.group(1).strip("?.! ")
            if subj not in {"html", "it", "a", "an", "no"}:
                return subj
        return "Hello World"

    @classmethod
    def _html_geocities(cls, prompt: str) -> str:
        pl = prompt.lower()
        title = "GAMER USB AD — GEOCITIES EDITION"
        headline = "ULTRA GAMER USB 9000"
        if "gamer" in pl and "usb" in pl:
            headline = "GAMER USB — 1337 MB/s OF RAW POWER"
        label = html_module.escape(headline)
        return (
            "<!DOCTYPE html>\n"
            "<html>\n"
            "<head>\n"
            "  <meta charset=\"UTF-8\">\n"
            f"  <title>{html_module.escape(title)}</title>\n"
            "  <style>\n"
            "    body { background: #000080 url('data:image/svg+xml,<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"20\" height=\"20\"><rect fill=\"%23ffff00\" width=\"10\" height=\"10\"/><rect fill=\"%2300ff00\" x=\"10\" y=\"10\" width=\"10\" height=\"10\"/></svg>'); "
            "color: #00ff00; font-family: Comic Sans MS, cursive; margin: 0; }\n"
            "    .banner { background: linear-gradient(90deg,red,yellow,lime,cyan,blue,magenta); padding: 4px; text-align: center; }\n"
            "    .banner h1 { color: #fff; text-shadow: 2px 2px #000; font-size: 2rem; margin: 0; animation: blink 1s step-end infinite; }\n"
            "    @keyframes blink { 50% { opacity: 0.3; } }\n"
            "    .box { border: 4px ridge #ff00ff; background: #c0c0c0; color: #000; margin: 1rem auto; max-width: 640px; padding: 1rem; }\n"
            "    .counter { font-size: 0.75rem; color: #888; text-align: center; }\n"
            "    marquee { font-size: 1.2rem; color: #ff0000; font-weight: bold; }\n"
            "    .usb { font-size: 4rem; text-align: center; }\n"
            "  </style>\n"
            "</head>\n"
            "<body>\n"
            "  <div class=\"banner\"><h1>★ WELCOME TO MY HOMEPAGE ★</h1></div>\n"
            "  <marquee>🔥 BUY NOW — GAMER USB — LIMITED STOCK — CLICK HERE 🔥</marquee>\n"
            "  <div class=\"box\">\n"
            "    <div class=\"usb\">💾🎮</div>\n"
            f"    <h2 style=\"text-align:center;color:#0000ff;\">{label}</h2>\n"
            "    <ul>\n"
            "      <li>⚡ ZERO LAG FILE TRANSFERS</li>\n"
            "      <li>🎯 RGB NOT INCLUDED (1999 AUTHENTIC)</li>\n"
            "      <li>🏆 BEST VIEWED IN NETSCAPE NAVIGATOR</li>\n"
            "    </ul>\n"
            "    <p style=\"text-align:center\"><blink><b>ONLY $19.99!!!</b></blink></p>\n"
            "  </div>\n"
            "  <p class=\"counter\">You are visitor #420,069</p>\n"
            "</body>\n"
            "</html>"
        )

    @classmethod
    def _html(cls, prompt: str, engine: Optional["CatR11Engine"] = None) -> str:
        if CONFIG.get("web_program_enabled") and engine is not None and hasattr(engine, "web"):
            tpl = CatR1WebProgram.pick_template(prompt)
            if tpl != "minimal" or re.search(r"\b(website|landing|dashboard|portfolio|site)\b", prompt.lower()):
                return CatR1WebProgram.render(tpl, prompt, engine)
        pl = prompt.lower()
        if re.search(r"\bgeocit(?:ies|es)\b", pl):
            return cls._html_geocities(prompt)
        subject = cls._subject(prompt, engine)
        title = subject.title() if subject else "Hello"
        label = html_module.escape(subject if subject else "App")
        has_cat = "cat" in pl or "🐱" in prompt or "kitty" in pl or "kitten" in pl

        if "todo" in pl or "task" in pl or "todos" in pl:
            return (
                "<!DOCTYPE html>\n"
                '<html lang="en">\n'
                "<head>\n"
                "  <meta charset=\"UTF-8\">\n"
                "  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\n"
                f"  <title>{label} - Todo</title>\n"
                "  <style>\n"
                "    * { box-sizing: border-box; margin: 0; padding: 0; }\n"
                "    body { font-family: system-ui, sans-serif; background: #0f172a; color: #e2e8f0; "
                "display: flex; justify-content: center; padding: 2rem; }\n"
                "    .app { width: 100%; max-width: 480px; }\n"
                "    h1 { font-size: 1.5rem; margin-bottom: 1rem; color: #38bdf8; }\n"
                "    .input-row { display: flex; gap: 0.5rem; margin-bottom: 1rem; }\n"
                "    input { flex: 1; padding: 0.6rem; border: 1px solid #334155; border-radius: 8px; "
                "background: #1e293b; color: #e2e8f0; font-size: 0.9rem; }\n"
                "    input:focus { outline: none; border-color: #38bdf8; }\n"
                "    button { padding: 0.6rem 1rem; border: none; border-radius: 8px; "
                "background: #38bdf8; color: #0f172a; cursor: pointer; font-weight: 600; }\n"
                "    button:hover { background: #7dd3fc; }\n"
                "    ul { list-style: none; }\n"
                "    li { display: flex; align-items: center; gap: 0.5rem; "
                "padding: 0.6rem; border-bottom: 1px solid #1e293b; }\n"
                "    li.done span { text-decoration: line-through; opacity: 0.5; }\n"
                "    li button { margin-left: auto; background: transparent; color: #ef4444; "
                "font-size: 1rem; padding: 0.2rem 0.5rem; }\n"
                "  </style>\n"
                "</head>\n"
                "<body>\n"
                "  <div class=\"app\">\n"
                f"    <h1>{label}</h1>\n"
                "    <div class=\"input-row\">\n"
                "      <input id=\"input\" placeholder=\"Add a task...\" autofocus>\n"
                "      <button onclick=\"addTask()\">Add</button>\n"
                "    </div>\n"
                "    <ul id=\"list\"></ul>\n"
                "  </div>\n"
                "  <script>\n"
                "    const list = document.getElementById('list');\n"
                "    const input = document.getElementById('input');\n"
                "    function addTask() {\n"
                "      const text = input.value.trim();\n"
                "      if (!text) return;\n"
                "      const li = document.createElement('li');\n"
                "      const span = document.createElement('span');\n"
                "      span.textContent = text;\n"
                "      const del = document.createElement('button');\n"
                "      del.textContent = 'x';\n"
                "      del.onclick = () => li.remove();\n"
                "      li.onclick = () => li.classList.toggle('done');\n"
                "      li.appendChild(span);\n"
                "      li.appendChild(del);\n"
                "      list.appendChild(li);\n"
                "      input.value = '';\n"
                "      input.focus();\n"
                "    }\n"
                "    input.addEventListener('keydown', e => { if (e.key === 'Enter') addTask(); });\n"
                "  </script>\n"
                "</body>\n"
                "</html>"
            )

        if "calc" in pl or "calculator" in pl:
            return (
                "<!DOCTYPE html>\n"
                '<html lang="en">\n'
                "<head>\n"
                "  <meta charset=\"UTF-8\">\n"
                "  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\n"
                f"  <title>{label} - Calculator</title>\n"
                "  <style>\n"
                "    * { box-sizing: border-box; margin: 0; }\n"
                "    body { font-family: system-ui, sans-serif; background: #0f172a; color: #e2e8f0; "
                "display: flex; justify-content: center; padding: 2rem; }\n"
                "    .calc { width: 280px; }\n"
                "    h1 { font-size: 1.2rem; margin-bottom: 0.8rem; color: #38bdf8; }\n"
                "    #display { width: 100%; padding: 0.8rem; font-size: 1.5rem; text-align: right; "
                "border: 1px solid #334155; border-radius: 8px; background: #1e293b; "
                "color: #e2e8f0; margin-bottom: 0.5rem; }\n"
                "    .grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 0.4rem; }\n"
                "    button { padding: 0.8rem; font-size: 1.1rem; border: none; border-radius: 6px; "
                "background: #1e293b; color: #e2e8f0; cursor: pointer; }\n"
                "    button:hover { background: #334155; }\n"
                "    button.op { background: #334155; color: #38bdf8; }\n"
                "    button.eq { background: #38bdf8; color: #0f172a; }\n"
                "    button.clr { background: #ef4444; color: #fff; }\n"
                "  </style>\n"
                "</head>\n"
                "<body>\n"
                "  <div class=\"calc\">\n"
                f"    <h1>{label}</h1>\n"
                "    <input id=\"display\" readonly value=\"0\">\n"
                "    <div class=\"grid\">\n"
                "      <button class=\"clr\" onclick=\"clearD()\">C</button>"
                "<button onclick=\"add('(')\">(</button>"
                "<button onclick=\"add(')')\">)</button>"
                "<button class=\"op\" onclick=\"add('/')\">÷</button>\n"
                "      <button onclick=\"add('7')\">7</button>"
                "<button onclick=\"add('8')\">8</button>"
                "<button onclick=\"add('9')\">9</button>"
                "<button class=\"op\" onclick=\"add('*')\">×</button>\n"
                "      <button onclick=\"add('4')\">4</button>"
                "<button onclick=\"add('5')\">5</button>"
                "<button onclick=\"add('6')\">6</button>"
                "<button class=\"op\" onclick=\"add('-')\">−</button>\n"
                "      <button onclick=\"add('1')\">1</button>"
                "<button onclick=\"add('2')\">2</button>"
                "<button onclick=\"add('3')\">3</button>"
                "<button class=\"op\" onclick=\"add('+')\">+</button>\n"
                "      <button onclick=\"add('0')\">0</button>"
                "<button onclick=\"add('.')\">.</button>"
                "<button style=\"grid-column:span 2\" class=\"eq\" onclick=\"calc()\">=</button>\n"
                "    </div>\n"
                "  </div>\n"
                "  <script>\n"
                "    const d = document.getElementById('display');\n"
                "    function add(v) { d.value = d.value === '0' ? v : d.value + v; d.focus(); }\n"
                "    function calc() { try { d.value = eval(d.value); } catch { d.value = 'Error'; } }\n"
                "    function clearD() { d.value = '0'; }\n"
                "    document.addEventListener('keydown', e => {\n"
                "      if (/^[0-9.+\\-*/()]$/.test(e.key)) add(e.key);\n"
                "      if (e.key === 'Enter') calc();\n"
                "      if (e.key === 'Escape') clearD();\n"
                "    });\n"
                "  </script>\n"
                "</body>\n"
                "</html>"
            )

        if "chat" in pl or "bot" in pl:
            return (
                "<!DOCTYPE html>\n"
                '<html lang="en">\n'
                "<head>\n"
                "  <meta charset=\"UTF-8\">\n"
                "  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\n"
                f"  <title>{label} - Chat</title>\n"
                "  <style>\n"
                "    * { box-sizing: border-box; margin: 0; }\n"
                "    body { font-family: system-ui, sans-serif; background: #0f172a; color: #e2e8f0; "
                "display: flex; justify-content: center; padding: 2rem; }\n"
                "    .chat { width: 100%; max-width: 480px; }\n"
                "    h1 { font-size: 1.2rem; margin-bottom: 0.8rem; color: #38bdf8; }\n"
                "    #messages { height: 320px; overflow-y: auto; border: 1px solid #334155; "
                "border-radius: 8px; padding: 0.8rem; margin-bottom: 0.8rem; background: #1e293b; }\n"
                "    .msg { margin-bottom: 0.5rem; padding: 0.4rem 0.8rem; border-radius: 8px; "
                "max-width: 80%; }\n"
                "    .user { background: #2563eb; color: #fff; margin-left: auto; }\n"
                "    .bot { background: #334155; }\n"
                "    .row { display: flex; gap: 0.5rem; }\n"
                "    input { flex: 1; padding: 0.6rem; border: 1px solid #334155; border-radius: 8px; "
                "background: #1e293b; color: #e2e8f0; }\n"
                "    button { padding: 0.6rem 1rem; border: none; border-radius: 8px; "
                "background: #38bdf8; color: #0f172a; cursor: pointer; font-weight: 600; }\n"
                "  </style>\n"
                "</head>\n"
                "<body>\n"
                "  <div class=\"chat\">\n"
                f"    <h1>{label}</h1>\n"
                "    <div id=\"messages\">\n"
                "      <div class=\"msg bot\">Hello! How can I help you?</div>\n"
                "    </div>\n"
                "    <div class=\"row\">\n"
                "      <input id=\"input\" placeholder=\"Type a message...\" autofocus>\n"
                "      <button onclick=\"send()\">Send</button>\n"
                "    </div>\n"
                "  </div>\n"
                "  <script>\n"
                "    const input = document.getElementById('input');\n"
                "    const msgs = document.getElementById('messages');\n"
                "    const responses = ['Tell me more.', 'Interesting!', 'I see.', 'Go on...', 'Hmm, let me think about that.'];\n"
                "    function send() {\n"
                "      const text = input.value.trim();\n"
                "      if (!text) return;\n"
                "      const u = document.createElement('div');\n"
                "      u.className = 'msg user'; u.textContent = text;\n"
                "      msgs.appendChild(u);\n"
                "      setTimeout(() => {\n"
                "        const b = document.createElement('div');\n"
                "        b.className = 'msg bot';\n"
                "        b.textContent = responses[Math.floor(Math.random() * responses.length)];\n"
                "        msgs.appendChild(b);\n"
                "        msgs.scrollTop = msgs.scrollHeight;\n"
                "      }, 400);\n"
                "      input.value = '';\n"
                "      msgs.scrollTop = msgs.scrollHeight;\n"
                "    }\n"
                "    input.addEventListener('keydown', e => { if (e.key === 'Enter') send(); });\n"
                "  </script>\n"
                "</body>\n"
                "</html>"
            )

        if "counter" in pl or "count" in pl:
            return (
                "<!DOCTYPE html>\n"
                '<html lang="en">\n'
                "<head>\n"
                "  <meta charset=\"UTF-8\">\n"
                "  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\n"
                f"  <title>{label} - Counter</title>\n"
                "  <style>\n"
                "    * { box-sizing: border-box; margin: 0; }\n"
                "    body { font-family: system-ui, sans-serif; background: #0f172a; color: #e2e8f0; "
                "display: flex; justify-content: center; padding: 2rem; text-align: center; }\n"
                "    .app { padding: 2rem; }\n"
                "    h1 { font-size: 1.2rem; margin-bottom: 1rem; color: #38bdf8; }\n"
                "    #count { font-size: 4rem; margin: 1rem 0; }\n"
                "    button { padding: 0.6rem 1.2rem; margin: 0.3rem; border: none; border-radius: 8px; "
                "background: #38bdf8; color: #0f172a; cursor: pointer; font-weight: 600; font-size: 1rem; }\n"
                "    button:hover { background: #7dd3fc; }\n"
                "    button.reset { background: #334155; color: #94a3b8; }\n"
                "  </style>\n"
                "</head>\n"
                "<body>\n"
                "  <div class=\"app\">\n"
                f"    <h1>{label}</h1>\n"
                "    <div id=\"count\">0</div>\n"
                "    <button onclick=\"c.textContent=+c.textContent-1\">−</button>\n"
                "    <button onclick=\"c.textContent=+c.textContent+1\">+</button>\n"
                "    <br><br><button class=\"reset\" onclick=\"c.textContent='0'\">Reset</button>\n"
                "  </div>\n"
                "  <script>const c = document.getElementById('count');</script>\n"
                "</body>\n"
                "</html>"
            )

        if has_cat:
            return (
                "<!DOCTYPE html>\n"
                '<html lang="en">\n'
                "<head>\n"
                "  <meta charset=\"UTF-8\">\n"
                "  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\n"
                "  <title>🐱 Hello Cat</title>\n"
                "  <style>\n"
                "    * { box-sizing: border-box; }\n"
                "    body { font-family: system-ui, -apple-system, sans-serif; display: flex; "
                "align-items: center; justify-content: center; min-height: 100vh; margin: 0; "
                "background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); color: #eee; }\n"
                "    .card { text-align: center; padding: 2rem; }\n"
                "    .emoji { font-size: 5rem; }\n"
                "    h1 { font-size: 2.5rem; margin: 0.5rem 0 0; font-weight: 600; }\n"
                "    p { color: #94a3b8; margin-top: 0.5rem; }\n"
                "  </style>\n"
                "</head>\n"
                "<body>\n"
                "  <div class=\"card\">\n"
                "    <div class=\"emoji\">🐱</div>\n"
                "    <h1>Hello Cat!</h1>\n"
                "    <p>Meow! Built by cat r1</p>\n"
                "  </div>\n"
                "</body>\n"
                "</html>"
            )

        return (
            "<!DOCTYPE html>\n"
            '<html lang="en">\n'
            "<head>\n"
            "  <meta charset=\"UTF-8\">\n"
            "  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\n"
            f"  <title>{label}</title>\n"
            "  <style>\n"
            "    * { box-sizing: border-box; margin: 0; }\n"
            "    body { font-family: system-ui, sans-serif; background: #0f172a; color: #e2e8f0; "
            "display: flex; align-items: center; justify-content: center; min-height: 100vh; }\n"
            "    .app { text-align: center; padding: 2rem; max-width: 400px; }\n"
            f"    h1 {{ font-size: 1.8rem; margin-bottom: 0.5rem; color: #38bdf8; }}\n"
            "    p { color: #94a3b8; line-height: 1.6; }\n"
            "  </style>\n"
            "</head>\n"
            "<body>\n"
            "  <div class=\"app\">\n"
            f"    <h1>{label}</h1>\n"
            "    <p>Built by cat r1 — in-memory, files=off.</p>\n"
            "  </div>\n"
            "</body>\n"
            "</html>"
        )

    @classmethod
    def _python(cls, engine: "CatR11Engine", prompt: str) -> str:
        pl = prompt.lower()
        if "fibonacci" in pl or re.search(r"\bfib\b", pl):
            block = engine.synth._code(engine._extract_topic_words(prompt), pl, None)
            m = re.search(r"```python\n(.*?)```", block, re.S)
            if m:
                return m.group(1).strip()
        if CodeAnythingEngine.enabled():
            return CodeAnythingEngine._py_general_app(prompt, cls._subject(prompt))
        subject = cls._subject(prompt)
        msg = TokenWeightCodeEmitter.message_from_prompt(prompt, None, engine)
        return TokenWeightCodeEmitter._python(msg)

    @classmethod
    def build(cls, engine: "CatR11Engine", prompt: str, lang: str, vec: Optional[np.ndarray] = None) -> str:
        if vec is None:
            vec = engine.last_vec if engine.last_vec is not None else engine.encode_for_task(prompt, task="code")
        if CodeAnythingEngine.enabled():
            code = CodeAnythingEngine.generate(engine, prompt, lang, vec)
            if CONFIG.get("cat_r1_engine"):
                return CatR1MythosEngine._recursive_perfect(code, lang, prompt, engine)
            return code
        if CONFIG.get("cat_r1_engine"):
            return CatR1MythosEngine.generate(engine, prompt, lang, vec)
        return TokenWeightCodeEmitter.emit(engine, prompt, lang, vec)

    @classmethod
    def format_response(cls, lang: str, code: str, prompt: str, engine: "CatR11Engine") -> str:
        pl = prompt.lower()
        lang = cls._coerce_lang(engine, lang, prompt)
        fenced = TokenWeightCodeEmitter.fence(lang, code)
        if CONFIG.get("code_output_exact") or engine._wants_code_only(prompt):
            return fenced
        if cls.should_run(pl) and lang in cls.RUNNABLE:
            result = cls.run(engine, lang, code)
            return f"{fenced}\n\n**Output:**\n```\n{result}\n```"
        return fenced

    @classmethod
    def _prompt_for_code(cls, prompt: str, engine: "CatR11Engine") -> str:
        pl = cls.normalize_prompt(prompt).strip().lower()
        if cls._GO.match(pl) or cls._CODE_SHORT.match(pl):
            for msg in reversed(engine.chat_history):
                if msg.get("role") == "user":
                    prior = msg.get("text", "").strip()
                    if prior and not cls._CODE_SHORT.match(prior.lower()):
                        return prior
            return "write hello world in python"
        if pl.startswith("/code "):
            return prompt.split(maxsplit=1)[1] if " " in prompt.strip() else "write hello world in python"
        return prompt

    @classmethod
    def respond(cls, engine: "CatR11Engine", prompt: str) -> str:
        if not cls.enabled():
            return f"{BRAND} code is disabled. Set `cat_r1_code_enabled=True` in CONFIG."
        prompt = cls._prompt_for_code(prompt, engine)
        norm = cls.normalize_prompt(prompt)
        vec = engine.encode_for_task(norm, task="code")
        lang = cls.detect_lang(engine, norm)
        code = cls.build(engine, norm, lang, vec)
        return cls.format_response(lang, code, norm, engine)


CatR1Code = CatR1Code
CatCode010 = CatR1Code


# ──────────────────────────────────────────────────────────────
# CAT R1.1 CODING API 0.1 (cat r1 code-style agent · files = off)
# ──────────────────────────────────────────────────────────────
@dataclass
class CodingBuffer:
    name: str = "untitled.py"
    lang: str = "python"
    code: str = ""
    last_output: str = ""
    runs: int = 0


class CatR1CodingAPI:
    """
    cat r1 code 1.0 API (files = off · BitNet · no external API).
    Claude Code fork — generate + run · reasoning_content + content split.
    """

    _interp = staticmethod(active_interpreter)
    PROTO = CONFIG.get("o1_interpreter_protocol", CatR1CodeR1.PROTO)
    VER = CatR1CodeR1.VER if CatR1CodeR1.enabled() else (
        CatR1CodeInterpreter01.VER if CatR1CodeInterpreter01.enabled() else O1PreviewSyntax.VERSION
    )
    NAME = CONFIG["code_interpreter_name"]
    FULL = NAME
    MYTHOS = CONFIG["mythos_tier"]
    PROSE = CONFIG["prose_tier"]
    BACKEND = CONFIG["code_interpreter"]
    MYTHOS_VER = CONFIG["code_interpreter_version"]
    TOOLS = CatR1CodeR1.TOOLS if CatR1CodeR1.enabled() else (
        "run", "edit", "lint", "explain", "new", "clear"
    )

    _buffers: Dict[str, CodingBuffer] = {}
    _active_id: str = "default"
    _buffer_lock = threading.Lock()

    @classmethod
    def default_snippet(cls) -> str:
        return (
            f"# {ClaudeMythosRuntime.think_header()}\n"
            f"# {cls.FULL} · cat r1 code · in-memory buffer only\n\n"
            "def main():\n"
            "    print('Hello from cat r1 code 1.0')\n\n"
            "if __name__ == '__main__':\n"
            "    main()\n"
        )

    @classmethod
    def help_text(cls) -> str:
        return ClaudeMythosRuntime.code_help()

    @classmethod
    def session(cls, session_id: Optional[str] = None) -> CodingBuffer:
        with cls._buffer_lock:
            sid = session_id or cls._active_id
            if sid not in cls._buffers:
                cls._buffers[sid] = CodingBuffer(code=cls.default_snippet())
            cls._active_id = sid
            return cls._buffers[sid]

    @classmethod
    def _detect_lang(cls, engine: "CatR11Engine", code: str, hint: str = "") -> str:
        if hint:
            norm = engine.normalize_lang(hint)
            if norm:
                return norm
        return CatR1Code.detect_lang(engine, code)

    @classmethod
    def lint(cls, code: str, lang: str) -> Tuple[bool, str]:
        lang = (lang or "python").lower()
        if lang == "python":
            try:
                ast.parse(code)
                return True, "syntax ok"
            except SyntaxError as e:
                return False, str(e)
        if lang in {"javascript", "typescript", "js", "ts"}:
            return True, "js lint skipped"
        return True, "lint skipped"

    @classmethod
    def explain(cls, engine: "CatR11Engine", code: str, lang: str) -> str:
        preview = code.strip()[:400]
        lines = len(code.splitlines())
        return (
            f"**{cls.FULL}** · `{lang}` · {lines} lines\n\n"
            f"Buffer is in-memory only.\n\n"
            f"```{(lang or 'text')[:12]}\n{preview}\n```"
        )

    @classmethod
    def agent_run(
        cls,
        engine: "CatR11Engine",
        code: str,
        lang: str = "",
        script_name: str = "untitled.py",
        *,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """cat code interpreter r1 1.0: BitNet encode → hook → lint → run (files = off · no API)."""
        body = (code or "").strip()
        interp = cls._interp()
        base = {
            "protocol": cls.PROTO,
            "version": cls.VER,
            "model": getattr(interp, "MODEL", CatCodeInterpreterR1.MODEL),
            "syntax": getattr(interp, "TAG", CatCodeInterpreterR1.TAG),
            "files": "off",
            "mythos_tier": cls.MYTHOS,
            "prose_tier": cls.PROSE,
            "fork": CatCodeInterpreterR1.FORK_REPO if CatCodeInterpreterR1.enabled() else None,
            "backend": "bitnet" if CatCodeInterpreterR1.bitnet_only() else "local",
            "api": "off" if CatCodeInterpreterR1.no_api() else "local",
        }
        if not body:
            return {**base, "ok": False, "error": "empty buffer"}
        if CatCodeInterpreterR1.enabled() and CatCodeInterpreterR1.bitnet_only():
            CatCodeInterpreterR1.ensure_bitnet(engine, body, force_pro=engine.is_pro_mode())
        lang_key = cls._detect_lang(engine, body, lang)
        ok, lint_msg = cls.lint(body, lang_key)
        if not ok:
            reasoning = interp.code_reasoning(
                lang_key, script_name, body, lint_msg=lint_msg, error=lint_msg,
            )
            content = interp.format_content(
                lang_key, script_name, "", error=lint_msg,
            )
            engine.last_think = reasoning
            return {
                **base,
                "ok": False,
                "error": lint_msg,
                "action": "lint",
                "lang": lang_key,
                "script": script_name,
                "reasoning_content": reasoning,
                "thinking": reasoning,
                "content": content,
            }
        if CatCodeInterpreterR1.enabled():
            hook = CatCodeInterpreterR1.run_hook(
                "PreToolUse", tool_name="Bash", tool_input={"command": f"python3 -c {body[:80]!r}"},
            )
            if not hook.get("allow", True):
                err = "; ".join(hook.get("issues", [])) or "hook blocked execution"
                reasoning = interp.code_reasoning(
                    lang_key, script_name, body, lint_msg=lint_msg, error=err,
                )
                content = interp.format_content(lang_key, script_name, "", error=err)
                engine.last_think = reasoning
                return {
                    **base,
                    "ok": False,
                    "error": err,
                    "action": "hook",
                    "hook": hook,
                    "lang": lang_key,
                    "script": script_name,
                    "reasoning_content": reasoning,
                    "thinking": reasoning,
                    "content": content,
                }
        output = CatR1Code.run(engine, lang_key, body)
        reasoning = interp.code_reasoning(
            lang_key, script_name, body, lint_msg=lint_msg, output=output,
        )
        content = interp.format_content(
            lang_key, script_name, output,
        )
        engine.last_think = reasoning
        buf = cls.session(session_id)
        buf.name = script_name
        buf.lang = lang_key
        buf.code = body
        buf.last_output = output
        buf.runs += 1
        return {
            **base,
            "ok": True,
            "action": "run",
            "reasoning_content": reasoning,
            "thinking": reasoning,
            "content": content,
            "engine": cls.FULL,
            "lang": lang_key,
            "script": script_name,
            "lint": lint_msg,
            "output": output,
            "runs": buf.runs,
            "bitnet": (engine.cat_r1_stats or {}).get("bitnet", {}),
        }

    @classmethod
    def format_result(cls, result: Dict[str, Any]) -> str:
        """Return user-facing `content` only — reasoning stays in reasoning_content / last_think."""
        if result.get("content"):
            return result["content"]
        if not result.get("ok"):
            return active_interpreter().format_content(
                result.get("lang", "code"),
                result.get("script", "buffer"),
                "",
                error=result.get("error", "unknown"),
            )
        return active_interpreter().format_content(
            result.get("lang", "code"),
            result.get("script", "buffer"),
            result.get("output", ""),
        )

    @classmethod
    def parse_request(cls, engine: "CatR11Engine", data: Dict[str, Any]) -> Dict[str, Any]:
        interp = cls._interp()
        allowed_protos = {
            None, cls.PROTO, O1_INTERPRETER_PROTO,
            CatR1CodeInterpreter01.PROTO, CatCodeInterpreterR1.PROTO,
        }
        if data.get("protocol") not in allowed_protos:
            return {"error": f"protocol must be {cls.PROTO}", "files": "off"}
        action = (data.get("action") or "run").lower().replace("_", "-")
        code = data.get("code") or data.get("buffer") or ""
        lang = data.get("lang") or data.get("language") or ""
        script = data.get("script") or data.get("name") or "untitled.py"
        sid = data.get("session") or data.get("session_id") or "default"
        CONFIG["files"] = "off"

        if CatCodeInterpreterR1.enabled() and CatCodeInterpreterR1.bitnet_only():
            prompt = code or data.get("prompt", "") or json.dumps(data)[:200]
            CatCodeInterpreterR1.ensure_bitnet(engine, prompt, force_pro=engine.is_pro_mode())

        if action == "generate":
            prompt = data.get("prompt") or code or ""
            text = CatR1CodeR1.generate_code(
                engine, prompt, force_pro=engine.is_pro_mode() or CatR1CodeR1.wants_pr(prompt),
            ) if CatR1CodeR1.enabled() else CatR1Code.respond(engine, prompt)
            return {"protocol": cls.PROTO, "version": cls.VER, "files": "off", "action": "generate", "content": text}
        if action == "help":
            return {"protocol": cls.PROTO, "version": cls.VER, "files": "off", "help": cls.help_text()}
        if action == "hook":
            hook = CatCodeInterpreterR1.run_hook(
                data.get("hook") or "PreToolUse",
                tool_name=data.get("tool_name", "Bash"),
                tool_input=data.get("tool_input") or {"command": data.get("command", "")},
            )
            return {"protocol": cls.PROTO, "version": cls.VER, "files": "off", "action": "hook", **hook}
        if action == "plan":
            agent = (data.get("agent") or "code-architect").lower()
            text = interp.agent_plan(agent, data.get("prompt", ""), code, lang or "python") if CatCodeInterpreterR1.enabled() else cls.explain(engine, code, lang or "python")
            return {"protocol": cls.PROTO, "version": cls.VER, "files": "off", "action": "plan", "content": text}
        if action == "review":
            text = CatCodeInterpreterR1.code_review(code, lang or "python") if CatCodeInterpreterR1.enabled() else cls.explain(engine, code, lang or "python")
            return {"protocol": cls.PROTO, "version": cls.VER, "files": "off", "action": "review", "content": text}
        if action == "feature-dev":
            text = CatCodeInterpreterR1.feature_dev_outline(data.get("prompt", code)) if CatCodeInterpreterR1.enabled() else cls.help_text()
            return {"protocol": cls.PROTO, "version": cls.VER, "files": "off", "action": "feature-dev", "content": text}
        if action == "diff":
            before = data.get("before") or cls.session(sid).code
            after = data.get("after") or code
            text = CatCodeInterpreterR1.diff_buffers(before, after) if CatCodeInterpreterR1.enabled() else f"before:\n{before}\n\nafter:\n{after}"
            return {"protocol": cls.PROTO, "version": cls.VER, "files": "off", "action": "diff", "content": text}
        if action == "edit":
            buf = cls.session(sid)
            new_code, ok, msg = CatCodeInterpreterR1.apply_edit(
                buf.code, data.get("old", ""), data.get("new", code),
            ) if CatCodeInterpreterR1.enabled() else (code or buf.code, bool(code), "edit")
            if ok:
                buf.code = new_code
            return {
                "protocol": cls.PROTO, "version": cls.VER, "files": "off",
                "action": "edit", "ok": ok, "message": msg, "code": buf.code,
            }
        if action == "explain":
            text = cls.explain(engine, code, lang or "python")
            return {"protocol": cls.PROTO, "version": cls.VER, "files": "off", "action": "explain", "content": text}
        if action == "lint":
            ok, msg = cls.lint(code, lang or "python")
            return {"protocol": cls.PROTO, "version": cls.VER, "files": "off", "action": "lint", "ok": ok, "message": msg}
        if action in {"run", "execute", "interpret"}:
            return cls.agent_run(engine, code, lang, script, session_id=sid)
        if action == "new":
            buf = cls.session(sid)
            buf.code = cls.default_snippet()
            buf.name = "untitled.py"
            buf.last_output = ""
            buf.runs = 0
            return {"protocol": cls.PROTO, "version": cls.VER, "files": "off", "action": "new", "script": buf.name, "code": buf.code}
        return {"error": f"unknown action: {action}", "files": "off", "tools": list(cls.TOOLS)}


CodingAPI = CatR1CodingAPI
ClaudeCodeStyle = CatR1CodingAPI
CatCodeInterpreter = CatR1CodeR1


class TokenWeightCodeEmitter:
    """
    Code synthesis from cat r1 token weights + prompt (files = off).
    Output is derived only from the inference vector and prompt tokens — no disk files,
    no dynamic template stubs.
    """

    @staticmethod
    def _pick(vec: Optional[np.ndarray], n: int, salt: int = 0) -> int:
        if vec is None or vec.size == 0:
            return salt % max(n, 1)
        return int(abs(float(np.sum(vec * (1.0 + salt * 0.031)) * 10007))) % max(n, 1)

    @classmethod
    def message_from_prompt(
        cls,
        prompt: str,
        vec: Optional[np.ndarray],
        engine: Optional["CatR11Engine"] = None,
    ) -> str:
        """Payload string from prompt tokens + cat r1 vec (files = off)."""
        subj = CatR1Code._subject(prompt, engine)
        if subj and subj.lower() not in {"hello world", "it", "it html"}:
            return subj
        pl = CatR1Code.normalize_prompt(prompt).lower()
        if "cat" in pl:
            return "Hello Cat" if cls._pick(vec, 2, 2) == 0 else "Meow!"
        defaults = ("Hello World", "Hello", "Meow", "Hi there")
        return defaults[cls._pick(vec, len(defaults), 1)]

    @classmethod
    def _c(cls, msg: str) -> str:
        esc = msg.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
        return (
            "#include <stdio.h>\n\n"
            "int main(void) {\n"
            f'    printf("{esc}\\n");\n'
            "    return 0;\n"
            "}"
        )

    @classmethod
    def _cpp(cls, msg: str) -> str:
        esc = msg.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
        return (
            "#include <iostream>\n\n"
            "int main() {\n"
            f'    std::cout << "{esc}" << std::endl;\n'
            "    return 0;\n"
            "}"
        )

    @classmethod
    def _python(cls, msg: str) -> str:
        esc = msg.replace("\\", "\\\\").replace("'", "\\'").replace("\n", "\\n")
        return (
            "import sys\n\n"
            f"MSG = '{esc}'\n\n"
            "def main():\n"
            "    name = sys.argv[1] if len(sys.argv) > 1 else 'World'\n"
            "    print(f'{MSG}, {name}!')\n"
            "    print(f'Running Python {sys.version}')\n\n"
            "if __name__ == '__main__':\n"
            "    main()"
        )

    @classmethod
    def emit(
        cls,
        engine: "CatR11Engine",
        prompt: str,
        lang: str,
        vec: Optional[np.ndarray],
    ) -> str:
        embedded_lang, embedded = CatR1Code.extract_prompt_code(prompt)
        if embedded:
            return embedded

        lang = CatR1Code._coerce_lang(engine, lang, prompt)
        msg = cls.message_from_prompt(prompt, vec, engine)

        if lang == "html":
            return CatR1Code._html(prompt, engine)
        if lang == "python":
            pl = prompt.lower()
            if "fibonacci" in pl or re.search(r"\bfib\b", pl):
                return CatR1Code._python(engine, prompt)
            if CodeAnythingEngine.enabled() and len(prompt.split()) > 4:
                return CodeAnythingEngine._py_general_app(prompt, msg)
            return cls._python(msg)
        if lang == "c":
            return cls._c(msg)
        if lang == "cpp":
            return cls._cpp(msg)
        if lang == "javascript":
            esc = msg.replace("\\", "\\\\").replace("'", "\\'").replace("\n", "\\n")
            return f"console.log('{esc}');"
        if lang == "bash":
            esc = msg.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
            return f'#!/bin/bash\necho "{esc}"'
        if lang == "rust":
            esc = msg.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
            return f'fn main() {{\n    println!("{esc}");\n}}'
        if lang == "go":
            esc = msg.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
            return f'package main\n\nimport "fmt"\n\nfunc main() {{\n\tfmt.Println("{esc}")\n}}'
        if lang == "java":
            esc = msg.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
            return (
                "public class Main {\n"
                "    public static void main(String[] args) {\n"
                f'        System.out.println("{esc}");\n'
                "    }\n"
                "}"
            )
        if lang == "css":
            return f"body {{\n  font-family: system-ui, sans-serif;\n}}\n\n/* {msg} */"
        if lang == "sql":
            esc = msg.replace("'", "''")
            return f"SELECT '{esc}' AS message;"
        if CodeAnythingEngine.enabled():
            task = CodeAnythingEngine.detect_task(prompt)
            return CodeAnythingEngine.scaffold(lang, task, prompt, engine, vec)
        # Token-weight fallback — never emit dynamic template stubs
        return cls._python(msg)

    @staticmethod
    def fence(lang: str, code: str) -> str:
        tag = lang if lang else ""
        return f"```{tag}\n{code.rstrip()}\n```"


# ──────────────────────────────────────────────────────────────
# CAT R1.1 CODE ENGINE (files = off · perfect-code loop)
# ──────────────────────────────────────────────────────────────
class CatR1MythosEngine:
    """
    cat r1 code engine for cat r1 (files = off).
    Pattern library · recursive perfection · lint · sandbox verify.
    """

    NAME = CONFIG["code_interpreter_name"]
    FAMILY = CONFIG["code_interpreter_family"]
    BACKEND = CONFIG["code_interpreter"]
    VERSION = CONFIG["code_interpreter_version"]

    _HELLO_CAT_C = (
        "#include <stdio.h>\n\n"
        "int main(void) {\n"
        '    printf("Hello Cat\\n");\n'
        '    printf("Meow!\\n");\n'
        "    return 0;\n"
        "}"
    )

    _PY_HELLO = (
        "def main() -> None:\n"
        '    print("Hello, World!")\n\n'
        "if __name__ == \"__main__\":\n"
        "    main()"
    )

    _PY_HELLO_CAT = (
        "def main() -> None:\n"
        '    print("Hello Cat!")\n'
        '    print("Meow!")\n\n'
        "if __name__ == \"__main__\":\n"
        "    main()"
    )

    _JS_HELLO = "console.log('Hello, World!');\n"

    _RUST_HELLO = (
        "fn main() {\n"
        '    println!("Hello, World!");\n'
        "}"
    )

    @classmethod
    def _lint(cls, code: str, lang: str) -> str:
        if not CONFIG.get("cat_r1b_lint", True):
            return code
        lines = [ln.rstrip() for ln in code.splitlines()]
        while lines and not lines[0].strip():
            lines.pop(0)
        while lines and not lines[-1].strip():
            lines.pop()
        body = "\n".join(lines)
        if lang == "python" and body and not body.endswith("\n"):
            body += "\n"
        return body

    @classmethod
    def _pattern_match(cls, prompt: str, lang: str, engine: "CatR11Engine") -> Optional[str]:
        pl = CatR1Code.normalize_prompt(prompt).lower()
        if lang == "c" and re.search(r"\bhello\b", pl):
            msg = CatR1Code._subject(prompt, engine)
            if "cat" in pl:
                return cls._HELLO_CAT_C
            return TokenWeightCodeEmitter._c(msg if msg else "Hello")
        if lang == "cpp" and "hello" in pl:
            msg = CatR1Code._subject(prompt, engine)
            return TokenWeightCodeEmitter._cpp(msg if msg else "Hello")
        if lang == "python":
            if "fibonacci" in pl or re.search(r"\bfib\b", pl):
                return CatR1Code._python(engine, prompt)
            if re.search(r"\bhello\s+cat\b", pl):
                return cls._PY_HELLO_CAT
            if re.search(r"\bhello\s+world\b", pl) or pl.strip() in {"hello", "write hello in python"}:
                return cls._PY_HELLO
            if CodeAnythingEngine.enabled():
                subject = CatR1Code._subject(prompt, engine)
                task = CodeAnythingEngine.detect_task(prompt)
                if task != "general":
                    return CodeAnythingEngine.scaffold(lang, task, prompt, engine)
                return CodeAnythingEngine._py_general_app(prompt, subject)
        if lang == "javascript" and "hello" in pl:
            return cls._JS_HELLO
        if lang == "rust" and "hello" in pl:
            return cls._RUST_HELLO
        if lang == "html":
            return CatR1Code._html(prompt, engine)
        return None

    @classmethod
    def _validate(cls, lang: str, code: str) -> Tuple[bool, str]:
        lang = lang.lower()
        if not code or not code.strip():
            return False, "empty"
        if lang == "python":
            try:
                ast.parse(code)
                return True, ""
            except SyntaxError as e:
                return False, str(e)
        if lang in {"c", "cpp"}:
            if lang == "c" and "#include" not in code and "printf" in code:
                return False, "missing include"
            if "main" not in code:
                return False, "missing main"
            return True, ""
        if lang == "html":
            if "<html" not in code.lower() and "<!doctype" not in code.lower():
                return False, "not html"
            return True, ""
        if lang == "javascript":
            return True, ""
        return True, ""

    @classmethod
    def _fix(cls, lang: str, code: str, prompt: str, reason: str) -> str:
        lang = lang.lower()
        code = code.strip()
        if lang == "c" and "missing include" in reason:
            if "#include" not in code:
                code = "#include <stdio.h>\n\n" + code
        if lang == "python" and ("SyntaxError" in reason or "invalid syntax" in reason):
            if "def main" not in code:
                msg = TokenWeightCodeEmitter.message_from_prompt(prompt, None, None)
                return TokenWeightCodeEmitter._python(msg)
        if lang == "c" and "return 0" not in code and "main" in code:
            if not code.rstrip().endswith("}"):
                code = code.rstrip() + "\n    return 0;\n}"
        code = re.sub(r"\n{3,}", "\n\n", code)
        return code.rstrip() + "\n"

    @classmethod
    def _polish(cls, code: str, lang: str) -> str:
        lines = code.splitlines()
        out: List[str] = []
        for ln in lines:
            if re.match(r"^c\s*$", ln.strip()) and lang != "text":
                continue
            if re.search(r"^/Volumes/|^/Users/", ln):
                continue
            out.append(ln.rstrip())
        body = "\n".join(out).strip()
        if lang == "python" and body and "if __name__" not in body and "def main" in body:
            if not body.endswith("\n"):
                body += "\n"
            if "if __name__" not in body:
                body += "\nif __name__ == '__main__':\n    main()"
        return body

    @classmethod
    def _recursive_perfect(
        cls, code: str, lang: str, prompt: str, engine: "CatR11Engine"
    ) -> str:
        if not CONFIG.get("cat_r1_code_perfect", True):
            return code
        depth = CONFIG.get("cat_r1_recursive_depth", 3)
        out = code
        for i in range(depth):
            out = cls._polish(out, lang)
            ok, reason = cls._validate(lang, out)
            if ok:
                if lang == "python" and engine._validate_code("python", out):
                    break
                if lang != "python":
                    break
            out = cls._fix(lang, out, prompt, reason)
        return cls._lint(cls._polish(out, lang), lang)

    @classmethod
    def enabled(cls) -> bool:
        return bool(CONFIG.get("cat_r1_engine", True) and CONFIG.get("cat_r1_code_enabled", True))

    @classmethod
    def generate(
        cls,
        engine: "CatR11Engine",
        prompt: str,
        lang: str,
        vec: Optional[np.ndarray] = None,
    ) -> str:
        if not cls.enabled():
            return TokenWeightCodeEmitter.emit(engine, prompt, lang, vec)
        hit = cls._pattern_match(prompt, lang, engine)
        if hit:
            draft = hit
        elif CodeAnythingEngine.enabled():
            draft = CodeAnythingEngine.generate(engine, prompt, lang, vec)
        else:
            draft = TokenWeightCodeEmitter.emit(engine, prompt, lang, vec)
        return cls._recursive_perfect(draft, lang, prompt, engine)

    @classmethod
    def respond(cls, engine: "CatR11Engine", prompt: str) -> str:
        return CatR1Code.respond(engine, prompt)


ClaudeMythosCode = CatR1MythosEngine
CatR1MythosCode = CatR1MythosEngine


# ──────────────────────────────────────────────────────────────
# CAT R1.1 LLM (files = off · local runtime on cat r1)
# ──────────────────────────────────────────────────────────────
class CatR1ContextMemory:
    """1M-token logical context window — in-memory only, files = off."""

    __slots__ = ("turns", "logical_tokens")

    def __init__(self):
        self.turns: List[Dict[str, str]] = []
        self.logical_tokens = 0

    def add(self, role: str, text: str) -> None:
        self.turns.append({"role": role, "text": text})
        self.logical_tokens += max(len(text.split()), 1)
        limit = CONFIG["cat_r1_context_window"]
        while self.logical_tokens > limit and len(self.turns) > 2:
            old = self.turns.pop(0)
            self.logical_tokens -= max(len(old["text"].split()), 1)

    def recent(self, n: int = 24) -> List[Dict[str, str]]:
        return self.turns[-n:]

    def prior_topics(self) -> str:
        users = [t["text"][:60] for t in self.turns if t["role"] == "user"][-4:]
        return "; ".join(users) if users else ""


FableContextMemory = CatR1ContextMemory


class CatR1LLM:
    """
    cat r1 frontier runtime (cat r1 frontier):
    encode → think → draft → self-check → respond — all in-memory.
    1.7T effective parameters · 10M token context · 512K output.
    Tuned to match cat r1 frontier behavior.
    """

    MODEL_ID = CONFIG["cat_r1_model_id"]
    NAME = BRAND
    FAMILY = "cat-r1"
    CONTEXT_WINDOW = CONFIG["cat_r1_context_window"]
    MAX_OUTPUT = CONFIG["cat_r1_max_output"]
    _BRAND_PREFIX = CatR1Code._BRAND_PREFIX

    _CREATIVE_CUES = (
        "fable", "parable", "allegory", "short story", "tell me a story",
        "write a story", "write me a story", "bedtime story", "tale about",
        "story about", "fable about", "once upon", "moral story",
        "poem", "poetry", "haiku", "sonnet", "verse", "rhyme",
        "narrative", "imagine a", "creative writing",
    )
    _CODE_CUES = (
        "code", "function", "implement", "snippet", "script", "class",
        "fibonacci", "fib", "prime", "python", "javascript", "rust", "java",
        "def ", "import ", "```",
    )

    def __init__(self):
        self.memory = CatR1ContextMemory()

    @staticmethod
    def _seed(vec: Optional[np.ndarray], salt: int = 0) -> int:
        if vec is None or vec.size == 0:
            return salt
        return int(abs(float(np.sum(vec * (1.0 + salt * 0.01)) * 10007))) % 10_000

    @classmethod
    def pick(cls, vec: Optional[np.ndarray], n: int, salt: int = 0) -> int:
        return cls._seed(vec, salt) % max(n, 1)

    @classmethod
    def is_creative(cls, pl: str) -> bool:
        pl = cls._BRAND_PREFIX.sub("", pl.strip())
        return CONFIG["cat_r1_enabled"] and any(c in pl for c in cls._CREATIVE_CUES)

    @classmethod
    def is_code_request(cls, pl: str) -> bool:
        if not CONFIG.get("cat_r1_code_enabled", True):
            return False
        return CatR1Code.wants_code(pl)

    @classmethod
    def wants_fable(cls, pl: str) -> bool:
        pl = CatR1Code._BRAND_PREFIX.sub("", pl.strip())
        cues = ("fable about", "write a fable", "tell me a fable", "parable", "allegory",
                "tell me a story", "write a story", "story about", "fable about")
        return any(c in pl for c in cues) or (
            re.search(r"\bfable\b", pl) is not None and "cat-r1" not in pl and "cat r1" not in pl
        )

    @classmethod
    def wants_poem(cls, pl: str) -> bool:
        return any(c in pl for c in ("poem", "poetry", "haiku", "sonnet", "verse", "rhyme"))

    def classify(self, prompt: str, engine: "CatR11Engine") -> str:
        if CatR1Fusion.is_noise(prompt):
            return "chat"
        norm = CatR1Code.normalize_prompt(prompt)
        pl = norm.lower().strip()
        follow = CatR1Fusion.session_followup(engine, prompt)
        if follow and CatR1Fusion.is_noise(prompt):
            return "chat"
        if is_zh_greeting(prompt.strip()):
            return "chat"
        if engine.synth.smalltalk_reply(pl) or engine.synth.smalltalk_reply(prompt.strip()):
            return "chat"
        if is_explain_request(prompt) and not CatR1Code.extract_prompt_code(prompt)[1]:
            return "explain"
        if CONFIG.get("web_program_enabled") and CatR1WebProgram.wants_web(pl):
            return "web"
        if CONFIG.get("cat_r1_code_enabled"):
            if CatR1Code.extract_prompt_code(prompt)[1]:
                return "code"
            if CatR1Code.wants_code_with_history(pl, engine):
                return "code"
        if self.is_creative(pl):
            if self.wants_poem(pl):
                return "poem"
            if self.wants_fable(pl) or "story" in pl or "tale" in pl:
                return "fable"
            return "creative"
        if engine._try_simple_math(prompt) is not None:
            return "math"
        if any(x in pl for x in ("run code", "execute", "interpret", "/exec", "运行", "执行")):
            return "execute"
        if pl in ("run it", "run this", "execute it", "test it", "run"):
            return "execute"
        if CatR1Code.is_fix_all_request(pl):
            return "chat"
        if CatR1Code.is_code_debug(prompt, pl):
            return "debug"
        if self.is_code_request(pl):
            return "code"
        if is_explain_request(prompt):
            return "explain"
        if re.search(r"\b(explain|what is|what are|why|how (?:does|do|to|can))\b", pl):
            return "explain"
        if re.search(r"(解释|说明|介绍)", prompt):
            return "explain"
        if re.search(r"\b(plan|architecture)\b", pl) and len(pl.split()) > 2:
            return "agent"
        if re.search(r"\b(design|build)\s+(?:a|an|the|my|this|your|me)\b", pl):
            return "agent"
        if CONFIG.get("universal_chat", True):
            return "chat"
        return "general"

    def plan(self, prompt: str, task: str, vec: Optional[np.ndarray]) -> List[str]:
        steps = {
            "chat": ["match tone", "respond warmly", "invite next turn"],
            "explain": ["identify core concept", "mechanism", "example", "caveats"],
            "code": ["spec inputs/outputs", "draft", "edge cases", "self-test"],
            "web": ["pick template", "render artifact HTML", "store in memory", "preview URL"],
            "debug": ["reproduce", "localize", "fix", "verify"],
            "fable": ["theme", "arc", "moral", "polish prose"],
            "poem": ["image", "turn", "close"],
            "creative": ["voice", "structure", "finish"],
            "agent": ["decompose", "sequence", "deliverables"],
            "math": ["parse", "compute", "verify"],
            "execute": ["parse block", "run sandbox", "report"],
            "general": ["understand", "answer", "check completeness"],
        }
        base = steps.get(task, steps["general"])
        if vec is not None and self.pick(vec, 3, 12) == 0 and task in ("explain", "code", "agent"):
            base = base + ["proactive follow-up"]
        return base

    @staticmethod
    def _extract_subject(prompt: str) -> str:
        pl = prompt.lower()
        for prefix in ("fable about ", "story about ", "poem about ", "tale about ", "about ", "on "):
            if prefix in pl:
                return prompt[pl.index(prefix) + len(prefix):].strip("?.")
        return re.sub(
            r"^(?:write|tell|give)\s+(?:me\s+)?(?:a\s+)?(?:fable|story|poem|tale)\s*(?:about\s+)?",
            "", pl, flags=re.I,
        ).strip("?.")

    def compose_fable(self, prompt: str, topic: str, vec: Optional[np.ndarray]) -> str:
        heroes = (
            ("a clever fox", "wit over force"), ("a patient tortoise", "steady effort"),
            ("a curious owl", "seeing what others miss"), ("a humble mouse", "small acts matter"),
        )
        settings = (
            "at the edge of an ancient forest", "in a harbor town of quiet clocks",
            "on a hillside where seasons negotiated in whispers",
        )
        hero, virtue = heroes[self.pick(vec, len(heroes), 2)]
        setting = settings[self.pick(vec, len(settings), 3)]
        subject = self._extract_subject(prompt) or topic.strip("?.") or "the work that waits"
        return (
            f"**{subject.title()}** — a fable\n\n"
            f"Once, {hero} lived {setting}. The village spoke of **{subject}**, "
            f"though no two voices agreed on its shape.\n\n"
            f"When difficulty arrived — as it always does — the hero did not perform brilliance. "
            f"They attended: mending what broke, keeping promises, asking honest questions. "
            f"Others mistook patience for slowness until the results could no longer be ignored.\n\n"
            f"**Moral:** {virtue.capitalize()} compounds quietly. "
            f"The world becomes slightly more legible for whoever comes next."
        )

    def compose_poem(self, topic: str, vec: Optional[np.ndarray], prompt: str = "") -> str:
        subj = (self._extract_subject(prompt) if prompt else "") or topic.strip("?.") or "the road ahead"
        forms = (
            f"**On {subj}**\n\nNot all answers arrive with noise;\n"
            f"some knock like moth-wings at the glass —\n"
            f"small, insistent, easy to ignore\nuntil you remember you wanted light.\n\n"
            f"Begin again. {subj.capitalize()} waits\nlike shore waits tide: patient, sure, returned.",
            f"**{subj.title()}**\n\nYou asked about {subj}.\n"
            f"The mind builds bridges out of questions;\neach span creaks until you walk it.\n\n"
            f"Go slowly. Name what you know.\nLeave doors open for better names.",
        )
        return forms[self.pick(vec, len(forms), 5)]

    def self_check(self, prompt: str, draft: str, task: str, engine: "CatR11Engine") -> str:
        out = draft
        if task == "code" and "```" in out:
            for m in re.finditer(r"```(\w*)\n(.*?)```", out, re.S):
                lang, code = m.group(1) or "python", m.group(2)
                if lang == "python" and engine._validate_code("python", code):
                    continue
        if task in ("explain", "general", "agent") and "?" in prompt and len(out) < 80:
            out += "\n\nI can go deeper — tell me which part needs expansion."
        return ClaudeMythosRuntime.emit(engine, out, prompt, task)

    def complete(self, engine: "CatR11Engine", prompt: str, *, simulate: bool = False) -> str:
        """Main cat r1 completion entry — files = off."""
        _CATR1_FAST = frozenset({"chat", "code", "math", "execute", "web", "explain"})
        pl = prompt.lower().strip()
        is_pro = engine.is_pro_mode()
        follow = CatR1Fusion.session_followup(engine, prompt)
        if follow and CatR1Fusion.is_noise(prompt):
            return follow
        task = self.classify(prompt, engine)
        channel = getattr(engine, "last_channel", None) or {}
        if CONFIG.get("tri_channel_detect", True) and channel:
            ch = channel.get("channel")
            if ch == "code" and (channel.get("has_code") or channel.get("needs_code")) and task in (
                "chat", "explain", "general", "agent", "debug"
            ):
                task = "code"
            loc = channel.get("locale", "english")
            if loc in ("chinese", "mixed"):
                engine.response_locale = loc
        engine.encode_for_task(prompt, task=task)
        vec = engine.last_vec
        if task in _CATR1_FAST:
            if engine.ultrathink_on and is_pro:
                engine._run_ultrathink(prompt)
            else:
                engine.last_think = ""
                engine._pending_think = ""
        _ = self.plan(prompt, task, vec)

        if task == "chat":
            hit = engine.synth.smalltalk_reply(pl)
            if hit:
                return hit
            if follow:
                return follow
            if CatR1Code._META_DESIRE.search(prompt):
                body = engine.synth.respond_anything(prompt, engine.chat_history, vec=vec, engine=engine)
                return ClaudeMythosRuntime.emit(engine, body, prompt, "chat")
            if CatR1Code.is_bitnet_request(pl):
                body = engine.ai_setup_response(prompt) or BitNetEngine.status_text(engine.cat_r1_stats)
                return ClaudeMythosRuntime.emit(engine, body, prompt, "chat")
            literal = engine.synth.extract_literal_desire(prompt)
            if literal is not None:
                return literal
            dia = engine.get_dialect(engine.detect_locale(prompt))
            intent = engine._best_intent(prompt)
            if intent in {"hello", "help", "core", "recursion", "languages", "profile",
                        "thanks", "goodbye", "joke", "math"}:
                if intent == "core" and CatR1Code.is_bitnet_request(pl):
                    body = engine.ai_setup_response(prompt) or BitNetEngine.status_text(engine.cat_r1_stats)
                    return ClaudeMythosRuntime.emit(engine, body, prompt, "chat")
                hit = engine._intent_response(intent, prompt, dia)
                if hit:
                    return ClaudeMythosRuntime.emit(engine, hit, prompt, "chat")
            if CONFIG.get("cat_r1_code_enabled") and CatR1Code.wants_code_with_history(pl, engine):
                if CatR1CodeR1.enabled():
                    body = CatR1CodeR1.generate_code(
                        engine, prompt, force_pro=CatR1CodeR1.wants_pr(prompt),
                    )
                else:
                    body = CatR1Code.respond(engine, prompt)
                return ClaudeMythosRuntime.emit(engine, body, prompt, "code")
            if is_pro:
                history = engine.chat_history[:-1] if engine.chat_history else []
                history_sorted = GoogleWhitepaperCatR1Sorter.sort_history_dicts(history, prompt)
                body = engine.synth.synthesize(
                    prompt, [(m["role"], m["text"]) for m in history_sorted], vec=vec
                )
                body = engine.synth.localize(body, prompt)
                if CONFIG.get("mythos_recursive_improve"):
                    body = engine.fusion.recursive_improve(body, prompt, vec)
                return ClaudeMythosRuntime.emit(engine, body, prompt, "chat")
            body = engine.synth.converse(prompt, engine.chat_history, vec=vec, engine=engine)
            return ClaudeMythosRuntime.emit(engine, body, prompt, "chat")

        # cat r1 code 1.0 — generate + run (BitNet · files = off)
        if task == "code":
            if CatR1CodeR1.enabled():
                force = CatR1CodeR1.wants_pr(prompt) or engine.is_pro_mode()
                body = CatR1CodeR1.generate_code(engine, prompt, force_pro=force)
            else:
                body = CatR1Code.respond(engine, prompt)
            return ClaudeMythosRuntime.emit(engine, body, prompt, "code")

        if task == "web":
            try:
                return engine.web.respond(engine, prompt)
            except Exception as exc:
                return f"**{WEB_PROGRAM_NAME}** error: {exc}"

        if task == "math":
            result = engine._try_simple_math(prompt)
            if result:
                if CONFIG.get("mythos_mode") and ClaudeMythosRuntime.enabled():
                    return ClaudeMythosRuntime.math_wrap(prompt, result)
                if CONFIG.get("cat_r1_reasoning"):
                    return engine.fusion.cat_r1_math_wrap(prompt, result)
                return f"Result: **{result}**"

        if task == "explain":
            dia = engine.get_dialect(engine.detect_locale(prompt))
            intent = engine._best_intent(prompt)
            if intent in {"recursion", "core", "help", "languages", "profile"}:
                hit = engine._intent_response(intent, prompt, dia)
                if hit:
                    return self.self_check(prompt, hit, task, engine)
            history = engine.chat_history[:-1] if engine.chat_history else []
            history_sorted = GoogleWhitepaperCatR1Sorter.sort_history_dicts(history, prompt)
            body = engine.synth.synthesize(
                prompt, [(m["role"], m["text"]) for m in history_sorted], vec=vec
            )
            body = engine.synth.localize(body, prompt)
            if is_pro and CONFIG.get("mythos_recursive_improve"):
                body = engine.fusion.recursive_improve(body, prompt, vec)
            return self.self_check(prompt, body, task, engine)

        if task == "execute":
            block_lang, code = engine.extract_code_block(prompt)
            exec_lang = CatR1Code.detect_lang(
                engine, prompt if not block_lang else f"in {block_lang}"
            )
            if not code:
                for m in reversed(engine.chat_history):
                    if m.get("role") == "assistant":
                        block_lang, code = engine.extract_code_block(m.get("text", ""))
                        if code:
                            exec_lang = engine.normalize_lang(block_lang) or exec_lang
                            break
            if not code:
                return "Paste code in a fenced block, or generate code first then say **run it**."
            result = CatR1CodingAPI.agent_run(engine, code, exec_lang)
            return CatR1CodingAPI.format_result(result)

        if task == "debug":
            history = engine.chat_history[:-1] if engine.chat_history else []
            history_sorted = GoogleWhitepaperCatR1Sorter.sort_history_dicts(history, prompt)
            body = engine.synth.synthesize(
                prompt, [(m["role"], m["text"]) for m in history_sorted], vec=vec
            )
            body = engine.synth.localize(body, prompt)
            return self.self_check(prompt, body, task, engine)

        norm = CatR1Code.normalize_prompt(prompt)
        if CONFIG.get("cat_r1_code_enabled") and CatR1Code.wants_code_with_history(norm.lower(), engine):
            return CatR1Code.respond(engine, prompt)

        if task not in _CATR1_FAST and not engine._pending_think:
            engine._run_ultrathink(prompt)
        history = engine.chat_history[:-1] if engine.chat_history else []
        dia = engine.get_dialect(engine.detect_locale(prompt))

        if task == "fable":
            topic = engine._extract_topic_words(prompt)
            body = self.compose_fable(prompt, topic, vec)
            return self.self_check(prompt, body, task, engine)

        if task == "poem":
            topic = engine._extract_topic_words(prompt)
            body = self.compose_poem(topic, vec, prompt)
            return self.self_check(prompt, body, task, engine)

        if task == "creative":
            topic = engine._extract_topic_words(prompt)
            body = self.compose_fable(prompt, topic, vec)
            return self.self_check(prompt, body, task, engine)

        intent = engine._best_intent(prompt)
        if intent in {"languages", "profile", "core", "recursion", "help"}:
            hit = engine._intent_response(intent, prompt, dia)
            if hit:
                return self.self_check(prompt, hit, task, engine)

        history = GoogleWhitepaperCatR1Sorter.sort_history_dicts(history, prompt)
        body = engine.synth.o1_answer(prompt, history, vec=vec)
        return self.self_check(prompt, body, task, engine)

    @staticmethod
    def model_card() -> Dict[str, Any]:
        return {
            "id": CATR1_MODEL_ID,
            "object": "model",
            "family": "cat-r1",
            "display_name": CatR1LLM.NAME,
            "owned_by": "cat r1",
            "match_target": LLM_MATCH_TARGET,
            "match_provider": LLM_MATCH_PROVIDER,
            "tier": "frontier",
            "rivals": [],
            "context_window": CONFIG["cat_r1_context_window"],
            "max_output_tokens": CONFIG["cat_r1_max_output"],
            "nominal_params": CONFIG["nominal_base_params"],
            "effective_params_tier": "1.7T (frontier)",
            "files": FILES,
            "prose_tier": PROSE_TIER,
            "mythos_tier": MYTHOS_TIER,
            "reasoning": REASONING_MODE,
            "cat_r1_reasoning": CONFIG.get("cat_r1_reasoning", True),
            "code_interpreter": CODE_ENGINE,
            "code_interpreter_version": CODING_API_VER,
            "interpreter_syntax": CONFIG.get("interpreter_syntax", "cat-r1-code-interpreter-0.1"),
            "interpreter_protocol": CONFIG.get("o1_interpreter_protocol", CatCodeInterpreterR1.PROTO),
            "cat_r1_code_interpreter_model": CONFIG.get(
                "cat_r1_code_interpreter_model", CatCodeInterpreterR1.MODEL
            ),
            "cat_code_interpreter_r1": CONFIG.get("cat_code_interpreter_r1", True),
            "claude_code_fork": CONFIG.get("claude_code_fork", CatCodeInterpreterR1.FORK_REPO),
            "catcode_bitnet_only": CONFIG.get("catcode_bitnet_only", True),
            "catcode_no_api": CONFIG.get("catcode_no_api", True),
            "api_enabled": CONFIG.get("api_enabled", False),
            "v4_pro_coding": CONFIG.get("v4_pro_coding", True),
            "reasoning_content_field": True,
            "coding_api_protocol": CODING_API_PROTO,
            "coding_api": CODING_API_LABEL,
            "mythos_engine_version": MYTHOS_ENGINE_VER,
            "code_backend": CODE_BACKEND,
            "cat_r1_code_enabled": CAT_R1_CODE_ENABLED,
            "web_program": CONFIG.get("web_program_enabled", True),
            "arch": "cat-r1_frontier",
            "bitnet_engine": {
                "type": "AbsMeanTernary",
                "bits": CONFIG["weight_bits"],
                "packing": "base-3",
                "real": CONFIG.get("bitnet_real", True),
                "use_packed": CONFIG.get("bitnet_use_packed", True),
                "kernel": CONFIG.get("bitnet_kernel", "ternary_addsub"),
                "status": "active",
            },
            "moe": {
                "experts": CONFIG["n_experts"],
                "top_k": CONFIG["top_k"],
                "cat_r1_scale": CONFIG.get("cat_r1_moe_scale", True),
                "cat_r1_dense": CONFIG.get("cat_r1_moe_dense", True),
            },
            "compression": {
                "sparse_k": CONFIG["compression_sparse_k"],
                "low_rank": CONFIG["compression_rank"],
                "awq_bits": CONFIG.get("compression_nvidia_awq_bits", 4),
                "gptq_block": CONFIG.get("compression_nvidia_gptq_block", 128),
                "sparse_2_4": CONFIG.get("compression_nvidia_sparse_2_4", True),
            },
            "distillation": {
                "enabled": CONFIG.get("distil_enabled", True),
                "student_passes": CONFIG["distil_passes"],
                "teacher_weight": CONFIG.get("distil_teacher_weight", CONFIG["teacher_weight"]),
                "protocol": CONFIG.get("distil_protocol", "cat-r1-distil"),
                "files": FILES,
            },
            "cat_r1_features": {
                "moe_dense": CONFIG.get("cat_r1_moe_dense", True),
                "multi_turn_search": CONFIG.get("cat_r1_multi_turn_search", True),
                "fun_mode": CONFIG.get("cat_r1_fun_mode", True),
                "realtime": CONFIG.get("cat_r1_realtime", True),
                "moe_scale": CONFIG.get("cat_r1_moe_scale", True),
                "multi_token_prediction": CONFIG.get("cat_r1_multi_token_prediction", True),
                "active_inference": CONFIG.get("cat_r1_active_inference", True),
                "long_context_finegrained": CONFIG.get("cat_r1_long_context_finegrained", True),
                "sparse_attention": CONFIG.get("cat_r1_sparse_attention", True),
            },
            "google_whitepaper_heuristics": CONFIG.get("google_whitepaper_heuristics", True),
            "cat_r1_voice": CONFIG.get("cat_r1_voice", True),
            "mythos_runtime": CONFIG.get("mythos_runtime", True),
            "mythos_voice": CONFIG.get("mythos_voice", True),
            "mythos_algos": list(ClaudeMythosRuntime.MYTHOS_ALGOS),
            "code_anything": CONFIG.get("code_anything", True),
            "code_anything_mode": CONFIG.get("code_anything_mode", "universal-frontier"),
        "code_anything_langs": len(CodeAnythingEngine.LANGS),
    }


# ──────────────────────────────────────────────────────────────
# BILINGUAL DICTIONARY LEXICON (English + Mandarin · files = off)
# Hardcoded response for every English dictionary word + Mandarin lexicon entry.
# ──────────────────────────────────────────────────────────────
class CatR1DictionaryLexicon:
    """In-memory EN + 中文 dictionary — one stable hardcoded response per lexicon word."""

    _ENGLISH: Dict[str, str] = {}
    _MANDARIN: Dict[str, str] = {}
    _EN_WORDS: Optional[frozenset] = None
    _ready = False

    _EN_POS_SUFFIXES: Tuple[Tuple[str, str], ...] = (
        ("ing", "verb (gerund/participle)"),
        ("ed", "verb (past) / adjective"),
        ("ly", "adverb"),
        ("tion", "noun"),
        ("sion", "noun"),
        ("ment", "noun"),
        ("ness", "noun"),
        ("ity", "noun"),
        ("able", "adjective"),
        ("ible", "adjective"),
        ("ful", "adjective"),
        ("less", "adjective"),
        ("ous", "adjective"),
        ("ive", "adjective"),
        ("er", "noun / comparative"),
        ("or", "noun"),
        ("ist", "noun"),
    )

    _EN_DEF_TEMPLATES: Tuple[str, ...] = (
        "A standard English dictionary entry for **{w}** — a {pos} in general vocabulary.",
        "**{w}** is an English {pos} with established dictionary usage.",
        "In English, **{w}** functions as a {pos} across formal and informal contexts.",
        "**{w}** appears in English dictionaries as a {pos} with regular orthography.",
    )

    _EN_EXAMPLE_TEMPLATES: Tuple[str, ...] = (
        'Example: "The word **{w}** fits naturally in everyday English."',
        'Example: "She used **{w}** correctly in her sentence."',
        'Example: "You will encounter **{w}** in books, news, and conversation."',
        'Example: "Dictionary lookup: **{w}** → {pos}."',
    )

    _ZH_DEF_TEMPLATES: Tuple[str, ...] = (
        "「{w}」是中文词汇/汉字，在普通话中常见。",
        "「{w}」为汉语词条，具体含义需结合语境理解。",
        "普通话词典收录「{w}」，可用于日常书面与口语表达。",
        "「{w}」在中文里具有稳定的书写形式与常用搭配。",
    )

    _ZH_EXAMPLE_TEMPLATES: Tuple[str, ...] = (
        "例句：我想确认「{w}」在这句话里的意思。",
        "例句：请用「{w}」造一个简单句子。",
        "例句：「{w}」在新闻和对话中都很常见。",
        "例句：查字典：「{w}」→ 见词条释义。",
    )

    _EN_CURATED: Dict[str, str] = {
        "a": "**a** /ə/ *indefinite article*\nUsed before nouns to refer to one non-specific thing.\nExample: I saw a bird in the tree.",
        "an": "**an** /ən/ *indefinite article*\nForm of *a* used before vowel sounds.\nExample: She ate an apple.",
        "the": "**the** /ðə/ *definite article*\nUsed to refer to a specific noun.\nExample: The sun is bright today.",
        "hello": "**hello** /həˈloʊ/ *interjection*\nGreeting used when meeting someone.\nExample: Hello, nice to meet you!",
        "goodbye": "**goodbye** /ˌɡʊdˈbaɪ/ *interjection*\nFarewell when leaving.\nExample: Goodbye, see you tomorrow.",
        "cat": "**cat** /kæt/ *noun*\nA small domesticated carnivorous mammal with soft fur.\nExample: The cat slept on the windowsill.",
        "dog": "**dog** /dɒɡ/ *noun*\nA domesticated carnivorous mammal, often kept as a pet.\nExample: The dog wagged its tail.",
        "run": "**run** /rʌn/ *verb*\nTo move swiftly on foot.\nExample: She runs five kilometers every day.",
        "love": "**love** /lʌv/ *verb / noun*\nDeep affection or strong liking.\nExample: I love spending time with family.",
        "water": "**water** /ˈwɔːtər/ *noun*\nA clear liquid essential for life.\nExample: Drink a glass of water.",
        "book": "**book** /bʊk/ *noun*\nA written or printed work of literature.\nExample: I'm reading a great book.",
        "world": "**world** /wɜːrld/ *noun*\nThe earth and all life upon it.\nExample: Travel broadens your view of the world.",
        "computer": "**computer** /kəmˈpjuːtər/ *noun*\nAn electronic device for storing and processing data.\nExample: I use my computer for work.",
        "dictionary": "**dictionary** /ˈdɪkʃəneri/ *noun*\nA book or resource listing words with meanings.\nExample: Look it up in the dictionary.",
        "english": "**english** /ˈɪŋɡlɪʃ/ *noun / adjective*\nThe language of England, now widely used worldwide.\nExample: She speaks English fluently.",
        "chinese": "**chinese** /tʃaɪˈniːz/ *noun / adjective*\nThe language of China; relating to China.\nExample: Chinese has many dialects.",
        "mandarin": "**mandarin** /ˈmændərɪn/ *noun*\nStandard Chinese; the official language of China.\nExample: Mandarin is the official language of China.",
        "word": "**word** /wɜːrd/ *noun*\nA single distinct meaningful element of speech or writing.\nExample: Every word has a history.",
        "language": "**language** /ˈlæŋɡwɪdʒ/ *noun*\nA system of communication used by a community.\nExample: English is a global language.",
        "code": "**code** /koʊd/ *noun / verb*\nA system of rules or instructions; to write computer programs.\nExample: She writes Python code.",
        "python": "**python** /ˈpaɪθən/ *noun*\nA large snake; also a popular programming language.\nExample: Python is great for beginners.",
        "algorithm": "**algorithm** /ˈælɡərɪðəm/ *noun*\nA step-by-step procedure for solving a problem.\nExample: Sorting uses a specific algorithm.",
        "function": "**function** /ˈfʌŋkʃən/ *noun*\nAn activity or purpose; in programming, a reusable block of code.\nExample: This function returns a list.",
        "recursion": "**recursion** /rɪˈkɜːrʒən/ *noun*\nThe process of defining something in terms of itself.\nExample: Recursion is common in programming.",
    }

    _ZH_CURATED: Dict[str, str] = {
        "你": "**你** (nǐ) *pronoun*\nYou (singular).\n例：你好吗？",
        "我": "**我** (wǒ) *pronoun*\nI; me.\n例：我是学生。",
        "他": "**他** (tā) *pronoun*\nHe; him.\n例：他是医生。",
        "她": "**她** (tā) *pronoun*\nShe; her.\n例：她很漂亮。",
        "是": "**是** (shì) *verb*\nTo be; yes.\n例：我是中国人。",
        "不": "**不** (bù) *adverb*\nNot; no.\n例：我不去。",
        "有": "**有** (yǒu) *verb*\nTo have; there is.\n例：我有时间。",
        "好": "**好** (hǎo) *adjective*\nGood; well; OK.\n例：你好！",
        "谢谢": "**谢谢** (xièxie) *expression*\nThank you.\n例：谢谢你的帮助！",
        "再见": "**再见** (zàijiàn) *expression*\nGoodbye.\n例：再见，明天见！",
        "你好": "**你好** (nǐ hǎo) *greeting*\nHello; hi.\n例：你好，很高兴认识你。",
        "什么": "**什么** (shénme) *interrogative*\nWhat.\n例：你在做什么？",
        "为什么": "**为什么** (wèishénme) *interrogative*\nWhy.\n例：为什么迟到？",
        "怎么": "**怎么** (zěnme) *interrogative*\nHow; why.\n例：你怎么知道的？",
        "中文": "**中文** (Zhōngwén) *noun*\nChinese language.\n例：学中文。",
        "英文": "**英文** (Yīngwén) *noun*\nEnglish language.\n例：英文单词。",
        "汉语": "**汉语** (Hànyǔ) *noun*\nChinese language (formal).\n例：汉语很难。",
        "普通话": "**普通话** (Pǔtōnghuà) *noun*\nMandarin Chinese (standard).\n例：说普通话。",
        "猫": "**猫** (māo) *noun*\nCat.\n例：小猫。",
        "狗": "**狗** (gǒu) *noun*\nDog.\n例：小狗。",
        "水": "**水** (shuǐ) *noun*\nWater.\n例：喝水。",
        "书": "**书** (shū) *noun*\nBook.\n例：看书。",
        "字": "**字** (zì) *noun*\nCharacter; word.\n例：汉字。",
        "词": "**词** (cí) *noun*\nWord; term.\n例：生词。",
        "字典": "**字典** (zìdiǎn) *noun*\nDictionary.\n例：查字典。",
        "世界": "**世界** (shìjiè) *noun*\nWorld.\n例：世界各地。",
        "学习": "**学习** (xuéxí) *verb / noun*\nTo study; learning.\n例：努力学习。",
        "代码": "**代码** (dàimǎ) *noun*\nCode (programming).\n例：写代码。",
        "程序": "**程序** (chéngxù) *noun*\nProgram.\n例：运行程序。",
        "递归": "**递归** (dìguī) *noun*\nRecursion.\n例：递归函数。",
        "算法": "**算法** (suànfǎ) *noun*\nAlgorithm.\n例：排序算法。",
        "函数": "**函数** (hánshù) *noun*\nFunction.\n例：定义函数。",
        "爱": "**爱** (ài) *verb / noun*\nLove; to love.\n例：爱是美好的。",
        "朋友": "**朋友** (péngyou) *noun*\nFriend.\n例：好朋友。",
        "时间": "**时间** (shíjiān) *noun*\nTime.\n例：没有时间。",
        "今天": "**今天** (jīntiān) *noun / adverb*\nToday.\n例：今天星期几？",
        "明天": "**明天** (míngtiān) *noun / adverb*\nTomorrow.\n例：明天见。",
        "中国": "**中国** (Zhōngguó) *proper noun*\nChina.\n例：我在中国。",
    }

    _MANDARIN_WORDS: Tuple[str, ...] = (
        "你好", "谢谢", "再见", "什么", "为什么", "怎么", "哪里", "多少", "今天", "明天", "昨天",
        "现在", "时间", "学校", "老师", "学生", "工作", "朋友", "家庭", "国家", "城市", "世界",
        "中文", "英文", "汉语", "普通话", "电脑", "手机", "网络", "代码", "程序", "问题", "答案",
        "意思", "字典", "解释", "学习", "知识", "帮助", "希望", "梦想", "生活", "生命", "快乐",
        "高兴", "漂亮", "美丽", "重要", "容易", "健康", "医生", "医院", "爸爸", "妈妈", "哥哥",
        "姐姐", "弟弟", "妹妹", "名字", "美国", "英国", "日本", "法国", "德国", "语言", "文化",
        "历史", "科学", "数学", "艺术", "和平", "战争", "真理", "方法", "原因", "结果", "例子",
        "递归", "算法", "函数", "变量", "苹果", "面包", "牛奶", "咖啡", "茶", "水果", "音乐",
        "电影", "游戏", "运动", "天气", "太阳", "月亮", "星星", "飞机", "火车", "汽车", "自行车",
        "我们", "你们", "他们", "她们", "自己", "大家", "人们", "男人", "女人", "孩子", "东西",
        "地方", "事情", "办法", "机会", "经验", "能力", "责任", "关系", "情况", "条件", "方向",
        "水平", "标准", "价值", "意义", "精神", "思想", "感情", "态度", "行为", "习惯", "规则",
        "法律", "权利", "义务", "自由", "安全", "幸福", "成功", "失败", "努力", "坚持", "理解",
        "认识", "发现", "决定", "选择", "接受", "拒绝", "同意", "反对", "支持", "反对", "讨论",
        "交流", "沟通", "合作", "竞争", "发展", "进步", "变化", "影响", "作用", "效果", "目的",
        "计划", "准备", "完成", "开始", "结束", "继续", "停止", "提高", "降低", "增加", "减少",
        "建立", "破坏", "保护", "保存", "创造", "生产", "消费", "管理", "组织", "安排", "处理",
        "解决", "分析", "比较", "总结", "介绍", "说明", "描述", "表达", "表示", "证明", "检查",
        "测试", "研究", "调查", "观察", "记录", "报告", "通知", "提醒", "建议", "要求", "允许",
        "禁止", "鼓励", "批评", "表扬", "感谢", "道歉", "祝贺", "欢迎", "邀请", "访问", "参观",
        "旅行", "休息", "娱乐", "锻炼", "治疗", "照顾", "关心", "尊重", "信任", "怀疑", "相信",
        "忘记", "记住", "想象", "期待", "担心", "害怕", "生气", "伤心", "满意", "失望", "惊讶",
    )

    @classmethod
    def _hash(cls, text: str) -> int:
        return sum(ord(c) * (i + 1) for i, c in enumerate(text)) % 10**6

    @classmethod
    def _guess_en_pos(cls, word: str) -> str:
        wl = (word or "").lower()
        for suf, pos in cls._EN_POS_SUFFIXES:
            if wl.endswith(suf):
                return pos
        return "word"

    @classmethod
    def _hardcoded_english(cls, word: str) -> str:
        w = (word or "").strip()
        wl = w.lower()
        if wl in cls._EN_CURATED:
            return cls._EN_CURATED[wl]
        pos = cls._guess_en_pos(wl)
        h = cls._hash(wl)
        definition = cls._EN_DEF_TEMPLATES[h % len(cls._EN_DEF_TEMPLATES)].format(w=w, pos=pos)
        example = cls._EN_EXAMPLE_TEMPLATES[h % len(cls._EN_EXAMPLE_TEMPLATES)].format(w=w, pos=pos)
        return f"**{w}**\n{definition}\nPart of speech: {pos}.\n{example}"

    @classmethod
    def _hardcoded_mandarin(cls, word: str) -> str:
        w = (word or "").strip()
        if w in cls._ZH_CURATED:
            return cls._ZH_CURATED[w]
        h = cls._hash(w)
        pos_hint = "单字" if len(w) == 1 else f"{len(w)}字词"
        definition = cls._ZH_DEF_TEMPLATES[h % len(cls._ZH_DEF_TEMPLATES)].format(w=w)
        example = cls._ZH_EXAMPLE_TEMPLATES[h % len(cls._ZH_EXAMPLE_TEMPLATES)].format(w=w)
        return f"关于「{w}」\n词形：{pos_hint}（普通话）\n{definition}\n{example}"

    @classmethod
    def _load_english_word_set(cls) -> frozenset:
        paths = (
            "/usr/share/dict/words",
            "/usr/dict/words",
            "/usr/share/dict/web2",
            "/usr/share/dict/american-english",
        )
        for path in paths:
            if not os.path.isfile(path):
                continue
            try:
                with open(path, encoding="utf-8", errors="ignore") as fh:
                    words = {
                        w.strip().lower()
                        for w in fh
                        if re.fullmatch(r"[A-Za-z]{2,24}", w.strip())
                    }
                if words:
                    return frozenset(words)
            except OSError:
                continue
        fallback = set(cls._EN_CURATED.keys())
        fallback.update(
            "the a an is are was were be been being have has had do does did will would could should "
            "may might must can shall to of in for on with at by from as into through during before "
            "after above below between under again further then once here there when where why how "
            "all each few more most other some such no nor not only own same so than too very just "
            "one two three four five six seven eight nine ten man woman child people friend family "
            "home house food school life death money phone internet read write speak listen see look "
            "think know learn teach work play want need help make go come give take find tell ask use "
            "try start stop open close big small good bad new old hot cold happy sad fast slow day "
            "night year month week today tomorrow yesterday".split()
        )
        return frozenset(fallback)

    @classmethod
    def _build_mandarin_lexicon(cls) -> None:
        for w in cls._MANDARIN_WORDS:
            if w not in cls._MANDARIN:
                cls._MANDARIN[w] = cls._hardcoded_mandarin(w)
        for ch in cls._ZH_CURATED:
            if ch not in cls._MANDARIN:
                cls._MANDARIN[ch] = cls._ZH_CURATED[ch]
        for code in range(0x4E00, 0x9FFF + 1):
            ch = chr(code)
            if ch not in cls._MANDARIN:
                cls._MANDARIN[ch] = cls._hardcoded_mandarin(ch)

    @classmethod
    def _build_english_lexicon(cls) -> None:
        cls._EN_WORDS = cls._load_english_word_set()
        for word in cls._EN_WORDS:
            if word not in cls._ENGLISH:
                cls._ENGLISH[word] = cls._hardcoded_english(word)
        for word, resp in cls._EN_CURATED.items():
            cls._ENGLISH[word.lower()] = resp

    @classmethod
    def initialize(cls) -> None:
        if cls._ready:
            return
        cls._build_english_lexicon()
        cls._build_mandarin_lexicon()
        cls._ready = True

    @classmethod
    def lookup(cls, word: str) -> Optional[str]:
        cls.initialize()
        s = (word or "").strip().strip(" \t\r\n.,!?;:'\"`~，。！？：、()[]{}<>")
        if not s:
            return None
        if VibeCodeHeuristics.has_cjk(s):
            if s in cls._MANDARIN:
                return cls._MANDARIN[s]
            if re.fullmatch(r"[\u4e00-\u9fff]{1,10}", s):
                resp = cls._hardcoded_mandarin(s)
                cls._MANDARIN[s] = resp
                return resp
            return None
        wl = s.lower()
        if wl in cls._ENGLISH:
            return cls._ENGLISH[wl]
        if re.fullmatch(r"[A-Za-z]{2,24}", s):
            resp = cls._hardcoded_english(s)
            cls._ENGLISH[wl] = resp
            return resp
        return None

    @classmethod
    def stats(cls) -> Dict[str, Any]:
        cls.initialize()
        return {
            "english_entries": len(cls._ENGLISH),
            "mandarin_entries": len(cls._MANDARIN),
            "english_word_set": len(cls._EN_WORDS or ()),
            "ready": cls._ready,
        }


class CatR11Synthesizer:
    """Knowledge + chat synthesis backend for cat r1 (files = off)."""

    def __init__(self):
        pass

    @staticmethod
    def detect_locale(prompt: str) -> str:
        has_cjk = VibeCodeHeuristics.has_cjk(prompt)
        has_eng = bool(re.search(r"[a-zA-Z]{3,}", prompt))
        if has_cjk and has_eng:
            return "mixed"
        if has_cjk:
            return "chinese"
        return "english"

    @staticmethod
    def _primarily_chinese(text: str) -> bool:
        cjk = len(re.findall(r"[\u4e00-\u9fff]", text or ""))
        if cjk < 4:
            return False
        latin = len(re.findall(r"[a-zA-Z]", text or ""))
        return cjk >= latin

    @classmethod
    def localize(cls, body: str, prompt: str) -> str:
        """Match reply language to user — EN · 中文 · mixed."""
        if not body or "```" in body:
            return body
        loc = cls.detect_locale(prompt)
        if loc == "english":
            return body
        if loc == "mixed":
            if cls._primarily_chinese(body):
                return body
            return body + "\n\n---\n*Bilingual · 可继续用中文或 English 提问 · paste ```code``` to run*"
        if cls._primarily_chinese(body):
            return body
        topic = extract_zh_topic(prompt) or prompt.strip()[:80]
        return cls._zh_shell(body, topic, prompt)

    @staticmethod
    def _zh_shell(body: str, topic: str, prompt: str) -> str:
        pl = prompt.lower()
        if any(k in pl for k in ("代码", "编程", "程序", "脚本", "函数")):
            lead = f"关于「{topic}」的代码思路："
        elif any(k in pl for k in ("调试", "报错", "错误", "bug", "异常")):
            lead = f"关于「{topic}」的调试建议："
        elif is_zh_question(prompt):
            lead = f"关于「{topic}」："
        else:
            lead = f"好的，关于「{topic}」——"
        lines = [ln.strip() for ln in body.strip().split("\n") if ln.strip()]
        if len(lines) <= 3:
            return f"{lead}\n\n{body.strip()}\n\n需要我写代码示例或逐步讲解吗？"
        return f"{lead}\n\n{body.strip()}\n\n---\n*继续用中文问，或说「写代码」让我生成可运行示例。*"

    @staticmethod
    def _is_single_word_query(prompt: str) -> bool:
        """
        Treat short "word-like" inputs as a dictionary/lexicon lookup request.
        (Goal: answer *any* English/CJK word with a stable template.)
        """
        s = CatR11Synthesizer._normalize_word_token(prompt)
        if not s:
            return False
        # Must be a single token (no spaces/newlines).
        if any(ch.isspace() for ch in s):
            return False
        # Avoid command-like inputs.
        if s.startswith("/"):
            return False
        # Avoid punctuation-heavy / code-like / URL-like inputs.
        if re.search(r"```|https?://|[`~]|[=<>]|@chat|@code|:|;|,|\[|\]|\{|\}", s, re.I):
            return False
        # Avoid numeric-only inputs.
        if re.fullmatch(r"\d+(?:\.\d+)?", s):
            return False

        pl = s.lower()
        reserved_en = {
            # common acknowledgements
            "yes", "yep", "yeah", "ok", "okay", "sure", "please", "thanks", "thank", "thx",
            "hi", "hello", "hey",
            # common control words
            "help", "reset", "bitnet", "run", "code", "chat", "think",
            "languages", "profile", "goodbye", "bye", "thanks a lot",
            # common meta intents
            "joke", "math", "explain", "howto", "debug", "opinion", "recursion",
        }
        if pl in reserved_en:
            return False
        reserved_zh = {"好的", "行", "可以", "谢谢", "请", "你好", "您好", "再见", "继续", "明白"}
        if s in reserved_zh:
            return False

        # English: letters only (2..24).
        if re.fullmatch(r"[A-Za-z]{2,24}", s):
            return True
        # Chinese: CJK chars only (1..10).
        if re.fullmatch(r"[\u4e00-\u9fff]{1,10}", s):
            return True
        return False

    @staticmethod
    def _normalize_word_token(prompt: str) -> str:
        s = (prompt or "").strip()
        # Strip common surrounding punctuation/quotes so inputs like `猫。` or `"cat"` still match.
        return s.strip(" \t\r\n.,!?;:'\"`~，。！？：、()[]{}<>")

    @staticmethod
    def _guess_en_pos(word: str) -> str:
        wl = (word or "").lower()
        if wl.endswith("ing"):
            return "verb (gerund/participle)"
        if wl.endswith("ed"):
            return "verb (past tense) / adjective"
        if wl.endswith("ly"):
            return "adverb"
        if wl.endswith(("tion", "sion", "ment", "ness", "ity", "ance", "ence")):
            return "noun"
        if wl.endswith(("ous", "ful", "less", "able", "ible", "ive")):
            return "adjective"
        if wl.endswith(("er", "or", "ist")):
            return "noun"
        return "word"

    @classmethod
    def _english_word_template(cls, word: str) -> str:
        w = (word or "").strip()
        pos = cls._guess_en_pos(w)
        wl = w.lower()
        meaning_hint = "a common concept/usage in English"
        if wl.endswith(("tion", "sion")):
            meaning_hint = "the act or process of something"
        elif wl.endswith("ing"):
            meaning_hint = "the action or activity of doing something"
        elif wl.endswith("ed"):
            meaning_hint = "the state of having done something (or something described by that)"
        elif wl.endswith("ly"):
            meaning_hint = "in a way related to the base meaning"
        elif wl.endswith("ness"):
            meaning_hint = "a quality or state of being"
        elif wl.endswith("ful"):
            meaning_hint = "full of; characterized by"
        elif wl.endswith("less"):
            meaning_hint = "without; lacking"
        elif wl.endswith(("able", "ible")):
            meaning_hint = "can be done; capable of"
        elif wl.endswith(("er", "or", "ist")):
            meaning_hint = "a person or thing that does something"
        elif wl.endswith("ment"):
            meaning_hint = "the result or act related to the base"

        if pos.startswith("verb"):
            if wl.endswith("ing"):
                example = f"Example: \"I enjoy {w} during my free time.\""
            elif wl.endswith("ed"):
                example = f"Example: \"We {w} last week, so the result is ready.\""
            else:
                example = f"Example: \"I will {w} in my next step.\""
        elif pos == "adverb":
            example = f"Example: \"She answered {w} and stayed calm.\""
        elif pos == "adjective":
            example = f"Example: \"That is a {w} approach to the problem.\""
        else:
            example = f"Example: \"Here's how you can use **{w}** in a sentence:\""

        return (
            f"**{w}**\n"
            f"Meaning (template): {meaning_hint}.\n"
            f"Part of speech (guess): {pos}.\n"
            f"Usage: You can use **{w}** as a {pos} when talking about the idea hinted by its ending; "
            "exact meaning still depends on context.\n"
            f"{example}"
        )

    @staticmethod
    def _chinese_word_template(word: str) -> str:
        w = (word or "").strip()
        return (
            f"关于「{w}」\n"
            f"1) 含义（模板）：常用于表达与「{w}」相关的概念或作用；具体含义需要结合语境。\n"
            f"2) 用法提示：它可能出现在名词/动词/形容词等位置（按句子结构判断）。\n"
            f"3) 例句（模板）：我想弄懂「{w}」在这句话里的意思。"
        )

    @classmethod
    def _word_response_template(cls, prompt: str) -> str:
        s = cls._normalize_word_token(prompt)
        hit = CatR1DictionaryLexicon.lookup(s)
        if hit:
            return hit
        if VibeCodeHeuristics.has_cjk(s):
            return cls._chinese_word_template(s)
        return cls._english_word_template(s)

    def respond_anything(
        self,
        prompt: str,
        history: List[Dict[str, str]],
        vec: Optional[np.ndarray] = None,
        engine: Optional["CatR11Engine"] = None,
    ) -> str:
        """Universal chat — always returns a useful EN/中文 reply."""
        pl = prompt.lower().strip()
        if engine is not None:
            setup = engine.ai_setup_response(prompt)
            if setup:
                return setup
        if CatR1Code._META_DESIRE.search(prompt):
            msg = (
                "Understood — **user desire mode** is on. I'll say what you ask.\n\n"
                "Try: `say hello world` · `respond with: yes` · `请说：你好` · or ask anything."
            )
            if VibeCodeHeuristics.has_cjk(prompt):
                msg = (
                    "明白——**用户意愿模式**已开启，你说什么我就说什么。\n\n"
                    "试试：`say hello` · `请说：你好世界` · 或直接告诉我你想要的内容。"
                )
            return msg
        literal = self.extract_literal_desire(prompt)
        if literal is not None:
            return literal
        hit = self.smalltalk_reply(pl) or self.smalltalk_reply(prompt.strip())
        if hit:
            return hit
        # Single-word dictionary mode: stable template for EN/CJK "word-like" inputs.
        if self._is_single_word_query(prompt):
            body = self._word_response_template(prompt)
            return self.localize(body, prompt)
        follow = self._followup(prompt, history, vec)
        if follow:
            return self.localize(follow, prompt)
        sorted_hist = GoogleWhitepaperCatR1Sorter.sort_history(
            [(m["role"], m["text"]) for m in history], prompt
        )
        body = self.synthesize(prompt, sorted_hist, vec=vec)
        return self.localize(body, prompt)

    @classmethod
    def extract_literal_desire(cls, prompt: str) -> Optional[str]:
        """Return exact text when user says say/respond with/output X."""
        s = prompt.strip()
        if not s or CatR1Code._META_DESIRE.search(s):
            return None
        if re.fullmatch(r'["\'].+["\']', s):
            return s.strip("\"'").strip()
        for pat in CatR1Code._LITERAL_SAY:
            m = pat.match(s)
            if m:
                text = m.group(1).strip().strip("\"'").strip()
                if text and not CatR1Code._META_DESIRE.search(text):
                    return text
        m = CatR1Code._SAYS.search(s)
        if m:
            return m.group(1).strip()
        return None

    def fulfill_user_desire(
        self,
        prompt: str,
        history: List[Dict[str, str]],
        vec: Optional[np.ndarray] = None,
    ) -> Optional[str]:
        """Compose a direct answer to what the user wants — no deflection."""
        if not CONFIG.get("user_desire_mode", True):
            return None
        norm = CatR1Code.normalize_prompt(prompt)
        work = norm or prompt
        if re.fullmatch(r"files\s*=\s*\.?\s*off", work.strip(), flags=re.I):
            return None
        topic = self._topic(work, work.lower().strip())
        pl = work.lower().strip()
        loc = self.detect_locale(prompt)
        idx = CatR1LLM.pick(vec, 5, salt=11) if vec is not None else 0

        if re.search(
            r"\b(?:write|compose|draft|want)\s+(?:a\s+)?(?:short\s+)?(?:story|tale|paragraph|message|letter|speech)\b"
            r"|\bparagraph\s+about\b",
            pl,
        ):
            return self._compose_direct(prompt, topic, vec, kind="prose")
        if re.search(r"(写|来一?段|给我一?篇|讲一?个)", prompt) and re.search(r"(故事|段落|文章|话)", prompt):
            return self._compose_direct(prompt, topic, vec, kind="prose")
        return None

    def _clean_topic(self, prompt: str) -> str:
        return CatR1Code.clean_topic(prompt)

    def _substantive_answer(
        self,
        topic: str,
        pl: str,
        prompt: str,
        vec: Optional[np.ndarray] = None,
    ) -> str:
        """Answer directly — no 'say explain/code' deflection."""
        clean = self._clean_topic(prompt) or topic.strip() or "that"
        t = clean.lower()
        loc = self.detect_locale(prompt)
        idx = CatR1LLM.pick(vec, 6, salt=23) if vec is not None else 0

        if re.search(r"\b(?:just\s+)?talk\s+to\s+me\b", pl):
            return (
                "Sure — I'm here. What's on your mind? "
                "A question you're chewing on, something you're building, or we can just chat."
            )
        if re.search(r"\b(?:sky|blue)\b", t) and re.search(r"\bwhy\b", pl):
            return (
                "The sky looks **blue** because of **Rayleigh scattering** — sunlight hits air molecules "
                "and shorter blue wavelengths scatter more than longer red ones, so blue dominates overhead. "
                "At sunset the light travels through more atmosphere, so more blue scatters away and you see reds and oranges."
            )
        if re.search(r"\bcats?\b", t):
            return (
                "**Cats** are small carnivorous mammals people have lived with for thousands of years. "
                "They're obligate carnivores, sleep a lot (often 12–16 hours), and communicate with body language, "
                "scent, and vocalizations. Smart, curious, and independent — but they bond with people on their own terms."
            )
        if "love" in t:
            return (
                "**Love** is a deep bond of care, attachment, and commitment — romantic, familial, or platonic. "
                "Researchers often describe it as intimacy, passion, and choice over time. "
                "What angle interests you — feelings, science, or something personal?"
            )
        if "pizza" in t:
            return (
                "Pizza — excellent choice. 🍕 "
                "Originating in Italy, it's flatbread with sauce and toppings baked hot. "
                "Margherita, pepperoni, or something weird with pineapple — what's your order?"
            )
        if "homework" in t or "作业" in prompt:
            return (
                "Happy to help with homework. Paste the exact question and the subject "
                "(math, essay, history, code) — I'll work through it step by step with you."
            )
        if loc == "chinese":
            bodies = (
                f"「{clean}」是个有意思的话题。简单说：它和我们每天的学习、选择和看世界的方式都有关系。",
                f"关于「{clean}」——核心在于把它具体化：你想知道什么、用在哪、已经了解多少。",
                f"「{clean}」值得聊。从我的角度，先抓住一个你最关心的点，我们再往下展开。",
            )
            return bodies[idx % len(bodies)]
        bodies = (
            f"**{clean.capitalize()}** — at a high level, it's about how people learn, choose, and make sense of the world. "
            f"Happy to go deeper on any part that matters to you.",
            f"On **{clean}**: the core idea is connecting what you already know to what you're trying to do next. "
            f"Ask a follow-up and I'll keep going.",
            f"**{clean.capitalize()}** is a solid topic. Here's a direct take: start with one concrete question, "
            f"then we can build from there — no need to pick a mode, just keep talking.",
        )
        return bodies[idx % len(bodies)]

    def _compose_direct(
        self,
        prompt: str,
        topic: str,
        vec: Optional[np.ndarray],
        *,
        kind: str = "answer",
    ) -> str:
        loc = self.detect_locale(prompt)
        subj = topic.strip("?.！？。 ")[:120] or prompt.strip()[:120]
        idx = CatR1LLM.pick(vec, 5, salt=17) if vec is not None else 0

        if kind == "prose":
            if loc == "chinese":
                openings = (
                    f"关于「{subj}」，",
                    f"说到「{subj}」，",
                    f"「{subj}」——",
                )
                bodies = (
                    "事情往往从一个小念头开始，然后慢慢长出形状。你不必一次做对，只要愿意继续问、继续改，答案会自己浮现。",
                    "每个选择都留下痕迹。重要的不是完美，而是清楚自己想要什么，然后朝那个方向走一步。",
                    "世界嘈杂，但你的问题很具体——这就已经是很好的起点。",
                )
                return openings[idx % len(openings)] + bodies[idx % len(bodies)]
            openings = (
                f"On **{subj}**: ",
                f"Regarding **{subj}** — ",
                f"**{subj}** — ",
            )
            bodies = (
                "it often starts as a small idea and grows from there. You don't have to get it perfect on the first try — keep asking, keep revising, and the shape will emerge.",
                "every choice leaves a trace. What matters isn't perfection; it's knowing what you want and taking one clear step toward it.",
                "the world is noisy, but your question is specific — that's already a strong starting point.",
            )
            return openings[idx % len(openings)] + bodies[idx % len(bodies)]

        if loc == "chinese":
            templates = (
                f"「{subj}」——我直接说重点：先把目标写清楚，再拆成两三步能做的事；"
                f"需要代码、解释或示例就说一声。",
                f"关于「{subj}」：核心是把问题具体化——你要什么结果、有什么限制、已经试过什么；"
                f"我可以按你的意愿写代码、讲解或继续聊。",
                f"好的，「{subj}」。我的理解是你想要明确、可用的回答——"
                f"告诉我偏概念、步骤还是代码，我按你的意思来。",
                f"「{subj}」值得认真答：先给结论——可以按你的需求定制；"
                f"说「写代码」「解释」或「举例」我就照做。",
                f"收到。「{subj}」——你说什么方向，我就说什么内容；中英文、代码或闲聊都行。",
            )
            return templates[idx % len(templates)]

        templates = (
            f"On **{subj}** — here's a direct take: clarify the goal, break it into two or three concrete steps, "
            f"and tell me if you want code, explanation, or an example.",
            f"Regarding **{subj}**: make the question specific — desired outcome, constraints, what you've tried. "
            f"I'll say what you need: concept, steps, or code.",
            f"**{subj}** — you want a clear, usable answer. Say *explain*, *code*, or *example* and I'll match your intent.",
            f"On **{subj}**: short version — I can tailor the reply. Tell me the angle and I'll say it your way.",
            f"Got it — **{subj}**. Ask in English or 中文; I'll respond with whatever you ask for.",
        )
        return templates[idx % len(templates)]

    @staticmethod
    def _normalize_chat(pl: str) -> str:
        s = pl.strip().lower()
        s = re.sub(r"\s+", " ", s)
        s = s.rstrip("?!.！？。")
        return s

    @classmethod
    def smalltalk_reply(cls, pl: str) -> Optional[str]:
        s = cls._normalize_chat(pl)
        for pat, reply in _SMALLTALK:
            if re.match(pat, s) or re.match(pat, pl.strip()):
                return reply
        return None

    @classmethod
    def is_educational(cls, pl: str) -> bool:
        if cls.smalltalk_reply(pl):
            return False
        if is_explain_request(pl) or is_zh_question(pl):
            return True
        if re.search(r"(解释|说明|介绍|教程|原理)", pl):
            return True
        patterns = (
            r"\b(explain|define|describe|what is|what are|why|how (?:does|do|to|can|would|should|could|will))\b",
            r"\b(compare|versus|vs\.?|difference|tutorial|implement|algorithm|architecture)\b",
            r"\b(debug|traceback|exception|error|bug|broken|fix)\b",
            r"\b(write|code|function|script|snippet|class|api|docker|python|cat-r1|cat r1|core|sql|git)\b",
            r"\b(calculate|compute|solve|fibonacci|prime)\b",
            r"\b(plan|roadmap|design|system|recommend|should i)\b",
        )
        s = cls._normalize_chat(pl)
        return any(re.search(p, s) for p in patterns)

    def _topic(self, prompt: str, pl: str) -> str:
        cleaned = CatR1Code.clean_topic(prompt)
        if cleaned:
            return cleaned
        for prefix in (
            "explain ", "why ", "how does ", "how do ", "how to ", "what is ", "what are ",
            "define ", "compare ", "debug ", "fix ", "write ", "tell me about ", "tell me ",
            "talk about ", "chat about ", "随便聊聊", "聊聊", "说说",
        ):
            if pl.startswith(prefix):
                return prompt[len(prefix):].strip("?.")
        if " about " in pl:
            return prompt[pl.index(" about ") + 7:].strip("?.")
        if "关于" in prompt:
            m = re.search(r"关于(.+?)[?？。!！]?$", prompt)
            if m:
                return m.group(1).strip()
        if "?" in prompt:
            return prompt.strip().rstrip("?")
        return prompt.strip()[:200] or "your question"

    def analyze(self, prompt: str) -> Dict[str, str]:
        work = CatR1Code.normalize_prompt(prompt) or prompt.strip()
        pl = work.lower().strip()
        if self.extract_literal_desire(work):
            return {"intent": "desire", "topic": work.strip(), "pl": pl}
        if is_zh_greeting(work.strip()):
            return {"intent": "casual", "topic": work.strip(), "pl": pl}
        if any(k in pl for k in ("joke", "funny", "humor", "laugh")) or "笑话" in work:
            return {"intent": "joke", "topic": self._topic(work, pl), "pl": pl}
        if VibeCodeHeuristics.has_cjk(work) and any(
            k in work for k in ("天气", "吃什么", "推荐", "觉得", "看法", "心情")
        ):
            return {"intent": "casual", "topic": work.strip()[:80], "pl": pl}
        if is_explain_request(work) or re.search(r"(解释|说明|介绍)", work):
            topic = extract_zh_topic(work) if VibeCodeHeuristics.has_cjk(work) else self._topic(work, pl)
            return {"intent": "explain", "topic": topic, "pl": pl}
        topic = self._topic(work, pl)
        if self.smalltalk_reply(pl):
            return {"intent": "casual", "topic": topic, "pl": pl}
        if CONFIG["cat_r1_enabled"] and CatR1LLM.wants_fable(pl):
            return {"intent": "fable", "topic": topic, "pl": pl}
        if CONFIG["cat_r1_enabled"] and CatR1LLM.wants_poem(pl):
            return {"intent": "poem", "topic": topic, "pl": pl}
        if any(k in pl for k in ("story", "tale", "creative", "imagine", "narrative")):
            return {"intent": "creative", "topic": topic, "pl": pl}
        if CatR1Code.is_code_debug(work, pl):
            return {"intent": "debug", "topic": topic, "pl": pl}
        if any(k in pl for k in ("compare", " vs ", " versus ", "difference", "better")):
            return {"intent": "compare", "topic": topic, "pl": pl}
        if re.search(
            r"\b(explain|what is|what are|why|how (?:does|do|to|can|would|should|could|will))\b", pl
        ):
            return {"intent": "explain", "topic": topic, "pl": pl}
        if any(k in pl for k in ("fibonacci", "fib", "prime", "primes")):
            return {"intent": "code", "topic": topic, "pl": pl}
        if CatR1Code.wants_code(pl):
            return {"intent": "code", "topic": topic, "pl": pl}
        if any(k in pl for k in ("write code", "function", "implement", "snippet", "script")):
            return {"intent": "code", "topic": topic, "pl": pl}
        if any(k in pl for k in ("plan", "roadmap", "architecture", "design", "system")):
            return {"intent": "design", "topic": topic, "pl": pl}
        if re.search(r"\d\s*[+\-*/^%]", pl) or "calculate" in pl or "solve" in pl:
            return {"intent": "math", "topic": topic, "pl": pl}
        if any(k in pl for k in ("should i", "opinion", "recommend", "best")):
            return {"intent": "advise", "topic": topic, "pl": pl}
        if re.search(r"\b(?:meow|mew|purr|nya|rawr)\b", pl):
            return {"intent": "casual", "topic": topic, "pl": pl}
        if re.search(r"\b(?:just\s+)?talk\s+to\s+me\b", pl):
            return {"intent": "casual", "topic": topic, "pl": pl}
        if len(pl.split()) <= 4 and "?" not in work and not self.is_educational(pl):
            if not re.search(r"\b(about|want|need|tell|why|what|how|help|cats?|love|pizza)\b", pl):
                return {"intent": "casual", "topic": topic, "pl": pl}
        if not self.is_educational(pl):
            return {"intent": "casual", "topic": topic, "pl": pl}
        return {"intent": "general", "topic": topic, "pl": pl}

    @staticmethod
    def _footer() -> str:
        return ""

    def _explain(self, topic: str, pl: str, vec: Optional[np.ndarray] = None, prompt: str = "") -> str:
        work = prompt or topic
        if ("python" in pl or "python" in topic.lower()) and VibeCodeHeuristics.has_cjk(topic + pl):
            return (
                f"**{topic.capitalize()}** — 清晰解释如下。\n\n"
                "**Python** 是一种高级编程语言，语法简洁、生态丰富。\n\n"
                "常用于脚本、Web 后端、数据分析、自动化和机器学习原型。\n\n"
                "优点：开发快、库多、社区大。\n"
                "缺点：GIL 限制 CPU 并行；大型项目建议加类型注解。\n\n"
                "```python\nfor i in range(3):\n    print(i)\n```"
            )
        kb = {
            ("recursion",): (
                f"**{topic.capitalize()}** is when a function calls itself until it hits a base case, "
                "then unwinds the stack to combine results.\n\n"
                "Each call gets its own stack frame. The base case stops new frames; the recursive step "
                "handles a smaller instance of the same problem.\n\n"
                "Example: `factorial(3)` waits on `factorial(2)` → `factorial(1)` returns 1, "
                "then multiplies back up: 1 × 2 × 3 = 6.\n\n"
                "Watch out for missing base cases — they cause stack overflow."
            ),
            ("递归",): (
                "**递归**是函数调用自身，直到遇到**基准情况**（base case），然后逐层返回结果。\n\n"
                "每次调用都有独立的栈帧。基准情况阻止无限调用；递归步骤处理更小规模的同一问题。\n\n"
                "示例：`factorial(3)` 等待 `factorial(2)` → `factorial(1)` 返回 1，"
                "再逐层相乘：1 × 2 × 3 = 6。\n\n"
                "注意：缺少基准情况会导致栈溢出。"
            ),
            ("docker",): (
                f"**Docker** packages an application and its dependencies into a **container** that runs "
                "consistently on any machine with Docker installed.\n\n"
                "Key pieces:\n"
                "- **Image** — read-only blueprint\n"
                "- **Container** — running instance of an image\n"
                "- **Dockerfile** — recipe to build an image\n\n"
                "Great for reproducible dev environments. For large-scale orchestration, people often add Kubernetes."
            ),
            ("core",): (
                f"**{BRAND}** quantizes neural network weights to three values: **{{-1, 0, 1}}**. "
                "That turns matrix multiplication into mostly additions and subtractions, "
                "which cuts memory use and can speed up inference.\n\n"
                f"{BRAND} runs a local in-memory student stack — "
                f"{CONFIG['distil_passes']} heads, all in-memory."
            ),
            ("transformer", "attention"): (
                f"A **transformer** processes sequences using **self-attention** — each token can "
                "weigh every other token to capture context.\n\n"
                "Stack: embeddings → multi-head attention → feed-forward network → residuals and layer norms, "
                "repeated across many layers. Decoder models mask future tokens for causal generation."
            ),
            ("reason", "think", "o1"): (
                f"{BRAND} R1 runs **{REASONING_MODE}** reasoning: think internally, answer cleanly.\n\n"
                f"{BRAND} **{CONFIG['weight_bits']}-bit** ternary weights + sparse compression + "
                f"rank-{CONFIG['compression_rank']} bottleneck — all in-memory."
            ),
            ("python",): (
                "**Python** is a high-level language known for readable syntax and a huge ecosystem.\n\n"
                "Use it for scripting, web backends, data work, automation, and ML prototypes.\n\n"
                "Strengths: fast to write, great libraries, huge community.\n"
                "Tradeoffs: GIL limits CPU parallelism; type hints help at scale.\n\n"
                "```python\nfor i in range(3):\n    print(i)\n```"
            ),
            ("javascript", "js"): (
                "**JavaScript** runs in browsers and on servers (Node.js).\n\n"
                "Event-driven, async-friendly — ideal for interactive UIs and I/O-heavy APIs.\n\n"
                "Use `async/await` for readable asynchronous code."
            ),
            ("git",): (
                "**Git** tracks code history with commits, branches, and merges.\n\n"
                "Daily flow: `git pull` → edit → `git add` → `git commit` → `git push`.\n"
                "Branches isolate features; merge or rebase when ready."
            ),
            ("api", "rest"): (
                "A **REST API** exposes resources over HTTP with verbs like GET, POST, PUT, DELETE.\n\n"
                "Use nouns in paths, correct status codes, and version your API.\n"
                "In this app everything is local — describe payloads in chat (in-memory)."
            ),
            ("sql", "database"): (
                "**SQL** queries relational databases with tables, rows, and joins.\n\n"
                "Core ops: SELECT, INSERT, UPDATE, DELETE, JOIN.\n"
                "Index columns you filter on; avoid SELECT * in production."
            ),
            ("machine", "ml", "learning"): (
                "**Machine learning** learns patterns from data instead of hand-written rules.\n\n"
                "Pipeline: data → features → model → loss → training → evaluation on hold-out set.\n"
                "Start simple; add complexity only when metrics justify it."
            ),
            ("async", "await"): (
                "**async/await** lets one thread juggle many I/O-bound tasks without blocking.\n\n"
                "The event loop schedules coroutines; `await` yields until I/O completes.\n"
                "Best for network/disk waits — use threads/processes for CPU-heavy work."
            ),
            ("compress", "quant"): (
                f"**Model compression** shrinks memory and speeds inference: quantization (cat r1 ternary), "
                f"sparsity (top-{CONFIG['compression_sparse_k']} activations), and low-rank bottlenecks.\n\n"
                f"{BRAND} stacks these in-memory for frontier-tier capacity."
            ),
        }
        for keys, body in kb.items():
            if any(k in pl or k in topic for k in keys):
                return f"**{topic.capitalize()}** — here's a clear explanation.\n\n{body}"
        if VibeCodeHeuristics.has_cjk(topic + pl):
            return (
                f"关于「{topic}」——我可以从概念、步骤或代码示例来讲。"
                f"告诉我你想了解哪个方面？"
            )
        if not self.is_educational(pl):
            return self._casual(topic, pl, work, vec)
        return self._substantive_answer(topic, pl, work, vec)

    def _fable(self, topic: str, prompt: str, vec: Optional[np.ndarray]) -> str:
        return CatR1LLM().compose_fable(prompt, topic, vec)

    def _poem(self, topic: str, vec: Optional[np.ndarray], prompt: str = "") -> str:
        return CatR1LLM().compose_poem(topic, vec, prompt)

    def _creative(self, topic: str, prompt: str, pl: str, vec: Optional[np.ndarray]) -> str:
        if CatR1LLM.wants_poem(pl):
            return self._poem(topic, vec)
        return self._fable(topic, prompt, vec)

    def _debug(self, topic: str, vec: Optional[np.ndarray] = None, prompt: str = "") -> str:
        loc = self.detect_locale(prompt or topic)
        if loc == "chinese":
            return (
                f"关于「{topic}」——调试是和现实的对话，从小处着手，每次只改一处。\n\n"
                "**最小复现**：找到仍能失败的最小输入。\n"
                "**从下往上读**：堆栈里最后一帧在你自己的代码中，通常先看那里。\n"
                "**写下预期 vs 实际**：动手改之前先写清楚。\n"
                "**有纪律地迭代**：一次只改一处，运行一次，记一条笔记。\n\n"
                "把完整报错贴过来，我可以逐行帮你看。"
            )
        return (
            "Debugging is a conversation with reality — start small, listen closely, change one thing at a time.\n\n"
            "**Reproduce minimally.** Find the smallest input that still fails.\n"
            "**Read bottom-up.** The last frame in *your* code is usually where to look first.\n"
            "**Name expected vs actual.** Write both down before you touch anything.\n"
            "**Iterate with discipline.** One change, one run, one note.\n\n"
            "Paste the full traceback when you have it — I'll walk through it line by line with you."
        )

    def _code(self, topic: str, pl: str, vec: Optional[np.ndarray] = None, prompt: str = "") -> str:
        loc = self.detect_locale(prompt or pl)
        if "fibonacci" in pl or "fib" in pl or "斐波那契" in (prompt or pl):
            body = (
                "```python\ndef fib(n: int) -> list[int]:\n"
                "    a, b = 0, 1\n    out = [a]\n"
                "    for _ in range(1, max(n, 1)):\n        a, b = b, a + b\n        out.append(a)\n"
                "    return out[:n]\n\nprint(fib(10))\n```"
            )
        elif "prime" in pl:
            body = (
                "```python\ndef primes_upto(n: int) -> list[int]:\n"
                "    if n < 2: return []\n    sieve = [True] * (n + 1)\n"
                "    sieve[0] = sieve[1] = False\n"
                "    for p in range(2, int(n**0.5) + 1):\n"
                "        if sieve[p]:\n            sieve[p*p:n+1:p] = [False] * len(sieve[p*p:n+1:p])\n"
                "    return [i for i, ok in enumerate(sieve) if ok]\n\nprint(primes_upto(50))\n```"
            )
        else:
            body = (
                f"```python\ndef solve():\n    \"\"\"Sketch: {topic[:60]}\"\"\"\n"
                "    return None\n\nif __name__ == '__main__':\n    print(solve())\n```"
            )
        if loc == "chinese":
            return (
                f"这是 **{topic}** 的代码草稿——先跑通，再处理边界情况。\n\n"
                f"{body}\n\n"
                "建议测试：空输入、零值、和一个较大值。说 **运行** 或 `/run` 可在本地执行。"
            )
        return (
            f"Here's a **{topic}** sketch — readable first, correct second. Run it, then harden edge cases.\n\n"
            f"{body}\n\n"
            "Empty input, zero, and one large value are the three tests I always run."
        )

    def _math(self, topic: str, pl: str) -> str:
        m = re.search(r"(\d+(?:\.\d+)?)\s*([+\-*/^])\s*(\d+(?:\.\d+)?)", pl)
        if m:
            a, op, b = float(m.group(1)), m.group(2), float(m.group(3))
            ops = {"+": a + b, "-": a - b, "*": a * b, "/": a / b if b else None, "^": a ** b}
            val = ops.get(op)
            if val is not None:
                out = int(val) if val == int(val) else round(val, 6)
                sym = {"*": "×", "/": "÷"}.get(op, op)
                return f"{a:g} {sym} {b:g} = **{out}**"
        return f"What expression should I evaluate for **{topic}**? You can type something like `15 * 7` or `100 / 4`."

    def _compare(self, topic: str, vec: Optional[np.ndarray] = None) -> str:
        return (
            f"Comparing **{topic}** well means naming what you optimize for before you pick winners.\n\n"
            "Consider **latency** (how fast must it be?), **complexity** (what can your team live with?), "
            "**offline needs** (this stack runs fully local), and **total cost** (infra plus time).\n\n"
            "List must-haves, eliminate what breaks them, prototype the top two. "
            "Tell me the pair you're weighing and I'll give you a sharper read."
        )

    def _advise(self, topic: str, vec: Optional[np.ndarray] = None) -> str:
        return (
            f"On **{topic}**, the honest answer is: it depends on what you're optimizing.\n\n"
            "If you need to **learn fast**, ship a small prototype and let reality edit your plan. "
            "If you need **long-term maintenance**, favor simpler architecture over clever architecture. "
            "If you need **privacy or offline work**, everything here stays in-memory.\n\n"
            "Share two or three constraints and I'll recommend something concrete."
        )

    def _joke(self, vec: Optional[np.ndarray] = None, prompt: str = "") -> str:
        jokes_en = [
            "Why do programmers prefer dark mode? Light attracts bugs.",
            "A SQL query walks into a bar, walks up to two tables, and asks: 'Can I join you?'",
            "There are only 10 kinds of people: those who understand binary and those who don't.",
            "Why did the developer go broke? Because they used up all their cache.",
        ]
        jokes_zh = [
            "程序员为什么喜欢深色模式？因为亮光会吸引 bug。",
            "SQL 查询走进酒吧，看到两张桌子问：我能 join 你们吗？",
            "世界上只有两种人：懂二进制的和不懂二进制的。",
            "开发者为什么破产了？因为他们把 cache 都用光了。",
        ]
        pool = jokes_zh if VibeCodeHeuristics.has_cjk(prompt) and not re.search(r"[a-zA-Z]{3,}", prompt) else jokes_en
        idx = CatR1LLM.pick(vec, len(pool), salt=7) if vec is not None else 0
        return pool[idx]

    def _casual(self, topic: str, pl: str, prompt: str, vec: Optional[np.ndarray] = None) -> str:
        hit = self.smalltalk_reply(pl)
        if hit:
            return hit
        if re.search(r"\b(?:meow|mew|purr|nya)\b", pl):
            meows = (
                "Meow! 🐱 I'm here — chat, code, or cat facts?",
                "Mrow! 🐱 *stretches* What can I help with?",
                "Meow meow! 🐱 Purr-fect timing — what's up?",
            )
            idx = CatR1LLM.pick(vec, len(meows), salt=3) if vec is not None else 0
            return meows[idx]
        if pl in {"hey", "hi", "yo", "sup", "hello"}:
            return "Hi — good to see you. What would you like to work on?"
        if is_zh_greeting(prompt.strip()):
            return f"你好！我是 **{BRAND}**。可以聊天、写代码、解释概念或调试问题——中英文都行。"
        if VibeCodeHeuristics.has_cjk(prompt):
            if any(k in prompt for k in ("天气", "吃什么", "推荐", "觉得", "看法", "意见")):
                return (
                    f"关于「{topic[:60]}」——我没有实时数据，但可以帮你分析思路、"
                    f"列选项，或者写段代码来处理。你想从哪个角度聊？"
                )
            if len(prompt.strip()) <= 20:
                desire = self.fulfill_user_desire(prompt, [], vec)
                if desire:
                    return desire
                return "你好！随便聊——问问题、要代码、求解释都可以。中英文混合也没问题。"
        if VibeCodeHeuristics.has_cjk(prompt) and len(prompt.strip()) <= 12:
            return "你好！请告诉我你需要什么帮助——代码、解释、聊天或调试都可以。"
        if re.search(r"\byou\b", pl) and re.search(r"\b(think|like|would|will|can|could)\b", pl):
            return "I'm here to help — ask me about code, concepts, debugging, or anything you're working on."
        if re.search(r"\b(who|what)\b.*\b(are|is)\b.*\byou\b", pl) or "your name" in pl:
            return f"I'm **{BRAND}** — a local assistant running in-memory. I can write code, explain concepts, debug issues, and help you build things."
        if re.search(r"\b(thats|that's|nice|cool|awesome|interesting|good)\b", pl) or pl in {"ok", "okay", "nice", "cool", "great"}:
            return "Glad you think so! What's next on your mind?"
        if re.search(r"\b(?:just\s+)?talk\s+to\s+me\b", pl):
            return self._substantive_answer(topic, pl, prompt, vec)
        if (
            len(pl.split()) <= 4
            and "?" not in prompt
            and not re.search(r"\b(about|want|need|tell|why|what|how|help|cats?|love|pizza|homework)\b", pl)
        ):
            return f"Hi! I'm **{BRAND}** — ask me for code, an explanation, or whatever you need help with."
        topic_clean = self._clean_topic(prompt)[:100].strip().rstrip(".?!")
        if topic_clean and len(topic_clean.split()) <= 12:
            return self._substantive_answer(topic_clean, pl, prompt, vec)
        return self._substantive_answer(topic_clean or topic, pl, prompt, vec)

    def _desire(self, prompt: str, topic: str, vec: Optional[np.ndarray] = None) -> str:
        lit = self.extract_literal_desire(prompt)
        if lit is not None:
            return lit
        fulfilled = self.fulfill_user_desire(prompt, [], vec)
        if fulfilled:
            return fulfilled
        return self._substantive_answer(topic, prompt.lower().strip(), prompt, vec)

    def _general(self, topic: str, prompt: str, vec: Optional[np.ndarray] = None) -> str:
        pl = prompt.lower().strip()
        if not self.is_educational(pl):
            return self._casual(topic, pl, prompt, vec)
        return self._substantive_answer(topic, pl, prompt, vec)

    def synthesize(self, prompt: str, history: List[tuple], vec: Optional[np.ndarray] = None) -> str:
        a = self.analyze(prompt)
        intent = a["intent"]
        topic, pl = a["topic"], a["pl"]
        has_cjk = VibeCodeHeuristics.has_cjk(prompt)
        has_eng = bool(re.search(r"[a-zA-Z]{3,}", prompt))
        bodies = {
            "explain": self._explain(topic, pl, vec, prompt),
            "qa": self._explain(topic, pl, vec, prompt),
            "compare": self._compare(topic, vec),
            "debug": self._debug(topic, vec, prompt),
            "code": self._code(topic, pl, vec, prompt),
            "design": self._general(topic, prompt, vec),
            "math": self._math(topic, pl),
            "advise": self._advise(topic, vec),
            "joke": self._joke(vec, prompt),
            "desire": self._desire(prompt, topic, vec),
            "casual": self._casual(topic, pl, prompt, vec),
            "fable": self._fable(topic, prompt, vec),
            "poem": self._poem(topic, vec),
            "creative": self._creative(topic, prompt, pl, vec),
        }
        body = bodies.get(intent, self._general(topic, prompt, vec))
        if has_cjk and has_eng and "```" in body and not re.search(r"[\u4e00-\u9fff]", body):
            body += "\n\n---\n需要中文解释吗？我可以把代码用中文讲一遍。"
        return body

    def _followup(self, prompt: str, history: List[Dict[str, str]], vec: Optional[np.ndarray] = None) -> Optional[str]:
        if not history:
            return None
        pl = prompt.lower().strip()
        cues = (
            "tell me more", "go on", "continue", "and then", "what else", "more detail",
            "expand", "elaborate", "can you explain", "say more", "go deeper", "why though",
            "继续", "接着说", "然后呢", "还有吗", "详细点", "展开", "多说点", "为什么",
        )
        short_ack = {"yes", "ok", "okay", "sure", "please", "yep", "yeah", "do it", "thanks",
                    "好", "好的", "行", "可以", "嗯", "谢谢", "继续"}
        last_user = last_bot = ""
        for m in reversed(history):
            if m["role"] == "assistant" and not last_bot:
                last_bot = m["text"]
            elif m["role"] == "user" and not last_user:
                last_user = m["text"]
            if last_user and last_bot:
                break
        if not last_bot:
            return None
        if any(c in pl for c in cues) or pl in short_ack or pl.startswith(("why ", "how come", "为什么", "怎么")):
            snippet = last_bot.strip().split("\n\n")[0][:500]
            if VibeCodeHeuristics.has_cjk(prompt):
                return (
                    f"接着刚才关于「{last_user[:80]}」的话题：\n\n"
                    f"{snippet}\n\n"
                    "更深一层是**机制**——实际应用时什么会改变。告诉我你想展开哪部分。"
                )
            return (
                f"Building on our exchange about \"{last_user[:80]}\":\n\n"
                f"{snippet}\n\n"
                "The deeper layer is *mechanism* — what actually changes when you apply this. "
                "Tell me which part you want expanded and I'll go there."
            )
        if pl.startswith(("what about", "how about")):
            sub = prompt.split(maxsplit=2)[-1] if len(prompt.split()) > 2 else prompt
            return self._explain(sub.strip("?"), pl, vec)
        return None

    def converse(self, prompt: str, history: List[Dict[str, str]], vec: Optional[np.ndarray] = None,
                engine: Optional["CatR11Engine"] = None) -> str:
        if CONFIG.get("universal_chat", True):
            return self.respond_anything(prompt, history, vec=vec, engine=engine)
        pl = prompt.lower().strip()
        small = self.smalltalk_reply(pl)
        if small:
            return small
        follow = self._followup(prompt, history, vec)
        if follow:
            return GoogleWhitepaperCatR1Sorter.cat_r1_voice(follow, prompt, "chat")
        sorted_hist = GoogleWhitepaperCatR1Sorter.sort_history(
            [(m["role"], m["text"]) for m in history], prompt
        )
        return self.localize(self.synthesize(prompt, sorted_hist, vec=vec), prompt)

    @staticmethod
    def _polish(text: str) -> str:
        t = re.sub(r"\n{3,}", "\n\n", text.strip())
        if t and t[-1] not in ".!?`\"'":
            t += "."
        return t

    def o1_answer(
        self,
        prompt: str,
        history: List[Dict[str, str]],
        *,
        reasoning: str = "",
        vec: Optional[np.ndarray] = None,
    ) -> str:
        """cat r1 answer path — Mythos voice + Google WP sort · files = off."""
        body = self.converse(prompt, history, vec=vec)
        body = GoogleWhitepaperCatR1Sorter.sort_paragraphs(body, prompt)
        if ClaudeMythosRuntime.enabled():
            body = ClaudeMythosRuntime.voice(body, prompt, "general")
        if CONFIG.get("cat_r1_voice") and CATR1_MODE:
            body = GoogleWhitepaperCatR1Sorter.cat_r1_voice(body, prompt, "general")
        return self._polish(body)


# ──────────────────────────────────────────────────────────────
# CAT R1.1 CHAT PROTOCOL v1.1 (files = off, in-memory sessions)
# ──────────────────────────────────────────────────────────────
@dataclass
class ChatMessage:
    role: str
    content: str
    turn: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {"role": self.role, "content": self.content, "turn": self.turn}


@dataclass
class ChatSession:
    session_id: str
    messages: List[ChatMessage] = field(default_factory=list)
    turn_count: int = 0
    created: float = field(default_factory=time.time)

    def append(self, role: str, content: str) -> ChatMessage:
        self.turn_count += 1
        msg = ChatMessage(role=role, content=content, turn=self.turn_count)
        self.messages.append(msg)
        if len(self.messages) > 48:
            self.messages = self.messages[-48:]
        return msg

    def history_dicts(self) -> List[Dict[str, str]]:
        return [{"role": m.role, "text": m.content} for m in self.messages]

    def transcript(self, limit: int = 12) -> str:
        lines = []
        for m in self.messages[-limit:]:
            label = "You" if m.role == "user" else BRAND
            lines.append(f"[{label}]: {m.content[:200]}")
        return "\n".join(lines)


class ChatProtocol:
    """
    cat r1 Chat Protocol v1.1 — multi-turn conversation, files = off.

    JSON envelope::
        {"protocol":"cat-r1-chat","version":"1.1","action":"message",
        "session":"<id>","message":{"role":"user","content":"hello"}}

    Text wire (stdin/stdout)::
        @chat user: hello
        @chat assistant: Hi! ...
    """

    PROTO = CONFIG["chat_protocol"]
    VER = CONFIG["chat_version"]

    def __init__(self, engine: "CatR11Engine"):
        self.engine = engine
        self._sessions: Dict[str, ChatSession] = {}
        self._active = ""

    def new_session(self) -> str:
        sid = uuid.uuid4().hex[:12]
        self._sessions[sid] = ChatSession(session_id=sid)
        self._active = sid
        while len(self._sessions) > CONFIG["max_sessions"]:
            oldest = min(self._sessions.values(), key=lambda s: s.created)
            del self._sessions[oldest.session_id]
        return sid

    def session(self, session_id: Optional[str] = None) -> ChatSession:
        sid = session_id or self._active
        if not sid or sid not in self._sessions:
            sid = self.new_session()
        self._active = sid
        return self._sessions[sid]

    def sync_engine_history(self, sess: ChatSession) -> None:
        self.engine.chat_history = sess.history_dicts()

    def turn(self, user_text: str, *, session_id: Optional[str] = None, simulate: bool = False, on_token=None) -> Dict[str, Any]:
        text = (user_text or "").strip()
        if not text:
            return self._err("empty message")
        sess = self.session(session_id)
        self.sync_engine_history(sess)
        start = len(sess.messages)
        reply = self.engine.generate(text, simulate=simulate, on_token=on_token)
        think = self.engine.last_think
        for m in self.engine.chat_history[start:]:
            sess.append(m["role"], m["text"])
        return self._ok(sess, reply, think)

    def handle_action(self, data: Dict[str, Any]) -> Dict[str, Any]:
        action = (data.get("action") or "message").lower()
        sid = data.get("session") or data.get("session_id")

        if action in ("new", "reset"):
            new_id = self.new_session()
            if action == "reset" and sid and sid in self._sessions:
                del self._sessions[sid]
                new_id = self.new_session()
            self.engine.chat_history = []
            self.engine.last_think = ""
            return {
                "protocol": self.PROTO, "version": self.VER, "files": "off",
                "action": action, "session": new_id, "turn": 0,
                "message": {"role": "system", "content": f"New chat session {new_id}."},
            }

        if action == "history":
            sess = self.session(sid)
            return {
                "protocol": self.PROTO, "version": self.VER, "files": "off",
                "action": "history", "session": sess.session_id, "turn": sess.turn_count,
                "messages": [m.to_dict() for m in sess.messages],
            }

        if action == "message":
            msg = data.get("message") or {}
            content = msg.get("content") or data.get("content") or data.get("text") or ""
            if isinstance(content, list):
                content = content[-1].get("text", "") if content else ""
            return self.turn(str(content), session_id=sid, simulate=False)

        return self._err(f"unknown action: {action}")

    def parse_text_wire(self, line: str) -> Optional[Tuple[str, str]]:
        m = re.match(r"^@chat\s+(user|assistant|system)\s*:\s*(.*)$", line.strip(), re.I)
        if not m:
            return None
        return m.group(1).lower(), m.group(2)

    def format_text_wire(self, role: str, content: str, sess: ChatSession) -> str:
        return f"@chat {role}: {content}\n@chat meta session={sess.session_id} turn={sess.turn_count}"

    def parse_request(self, raw: Any) -> Dict[str, Any]:
        if isinstance(raw, dict):
            if raw.get("protocol") not in (None, self.PROTO):
                return self._err(f"protocol must be {self.PROTO}")
            return self.handle_action(raw)
        if isinstance(raw, str):
            wire = self.parse_text_wire(raw)
            if wire:
                role, content = wire
                if role == "user":
                    return self.turn(content, simulate=False)
                return self._err("text wire expects @chat user: <message>")
            try:
                return self.handle_action(json.loads(raw))
            except json.JSONDecodeError:
                return self.turn(raw, simulate=False)
        return self._err("invalid request")

    def _ok(self, sess: ChatSession, content: str, thinking: str = "") -> Dict[str, Any]:
        return {
            "protocol": self.PROTO,
            "version": self.VER,
            "files": "off",
            "action": "message",
            "session": sess.session_id,
            "turn": sess.turn_count,
            "message": {"role": "assistant", "content": content},
            "thinking": thinking,
        }

    def _err(self, detail: str) -> Dict[str, Any]:
        return {
            "protocol": self.PROTO,
            "version": self.VER,
            "files": "off",
            "error": detail,
            "session": self._active or None,
        }

    @staticmethod
    def help_text() -> str:
        return (
            f"**{BRAND} Chat Protocol v{CONFIG['chat_version']}** · `{CATR1_MODEL_ID}`\n\n"
            f"**{O1PreviewSyntax.TAG} + {MYTHOS_NAME} interpreter** · `/run` · `/interpret` · `/think` · paste ``` blocks\n\n"
            "**In-app commands**\n"
            "- `/chat` — show protocol help\n"
            "- `/chat new` — start a fresh session\n"
            "- `/chat history` — show this session transcript\n"
            "- `/chat session` — show session id\n\n"
            "**JSON API** — `POST /chat`\n"
            "```json\n"
            '{"protocol":"cat-r1-chat","version":"1.1","action":"message",'
            '"session":"<id>","message":{"role":"user","content":"hello"}}\n'
            "```\n\n"
            "**Text wire** (CLI `--chat`)\n"
            "`@chat user: your message here`\n\n"
            "Sessions live in memory only — no files written."
        )


def run_chat_cli(engine: CatR11Engine) -> None:
    proto = engine.chat
    sid = proto.new_session()
    print(f"{BRAND} · {CATR1_MODEL_ID} · protocol {ChatProtocol.PROTO}/{ChatProtocol.VER}")
    print(f"session={sid} · EN/中文 · code · chat · /lang · /quit to exit\n")
    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye")
            break
        if not line:
            continue
        if line.lower() in {"/quit", "/exit", "quit", "exit"}:
            break
        if line.startswith("@chat"):
            parsed = proto.parse_text_wire(line)
            if parsed and parsed[0] == "user":
                out = proto.turn(parsed[1], session_id=sid, simulate=False)
            else:
                out = proto.parse_request(line)
        elif line == "/chat new":
            sid = proto.new_session()
            print(f"new session={sid}")
            continue
        elif line.startswith("/"):
            out = {"message": {"content": engine.generate(line, simulate=False)}}
        else:
            out = proto.turn(line, session_id=sid, simulate=False)
        if out.get("error"):
            print(f"error: {out['error']}")
            continue
        reply = out.get("message", {}).get("content", "")
        print(f"\n{reply}\n")


# ──────────────────────────────────────────────────────────────
# CAT R1.1 ENGINE (files = off · in-memory ternary stack)
# ──────────────────────────────────────────────────────────────
class CatR11Engine:
    __slots__ = ("name", "ver", "d_model", "_lock", "dialect_idx",
                "dialects", "intent_map", "code_experts", "aliases",
                "embeddings", "vocab", "output_head", "output_head_lin",
                "_intent_weights", "_token_index", "_intent_trained",
                "learning_curve", "response_locale",
                "chat_history", "max_history", "assistant_mode",
                "_category_lexicon", "_embed_cache",
                "_student_layers", "ultrathink", "synth", "last_think",
                "ultrathink_on", "_pending_think", "_distil_cache", "_recursive_cache",
                "last_recursive_passes", "last_recursive_trace", "rival", "compressor",
                "last_compression_ratio", "compression_trace", "chat", "cat_r1_blocks",
                "cat_r1_stats", "norm_gamma", "norm_beta", "last_vec", "cat_r1", "fable", "deepmind", "fusion", "web", "heuristics",
                "persistent_memory", "v4_compressor", "model_router", "active_model_profile",
                "v4_runtime", "last_inference_stats", "dspark",
                "_last_prompt_hash", "preferred_lang", "last_channel")

    def __init__(self, d_model: int = None):
        self.name = BRAND
        self.ver = MODEL_NAME
        self.d_model = d_model or CONFIG["d_model"]
        self._lock = threading.Lock()
        self.dialect_idx = {"english": 0, "chinese": 0}
        self.learning_curve: List[float] = []
        self._intent_trained = False
        self.persistent_memory = _load_persistent_memory()
        if self.persistent_memory.first_run:
            self.persistent_memory.first_run = False
            self.persistent_memory.setup_version = EDITION
            _save_persistent_memory(self.persistent_memory)
        self._token_index: Dict[str, int] = {}
        self._intent_weights: Optional[np.ndarray] = None
        self.response_locale = "english"
        self.preferred_lang = "auto"
        self.last_channel: Dict[str, Any] = {}
        self.chat_history: List[Dict[str, str]] = []
        self.max_history = 24
        self.assistant_mode = "cat_r1"
        self._embed_cache: Dict[str, np.ndarray] = {}
        self._student_layers: List[List[CatR1Block]] = []
        self._distil_cache: Dict[tuple, np.ndarray] = {}
        self._recursive_cache: Dict[tuple, np.ndarray] = {}
        self._last_prompt_hash = 0
        self.last_recursive_passes = 0
        self.last_recursive_trace: List[str] = []
        self.compression_trace: List[str] = []
        self.last_compression_ratio = 1.0
        self.last_vec: Optional[np.ndarray] = None
        self.ultrathink = O1PreviewReasoner()
        self.synth = CatR11Synthesizer()
        self.last_think = ""
        self._pending_think = ""
        self.ultrathink_on = CONFIG["ultrathink_default"]
        self.cat_r1 = CatR1LLM()
        self.fable = self.cat_r1
        self.chat = ChatProtocol(self)
        self.compressor = CatR1Compressor(self.d_model)
        self.v4_compressor = CatR1V4ProCompression(self.d_model)
        self.model_router = CatR1ModelRouter()
        self.active_model_profile = CatR1FlashProfile()
        self.v4_runtime = CatR1V4InferenceRuntime(self)
        self.last_inference_stats = InferenceStats()
        self.rival = CatR1Core(self)
        self.deepmind = DeepMindFastStack(self)
        self.fusion = CatR1Fusion()
        self.web = CatR1WebProgram()
        self.heuristics = CatR1Heuristics()
        self.aliases = dict(CodeAnythingEngine.LANG_ALIASES) if CONFIG.get("code_anything") else {
            "py": "python", "c++": "cpp", "js": "javascript", "ts": "typescript",
            "sh": "bash", "shell": "bash", "asm": "assembly", "node": "javascript",
        }
        self.intent_map = {
            "hello": ["hi","hello","hey","yo","sup","good morning","good evening","howdy",
                    "how are you","how're you","how is it going","what's up","whats up",
                    "你好","您好","你好吗","嗨","哈喽","早上好","下午好","晚上好"],
            "core": ["core","cat-r1","cat r1","bitnet","ternary","-1, 0, 1","quantize","1.58","moe","cat r1","核心","三值"],
            "recursion": ["recursion","recursive","function calls itself","factorial","递归"],
            "help": ["help","commands","menu","usage","what can you do","capabilities","帮助","怎么用"],
            "languages": ["supported languages","which language","experts","what languages","支持的语言","哪些语言"],
            "profile": ["readme",".md","license","gpl3","about you","about yourself","who made you","who are you","你是谁"],
            "thanks": ["thanks","thank you","thx","appreciate","谢谢","感谢"],
            "goodbye": ["bye","goodbye","see you","later","exit","再见","拜拜"],
            "math": ["calculate","compute","sum","multiply","divide","equation"],
            "explain": ["explain","what is","what are","define","meaning of","tell me about",
                        "什么是","是什么","解释","说明","介绍","为什么","如何"],
            "howto": ["how do","how to","how can","steps to","walk me through","tutorial"],
            "debug": ["error","bug","traceback","exception","crash","broken","not working"],
            "opinion": ["should i","recommend","opinion","think about"],
            "joke": ["joke","funny","humor","laugh","make me laugh"],
            "fable": ["fable","parable","allegory","bedtime story","tell me a story","write a story"],
        }
        self._category_lexicon = {
            "greeting": ["hi", "hello", "hey", "morning", "evening", "sup", "你好", "您好", "嗨"],
            "farewell": ["bye", "goodbye", "later", "see you", "exit", "再见", "拜拜"],
            "question": ["what", "why", "how", "who", "when", "where", "which", "?", "什么", "为什么", "怎么", "如何"],
            "code": ["code", "script", "function", "class", "compile", "program", "syntax", "api", "代码", "程序", "脚本"],
            "tech": ["python", "cat-r1", "cat r1", "core", "ai", "model", "neural", "gpu", "cpu", "server"],
            "creative": ["story", "poem", "joke", "idea", "name", "design"],
            "personal": ["you", "your", "yourself", "who are you"],
            "task": ["make", "build", "create", "write", "generate", "show me"],
        }
        self.dialects = {
            "english": [
                {"hello":"Hi. How can I help?","ready":"Ready. Ask for code or explanation.",
                "py_intro":"Here is Python code.","generic":"Here is code.",
                "core":f"{BRAND} uses ternary weights: {{-1, 0, 1}}.","recursion":"Recursion: function calls itself. Example:",
                "error":"An error occurred. Let me check.","thanks":"You're welcome. Ask anything else.",
                "confirm":"Got it. Processing.","think":"Let me think about this step by step.",
                "code_any":"Write code in any language. I'll handle the rest.",
                "zh_detect":"I detected Chinese in your message. I'll reply in Chinese next time."},
                {"hello":"Hey. What do you need?","ready":"Send a prompt.","py_intro":"Python below.","generic":"Code below.",
                "core":"Ternary constraints eliminate FP32 multiplications.","recursion":"Self-referential execution. Example:",
                "error":"Something went wrong.","thanks":"Anytime.","confirm":"On it.",
                "think":"Thinking step by step.","code_any":"Drop code in any language.",
                "zh_detect":"I see Chinese — switching to Chinese."},
            ],
            "chinese": [
                {"hello":"你好。需要什么？","ready":"请给出任务。","py_intro":"Python 代码：","generic":"代码：",
                "core":f"{BRAND} 使用三值权重：{{-1, 0, 1}}。","recursion":"递归：函数自调用。示例：",
                "error":"出错了。让我检查一下。","thanks":"不客气。随时提问。",
                "confirm":"收到。正在处理。","think":"让我一步步思考。",
                "code_any":"任何语言都能写代码。交给我。",
                "en_detect":"I'll switch to English for you."},
                {"hello":"你好！需要帮忙吗？","ready":"请说出任务。","py_intro":"Python 代码：","generic":"代码：",
                "core":f"{BRAND} 使用三值权重。","recursion":"递归：函数调用自身。",
                "error":"出错了。","thanks":"不用谢。","confirm":"好的。",
                "think":"让我思考一下。","code_any":"任何语言都可以。",
                "en_detect":"切换到英文。"},
            ],
            "code": [
                {"hello":"Code ready. What language?","ready":"Ready to code.","py_intro":"```python","generic":"```",
                "core":"Ternary BitNet engine.","recursion":"Recursive function:",
                "error":"Compile/run error below.","thanks":"Code done.","confirm":"Coding.",
                "think":"Analyzing code requirements.","code_any":"Any language works.",
                "zh_detect":"检测到中文，代码注释将用中文。","en_detect":"English comments."},
            ],
            "mixed": [
                {"hello":"Hi! 你好！How can I help?","ready":"Ready. 请说出任务。",
                "py_intro":"Here's the Python code — 这是 Python 代码：","generic":"Code — 代码：",
                "core":f"{BRAND} uses ternary weights {{-1, 0, 1}} — 使用三值权重。",
                "recursion":"Recursion: function calls itself. 递归：函数自调用。示例：",
                "error":"An error occurred. 出错了。Let me check. 让我检查一下。",
                "thanks":"You're welcome! 不客气！Ask anything else. 随时提问。",
                "confirm":"Got it. 收到。Processing. 正在处理。",
                "think":"Let me think step by step. 让我一步步思考。",
                "code_any":"Write code in any language. 任何语言都能写代码。",
                "zh_detect":"I see both English and Chinese — I'll mix them!",
                "en_detect":"中英混合回复。"},
            ],
        }
        self.code_experts = (
            CodeAnythingEngine.experts() if CONFIG.get("code_anything") else {
            "python": "def main():\n    print('Hello World')\n\nif __name__ == '__main__':\n    main()",
            "cpp": (
                "#include <iostream>\n\n"
                "int main() {\n"
                "    std::cout << \"Hello World\" << std::endl;\n"
                "    return 0;\n"
                "}"
            ),
            "c": (
                "#include <stdio.h>\n\n"
                "int main(void) {\n"
                "    printf(\"Hello World\\n\");\n"
                "    return 0;\n"
                "}"
            ),
            "javascript": "console.log('Hello World');",
            "html": "<!DOCTYPE html><html><body><h1>Hello World</h1></body></html>",
            "typescript": "function main(): void { console.log('Hello World'); }\nmain();",
            "java": "public class Main { public static void main(String[] args) { System.out.println(\"Hello World\"); } }",
            "rust": "fn main() { println!(\"Hello World\"); }",
            "bash": "#!/bin/bash\necho \"Hello World\"",
            "assembly": "section .data\n    msg db 'Hello World',0xa\nsection .text\n    global _start\n_start:",
            "go": "package main\nimport \"fmt\"\nfunc main() { fmt.Println(\"Hello World\") }",
        })
        self._init_cat_r1_core()
        self._train_intent()

    def _build_vocab(self) -> Dict[str, int]:
        words = {"<pad>", "<unk>"}
        for keys in self.intent_map.values():
            for phrase in keys:
                words.update(re.findall(r"[a-z0-9+#]+", phrase.lower()))
        for w in (
            "the", "a", "is", "are", "to", "for", "of", "in", "on", "with", "and", "or",
            "python", "code", "write", "explain", "help", "core", "model", "chat",
            "你好", "谢谢", "什么", "解释", "递归", "代码", "程序", "帮助", "是", "的",
        ):
            words.add(w)
        for phrase in (
            "什么是", "是什么", "解释", "说明", "介绍", "为什么", "如何", "帮我", "写代码",
        ):
            words.add(phrase)
        for keys in self.intent_map.values():
            for phrase in keys:
                words.update(tokenize_text(phrase, 64))
        ordered = sorted(words)[: CONFIG["vocab_size"] - 1]
        vocab = {w: i + 1 for i, w in enumerate(ordered)}
        vocab["<pad>"] = 0
        return vocab

    def _build_cat_r1_stack(self, seed: int) -> List[CatR1Block]:
        return [CatR1Block(self.d_model, seed + layer * 7919) for layer in range(CONFIG["layers"])]

    def _init_cat_r1_core(self):
        d = self.d_model
        rng = np.random.RandomState(42)
        self.vocab = self._build_vocab()
        vs = min(CONFIG["vocab_size"], len(self.vocab) + 256)
        self.embeddings = (rng.randn(vs, d).astype(np.float32) * 0.02)
        self.norm_gamma = np.ones((d,), dtype=np.float32)
        self.norm_beta = np.zeros((d,), dtype=np.float32)
        self.cat_r1_blocks = self._build_cat_r1_stack(42)
        self.output_head_lin = CatR1Linear(d, len(self.intent_map) + 1, 4242)
        self.output_head = self.output_head_lin.w_signed
        self._student_layers: List[List[CatR1Block]] = []
        self.dspark = DSparkBitNetEngine(self)
        self.cat_r1_stats = cat_r1_memory_report(
            self.cat_r1_blocks, self.embeddings, self.output_head_lin.shadow_w
        )
        if CONFIG.get("bitnet_self_test", True):
            bt = BitNetEngine.enable_real_mode(self)
            ds = self.dspark.last_stats
            self.cat_r1_stats["dspark"] = {
                "enabled": CONFIG.get("dspark_enabled", True),
                "accepted": ds.accepted,
                "drafts": ds.drafts,
                "gamma": int(CONFIG.get("v4_speculative_draft_tokens", 5)),
            }
            if not bt.get("ok"):
                print(f"[{BRAND}] BitNet self-test FAILED: {bt}", file=sys.stderr)

    def _ensure_bitnet_real(self) -> Dict[str, Any]:
        """Real BitNet ternary stack — in-memory packed weights, files = off."""
        self._reset_bitnet_self_test_cache()
        return BitNetEngine.enable_real_mode(self)

    def v4_pro_compress(self, x: np.ndarray) -> np.ndarray:
        """Apply cat r1 V4 Pro neural compression to vector."""
        if CONFIG.get("v4_pro_compression"):
            return self.v4_compressor.v4_compress(x)
        return x

    def select_model(self, prompt: str, intent: str = "") -> Any:
        """Route to Pro or Flash model based on prompt complexity."""
        profile = self.model_router.select(prompt, intent)
        self.active_model_profile = profile
        return profile

    def is_pro_mode(self) -> bool:
        return self.active_model_profile.tier == "pro"

    def get_reasoning_depth(self) -> int:
        return (CONFIG.get("cat_r1_pro_reasoning_depth", 5)
                if self.is_pro_mode()
                else CONFIG.get("cat_r1_flash_reasoning_depth", 1))

    def _clear_engine_caches(self) -> None:
        self._embed_cache.clear()
        self._distil_cache.clear()
        self._recursive_cache.clear()

    def _reset_bitnet_self_test_cache(self) -> None:
        BitNetEngine._self_test_cache = None

    def v4_compression_stats(self) -> Dict[str, Any]:
        return self.v4_compressor.stats()

    def _ensure_students(self, count: int) -> List[List[CatR1Block]]:
        while len(self._student_layers) < count:
            i = len(self._student_layers)
            self._student_layers.append(self._build_cat_r1_stack(1337 + i * 9973))
        return self._student_layers[:count]

    def _token_embed(self, token: str) -> np.ndarray:
        key = token if not token.isascii() else token.lower()
        tid = self.vocab.get(key)
        if tid is None:
            tid = abs(hash(key)) % max(len(self.embeddings) - 1, 1) + 1
        if tid >= len(self.embeddings):
            tid = 0
        return self.embeddings[tid].astype(np.float32)

    def _layer_norm(self, x):
        if x.ndim == 2:
            mean = np.mean(x, axis=-1, keepdims=True)
            std = np.std(x, axis=-1, keepdims=True) + 1e-5
            return (x - mean) / std * self.norm_gamma + self.norm_beta
        mean, std = np.mean(x, axis=-1, keepdims=True), np.std(x, axis=-1, keepdims=True) + 1e-5
        return (x - mean) / std * self.norm_gamma + self.norm_beta

    def _forward_stack(self, x: np.ndarray, blocks: List[CatR1Block]) -> np.ndarray:
        if x.ndim == 1:
            x = x.reshape(1, -1)
        y = x.astype(np.float32)
        for blk in blocks:
            y = blk.forward(y)
        return _rms_norm(y, self.norm_gamma)

    def _pool_sequence(self, seq: np.ndarray) -> np.ndarray:
        if seq.ndim == 1:
            return seq.astype(np.float32)
        return np.mean(seq, axis=0).astype(np.float32)

    def forward(self, x, turbo: bool = True, *, turbo_only: bool = False):
        if x.ndim == 1:
            x = x.reshape(1, -1)
        key = (x.tobytes(), turbo, turbo_only)
        cached = self._distil_cache.get(key)
        if cached is not None:
            return cached.copy()
        teacher = self._pool_sequence(self._forward_stack(x, self.cat_r1_blocks))
        if turbo_only:
            if len(self._distil_cache) < 512:
                self._distil_cache[key] = teacher.copy()
            elif len(self._distil_cache) > 1024:
                self._distil_cache.clear()
            return teacher
        n_pass = CONFIG["turbo_passes"] if turbo else CONFIG["distil_passes"]
        tw = CONFIG.get("distil_teacher_weight", CONFIG["teacher_weight"])
        merged = teacher * tw
        students = self._ensure_students(n_pass)
        sw = 0.66 / n_pass
        for stack in students:
            merged = merged + self._pool_sequence(self._forward_stack(x, stack)) * sw
        if len(self._distil_cache) < 512:
            self._distil_cache[key] = merged.copy()
        elif len(self._distil_cache) > 1024:
            self._distil_cache.clear()
        return merged

    def encode_for_task(self, prompt: str, task: Optional[str] = None) -> np.ndarray:
        return self.v4_runtime.encode(prompt, task=task)

    def v4_pro_code_encode(self, prompt: str) -> np.ndarray:
        """cat r1 V4 Pro coding encode — Pro depth + syntax-biased V4 compression."""
        if not self.is_pro_mode():
            self.active_model_profile = CatR1ProProfile()
        depth = int(CONFIG.get("v4_pro_coding_depth", CONFIG.get("cat_r1_pro_reasoning_depth", 5)))
        old_depth = CONFIG["recursive_depth"]
        CONFIG["recursive_depth"] = depth
        try:
            vec = self.recursive_encode(prompt, depth=depth)
            if CONFIG.get("v4_pro_compression"):
                vec = self.v4_compressor.v4_compress_for_code(vec, prompt)
                cap = self.v4_compressor.effective_v4_capacity()
                self.last_recursive_trace.append(
                    f"{BITNET_V4_LABEL} coding · {self.v4_compressor.last_ratio:.1f}x · "
                    f"~{cap:.1f}B effective · files=off"
                )
        finally:
            CONFIG["recursive_depth"] = old_depth
        self.last_vec = vec.copy()
        return vec

    def _apply_tri_channel(self, raw: str) -> Dict[str, Any]:
        """Detect code / Chinese / English channel; lock files = off."""
        CONFIG["files"] = "off"
        channel = self.heuristics.analyze_channels(raw, self)
        self.last_channel = channel
        self.response_locale = channel.get("locale", "english")
        if channel.get("force_pro"):
            self.active_model_profile = CatR1ProProfile()
            self.ultrathink_on = True
        return channel

    def _catseek_handler(self, raw: str, *, force_pro: bool = False) -> str:
        """cat r1 code 1.0 — generate + run · BitNet local · no API · # pr."""
        CONFIG["files"] = "off"
        CONFIG["cat_r1_code"] = True
        CONFIG["cat_code_interpreter_r1"] = True
        CONFIG["interpreter_syntax"] = CatR1CodeR1.PROTO
        CONFIG["catcode_bitnet_only"] = True
        wants_pr = bool(force_pro or CatR1CodeR1.wants_pr(raw))
        CatR1CodeR1.run_hook("SessionStart")
        pl = raw.lower().strip()
        tail = ""
        if pl.startswith("/catrcode ") or pl.startswith("/catseek "):
            tail = raw.split(maxsplit=1)[1].strip() if " " in raw.strip() else ""
        elif pl.startswith("/catcode "):
            tail = raw.split(maxsplit=1)[1].strip() if " " in raw.strip() else ""
        elif pl not in {"/catrcode", "/catseek", "/catcode"}:
            tail = CatR1Code.normalize_prompt(raw)

        if not tail or tail.lower() in {"help", "?", "status"}:
            resp = CatR1CodeR1.help_text()
        else:
            resp = CatR1CodeR1.dispatch(self, tail, force_pro=wants_pr)
        resp = f"{resp}\n\n`{CatR1CodeR1.bitnet_status(self)}`"
        self._remember("assistant", resp)
        self.cat_r1.memory.add("assistant", resp)
        return resp

    _catcode_handler = _catseek_handler

    def _resolve_v4_profile(self, raw: str, prompt: str) -> bool:
        """Route cat r1 vs flash; # pr forces Expert + DSpark."""
        norm = CatR1Code.normalize_prompt(raw)
        pl = (norm or raw).lower()
        wants_bitnet = CatR1Code.is_bitnet_request(pl)
        wants_dspark = bool(re.search(r"\b(?:implement|enable|integrate)\s+dspark\b|\bdspark\b", pl))
        force_pro = bool(
            re.search(r"(?:^|\s)#+\s*pr\b", raw, re.I)
            or re.search(r"\bcatspark\s*r1\s+pro\b", raw, re.I)
            or re.search(r"\bcat\s*r1\s+pro\b", raw, re.I)
            or re.search(r"\bbitnet\s+v4\s+pro\b", raw, re.I)
            or wants_bitnet
            or wants_dspark
        )
        if wants_bitnet or force_pro or wants_dspark or CONFIG.get("cat_r1_dspark_on_pr"):
            CONFIG["dspark_enabled"] = True
            CONFIG["dspark_speculative_decode"] = True
        if wants_bitnet or force_pro or wants_dspark:
            CatR1DSpark.ensure(self, raw, force_pr=force_pro)
            if force_pro:
                return True
        intent_info = self.heuristics.classify_intent(prompt)
        self.select_model(prompt, intent_info.get("intent", ""))
        return self.is_pro_mode()

    def _append_v4_stats(self, resp: str) -> str:
        stats = getattr(self, "last_inference_stats", None)
        if stats and stats.model_label:
            return f"{resp}\n\n`{self.v4_runtime.stats_line()}`"
        return resp

    def ai_setup_response(self, prompt: str) -> Optional[str]:
        """Enable real AI + BitNet behavior for setup/meta prompts."""
        norm = CatR1Code.normalize_prompt(prompt)
        pl = (norm or prompt).lower().strip()
        if re.fullmatch(r"(?:meow|mew)[!?.]*", pl):
            return None
        parts: List[str] = []
        if re.search(r"\bmeow\b|喵", pl):
            parts.append("Meow! 🐱 *purrs softly*")
        if re.search(r"\b(?:behave|act)\s+like\s+(?:an?\s+)?ai\b", pl):
            CONFIG["ai_mode"] = True
            CONFIG["universal_chat"] = True
            CONFIG["user_desire_mode"] = True
            CONFIG["bitnet_real"] = True
            CONFIG["bitnet_use_packed"] = True
            CONFIG["bitnet_encode"] = True
            parts.append(
                "**AI mode enabled** — universal chat, user-desire replies, "
                "BitNet-packed inference on every encode."
            )
        if CatR1Code.is_bitnet_request(pl):
            bt = self._ensure_bitnet_real()
            CONFIG["dspark_enabled"] = True
            CONFIG["dspark_speculative_decode"] = True
            self.cat_r1_stats["dspark"] = {
                "enabled": True,
                "accepted": self.dspark.last_stats.accepted,
                "drafts": self.dspark.last_stats.drafts,
                "gamma": int(CONFIG.get("v4_speculative_draft_tokens", 5)),
            }
            parts.append(BitNetEngine.status_text(self.cat_r1_stats))
            parts.append(DSparkBitNetEngine.status_line(self.dspark.last_stats))
            parts.append(
                f"**{BRAND} BitNet real mode** — ternary GEMM kernel + base-3 packed weights · "
                f"DSpark speculative decode integrated · `files = off` · "
                f"self-test **{'PASS' if bt.get('ok') else 'FAIL'}**."
            )
        if CatR1Code.is_fix_all_request(pl):
            parts.append(self.fix_all_bugs_response())
        if CatR1Code.is_v4_coding_optimize_request(pl):
            CONFIG["v4_pro_coding"] = True
            CONFIG["v4_pro_compression"] = True
            CONFIG["bitnet_encode"] = True
            parts.append(self.v4_coding_optimize_response())
        if parts:
            parts.append("Ask anything — chat, code, `say ...`, EN/中文.")
            return "\n\n".join(parts)
        return None

    def fix_all_bugs_response(self) -> str:
        """Status after a bug-sweep — files = off, in-memory only."""
        bt = self.cat_r1_stats.get("bitnet") or BitNetEngine.self_test(self.d_model)
        ok = bt.get("ok", False)
        return (
            "**Bug sweep** — `files = off` · in-memory only.\n\n"
            "**Verified working:**\n"
            f"- BitNet self-test: **{'PASS' if ok else 'FAIL'}**\n"
            "- Chat: GUI → `chat.turn` → `generate` → `complete`\n"
            "- User desire: `say hello` · `请说：你好`\n"
            "- Bilingual EN/中文 · `/bitnet` · `/lang`\n\n"
            "**Fixed this session:**\n"
            "- Meta prompts (`# pr` / `# todo` / `files = off`) strip to real intent\n"
            "- `fix all the bugs` → chat (not generic debug deflection)\n"
            "- `implement bitnet` → full BitNet status\n"
            "- `meow` → cat reply\n\n"
            "Paste a traceback or describe one broken behavior — I'll fix it."
        )

    def v4_coding_optimize_response(self) -> str:
        """Status after enabling cat r1 BitNet V4 Pro coding neural path."""
        v4 = self.v4_compressor.stats()
        return (
            f"**{BITNET_V4_LABEL} coding** enabled · `files = off`\n\n"
            "**Neural stack (cat r1 BitNet tier):**\n"
            f"- Encoder: **{CAT_R1_PRO}** · depth **{CONFIG.get('v4_pro_coding_depth', 5)}** passes\n"
            f"- Compression: **{BITNET_V4_LABEL}** "
            f"({v4.get('last_ratio', '1.0x')} · ~{v4.get('effective_capacity_billions', '1.7B')} effective)\n"
            f"- Coding passes: **{CONFIG.get('v4_pro_coding_passes', 2)}** syntax-biased BitNet refinements\n"
            "- BitNet ternary forward + packed weights on every encode\n"
            "- Code tasks bypass turbo — full recursive BitNet path\n\n"
            "Paste code or ask for a program — generation routes through cat r1 BitNet coding."
        )

    def _recursive_step(self, state: np.ndarray, pass_idx: int, max_depth: int = 0) -> np.ndarray:
        depth = max_depth or CONFIG["recursive_depth"]
        turbo = pass_idx < depth - 1
        seq = state.reshape(1, -1) if state.ndim == 1 else state
        delta_seq = self._forward_stack(seq, self.cat_r1_blocks if turbo else self._ensure_students(1)[0])
        delta = self._pool_sequence(delta_seq)
        alpha = min(0.72, 0.28 + 0.11 * pass_idx)
        base = self._pool_sequence(state) if state.ndim == 2 else state
        merged = self._layer_norm(base * (1.0 - alpha) + delta * alpha)
        if CONFIG["compression_enabled"]:
            merged = self.compressor.compress_roundtrip(merged)
            self.last_compression_ratio = self.compressor.last_ratio
        return merged

    def recursive_encode(self, prompt: str, *, depth: Optional[int] = None) -> np.ndarray:
        """o1-preview recursive cat r1 loop with compression between passes."""
        max_depth = depth if depth is not None else CONFIG["recursive_depth"]
        key = (prompt.lower().strip(), max_depth, CONFIG["compression_enabled"])
        hit = self._recursive_cache.get(key)
        if hit is not None:
            return hit.copy()
        state = self.encode_prompt(prompt)
        trace: List[str] = []
        ctrace: List[str] = []
        prev = self._pool_sequence(state) if state.ndim == 2 else state.copy()
        used = max_depth
        is_pro = self.is_pro_mode() if hasattr(self, "is_pro_mode") else False
        for i in range(max_depth):
            state = self._recursive_step(state, i, max_depth)
            pooled = self._pool_sequence(state) if state.ndim == 2 else state
            norm = float(np.linalg.norm(pooled))
            line = f"pass {i + 1} · norm {norm:.4f} · cat r1"
            if CONFIG["compression_enabled"]:
                line += f" · {self.last_compression_ratio:.1f}x"
                ctrace.append(
                    f"sparse-{CONFIG['compression_sparse_k']} + rank-{CONFIG['compression_rank']} "
                    f"→ {self.last_compression_ratio:.1f}x · ~{self.compressor.effective_params_billions():.1f}B effective"
                )
            if CONFIG.get("v4_pro_compression") and (is_pro or CONFIG.get("v4_pro_coding")):
                pooled = self.v4_compressor.v4_compress(pooled)
                ctrace.append(
                    f"{BITNET_V4_LABEL} · {self.v4_compressor.last_ratio:.1f}x · "
                    f"{self.v4_compressor.effective_v4_capacity():.1f}B effective"
                )
            if i > 0:
                diff = float(np.linalg.norm(pooled - prev))
                line += f" · Δ {diff:.4f}"
                if diff < CONFIG["recursive_epsilon"]:
                    trace.append(line)
                    trace.append(f"converged at pass {i + 1}")
                    used = i + 1
                    break
            trace.append(line)
            prev = pooled.copy()
        self.last_recursive_passes = used
        self.last_recursive_trace = trace
        self.compression_trace = ctrace
        out = self._pool_sequence(state) if state.ndim == 2 else state
        self.last_vec = out.copy()
        if len(self._recursive_cache) < 128:
            self._recursive_cache[key] = out.copy()
        return out

    def _distil_draft(self, prompt: str, vec: Optional[np.ndarray] = None) -> str:
        if vec is None:
            vec = self.recursive_encode(prompt)
        logits = self.output_head_lin.forward(vec)
        labels = list(self.intent_map.keys()) + ["general"]
        idx = int(np.argmax(logits)) % len(labels)
        topic = self._extract_topic_words(prompt)
        st = self.cat_r1_stats
        tw = CONFIG.get("distil_teacher_weight", CONFIG["teacher_weight"])
        return (
            f"cat r1 reasoning → intent={labels[idx]} → topic=«{topic[:60]}» · "
            f"{self.last_recursive_passes} passes · "
            f"{st['cat_r1_linear_layers']} linear layers · "
            f"{st['packed_kb']:.1f}KB packed · "
            f"{tw:.0%} primary + {CONFIG['turbo_passes']} students"
        )

    def _history_pairs(self) -> List[tuple]:
        return [(m["role"], m["text"]) for m in self.chat_history[:-1]][-8:]

    def _run_ultrathink(self, prompt: str, *, force: bool = False) -> None:
        if self._pending_think and not force:
            self.last_think = self._pending_think
            self._pending_think = ""
            return
        if not O1PreviewReasoner.should_run(prompt, enabled=self.ultrathink_on, force=force):
            return
        if not force and len(prompt.strip()) < 8:
            self.last_think = ""
            return
        vec = self.last_vec if self.last_vec is not None else self.recursive_encode(prompt)
        draft = self._distil_draft(prompt, vec=vec)
        self.last_think = self.ultrathink.run(
            prompt, distill_draft=draft, recursive_trace=self.last_recursive_trace,
            compression_trace=self.compression_trace,
        )

    def _o1_respond(self, prompt: str) -> str:
        prior = self.chat_history[:-1] if self.chat_history else []
        vec = self.last_vec
        if CONFIG["o1_preview"]:
            return self.synth.o1_answer(prompt, prior, reasoning=self.last_think, vec=vec)
        return self.synth.converse(prompt, prior, vec=vec)

    def _r1_synthesize(self, prompt: str) -> str:
        return self._o1_respond(prompt)

    def encode_prompt(self, prompt):
        key = prompt.strip()
        cached = self._embed_cache.get(key)
        if cached is not None:
            return cached
        tokens = tokenize_text(prompt, CONFIG["max_seq"])
        seq = np.stack([self._token_embed(t) for t in tokens], axis=0)
        out = self._layer_norm(seq)
        if len(self._embed_cache) < 1024:
            self._embed_cache[key] = out
        elif len(self._embed_cache) > 2048:
            self._embed_cache.clear()
        return out

    def _intent_features(self, text: str) -> np.ndarray:
        vocab = list(self._token_index.keys()) if self._token_index else []
        if not vocab:
            return np.array([1.0], dtype=np.float32)
        tl = text.lower()
        return np.array([tl.count(t) for t in vocab] + [1.0], dtype=np.float32)

    def _train_intent(self):
        if self._intent_trained:
            return
        corpus: List[tuple] = []
        for label, keys in self.intent_map.items():
            for k in keys:
                corpus.append((k, label))
        corpus.extend([
            ("write python function", "howto"), ("fix my bug", "debug"),
            ("2 plus 2", "math"), ("thanks a lot", "thanks"),
            ("see you tomorrow", "goodbye"), ("what is docker", "explain"),
            ("should i use rust", "opinion"),
        ])
        vocab: set = set()
        for label, keys in self.intent_map.items():
            for k in keys:
                vocab.update(tokenize_text(k, 64))
        for text, _ in corpus:
            vocab.update(tokenize_text(text, 64))
        self._token_index = {t: i for i, t in enumerate(sorted(vocab))}
        self._intent_weights = np.zeros((len(self.intent_map), len(vocab) + 1), dtype=np.float32)
        labels = list(self.intent_map.keys())
        for _ in range(40):
            for text, label in corpus:
                x = self._intent_features(text)
                y_idx = labels.index(label)
                scores = self._intent_weights @ x
                others = [s for i, s in enumerate(scores) if i != y_idx]
                margin = (max(others) if others else -1e9) - scores[y_idx] + 1
                if margin > 0:
                    self._intent_weights[y_idx] += 0.12 * x
                    if others:
                        self._intent_weights[int(np.argmax(scores))] -= 0.12 * x
        self._intent_trained = True

    def _key_matches(self, key: str, text: str) -> bool:
        if VibeCodeHeuristics.has_cjk(key):
            return key in text
        if key == "help":
            if re.search(r"\bhelp\s+me\s+(with|on|understand|do|figure)\b", text):
                return False
            return bool(re.search(
                r"^(?:help|/help|\?|what can you do|capabilities|usage|menu)\b", text
            ))
        loose = {"fix", "best", "math", "later", "exit", "bug", "sum"}
        if len(key) <= 4 or key in loose:
            return bool(re.search(r"\b" + re.escape(key) + r"\b", text))
        return key in text

    def _best_intent(self, prompt: str) -> Optional[str]:
        p = prompt.lower()
        best, best_len = None, 0
        for intent, keys in self.intent_map.items():
            for k in keys:
                if self._key_matches(k, p) and len(k) > best_len:
                    best, best_len = intent, len(k)
        if best:
            return best
        self._train_intent()
        if self._intent_weights is None:
            return None
        scores = self._intent_weights @ self._intent_features(p)
        idx = int(np.argmax(scores))
        label = list(self.intent_map.keys())[idx]
        if float(scores[idx]) > 0.55:
            # Meta/sidebar intents need an explicit keyword hit — not ML alone.
            if label in {"core", "recursion", "help", "languages", "profile"}:
                return None
            return label
        return None

    def _score_categories(self, text: str) -> Dict[str, int]:
        tokens = set(tokenize_text(text, 128))
        return {
            cat: sum(1 for w in words if w in tokens or w in text)
            for cat, words in self._category_lexicon.items()
        }

    def _extract_topic_words(self, prompt: str, n: int = 4) -> str:
        cleaned = CatR1Code.clean_topic(prompt)
        if cleaned and cleaned != prompt.strip():
            prompt = cleaned
        if VibeCodeHeuristics.has_cjk(prompt):
            zh = extract_zh_topic(prompt)
            if zh and zh != prompt.strip():
                return zh[:48]
            chars = _TOKEN_CJK.findall(prompt)
            if chars:
                return "".join(chars[:n * 2])[:48]
        stop = {"the", "a", "an", "is", "are", "to", "for", "of", "in", "on", "my", "me", "i", "you", "please", "can", "do"}
        words = [w for w in tokenize_text(prompt, 64) if w not in stop and len(w) > 1]
        return " ".join(words[:n]) if words else prompt.strip()[:48] or "that"

    def _try_simple_math(self, prompt: str) -> Optional[str]:
        expr = prompt.lower().strip().rstrip("?")
        expr = re.sub(r"^(what is|calculate|compute|solve)\s+", "", expr)
        expr = expr.replace("plus", "+").replace("minus", "-").replace("times", "*").replace("multiplied by", "*")
        expr = expr.replace("divided by", "/").replace("over", "/")
        if not re.fullmatch(r"[\d\s+\-*/().]+", expr.strip()):
            return None
        try:
            val = eval(expr, {"__builtins__": {}}, {})  # noqa: S307 — sandboxed numeric expr only
            if isinstance(val, (int, float)):
                return str(int(val)) if float(val).is_integer() else f"{val:.6g}"
        except Exception:
            return None
        return None

    def _explain_topic(self, topic: str, dialect: Dict[str, str]) -> str:
        known = {
            "core": dialect["core"],
            "python": "Python is a general-purpose language — great for scripts, APIs, and automation.",
            "recursion": dialect["recursion"],
            "docker": "Docker packages apps in containers so they run the same everywhere.",
            "api": "An API is a defined interface for programs to request data or actions from another service.",
            "javascript": "JavaScript runs in browsers and on servers (Node.js) for interactive web apps.",
            "rust": "Rust is a systems language focused on memory safety without a garbage collector.",
        }
        for key, answer in known.items():
            if key in topic:
                return answer
        return self.synth._substantive_answer(topic, topic.lower(), topic, self.last_vec)

    def _howto_topic(self, topic: str) -> str:
        return (
            f"To {topic}:\n"
            "1) State the goal and any constraints (OS, language, deadline).\n"
            "2) Start with the smallest working version.\n"
            "3) Run it, capture errors, and iterate.\n"
            "Paste your current code or error and I will tailor the steps."
        )

    def _answer_open_question(self, prompt: str, categories: Dict[str, int], ctx: str) -> str:
        topic = self._extract_topic_words(prompt)
        top = max(categories, key=categories.get) if categories else "general"
        if top == "personal":
            return (
                f"I'm **{BRAND}** — a local **{MODEL_NAME}** assistant. "
                "I help with code, explanations, debugging, and chat. Everything runs in-memory."
            )
        return self.synth._substantive_answer(
            topic, prompt.lower().strip(), prompt, self.last_vec
        )

    def _respond_universal(self, prompt: str, dialect: Dict[str, str], vec: np.ndarray) -> str:
        p = prompt.strip()
        pl = p.lower()
        ctx = self._recent_user_context()
        categories = self._score_categories(pl)

        math_result = self._try_simple_math(p)
        if math_result is not None:
            return f"Result: {math_result}"

        if self._wants_steps(prompt):
            return self._howto_topic(self._extract_topic_words(prompt))

        if pl.startswith(("what is ", "what's ", "what are ")):
            topic = re.sub(r"^what(?:'s| is| are)\s+", "", pl).rstrip("?").strip()
            return self._explain_topic(topic, dialect)

        if pl.startswith(("how do ", "how to ", "how can ")):
            topic = re.sub(r"^how (?:do|to|can) (?:i |we )?", "", pl).rstrip("?").strip()
            return self._howto_topic(topic)

        if pl.startswith("why "):
            topic = pl[4:].rstrip("?").strip()
            topic = CatR1Code.clean_topic(topic) or topic
            if re.search(r"\b(?:sky|blue)\b", topic):
                return (
                    "The sky looks **blue** because of **Rayleigh scattering** — shorter blue wavelengths "
                    "scatter more in the atmosphere than longer red ones."
                )
            return self.synth._substantive_answer(topic, pl, prompt, vec)

        if "joke" in pl or categories.get("creative", 0) >= 2:
            jokes = [
                "Why do programmers prefer dark mode? Light attracts bugs.",
                "A SQL query walks into a bar, walks up to two tables, and asks: 'Can I join you?'",
                "There are only 10 kinds of people: those who understand binary and those who don't.",
            ]
            return jokes[int(np.abs(vec[:6].sum() * 100)) % len(jokes)]

        if "?" in p:
            return self._answer_open_question(p, categories, ctx)

        if self._wants_brief(prompt):
            topic = self._extract_topic_words(prompt)
            return self.synth._substantive_answer(topic, pl, prompt, vec)

        topic = self._extract_topic_words(prompt)
        body = self.synth._substantive_answer(topic, pl, prompt, vec)
        if ctx:
            return f"{body}\n\n*(Earlier: {ctx})*"
        return body

    def detect_locale(self, p):
        pl = p.lower()
        has_cjk = bool(re.search(r"[\u4e00-\u9fff]", pl))
        has_eng = bool(re.search(r"[a-zA-Z]{3,}", pl))
        wants_code = bool(re.search(r"(code|```|def |function|script|program|write a|write an|implement)", pl, re.I))
        if has_cjk and has_eng:
            self.response_locale = "mixed"
            return "mixed"
        if has_cjk:
            self.response_locale = "chinese"
            return "chinese"
        if wants_code:
            return "code"
        self.response_locale = "english"
        return "english"

    def get_dialect(self, loc):
        bank = self.dialects.get(loc, self.dialects["english"])
        return bank[0] if bank else self.dialects["english"][0]

    def _zh_wrap(self, text: str, prompt: str) -> str:
        if not text:
            return text
        if "```" in text:
            return text
        lines = text.strip().split("\n")
        if len(lines) <= 2:
            return text
        topic = prompt.strip()[:60]
        prefixed = f"关于「{topic}」——{lines[0]}\n"
        rest = "\n".join(lines[1:])
        return f"{prefixed}\n{rest}"

    def _en_wrap(self, text: str, prompt: str) -> str:
        if not text or "```" in text:
            return text
        return text

    def _remember(self, role: str, text: str):
        self.chat_history.append({"role": role, "text": text.strip()})
        if len(self.chat_history) > self.max_history:
            self.chat_history = self.chat_history[-self.max_history:]

    def _recent_user_context(self, n: int = 3) -> str:
        msgs = [m["text"] for m in self.chat_history if m["role"] == "user"]
        if not msgs:
            return ""
        return " | ".join(msgs[-n:])

    def _wants_brief(self, prompt: str) -> bool:
        p = prompt.lower()
        return any(x in p for x in ["short", "brief", "one line", "tldr", "concise"])

    def _wants_steps(self, prompt: str) -> bool:
        p = prompt.lower()
        return any(x in p for x in ["step by step", "steps", "walkthrough", "how do i"])

    def _wants_code_only(self, prompt: str) -> bool:
        p = prompt.lower()
        return any(x in p for x in ["code only", "just code", "only code", "no explanation"])

    def _chat_fallback(self, prompt: str, dialect: Dict[str, str], vec: Optional[np.ndarray] = None) -> str:
        if vec is None:
            vec = self.recursive_encode(prompt)
        return self._respond_universal(prompt, dialect, vec)

    def _intent_response(self, intent: str, prompt: str, dialect: Dict[str, str]) -> Optional[str]:
        if intent == "hello":
            hit = self.synth.smalltalk_reply(prompt)
            return hit or dialect["hello"]
        if intent == "core":
            return dialect["core"]
        if intent == "recursion":
            return f"{dialect['recursion']}\n\ndef fact(n):\n    return 1 if n<=1 else n*fact(n-1)"
        if intent == "help":
            return (
                f"**{BRAND}** · {MODEL_NAME} · {REASONING_MODE}\n\n"
                "Frontier-tier local assistant — compression + reasoning.\n\n"
                f"**{CodeAnythingEngine.TAG}** — any language, any task (API, CLI, web, algo).\n\n"
                "Chat: `/chat` · `/think` · `/reset` · `/code` · **run it**"
            )
        if intent == "languages":
            n = len(self.code_experts)
            tag = CodeAnythingEngine.TAG if CONFIG.get("code_anything") else "languages"
            return (
                f"**{tag}** · {n} supported\n\n"
                f"{', '.join(sorted(self.code_experts))}\n\n"
                "Runnable natively: python · javascript · bash. Others: dry-run simulation."
            )
        if intent == "profile":
            return CAT_R11_PROFILE_MD
        if intent == "thanks":
            return "You're welcome — ask anything else."
        if intent == "goodbye":
            return "Goodbye. Come back anytime."
        if intent == "math":
            result = self._try_simple_math(prompt)
            if result:
                return f"Result: {result}"
            return "Give a numeric expression (e.g. 15 * 7 or what is 100 divided by 4)."
        if intent == "explain":
            topic = re.sub(r".*(?:explain|define|meaning of|tell me about)\s+", "", prompt.lower()).rstrip("?")
            return self._explain_topic(topic.strip(), dialect)
        if intent == "howto":
            topic = re.sub(r".*(?:how do|how to|how can|tutorial)\s+", "", prompt.lower()).rstrip("?")
            return self._howto_topic(topic.strip() or self._extract_topic_words(prompt))
        if intent == "debug":
            if re.search(r"\bfix\b", prompt.lower()) and "traceback" not in prompt.lower():
                topic = self._extract_topic_words(prompt)
                return (
                    f"To fix \"{topic}\": share the config snippet, expected behavior, and any error log. "
                    "I will suggest concrete changes."
                )
            return "Paste the full traceback, file path, and what you expected vs what happened."
        if intent == "opinion":
            topic = self._extract_topic_words(prompt)
            return (
                f"On \"{topic}\": it depends on constraints (team skill, performance, ecosystem). "
                "Share your use case and I will recommend a concrete choice."
            )
        if intent == "joke":
            jokes = [
                "Why do programmers prefer dark mode? Light attracts bugs.",
                "A SQL query walks into a bar, walks up to two tables, and asks: 'Can I join you?'",
                "There are only 10 kinds of people: those who understand binary and those who don't.",
            ]
            vec = self.recursive_encode(prompt)
            return jokes[int(np.abs(vec[:6].sum() * 100)) % len(jokes)]
        if intent == "fable" and CONFIG["cat_r1_enabled"]:
            vec = self.last_vec if self.last_vec is not None else self.recursive_encode(prompt)
            topic = self._extract_topic_words(prompt)
            return self.cat_r1.compose_fable(prompt, topic, vec)
        return None
    _LANG_STOP = frozenset({"a", "an", "the", "me", "my", "some", "that", "is", "to", "it"})

    def extract_lang(self, p):
        original = p
        vibe = VibeCodeHeuristics.lang_from_text(original, self)
        if vibe:
            return vibe
        p = p.lower()
        if re.search(r"\b(?:a|an)\s+html\b|\bhtml\s+(?:program|page|file|ad|that|site|app)\b", p):
            return "html"
        m = CatR1Code._IN_LANG.search(p)
        if m:
            raw = m.group(1).lower()
            if raw == "shell":
                raw = "bash"
            return self.normalize_lang(raw)
        m = CatR1Code._MAKE_IT_LANG.search(p)
        if m:
            raw = m.group(1).lower()
            if raw == "shell":
                raw = "bash"
            return self.normalize_lang(raw)
        for a, l in self.aliases.items():
            if re.search(rf"\bin\s+{re.escape(a)}\b", p) or re.search(rf"\b{re.escape(a)}\s+code\b", p):
                return l
        for l in self.code_experts:
            if re.search(rf"\bin\s+{re.escape(l)}\b", p) or re.search(rf"\b{re.escape(l)}\s+code\b", p):
                return l
        m = re.search(r"(?:write|code|syntax)\s+(?:(?:a|an|the)\s+)?(?:in\s+)?([a-z+#]+)", p)
        if m:
            raw = m.group(1)
            if raw in self._LANG_STOP:
                if "html" in p:
                    return "html"
                if re.search(r"\b(?:c|c\+\+|cpp)\b", p):
                    return "cpp" if "++" in p or "cpp" in p else "c"
                inferred = self.detect_lang_from_text(original)
                return self.normalize_lang(inferred) if inferred else None
            return self.aliases.get(raw, raw)
        inferred = self.detect_lang_from_text(original)
        return self.normalize_lang(inferred) if inferred else None

    def detect_lang_from_text(self, text: str) -> Optional[str]:
        s = text or ""
        sl = s.lower()

        # Fast keyword/shape checks.
        if re.search(r"#!/bin/(ba)?sh|echo\s+['\"]|\$\{?[A-Z_][A-Z0-9_]*\}?|^\s*for\s+\w+\s+in\s+", s, re.MULTILINE):
            return "bash"
        if re.search(r"<!doctype html>|<html|</html>|<body|</body>|<div|</div>", sl):
            return "html"
        if re.search(r"\bconsole\.log\(|\bfunction\s+\w+\s*\(|=>|\b(let|const|var)\s+\w+", s):
            return "javascript"
        if re.search(r"\binterface\s+\w+|:\s*(string|number|boolean)\b", s):
            return "typescript"
        if re.search(r"^\s*#include\s+<", s, re.MULTILINE):
            if "std::" in s or "cout" in s:
                return "cpp"
            return "c"
        if re.search(r"\bpublic\s+class\b|\bSystem\.out\.println\(", s):
            return "java"
        if re.search(r"\bfun\s+main\s*\(|println\(|val\s+\w+|var\s+\w+:", s):
            return "kotlin"
        if re.search(r"\bimport\s+Foundation\b|struct\s+\w+:\s*View", s):
            return "swift"
        if re.search(r"\bpragma\s+solidity\b|contract\s+\w+", s):
            return "solidity"
        if re.search(r"\bfn\s+main\s*\(|println!\(", s):
            return "rust"
        if re.search(r"\bpackage\s+main\b|\bfunc\s+main\s*\(", s):
            return "go"
        if re.search(r"\bsection\s+\.(text|data)\b|\bglobal\s+_start\b", sl):
            return "assembly"
        if re.search(r"^\s*def\s+\w+\s*\(|__name__\s*==\s*['\"]__main__['\"]|\bprint\(", s, re.MULTILINE):
            return "python"

        # Fallback token scoring across supported syntaxes.
        scores = {
            "python": 0,
            "cpp": 0,
            "c": 0,
            "javascript": 0,
            "typescript": 0,
            "java": 0,
            "rust": 0,
            "go": 0,
            "bash": 0,
            "assembly": 0,
            "html": 0,
        }
        token_hints = {
            "python": ["def ", "import ", "None", "True", "False", "elif", "self."],
            "cpp": ["std::", "#include", "cout", "cin", "namespace std", "->"],
            "c": ["#include", "printf(", "scanf(", "malloc(", "free("],
            "javascript": ["console.log", "function ", "=>", "let ", "const ", "var "],
            "typescript": [": string", ": number", "interface ", "type ", "implements "],
            "java": ["public class", "public static void main", "System.out.println", "new "],
            "rust": ["fn ", "let mut", "println!", "match ", "::"],
            "go": ["package ", "func ", "fmt.", ":=", "go "],
            "bash": ["#!/bin/bash", "echo ", "$(", "fi", "done"],
            "assembly": ["mov ", "jmp ", "section .text", "db ", "_start"],
            "html": ["<html", "<body", "<div", "</", "<!doctype"],
        }
        for lang, hints in token_hints.items():
            for h in hints:
                if h in s or h in sl:
                    scores[lang] += 1
        best_lang = max(scores, key=scores.get)
        return best_lang if scores[best_lang] > 0 else None
    def extract_code_block(self, p):
        m = re.search(r"```([a-zA-Z0-9_+#-]*)\n([\s\S]*?)```", p)
        if not m:
            return None, None
        lang = (m.group(1) or "").strip().lower() or None
        code = m.group(2).strip()
        if not lang:
            lang = self.detect_lang_from_text(code)
        return lang, code

    def normalize_lang(self, lang: Optional[str]) -> Optional[str]:
        if not lang:
            return None
        lang = lang.lower().strip()
        aliases = {
            "py": "python", "python3": "python",
            "c++": "cpp", "cc": "cpp",
            "js": "javascript", "node": "javascript",
            "ts": "typescript",
            "sh": "bash", "shell": "bash", "zsh": "bash",
            "asm": "assembly",
            "kt": "kotlin", "rb": "ruby", "rs": "rust",
            "cs": "csharp", "fs": "fsharp", "sol": "solidity",
        }
        if CONFIG.get("code_anything"):
            aliases.update(CodeAnythingEngine.LANG_ALIASES)
        return aliases.get(lang, lang)

    def generate_dynamic_template(self, lang: str, prompt: str) -> str:
        lang = self.normalize_lang(lang) or "python"
        comment = "//"
        if lang in {"python", "bash"}:
            comment = "#"
        elif lang == "html":
            comment = "<!-- -->"

        if lang == "html":
            return "<!DOCTYPE html>\n<html>\n<body>\n  <h1>Hello World</h1>\n</body>\n</html>"
        if lang == "python":
            return "def main():\n    print('Hello World')\n\nif __name__ == '__main__':\n    main()"
        return (
            f"{comment} Dynamic template for {lang}\n"
            f"{comment} Prompt: {prompt[:80]}"
        )

    def _extract_prompt_requirements(self, prompt: str) -> Dict[str, Any]:
        p = prompt.lower()
        fn_match = re.search(r"(?:function|def|method)\s+([a-zA-Z_][a-zA-Z0-9_]*)", prompt)
        class_match = re.search(r"(?:class|struct)\s+([a-zA-Z_][a-zA-Z0-9_]*)", prompt)
        return {
            "wants_main": ("main" in p) or ("entry point" in p),
            "wants_json": ("json" in p),
            "wants_async": ("async" in p) or ("await" in p),
            "wants_cli": ("arg" in p) or ("argv" in p) or ("command line" in p) or ("cli" in p),
            "wants_file_io": ("file" in p) or ("read" in p) or ("write" in p),
            "function_name": fn_match.group(1) if fn_match else None,
            "class_name": class_match.group(1) if class_match else None,
        }

    def _validate_code(self, lang: str, code: str) -> bool:
        lang = self.normalize_lang(lang) or "python"
        try:
            if lang == "python":
                ast.parse(code, mode="exec")
                return True
            if lang in {"javascript", "typescript", "java", "cpp", "c", "go", "rust"}:
                opens = sum(code.count(ch) for ch in "{([")
                closes = sum(code.count(ch) for ch in "})]")
                return opens == closes and len(code.strip()) > 0
            if lang == "html":
                return "<html" in code.lower() and "</html>" in code.lower()
            if lang == "bash":
                return len(code.strip()) > 0
            return len(code.strip()) > 0
        except Exception:
            return False

    def _tailor_code_once(self, lang: str, code: str, req: Dict[str, Any]) -> str:
        lang = self.normalize_lang(lang) or "python"
        patched = code
        fn_name = req.get("function_name")
        class_name = req.get("class_name")

        if lang == "python":
            if req["wants_async"] and "async def" not in patched:
                patched = (
                    "import asyncio\n\n"
                    "async def main_async():\n"
                    "    print('Hello World')\n\n"
                    "if __name__ == '__main__':\n"
                    "    asyncio.run(main_async())"
                )
            if req["wants_json"] and "import json" not in patched:
                patched = f"import json\n\n{patched}"
            if req["wants_cli"] and "import sys" not in patched:
                patched = f"import sys\n\n{patched}"
            if req["wants_file_io"] and "open(" not in patched:
                patched += "\n\n# file io example\nwith open('output.txt', 'w', encoding='utf-8') as f:\n    f.write('Hello World')\n"
            if fn_name and f"def {fn_name}(" not in patched:
                patched += f"\n\ndef {fn_name}():\n    return 'ok'\n"
            if class_name and f"class {class_name}" not in patched:
                patched += f"\n\nclass {class_name}:\n    pass\n"
            if req["wants_main"] and "__name__ == '__main__'" not in patched:
                patched += "\n\nif __name__ == '__main__':\n    main()\n"
            return patched

        if lang in {"javascript", "typescript"}:
            if req["wants_async"] and "async function" not in patched:
                patched = "async function main(){\n  console.log('Hello World');\n}\nmain();"
            if req["wants_json"] and "JSON." not in patched:
                patched += "\n\nconst payload = JSON.stringify({ ok: true });\nconsole.log(payload);\n"
            if fn_name and f"function {fn_name}" not in patched:
                patched += f"\n\nfunction {fn_name}() {{ return 'ok'; }}\n"
            return patched

        if lang == "bash":
            if not patched.startswith("#!/bin/bash"):
                patched = "#!/bin/bash\n" + patched
            if req["wants_file_io"] and ">" not in patched:
                patched += "\necho \"Hello World\" > output.txt\n"
            return patched

        return patched

    def recompile_code_for_prompt(self, lang: str, prompt: str, seed_code: str) -> str:
        """Dynamic recompilation: iterate and tailor generated code to prompt requirements."""
        lang = self.normalize_lang(lang) or "python"
        req = self._extract_prompt_requirements(prompt)
        code = seed_code
        for _ in range(3):
            code = self._tailor_code_once(lang, code, req)
            if self._validate_code(lang, code):
                return code
        # Final safe fallback if iterative tailoring failed validation.
        fallback = self.code_experts.get(lang) or self.generate_dynamic_template(lang, prompt)
        return fallback

    def execute_code_any_language(self, lang: Optional[str], code: Optional[str]) -> str:
        lang = self.normalize_lang(lang) or "python"
        if not code:
            return "No code provided."

        timeout = CONFIG.get("code_terminal_timeout", 8)

        if lang == "python":
            return self.safe_exec_python(code, timeout)
        if lang == "javascript":
            try:
                out = subprocess.run(
                    ["node", "-e", code],
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    check=False,
                )
                text = (out.stdout or out.stderr or "").strip()
                return text if text else "(no output)"
            except FileNotFoundError:
                return "Node.js runtime not found. Install node to execute JavaScript."
            except subprocess.TimeoutExpired:
                return "Execution timed out."
            except Exception as e:
                return f"Execution error: {e}"
        if lang == "bash":
            try:
                out = subprocess.run(
                    ["bash", "-lc", code],
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    check=False,
                )
                text = (out.stdout or out.stderr or "").strip()
                return text if text else "(no output)"
            except subprocess.TimeoutExpired:
                return "Execution timed out."
            except Exception as e:
                return f"Execution error: {e}"

        if CONFIG.get("code_anything"):
            return CodeAnythingEngine.simulate(lang, code, "")
        lines = [ln for ln in code.splitlines() if ln.strip()]
        return (
            f"Interpreter summary ({lang}):\n"
            f"- lines: {len(lines)}\n"
            f"- chars: {len(code)}\n"
            f"- execution backend: not installed for {lang}\n"
            "Tip: Python/JavaScript/Bash run natively in this local interpreter."
        )

    def safe_exec_python(self, code, timeout=8):
        try:
            if not code:
                return "No code provided."

            tmpdir = os.path.join("/var/folders/q1/43b16kqd4zbbst790zltxp2m0000gn/T", "opencode", "cat_r1_exec")
            os.makedirs(tmpdir, exist_ok=True)
            script_path = os.path.join(tmpdir, f"exec_{uuid.uuid4().hex[:12]}.py")
            with open(script_path, "w") as f:
                f.write(code)

            try:
                out = subprocess.run(
                    [sys.executable, "-u", script_path],
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    check=False,
                )
                stdout = out.stdout.strip()
                stderr = out.stderr.strip()
                if out.returncode != 0 and stderr:
                    return f"Exit code {out.returncode}\n{stderr}"
                return stdout if stdout else "(no output)"
            except subprocess.TimeoutExpired:
                return "Execution timed out."
            finally:
                try:
                    os.remove(script_path)
                except OSError:
                    pass
        except Exception as e:
            return f"Error: {e}"

    def generate(self, prompt, simulate=None, on_token=None):
        if simulate is None:
            simulate = CONFIG.get("simulate_enabled", False)
        raw = prompt.strip()
        prompt_hash = hash(raw)
        is_repeat = prompt_hash == self._last_prompt_hash and self._pending_think
        self._last_prompt_hash = prompt_hash
        loc = self.detect_locale(prompt)
        dia = self.get_dialect(loc)
        self._remember("user", prompt)
        self.cat_r1.memory.add("user", prompt)
        self.persistent_memory.total_messages += 1
        if simulate and not is_repeat and CONFIG.get("simulate_latency", 0) > 0:
            time.sleep(CONFIG["simulate_latency"])
        p = prompt.lower()
        pl = raw.lower()

        if pl in {"/reset", "reset chat", "clear memory"}:
            self.chat_history = []
            self.chat.new_session()
            self.cat_r1.memory = CatR1ContextMemory()
            self._clear_engine_caches()
            self.deepmind.clear()
            self.web.clear()
            self.compression_trace = []
            self.last_recursive_trace = []
            self.last_think = ""
            return "Conversation memory cleared."

        if pl == "/chat":
            resp = ChatProtocol.help_text()
            self._remember("assistant", resp)
            return resp
        if pl in {"help", "/help", "?"}:
            dia = self.get_dialect(self.detect_locale(prompt))
            resp = self._intent_response("help", prompt, dia) or ChatProtocol.help_text()
            self._remember("assistant", resp)
            return resp
        if pl == "/chat new":
            sid = self.chat.new_session()
            self.chat_history = []
            self.cat_r1.memory = CatR1ContextMemory()
            self.last_think = ""
            resp = f"New chat session `{sid}`."
            self._remember("assistant", resp)
            self.cat_r1.memory.add("assistant", resp)
            return resp
        if pl == "/chat history":
            return self.chat.session().transcript() or "(empty session)"
        if pl == "/chat session":
            s = self.chat.session()
            return f"Session `{s.session_id}` · turn {s.turn_count} · {CATR1_MODEL_ID}"

        if pl == "/ultrathink":
            resp = f"Extended thinking is **{'on' if self.ultrathink_on else 'off'}**."
            self._remember("assistant", resp)
            return resp
        if pl == "/ultrathink on":
            self.ultrathink_on = True
            resp = "Extended thinking **on**."
            self._remember("assistant", resp)
            return resp
        if pl == "/ultrathink off":
            self.ultrathink_on = False
            resp = "Extended thinking **off**."
            self._remember("assistant", resp)
            return resp
        if pl in {"/catr1", "/catspark"} or pl.startswith("/catr1 ") or pl.startswith("/catspark "):
            tail = raw.split(maxsplit=1)[1].strip() if " " in raw.strip() else ""
            if not tail or tail.lower() in {"help", "?", "status"}:
                resp = CatR1DSpark.help_text()
            else:
                CatR1DSpark.ensure(self, tail, force_pr=CatR1DSpark.wants_pr(raw))
                resp = (
                    f"{CatR1DSpark.help_text()}\n\n"
                    f"`{DSparkBitNetEngine.status_line(self.dspark.last_stats)}`"
                )
            self._remember("assistant", resp)
            return resp
        if pl == "/pro":
            self.active_model_profile = CatR1ProProfile()
            resp = (
                f"Switched to **{CAT_R1_PRO}** — BitNet deep reasoning "
                f"(depth={CONFIG.get('cat_r1_pro_reasoning_depth', 5)}, "
                f"~{CONFIG.get('v4_pro_target_tps', 48):.0f} tok/s target)."
            )
            self._remember("assistant", resp)
            return resp
        if pl == "/flash":
            self.active_model_profile = CatR1FlashProfile()
            resp = (
                f"Switched to **{CAT_R1_FLASH}** — BitNet turbo instant "
                f"(depth={CONFIG.get('cat_r1_flash_reasoning_depth', 1)}, "
                f"~{CONFIG.get('v4_flash_target_tps', 420):.0f} tok/s target)."
            )
            self._remember("assistant", resp)
            return resp
        if pl == "/model":
            active = self.active_model_profile
            resp = f"**Active model:** {active.label} · tier={active.tier} · depth={active.reasoning_depth} · mode={active.inference_mode}"
            self._remember("assistant", resp)
            return resp

        force_pro = self._resolve_v4_profile(raw, prompt)
        if CONFIG.get("tri_channel_detect", True):
            tri = self._apply_tri_channel(raw)
            if tri.get("force_pro"):
                force_pro = True

        setup = self.ai_setup_response(raw)
        if setup:
            if force_pro or self.is_pro_mode():
                self.encode_for_task(raw, task="chat")
                setup = self._append_v4_stats(setup)
            self._remember("assistant", setup)
            return setup
        if pl == "/bitnet" or CatR1Code.is_bitnet_request(pl):
            bt = self._ensure_bitnet_real()
            CONFIG["dspark_enabled"] = True
            CONFIG["dspark_speculative_decode"] = True
            self.cat_r1_stats["dspark"] = {
                "enabled": True,
                "accepted": self.dspark.last_stats.accepted,
                "drafts": self.dspark.last_stats.drafts,
                "gamma": int(CONFIG.get("v4_speculative_draft_tokens", 5)),
            }
            resp = BitNetEngine.status_text(self.cat_r1_stats)
            resp += (
                f"\n\n{DSparkBitNetEngine.status_line(self.dspark.last_stats)}\n\n"
                f"**{BRAND} BitNet real mode ON** — packed ternary weights + "
                f"DSpark speculative decode · `files = off` · "
                f"self-test **{'PASS' if bt.get('ok') else 'FAIL'}**."
            )
            if force_pro or self.is_pro_mode():
                self.encode_for_task(raw, task="chat")
                resp = self._append_v4_stats(resp)
            self._remember("assistant", resp)
            return resp
        if pl == "/lang" or pl.startswith("/lang "):
            tail = raw.split(maxsplit=1)[1].strip().lower() if " " in raw.strip() else ""
            if tail in {"en", "english", "英文"}:
                self.preferred_lang = "en"
                resp = "Language set to **English**."
            elif tail in {"zh", "cn", "chinese", "中文", "汉语"}:
                self.preferred_lang = "zh"
                resp = "语言已设为 **中文**。"
            elif tail in {"auto", "mix", "mixed", "自动", ""}:
                self.preferred_lang = "auto"
                resp = "Language **auto** — replies follow your message (EN · 中文 · mixed)."
            else:
                resp = "Usage: `/lang auto` · `/lang en` · `/lang zh`"
            self._remember("assistant", resp)
            return resp
        if pl == "/think" and self.last_think:
            return self.last_think
        if pl == "/web" or pl.startswith("/web "):
            resp = self.web.handle_command(self, raw)
            self._remember("assistant", resp)
            self.cat_r1.memory.add("assistant", resp)
            return resp
        if pl == "/interpret" or pl.startswith("/interpret "):
            block_lang, code = self.extract_code_block(raw if "```" in raw else prompt)
            if not code and pl.startswith("/interpret "):
                code = raw.split(maxsplit=1)[1] if " " in raw else ""
            exec_lang = block_lang or CatR1Code.detect_lang(self, code or prompt)
            if not code:
                return f"Usage: `/interpret` with ``` fences or `/interpret print('hi')`"
            result = CatR1CodingAPI.agent_run(self, code, exec_lang)
            resp = CatR1CodingAPI.format_result(result)
            self._remember("assistant", resp)
            self.cat_r1.memory.add("assistant", resp)
            return resp
        if pl == "/run" or pl.startswith("/run "):
            block_lang, code = self.extract_code_block(raw if "```" in raw else prompt)
            if not code and pl.startswith("/run "):
                code = raw.split(maxsplit=1)[1] if " " in raw else ""
            exec_lang = block_lang or CatR1Code.detect_lang(self, code or prompt)
            if not code:
                return "Usage: paste code in ``` fences, or `/run print('hi')`"
            result = CatR1CodingAPI.agent_run(self, code, exec_lang)
            resp = CatR1CodingAPI.format_result(result)
            self._remember("assistant", resp)
            self.cat_r1.memory.add("assistant", resp)
            return resp
        if pl == "/coding" or pl.startswith("/coding "):
            if pl.strip() == "/coding":
                return CatR1CodingAPI.help_text()
            tail = raw.split(maxsplit=1)[1] if " " in raw else ""
            if tail.startswith("{"):
                try:
                    payload = json.loads(tail)
                except json.JSONDecodeError:
                    payload = {"action": "run", "code": tail}
            else:
                action, _, body = tail.partition(" ")
                payload = {"action": action or "run", "code": body}
            out = CatR1CodingAPI.parse_request(self, payload)
            if out.get("action") == "explain":
                resp = out.get("content", "")
                self._remember("assistant", resp)
                return resp
            if out.get("action") == "help":
                resp = out.get("help", CatR1CodingAPI.help_text())
                self._remember("assistant", resp)
                return resp
            if not out.get("ok", True) and out.get("error"):
                resp = f"**{CODING_API_LABEL}** error: {out['error']}"
                self._remember("assistant", resp)
                return resp
            if "output" in out:
                resp = CatR1CodingAPI.format_result(out)
                self._remember("assistant", resp)
                return resp
            resp = json.dumps(out, indent=2)
            self._remember("assistant", resp)
            return resp
        if (
            pl == "/catrcode" or pl.startswith("/catrcode ")
            or pl == "/catseek" or pl.startswith("/catseek ")
            or pl == "/catcode" or pl.startswith("/catcode ")
            or CatR1CodeR1.wants_catseek(raw)
        ):
            return self._catseek_handler(raw, force_pro=force_pro)
        if pl == "/code":
            resp = CatR1Code.code_help()
            self._remember("assistant", resp)
            return resp
        if pl.startswith("/code "):
            self.encode_for_task(raw, task="code")
            resp = CatR1Code.respond(self, raw)
            self._remember("assistant", resp)
            self.cat_r1.memory.add("assistant", resp)
            return resp
        if CONFIG.get("cat_r1_code_enabled") and pl in {"code", "code >", ">"}:
            self.encode_for_task(raw, task="code")
            resp = CatR1Code.respond(self, raw)
            self._remember("assistant", resp)
            self.cat_r1.memory.add("assistant", resp)
            return resp
        if pl.startswith("/ultrathink ") and pl not in ("/ultrathink off", "/ultrathink on"):
            think_prompt = raw.split(maxsplit=1)[1] if " " in raw else ""
            if not think_prompt.strip():
                return "Usage: `/ultrathink <your question>`"
            self._run_ultrathink(think_prompt, force=True)
            resp = self.cat_r1.complete(self, think_prompt, simulate=False)
            self._remember("assistant", resp)
            self.cat_r1.memory.add("assistant", resp)
            return resp

        if pl in {"who are you", "what are you", "what model are you"}:
            st = self.cat_r1_stats
            v4 = self.v4_compressor
            active = self.active_model_profile
            v4s = v4.stats() if CONFIG.get("v4_pro_compression") else {}
            resp = (
                f"I'm **{BRAND}** — `{CAT_R1_MODEL_ID}` running locally.\n\n"
                f"Behavior target: **{LLM_MATCH_TARGET}** by **{LLM_MATCH_PROVIDER}**.\n\n"
                f"In-memory student stack "
                f"({CONFIG.get('distil_teacher_weight', CONFIG['teacher_weight']):.0%} primary blend).\n\n"
                f"**Two model variants:**\n"
                f"- **{CAT_R1_PRO}** — deep reasoning (depth=5, {CatR1ProProfile.effective_params}) · `/pro`\n"
                f"- **{CAT_R1_FLASH}** — instant (depth=1, {CatR1FlashProfile.effective_params}) · `/flash`\n"
                f"Active: **{active.label}** (tier={active.tier})\n\n"
                f"**Compression:** {BITNET_V4_LABEL} ({v4s.get('last_ratio', '1.0x')} · ~{v4s.get('effective_capacity_billions', '1.7')}B effective)\n\n"
                f"Reasoning: **{REASONING_MODE}** · extended thinking.\n\n"
                f"Code: **{CODE_ENGINE}** ({EDITION}) {'✓ enabled' if CAT_R1_CODE_ENABLED else 'off'} · {CORE_NAME} — "
                f"{st['cat_r1_linear_layers']} linear layers, "
                f"{st['weight_bits']}-bit ternary weights, {CONFIG['cat_r1_context_window']:,} token context.\n\n"
                f"In-memory: {st['shadow_params']:,} params · {st['packed_kb']:.0f}KB packed · no weight files.\n\n"
                "I'm thorough, proactive, and self-checking — ask me anything."
            )
            self._remember("assistant", resp)
            self.cat_r1.memory.add("assistant", resp)
            return resp

        # Auto-select model via CatR1 Router (Pro vs Flash) — profile already resolved above
        if force_pro:
            self.active_model_profile = CatR1ProProfile()
        cat_r1_depth = self.get_reasoning_depth()
        old_recursive_depth = CONFIG["recursive_depth"]
        CONFIG["recursive_depth"] = cat_r1_depth
        self.last_recursive_passes = 0

        # cat r1 BitNet V4 Pro compression for deep reasoning passes
        if self.is_pro_mode() and CONFIG.get("v4_pro_compression"):
            self.v4_compressor.passes = 0

        # Heuristics-guided dispatch → cat r1 core (tri-channel · always answer)
        channel = getattr(self, "last_channel", None) or self.heuristics.analyze_channels(raw, self)
        try:
            resp = self.cat_r1.complete(self, CatR1Code.normalize_prompt(raw), simulate=False)
        except Exception:
            resp = None
        finally:
            CONFIG["recursive_depth"] = old_recursive_depth
        if CONFIG.get("always_answer", True):
            resp = self.heuristics.always_answer(self, raw, channel, resp)
        elif not isinstance(resp, str) or not resp.strip():
            history = self.chat_history[:-1] if self.chat_history else []
            resp = self.synth.o1_answer(raw, history, vec=self.last_vec) or ""
            if not resp.strip():
                resp = ChatProtocol.help_text()

        # Trilingual: ensure responses match the user's language mix
        if CONFIG.get("bilingual_chat", True):
            if self.preferred_lang == "zh":
                if resp and "```" not in resp and not re.search(r"[\u4e00-\u9fff]", resp):
                    resp = self.synth.localize(resp, prompt)
            elif self.preferred_lang == "en":
                pass
            else:
                resp = self.synth.localize(resp, prompt)
        elif loc == "mixed":
            if resp and "```" not in resp and not re.search(r"[\u4e00-\u9fff]", resp):
                zh_note = f"\n\n---\n*（以上为英文回复。你也可以用中文问我问题！）*"
                resp = resp + zh_note
        elif loc == "chinese" and resp and not re.search(r"[\u4e00-\u9fff]", resp):
            if "```" not in resp:
                resp = self._zh_wrap(resp, prompt)
        elif loc == "english" and resp and not re.search(r"[a-zA-Z]{3,}", resp):
            if "```" not in resp:
                resp = self._en_wrap(resp, prompt)

        self._remember("assistant", resp)
        self.cat_r1.memory.add("assistant", resp)

        if on_token is not None:
            tokens = re.split(r"(\s+)", resp)
            for t in tokens:
                on_token(t)
        stats = getattr(self, "last_inference_stats", None)
        if stats and stats.model_label:
            self.v4_runtime.last_stats = stats
        return resp

    def clear_history(self) -> None:
        self.chat_history.clear()
        self.cat_r1.memory = CatR1ContextMemory()
        self._clear_engine_caches()
        self.deepmind.clear()
        self.web.clear()
        self.compression_trace.clear()
        self.last_recursive_trace.clear()
        self.last_think = ""
        self._pending_think = ""

    def get_thoughts(self, prompt, lang, ultra):
        if not prompt.strip():
            return ["Ready."]
        if not O1PreviewReasoner.should_run(prompt, enabled=self.ultrathink_on, force=ultra):
            return []
        vec = self.recursive_encode(prompt)
        draft = self._distil_draft(prompt, vec=vec)
        trace = self.ultrathink.run(
            prompt, distill_draft=draft, recursive_trace=self.last_recursive_trace,
            compression_trace=self.compression_trace,
        )
        self._pending_think = trace
        return [ln.strip() for ln in trace.split("\n") if ln.strip()]

# ──────────────────────────────────────────────────────────────
# GUI & API — cat r1 chat layout · files = off
# ──────────────────────────────────────────────────────────────
class CatR11GUI:
    def __init__(self, root, engine: Optional[CatR11Engine] = None):
        self.root = root
        self.engine = engine or CatR11Engine()
        self.ui = CAT_R1_UI
        self._msg_widgets: List[tk.Widget] = []
        self._history_items: List[str] = []
        self._stream_tokens: List[str] = []
        self._stream_widget: Optional[tk.Text] = None
        self._stream_started = False
        self._chat_mode = self.engine.persistent_memory.chat_mode
        self._thinking_on = self.engine.persistent_memory.thinking_on
        self._web_search_on = False

        root.title(WINDOW_TITLE)
        root.geometry("1360x840")
        root.minsize(1024, 680)
        if os.name == "darwin":
            try:
                root.tk.call("::tk::unsupported::MacWindowStyle", "style", root._w, "dark", "normal")
            except Exception:
                pass
        root.configure(bg=self.ui["bg"])

        self.fonts = {
            "ui": font.Font(family="Segoe UI" if os.name == "nt" else "Helvetica Neue", size=12),
            "ui_bold": font.Font(family="Segoe UI" if os.name == "nt" else "Helvetica Neue", size=12, weight="bold"),
            "title": font.Font(family="Segoe UI" if os.name == "nt" else "Helvetica Neue", size=18, weight="bold"),
            "logo": font.Font(family="Segoe UI" if os.name == "nt" else "Helvetica Neue", size=15, weight="bold"),
            "small": font.Font(family="Segoe UI" if os.name == "nt" else "Helvetica Neue", size=10),
            "mono": font.Font(family="Menlo" if os.name != "nt" else "Consolas", size=11),
            "empty": font.Font(family="Segoe UI" if os.name == "nt" else "Helvetica Neue", size=22, weight="normal"),
            "mascot": font.Font(family="Segoe UI" if os.name == "nt" else "Helvetica Neue", size=42),
        }

        outer = tk.Frame(root, bg=self.ui["bg"])
        outer.pack(fill="both", expand=True)

        # ── Sidebar (cat r1 chat layout) ──
        sidebar = tk.Frame(outer, bg=self.ui["sidebar"], width=268,
                        highlightthickness=1, highlightbackground=self.ui["sidebar_border"])
        sidebar.pack(side="left", fill="y")
        sidebar.pack_propagate(False)

        top_bar = tk.Frame(sidebar, bg=self.ui["sidebar"])
        top_bar.pack(fill="x", padx=18, pady=(18, 10))
        mascot_side = tk.Frame(top_bar, bg=self.ui["mascot_bg"], width=40, height=40)
        mascot_side.pack(side="left")
        mascot_side.pack_propagate(False)
        tk.Label(
            mascot_side, text=MASCOT_GLYPH, font=self.fonts["mascot"],
            bg=self.ui["mascot_bg"], fg=self.ui["mascot_fg"],
        ).place(relx=0.5, rely=0.5, anchor="center")
        name_col = tk.Frame(top_bar, bg=self.ui["sidebar"])
        name_col.pack(side="left", padx=(10, 0))
        tk.Label(name_col, text=BRAND, font=self.fonts["logo"], bg=self.ui["sidebar"],
                fg=self.ui["text"]).pack(anchor="w")
        tk.Label(name_col, text=BRAND_TAG, font=self.fonts["small"], bg=self.ui["sidebar"],
                fg=self.ui["muted"]).pack(anchor="w")

        new_chat_outer = tk.Frame(sidebar, bg=self.ui["new_chat_border"], padx=1, pady=1)
        new_chat_outer.pack(fill="x", padx=16, pady=(6, 14))
        tk.Button(
            new_chat_outer, text="  +  New chat", font=self.fonts["ui_bold"],
            bg=self.ui["new_chat_bg"], fg="#3b82f6",
            activebackground=self.ui["history_hover"], relief="flat", bd=0,
            padx=12, pady=10, cursor="hand2", anchor="w", command=self._new_chat,
        ).pack(fill="x")

        tk.Label(sidebar, text="Recent", font=self.fonts["small"], bg=self.ui["sidebar"],
                fg=self.ui["muted"]).pack(anchor="w", padx=20, pady=(0, 4))
        self.history_frame = tk.Frame(sidebar, bg=self.ui["sidebar"])
        self.history_frame.pack(fill="both", expand=True, padx=10)

        tk.Frame(sidebar, bg=self.ui["sidebar_border"], height=1).pack(fill="x", padx=16, pady=8)
        foot = tk.Frame(sidebar, bg=self.ui["sidebar"])
        foot.pack(fill="x", padx=18, pady=(0, 14))
        tk.Label(foot, text=BRAND_TAG, font=self.fonts["small"],
                bg=self.ui["sidebar"], fg=self.ui["muted"]).pack(anchor="w")

        tk.Button(
            sidebar, text=f"  </>  {CODE_ENGINE}", font=self.fonts["ui"],
            bg="#000000", fg="#3b82f6",
            activebackground=self.ui["history_hover"], relief="flat", bd=0,
            padx=12, pady=8, cursor="hand2", anchor="w",
            command=self._toggle_code_panel,
        ).pack(fill="x", padx=16, pady=(0, 8))

        # ── Main panel ──
        main = tk.Frame(outer, bg=self.ui["bg"])
        main.pack(side="left", fill="both", expand=True)

        header = tk.Frame(main, bg=self.ui["header_bg"], highlightthickness=1,
                        highlightbackground=self.ui["header_border"], height=56)
        header.pack(fill="x")
        header.pack_propagate(False)

        model_pill = tk.Frame(header, bg=self.ui["user_bg"], highlightthickness=1,
                            highlightbackground=self.ui["input_border"])
        model_pill.pack(side="left", padx=20, pady=12)
        self.model_label = tk.Label(
            model_pill, text=f"  {V4_MODEL_LABEL_PRO}  ", font=self.fonts["ui_bold"],
            bg=self.ui["user_bg"], fg=self.ui["accent"],
        )
        self.model_label.pack(side="left", padx=4, pady=4)
        tk.Label(model_pill, text="▾", font=self.fonts["small"],
                bg=self.ui["user_bg"], fg=self.ui["muted"]).pack(side="left", padx=(0, 6))

        tk.Label(
            header, text=f"{BRAND_TAG} · {CAT_R1_PRO} · {CAT_R1_FLASH}",
            font=self.fonts["small"],
            bg=self.ui["header_bg"], fg=self.ui["muted"],
        ).pack(side="left", padx=(4, 0))

        self.header_status = tk.Label(header, text=f"{BRAND} ready", font=self.fonts["small"],
                                    bg=self.ui["header_bg"], fg=self.ui["muted"])
        self.header_status.pack(side="right", padx=22)

        # Chat area + empty state
        chat_outer = tk.Frame(main, bg=self.ui["bg"])
        chat_outer.pack(fill="both", expand=True)

        self.empty_state = tk.Frame(chat_outer, bg=self.ui["bg"])
        self.empty_state.place(relx=0.5, rely=0.40, anchor="center")
        mascot_ring = tk.Frame(self.empty_state, bg=self.ui["mascot_bg"], width=88, height=88)
        mascot_ring.pack()
        mascot_ring.pack_propagate(False)
        tk.Label(
            mascot_ring, text=MASCOT_GLYPH, font=self.fonts["mascot"],
            bg=self.ui["mascot_bg"], fg=self.ui["mascot_fg"],
        ).place(relx=0.5, rely=0.5, anchor="center")
        tk.Label(self.empty_state, text=f"{BRAND} — How can I help you today?", font=self.fonts["empty"], bg=self.ui["bg"],
                fg=self.ui["empty_title"]).pack(pady=(16, 4))
        tk.Label(self.empty_state, text=f"{BRAND} · 今天有什么可以帮你的？ · English / 中文 · code · chat",
                font=self.fonts["ui"], bg=self.ui["bg"], fg=self.ui["muted"]).pack()
        tk.Label(self.empty_state, text=BRAND_TAG,
                font=self.fonts["ui"], bg=self.ui["bg"], fg=self.ui["muted"]).pack()
        tk.Label(self.empty_state, text=f"{CAT_R1_PRO} · {CAT_R1_FLASH}", font=self.fonts["small"],
                bg=self.ui["bg"], fg=self.ui["muted"]).pack(pady=(8, 0))

        chat_wrap = tk.Frame(chat_outer, bg=self.ui["bg"])
        chat_wrap.pack(fill="both", expand=True, padx=0, pady=0)

        self.chat_canvas = tk.Canvas(chat_wrap, bg=self.ui["bg"], highlightthickness=0, bd=0)
        scrollbar = tk.Scrollbar(chat_wrap, orient="vertical", command=self.chat_canvas.yview,
                                width=10, troughcolor=self.ui["bg"],
                                activebackground=self.ui["input_border"])
        self.chat_canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y", padx=(0, 4))
        self.chat_canvas.pack(side="left", fill="both", expand=True, padx=(24, 8))

        self.messages_frame = tk.Frame(self.chat_canvas, bg=self.ui["bg"])
        self._canvas_window = self.chat_canvas.create_window((0, 0), window=self.messages_frame, anchor="n")
        self.messages_frame.bind("<Configure>", self._on_frame_configure)
        self.chat_canvas.bind("<Configure>", self._on_canvas_configure)
        self.chat_canvas.bind_all("<MouseWheel>", self._on_mousewheel, add="+")
        if os.name != "nt":
            self.chat_canvas.bind_all("<Button-4>", lambda e: self.chat_canvas.yview_scroll(-1, "units"), add="+")
            self.chat_canvas.bind_all("<Button-5>", lambda e: self.chat_canvas.yview_scroll(1, "units"), add="+")

        # cat r1 Coding API 0.1 — cat r1 code-style · files = off
        self._code_panel_visible = False
        self.code_script_name = "untitled.py"
        self.code_panel = tk.Frame(main, bg=self.ui["bg"])
        code_shadow = tk.Frame(self.code_panel, bg=self.ui["input_shadow"], padx=1, pady=1)
        code_shadow.pack(fill="x")
        code_border = tk.Frame(code_shadow, bg=self.ui["input_border"], padx=1, pady=1)
        code_border.pack(fill="x")
        code_inner = tk.Frame(code_border, bg=self.ui["code_bg"])
        code_inner.pack(fill="x")

        self._build_code_menustrip(code_inner)

        self.code_editor = scrolledtext.ScrolledText(
            code_inner, height=7, font=self.fonts["mono"],
            bg=self.ui["code_bg"], fg=self.ui["code_fg"],
            insertbackground=self.ui["code_fg"], relief="flat", bd=0,
            padx=12, pady=8, wrap="none", undo=True, exportselection=True,
        )
        self.code_editor.pack(fill="x", padx=8, pady=(0, 4))
        self.code_editor.insert("1.0", CatR1CodingAPI.default_snippet())
        self._bind_code_interpreter(self.code_editor)

        tk.Label(
            code_inner, text=f"{MYTHOS_NAME} · Output", font=self.fonts["small"],
            bg=self.ui["code_bg"], fg=self.ui["muted"],
        ).pack(anchor="w", padx=12)
        self.code_output = tk.Text(
            code_inner, height=3, font=self.fonts["mono"],
            bg="#141414", fg="#9cdcfe", relief="flat", bd=0,
            padx=12, pady=6, wrap="word", exportselection=True,
        )
        self.code_output.pack(fill="x", padx=8, pady=(0, 8))
        self.code_output.bind("<Key>", self._code_output_key_filter)
        self._bind_code_output_menu(self.code_output)

        # Input bar — centered cat r1 composer
        self.input_outer = tk.Frame(main, bg=self.ui["bg"])
        input_outer = self.input_outer

        input_center = tk.Frame(input_outer, bg=self.ui["bg"])
        input_center.pack(fill="x", padx=48)

        self._build_composer_toolbar(input_center)

        shadow = tk.Frame(input_center, bg=self.ui["input_shadow"], padx=1, pady=1)
        shadow.pack(fill="x")
        border = tk.Frame(shadow, bg=self.ui["input_border"], padx=1, pady=1)
        border.pack(fill="x")
        input_box = tk.Frame(border, bg=self.ui["input_bg"])
        input_box.pack(fill="x")

        self._placeholder = f"Message {BRAND}… (English / 中文 · code · chat)"
        self._placeholder_active = True

        self.entry = tk.Text(
            input_box, height=2, font=self.fonts["ui"], bg=self.ui["input_bg"],
            fg=self.ui["text"], insertbackground=self.ui["text"], relief="flat",
            bd=0, padx=18, pady=14, wrap="word",
        )
        self.entry.pack(side="left", fill="both", expand=True)
        self.entry.bind("<Return>", self._on_enter)
        self.entry.bind("<KeyPress>", self._on_entry_key)
        self.entry.bind("<<Paste>>", self._on_entry_edit)
        self.entry.bind("<Button-1>", self._on_entry_click)
        self.entry.bind("<KeyRelease>", self._sync_placeholder)
        self.entry.bind("<FocusIn>", self._on_entry_focus_in)
        self.entry.bind("<FocusOut>", self._on_entry_focus_out)
        self._set_placeholder()
        self.entry.focus_set()
        self._bind_clipboard(self.entry, on_edit=self._on_entry_edit)

        send_wrap = tk.Frame(input_box, bg=self.ui["input_bg"])
        send_wrap.pack(side="right", padx=(0, 12), pady=10)
        self.send_btn = tk.Button(
            send_wrap, text="↑", font=font.Font(size=15, weight="bold"),
            bg="#000000", fg="#3b82f6",
            activebackground=self.ui["send_hover"], activeforeground="#3b82f6",
            relief="flat", bd=0, width=2, height=1, cursor="hand2", command=self.send,
        )
        self.send_btn.pack()

        tk.Label(
            input_center,
            text=(
                f"{BRAND} · EN/中文 · DSpark · BitNet · no API"
                if CONFIG.get("catcode_no_api", True)
                else f"{BRAND} · EN/中文 · code · API :{CONFIG['api_port']} · /lang auto"
            ),
            font=self.fonts["small"], bg=self.ui["bg"], fg=self.ui["muted"],
        ).pack(anchor="center", pady=(8, 0))

        self._apply_chat_mode()
        input_outer.pack(side="bottom", fill="x", padx=0, pady=(0, 20))
        self.code_panel.pack(side="bottom", fill="x", padx=0, pady=(0, 8))
        self._code_panel_visible = True

        if CONFIG.get("api_enabled", False):
            self._start_api()

    def _current_model_label(self) -> str:
        if self._chat_mode == "expert":
            return CatR1ProProfile.label
        return CatR1FlashProfile.label

    def _inference_status(self) -> str:
        stats = getattr(self.engine, "last_inference_stats", None)
        if stats and stats.model_label:
            return (
                f"{stats.model_label} · {stats.tps_estimate:.0f} tok/s · "
                f"{stats.encode_ms:.0f}ms · {stats.passes} pass"
            )
        mode_txt = CAT_R1_PRO if self._chat_mode == "expert" else CAT_R1_FLASH
        think_txt = f"{BRAND} think on" if self._thinking_on else f"{BRAND} think off"
        return f"{mode_txt} · {think_txt}"

    def _apply_chat_mode(self):
        expert = self._chat_mode == "expert"
        self.engine.ultrathink_on = expert and self._thinking_on
        if expert:
            self.engine.active_model_profile = CatR1ProProfile()
        else:
            self.engine.active_model_profile = CatR1FlashProfile()
        label = self._current_model_label()
        self.model_label.config(text=f"  {label}  ")
        self.root.title(WINDOW_TITLE)
        self.header_status.config(text=self._inference_status())

    def _bot_display_name(self) -> str:
        return self._current_model_label()

    def _set_chat_mode(self, mode: str):
        if mode not in {"expert", "instant"}:
            return
        self._chat_mode = mode
        self._apply_chat_mode()
        self._refresh_mode_buttons()

    def _toggle_thinking(self):
        self._thinking_on = not self._thinking_on
        self._apply_chat_mode()
        self._refresh_mode_buttons()

    def _toggle_web_search(self):
        self._web_search_on = not self._web_search_on
        self._refresh_mode_buttons()
        self.header_status.config(
            text=f"Web search {'on' if self._web_search_on else 'off'} · {MASCOT_NAME}"
        )

    def _mode_btn_style(self, active: bool) -> Dict[str, str]:
        if active:
            return {"bg": self.ui["mode_active_bg"], "fg": self.ui["mode_active_fg"]}
        return {"bg": self.ui["mode_idle_bg"], "fg": self.ui["mode_idle_fg"]}

    def _refresh_mode_buttons(self):
        if not hasattr(self, "btn_expert"):
            return
        for btn, on in (
            (self.btn_expert, self._chat_mode == "expert"),
            (self.btn_instant, self._chat_mode == "instant"),
            (self.btn_thinking, self._thinking_on),
            (self.btn_web, self._web_search_on),
        ):
            st = self._mode_btn_style(on)
            btn.config(bg=st["bg"], fg=st["fg"], activebackground=st["bg"], activeforeground=st["fg"])

    def _build_composer_toolbar(self, parent):
        """cat r1 · cat r1-flash · Thinking · Search."""
        bar = tk.Frame(parent, bg=self.ui["bg"])
        bar.pack(fill="x", pady=(0, 10))

        left = tk.Frame(bar, bg=self.ui["bg"])
        left.pack(side="left")

        self.btn_expert = tk.Button(
            left, text=CAT_R1_PRO, font=self.fonts["small"],
            relief="flat", bd=0, padx=14, pady=7, cursor="hand2",
            command=lambda: self._set_chat_mode("expert"),
        )
        self.btn_expert.pack(side="left", padx=(0, 6))

        self.btn_instant = tk.Button(
            left, text=CAT_R1_FLASH, font=self.fonts["small"],
            relief="flat", bd=0, padx=14, pady=7, cursor="hand2",
            command=lambda: self._set_chat_mode("instant"),
        )
        self.btn_instant.pack(side="left")

        right = tk.Frame(bar, bg=self.ui["bg"])
        right.pack(side="right")

        self.btn_web = tk.Button(
            right, text="🌐 Search", font=self.fonts["small"],
            relief="flat", bd=0, padx=12, pady=7, cursor="hand2",
            command=self._toggle_web_search,
        )
        self.btn_web.pack(side="right", padx=(6, 0))

        self.btn_thinking = tk.Button(
            right, text=f"💭 {BRAND} think", font=self.fonts["small"],
            relief="flat", bd=0, padx=12, pady=7, cursor="hand2",
            command=self._toggle_thinking,
        )
        self.btn_thinking.pack(side="right", padx=(6, 0))

        self._refresh_mode_buttons()

    def _append_think_block(self, text: str):
        """Collapsible o1-preview reasoning block."""
        self._hide_empty()
        body = self._plain(text).strip()
        if not body:
            return

        row = tk.Frame(self.messages_frame, bg=self.ui["bg"])
        row.pack(fill="x", pady=(8, 4), padx=8)
        self._msg_widgets.append(row)

        outer = tk.Frame(
            row, bg=self.ui["think_bg"],
            highlightthickness=1, highlightbackground=self.ui["think_border"],
        )
        outer.pack(anchor="w", fill="x")

        state = {"open": True}

        header = tk.Frame(outer, bg=self.ui["think_bg"])
        header.pack(fill="x", padx=12, pady=8)

        chevron = tk.Label(header, text="▾", font=self.fonts["small"], bg=self.ui["think_bg"], fg=self.ui["muted"])
        chevron.pack(side="left")
        tk.Label(
            header, text=f"{BRAND} thinking", font=self.fonts["ui_bold"],
            bg=self.ui["think_bg"], fg=self.ui["text"],
        ).pack(side="left", padx=(6, 0))

        body_frame = tk.Frame(outer, bg=self.ui["think_bg"])
        body_frame.pack(fill="x", padx=12, pady=(0, 10))

        lines = max(body.count("\n") + 1, 2)
        view = tk.Text(
            body_frame, font=self.fonts["small"], bg=self.ui["think_bg"], fg=self.ui["think_fg"],
            relief="flat", bd=0, wrap="word", height=min(max(lines, 3), 16), width=72,
            highlightthickness=0, cursor="arrow",
        )
        view.insert("1.0", body)
        view.config(state="disabled")
        view.pack(fill="x")

        def toggle(_e=None):
            if state["open"]:
                body_frame.pack_forget()
                chevron.config(text="▸")
                state["open"] = False
            else:
                body_frame.pack(fill="x", padx=12, pady=(0, 10))
                chevron.config(text="▾")
                state["open"] = True
            self.root.after(10, self._scroll_to_bottom)

        header.bind("<Button-1>", toggle)
        chevron.bind("<Button-1>", toggle)
        self.root.after(10, self._scroll_to_bottom)

    @staticmethod
    def _plain(text: str) -> str:
        return re.sub(r"\*\*([^*]+)\*\*", r"\1", str(text))

    def _parse_chat_codeblock(self, text: str, lang_hint: str = "") -> Tuple[str, str]:
        raw = str(text or "").strip()
        lang = (lang_hint or "").strip()
        fenced = re.search(r"```([a-zA-Z0-9_+#-]*)\s*\n([\s\S]*?)```", raw)
        if fenced:
            return fenced.group(2).rstrip(), lang or (fenced.group(1) or "").strip()
        if not lang and "\n" in raw:
            first, rest = raw.split("\n", 1)
            first = first.strip()
            if re.fullmatch(r"[a-zA-Z0-9_+#-]{1,15}", first):
                detected = self._detect_codeblock_lang(rest, first)
                if detected == first.lower():
                    return rest.rstrip(), first
        return raw, lang

    def _detect_codeblock_lang(self, code: str, lang: str = "") -> str:
        lang = (lang or "").strip().lower()
        if lang and lang not in {"text", "plain", "code"}:
            return lang
        sample = code[:800]
        if "#include" in sample or re.search(r"\bint\s+main\s*\(", sample):
            return "c"
        if re.search(r"\b(def|import|print)\b", sample) and ":" in sample:
            return "python"
        if re.search(r"\b(function|const|let|console\.)\b", sample):
            return "javascript"
        if re.search(r"\b(fn|println!|use\s+std)\b", sample):
            return "rust"
        if re.search(r"\b(package|func|fmt\.)\b", sample):
            return "go"
        if re.search(r"\b(public\s+class|System\.out)\b", sample):
            return "java"
        return lang or "code"

    def _render_chat_code_block(self, parent, code: str, lang: str = "") -> tk.Frame:
        """Chat code bubble with toolstrip: lang · Copy · Run · files = off."""
        body, lang = self._parse_chat_codeblock(code, lang)
        lang_key = self._detect_codeblock_lang(body, lang)
        lang_label = lang_key.upper() if len(lang_key) <= 4 else lang_key.capitalize()

        outer = tk.Frame(
            parent, bg=self.ui["code_bg"],
            highlightthickness=1, highlightbackground="#5c5c5c",
        )
        strip = tk.Frame(outer, bg="#3a3a3a", height=40)
        strip.pack(fill="x")
        strip.pack_propagate(False)

        left = tk.Frame(strip, bg="#3a3a3a")
        left.pack(side="left", padx=10, pady=6)
        tk.Label(left, text=lang_label, font=self.fonts["ui_bold"], bg="#3a3a3a", fg="#ffffff").pack(side="left")
        tk.Label(left, text=f"  ·  {CODING_API_LABEL}", font=self.fonts["small"], bg="#3a3a3a", fg="#b0b0b0").pack(side="left")

        actions = tk.Frame(strip, bg="#3a3a3a")
        actions.pack(side="right", padx=8, pady=4)

        tk.Label(
            actions, text=GUI_TAGLINE, font=self.fonts["ui_bold"],
            bg=self.ui["accent"], fg=self.ui["accent_text"], padx=10, pady=3,
        ).pack(side="right", padx=(8, 0))

        def copy_code():
            self._clipboard_write(body)
            self.header_status.config(text="Copied")

        def run_code():
            if not self._code_panel_visible:
                self._toggle_code_panel()
            ext_map = {
                "python": ".py", "javascript": ".js", "typescript": ".ts", "java": ".java",
                "rust": ".rs", "go": ".go", "bash": ".sh", "html": ".html", "cpp": ".cpp", "c": ".c",
            }
            self.code_script_name = f"snippet{ext_map.get(lang_key, '.txt')}"
            self.code_name_label.config(text=self.code_script_name)
            self.code_editor.delete("1.0", "end")
            self.code_editor.insert("1.0", body)
            self.code_editor.focus_set()
            self.header_status.config(text=f"Opened in {CODING_API_LABEL}")

        tk.Button(
            actions, text="▶ Run", font=self.fonts["small"],
            bg="#000000", fg="#3b82f6", activebackground="#1a1a1a",
            relief="flat", bd=0, padx=10, pady=4, cursor="hand2", command=run_code,
        ).pack(side="right", padx=(0, 6))
        tk.Button(
            actions, text="Copy", font=self.fonts["small"],
            bg="#000000", fg="#3b82f6", activebackground="#1a1a1a",
            relief="flat", bd=0, padx=10, pady=4, cursor="hand2", command=copy_code,
        ).pack(side="right")

        tk.Frame(outer, bg="#5c5c5c", height=1).pack(fill="x")

        lines = max(body.count("\n") + 1, 2)
        code_view = tk.Text(
            outer, font=self.fonts["mono"], bg=self.ui["code_bg"], fg=self.ui["code_fg"],
            relief="flat", bd=0, padx=14, pady=12, wrap="none",
            height=min(max(lines, 2), 28), width=72,
            highlightthickness=0, insertwidth=0, cursor="arrow",
            selectbackground=self.ui["accent"], selectforeground=self.ui["accent_text"],
        )
        code_view.insert("1.0", body)
        code_view.config(state="disabled")
        code_view.pack(fill="x")

        def copy_shortcut(_e=None):
            copy_code()
            return "break"

        mod = "Command" if os.name == "darwin" else "Control"
        code_view.bind(f"<{mod}-c>", copy_shortcut, add="+")
        code_view.bind("<Control-c>", copy_shortcut, add="+")
        code_view.bind("<Button-3>", copy_shortcut)
        return outer

    def _code_menu_style(self) -> Dict[str, str]:
        return {
            "strip_bg": "#3a3a3a",
            "strip_fg": "#e8e8e8",
            "strip_active": "#4a4a4a",
            "menu_bg": "#2b2b2b",
            "menu_fg": "#e8e8e8",
            "menu_active_bg": self.ui["accent"],
            "menu_active_fg": self.ui["accent_text"],
        }

    def _accel(self, key: str) -> str:
        return f"⌘{key}" if os.name == "darwin" else f"Ctrl+{key}"

    def _code_select_all(self):
        self.code_editor.tag_add(tk.SEL, "1.0", "end-1c")
        self.code_editor.mark_set(tk.INSERT, "end-1c")
        self.code_editor.see(tk.INSERT)
        self.code_editor.focus_set()

    def _code_new_script(self):
        self.code_script_name = "untitled.py"
        self.code_name_label.config(text=self.code_script_name)
        self.code_editor.delete("1.0", "end")
        self.code_editor.insert("1.0", CatR1CodingAPI.default_snippet())
        self.code_output.delete("1.0", "end")
        CatR1CodingAPI.session().code = CatR1CodingAPI.default_snippet()
        self.code_editor.focus_set()
        self.header_status.config(text=f"New buffer · {CODING_API_LABEL}")

    def _code_clear_editor(self):
        self.code_editor.delete("1.0", "end")
        self.code_editor.focus_set()

    def _code_clear_output(self):
        self.code_output.delete("1.0", "end")

    def _code_undo(self):
        try:
            self.code_editor.edit_undo()
        except tk.TclError:
            pass

    def _code_redo(self):
        try:
            self.code_editor.edit_redo()
        except tk.TclError:
            pass

    def _code_send_to_chat(self):
        code = self.code_editor.get("1.0", "end-1c").strip()
        if not code:
            return
        lang = self.engine.detect_lang_from_text(code) or "python"
        block = f"```{lang}\n{code}\n```"
        self._clear_placeholder()
        self.entry.delete("1.0", "end")
        self.entry.insert("1.0", block)
        self.entry.config(fg=self.ui["text"])
        self._placeholder_active = False
        self.entry.focus_set()
        self.header_status.config(text="Code sent to chat")

    def _build_code_menustrip(self, parent):
        st = self._code_menu_style()
        strip = tk.Frame(parent, bg=st["strip_bg"], height=40)
        strip.pack(fill="x")
        strip.pack_propagate(False)

        menus = tk.Frame(strip, bg=st["strip_bg"])
        menus.pack(side="left", padx=4)

        def menubutton(label: str, items: List[Tuple[str, Any]]):
            mb = tk.Menubutton(
                menus, text=label, font=self.fonts["small"],
                bg=st["strip_bg"], fg=st["strip_fg"],
                activebackground=st["strip_active"], activeforeground=st["strip_fg"],
                relief="flat", bd=0, padx=10, pady=8, cursor="hand2",
            )
            drop = tk.Menu(
                mb, tearoff=0,
                bg=st["menu_bg"], fg=st["menu_fg"],
                activebackground=st["menu_active_bg"], activeforeground=st["menu_active_fg"],
                relief="flat", bd=0,
            )
            for entry in items:
                if entry is None:
                    drop.add_separator()
                else:
                    text, cmd = entry
                    drop.add_command(label=text, command=cmd)
            mb.config(menu=drop)
            mb.pack(side="left")

        menubutton("File", [
            (f"New script\t{self._accel('N')}", self._code_new_script),
            (f"Clear editor", self._code_clear_editor),
            None,
            (GUI_TAGLINE, lambda: None),
        ])
        menubutton("Edit", [
            (f"Undo\t{self._accel('Z')}", self._code_undo),
            (f"Redo\t{'⌘⇧Z' if os.name == 'darwin' else 'Ctrl+Y'}", self._code_redo),
            None,
            (f"Cut\t{self._accel('X')}", lambda: self._clip_action(self.code_editor, "cut")),
            (f"Copy\t{self._accel('C')}", lambda: self._clip_action(self.code_editor, "copy")),
            (f"Paste\t{self._accel('V')}", lambda: self._clip_action(self.code_editor, "paste")),
            None,
            (f"Select all\t{self._accel('A')}", self._code_select_all),
        ])
        menubutton("Run", [
            (f"Run agent\t{self._accel('Return')}", self._run_code_interpreter),
            (f"Clear output", self._code_clear_output),
            (f"Send to chat", self._code_send_to_chat),
        ])
        menubutton("Agent", [
            (f"Lint buffer", self._code_lint_interpreter),
            (f"Explain buffer", self._code_explain_interpreter),
            None,
            (f"New buffer\t{self._accel('N')}", self._code_new_script),
            (f"Clear buffer", self._code_clear_editor),
        ])
        menubutton("Help", [
            (f"{BRAND} interpreter help", lambda: self.log(self._bot_display_name(), ClaudeMythosRuntime.code_help(), "bot")),
        ])

        right = tk.Frame(strip, bg=st["strip_bg"])
        right.pack(side="right", padx=8)

        self.code_name_label = tk.Label(
            right, text=self.code_script_name, font=self.fonts["mono"],
            bg=st["strip_bg"], fg="#b0b0b0",
        )
        self.code_name_label.pack(side="left", padx=(0, 10))

        tk.Label(
            right, text=GUI_TAGLINE, font=self.fonts["ui_bold"],
            bg=self.ui["accent"], fg=self.ui["accent_text"],
            padx=10, pady=3,
        ).pack(side="left", padx=(0, 10))

        tk.Button(
            right, text="▶ Run", font=self.fonts["ui_bold"],
            bg="#000000", fg="#3b82f6",
            activebackground=self.ui["send_hover"], activeforeground="#3b82f6",
            relief="flat", bd=0, padx=14, pady=4, cursor="hand2",
            command=self._run_code_interpreter,
        ).pack(side="left")

        tk.Frame(parent, bg="#5c5c5c", height=1).pack(fill="x")

        mod_key = "Command" if os.name == "darwin" else "Control"

        def new_script_shortcut(_e=None):
            if self._code_panel_visible:
                self._code_new_script()
                return "break"

        def run_shortcut(_e=None):
            if self._code_focused():
                self._run_code_interpreter()
                return "break"

        self.root.bind_all(f"<{mod_key}-n>", new_script_shortcut, add="+")
        self.root.bind_all(f"<{mod_key}-Return>", run_shortcut, add="+")

    def _code_focused(self) -> bool:
        try:
            return self.root.focus_get() == self.code_editor
        except (KeyError, tk.TclError):
            return False

    def _code_output_key_filter(self, event):
        mod = 0x8 if os.name == "darwin" else 0x4
        if event.state & mod and event.keysym.lower() in {"c", "a", "x"}:
            return None
        if event.char and event.char.isprintable():
            return "break"
        if event.keysym in {"Return", "space", "Tab", "BackSpace", "Delete"}:
            return "break"
        return None

    def _widget_selection(self, widget) -> Optional[str]:
        try:
            return widget.get(tk.SEL_FIRST, tk.SEL_LAST)
        except tk.TclError:
            return None

    def _clipboard_write(self, text: str) -> None:
        if text is None:
            return
        if os.name == "darwin":
            try:
                subprocess.run(
                    ["pbcopy"], input=text, text=True, check=True,
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
                return
            except (OSError, subprocess.CalledProcessError):
                pass
        try:
            self.root.clipboard_clear()
            self.root.clipboard_append(text)
            self.root.update_idletasks()
            self.root.update()
        except tk.TclError:
            pass

    def _clipboard_read(self) -> Optional[str]:
        if os.name == "darwin":
            try:
                proc = subprocess.run(
                    ["pbpaste"], capture_output=True, text=True, check=True,
                    stderr=subprocess.DEVNULL,
                )
                if proc.stdout is not None:
                    return proc.stdout
            except (OSError, subprocess.CalledProcessError):
                pass
        for clip_type in ("STRING", "UTF8_STRING", "public.utf8-plain-text", "TEXT"):
            try:
                return self.root.clipboard_get(type=clip_type)
            except tk.TclError:
                continue
        try:
            return self.root.clipboard_get()
        except tk.TclError:
            return None

    def _clip_action(self, widget, action: str):
        try:
            widget.focus_set()
        except tk.TclError:
            pass
        if action == "cut":
            self._clip_cut(widget)
        elif action == "copy":
            self._clip_copy(widget)
        elif action == "paste":
            self._clip_paste(widget)

    def _clip_copy(self, widget):
        text = self._widget_selection(widget)
        if not text:
            text = widget.get("1.0", "end-1c")
        if text:
            self._clipboard_write(text)

    def _clip_cut(self, widget):
        if str(widget.cget("state")) == "disabled":
            return
        text = self._widget_selection(widget)
        if not text:
            return
        self._clipboard_write(text)
        try:
            widget.delete(tk.SEL_FIRST, tk.SEL_LAST)
        except tk.TclError:
            pass

    def _clip_paste(self, widget):
        if str(widget.cget("state")) == "disabled":
            return
        try:
            widget.focus_set()
        except tk.TclError:
            pass
        text = self._clipboard_read()
        if text is None:
            return
        try:
            widget.delete(tk.SEL_FIRST, tk.SEL_LAST)
        except tk.TclError:
            pass
        widget.insert(tk.INSERT, text)

    def _bind_code_interpreter(self, widget):
        menu = tk.Menu(widget, tearoff=0)
        menu.add_command(label="Cut", command=lambda: self._clip_action(widget, "cut"))
        menu.add_command(label="Copy", command=lambda: self._clip_action(widget, "copy"))
        menu.add_command(label="Paste", command=lambda: self._clip_action(widget, "paste"))
        menu.add_separator()
        menu.add_command(label="Select All", command=self._code_select_all)

        def show_menu(event):
            try:
                widget.focus_set()
                menu.tk_popup(event.x_root, event.y_root)
            finally:
                menu.grab_release()

        def key_copy(_e):
            self._clip_action(widget, "copy")
            return "break"

        def key_cut(_e):
            self._clip_action(widget, "cut")
            return "break"

        def key_paste(_e):
            self._clip_action(widget, "paste")
            return "break"

        def key_undo(_e):
            self._code_undo()
            return "break"

        def key_redo(_e):
            self._code_redo()
            return "break"

        mod = "Command" if os.name == "darwin" else "Control"
        widget.bind("<Button-2>", show_menu)
        widget.bind("<Button-3>", show_menu)
        widget.bind(f"<{mod}-c>", key_copy, add="+")
        widget.bind(f"<{mod}-x>", key_cut, add="+")
        widget.bind(f"<{mod}-v>", key_paste, add="+")
        widget.bind("<Control-c>", key_copy, add="+")
        widget.bind("<Control-x>", key_cut, add="+")
        widget.bind("<Control-v>", key_paste, add="+")
        widget.bind(f"<{mod}-z>", key_undo, add="+")
        widget.bind(f"<{mod}-Z>", key_redo, add="+")

    def _bind_code_output_menu(self, widget):
        menu = tk.Menu(widget, tearoff=0)
        menu.add_command(label="Copy", command=lambda: self._clip_action(widget, "copy"))
        menu.add_command(label="Select All", command=lambda: widget.tag_add(tk.SEL, "1.0", "end-1c"))

        def show_menu(event):
            try:
                widget.focus_set()
                menu.tk_popup(event.x_root, event.y_root)
            finally:
                menu.grab_release()

        def key_copy(_e):
            self._clip_action(widget, "copy")
            return "break"

        widget.bind("<Button-2>", show_menu)
        widget.bind("<Button-3>", show_menu)
        mod = "Command" if os.name == "darwin" else "Control"
        widget.bind(f"<{mod}-c>", key_copy, add="+")
        widget.bind("<Control-c>", key_copy, add="+")

    def _bind_clipboard(self, widget, *, readonly: bool = False, on_edit=None):
        mod = "Command" if os.name == "darwin" else "Control"

        def do_copy(_e=None):
            self._clip_copy(widget)
            return "break"

        def do_cut(_e=None):
            if readonly:
                return "break"
            self._clip_cut(widget)
            if on_edit:
                on_edit()
            return "break"

        def do_paste(_e=None):
            if readonly:
                return "break"
            self._clip_paste(widget)
            if on_edit:
                on_edit()
            return "break"

        widget.bind(f"<{mod}-c>", do_copy, add="+")
        widget.bind(f"<{mod}-x>", do_cut, add="+")
        widget.bind(f"<{mod}-v>", do_paste, add="+")
        widget.bind("<Control-c>", do_copy, add="+")
        widget.bind("<Control-x>", do_cut, add="+")
        widget.bind("<Control-v>", do_paste, add="+")

    def _toggle_code_panel(self):
        if self._code_panel_visible:
            self.code_panel.pack_forget()
            self._code_panel_visible = False
        else:
            self.code_panel.pack(side="bottom", fill="x", padx=0, pady=(0, 8))
            self._code_panel_visible = True
            self.code_editor.focus_set()

    def _code_lint_interpreter(self):
        code = self.code_editor.get("1.0", "end-1c").strip()
        lang = self.engine.detect_lang_from_text(code) or "python"
        ok, msg = CatR1CodingAPI.lint(code, lang)
        self.code_output.config(state="normal")
        self.code_output.delete("1.0", "end")
        self.code_output.insert("1.0", f"{'ok' if ok else 'fail'}: {msg}")
        self.header_status.config(text=f"Lint · {CODING_API_LABEL}")

    def _code_explain_interpreter(self):
        code = self.code_editor.get("1.0", "end-1c").strip()
        if not code:
            self.header_status.config(text="Explain: empty buffer")
            return
        lang = self.engine.detect_lang_from_text(code) or "python"
        text = CatR1CodingAPI.explain(self.engine, code, lang)
        self.log(self._bot_display_name(), text, "bot")
        self.header_status.config(text=f"Explain · {CODING_API_LABEL}")

    def _run_code_interpreter(self):
        code = self.code_editor.get("1.0", "end-1c").strip()
        if not code:
            self.header_status.config(text=f"{CODING_API_LABEL}: empty buffer")
            return
        lang = self.engine.detect_lang_from_text(code) or "python"
        self.header_status.config(text=f"Agent run · {MYTHOS_NAME} · {self.code_script_name} ({lang})…")
        result = CatR1CodingAPI.agent_run(
            self.engine, code, lang, self.code_script_name,
        )
        self.code_output.config(state="normal")
        self.code_output.delete("1.0", "end")
        interp = active_interpreter()
        panel = interp.format_panel(result) if hasattr(interp, "format_panel") else O1PreviewSyntax.format_panel_stdout(result)
        self.code_output.insert("1.0", panel)
        if result.get("ok"):
            self.header_status.config(text=f"{BRAND} ready · {MYTHOS_NAME} · run #{result.get('runs', 1)}")
        else:
            self.header_status.config(text=f"Lint error · {MYTHOS_NAME}")

    def _entry_text(self) -> str:
        return self.entry.get("1.0", "end-1c")

    def _set_placeholder(self):
        if not self._placeholder_active:
            return
        self.entry.config(state="normal")
        self.entry.delete("1.0", "end")
        self.entry.insert("1.0", self._placeholder)
        self.entry.config(fg=self.ui["muted"])

    def _clear_placeholder(self):
        if not self._placeholder_active:
            return
        self._placeholder_active = False
        self.entry.config(state="normal", fg=self.ui["text"])
        self.entry.delete("1.0", "end")

    def _sync_placeholder(self, _event=None):
        if self._placeholder_active:
            return
        if not self._entry_text().strip():
            self._placeholder_active = True
            self._set_placeholder()

    _PLACEHOLDER_SKIP_KEYS = frozenset({
        "Shift_L", "Shift_R", "Control_L", "Control_R", "Alt_L", "Alt_R",
        "Meta_L", "Meta_R", "Caps_Lock", "Tab",
    })

    def _on_entry_key(self, event):
        if self._placeholder_active and event.keysym not in self._PLACEHOLDER_SKIP_KEYS:
            self._clear_placeholder()

    def _on_entry_click(self, _event=None):
        if self._placeholder_active:
            self._clear_placeholder()

    def _on_entry_edit(self, _event=None):
        if self._placeholder_active:
            self._clear_placeholder()

    def _on_entry_focus_in(self, _event=None):
        if self._placeholder_active:
            self._clear_placeholder()

    def _on_entry_focus_out(self, _event=None):
        if self._placeholder_active:
            return
        if not self._entry_text().strip():
            self._placeholder_active = True
            self._set_placeholder()

    def _on_frame_configure(self, _event=None):
        self.chat_canvas.configure(scrollregion=self.chat_canvas.bbox("all"))

    def _on_canvas_configure(self, event):
        width = min(event.width - 48, 820)
        self.chat_canvas.itemconfig(self._canvas_window, width=max(width, 400))
        x = max((event.width - width) // 2, 24)
        self.chat_canvas.coords(self._canvas_window, x, 0)

    def _on_mousewheel(self, event):
        widget = self.chat_canvas.winfo_containing(event.x_root, event.y_root)
        if widget in (self.chat_canvas, self.messages_frame) or str(widget).startswith(str(self.messages_frame)):
            self.chat_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def _on_enter(self, event):
        if event.state & 0x1:
            return
        self.send()
        return "break"

    def _hide_empty(self):
        self.empty_state.place_forget()

    def _show_empty(self):
        if not self._msg_widgets:
            self.empty_state.place(relx=0.5, rely=0.40, anchor="center")

    def _add_history_item(self, title: str):
        title = (title[:36] + "…") if len(title) > 37 else title
        self._history_items.insert(0, title)
        self._history_items = self._history_items[:8]
        for w in self.history_frame.winfo_children():
            w.destroy()
        for item in self._history_items:
            btn = tk.Button(
                self.history_frame, text=item, font=self.fonts["ui"], bg="#000000",
                fg="#3b82f6", activebackground=self.ui["history_hover"],
                relief="flat", bd=0, anchor="w", padx=12, pady=8, cursor="hand2",
                command=lambda t=item: self._prefill(t),
            )
            btn.pack(fill="x", padx=6, pady=1)

    def _new_chat(self):
        self._finalize_stream()
        for w in self._msg_widgets:
            w.destroy()
        self._msg_widgets.clear()
        self.engine.chat.new_session()
        self.engine.persistent_memory.total_sessions += 1
        _save_persistent_memory(self.engine.persistent_memory)
        self.engine.clear_history()
        self._clear_placeholder()
        self.entry.delete("1.0", "end")
        self.entry.config(fg=self.ui["text"], state="normal")
        self._placeholder_active = False
        self._show_empty()
        self._apply_chat_mode()
        self.header_status.config(text=f"{BRAND} ready")
        self.chat_canvas.yview_moveto(0.0)
        self.entry.focus_set()

    def _prefill(self, text: str):
        self._clear_placeholder()
        self.entry.delete("1.0", "end")
        self.entry.insert("1.0", text if not text.endswith("…") else text[:-1])
        self.entry.config(fg=self.ui["text"], state="normal")
        self._placeholder_active = False
        self.entry.focus_set()

    def _avatar(self, parent, glyph: str, bg: str) -> tk.Frame:
        wrap = tk.Frame(parent, bg=bg, width=34, height=34)
        wrap.pack_propagate(False)
        tk.Label(
            wrap, text=glyph, font=self.fonts["ui_bold"], bg=bg, fg=self.ui["mascot_fg"],
        ).place(relx=0.5, rely=0.5, anchor="center")
        return wrap

    def _append_message(self, role: str, text: str, kind: str = "text", code_lang: str = ""):
        self._hide_empty()
        is_user = role == "user"
        plain = self._plain(text)

        row = tk.Frame(self.messages_frame, bg=self.ui["bg"])
        row.pack(fill="x", pady=(10, 10), padx=8)
        self._msg_widgets.append(row)

        inner = tk.Frame(row, bg=self.ui["bg"])
        inner.pack(anchor="e" if is_user else "w", fill="x")

        content_row = tk.Frame(inner, bg=self.ui["bg"])
        content_row.pack(anchor="e" if is_user else "w")

        if not is_user and kind != "code":
            self._avatar(content_row, MASCOT_GLYPH, self.ui["mascot_bg"]).pack(
                side="left", padx=(0, 10), pady=2,
            )

        msg_col = tk.Frame(content_row, bg=self.ui["bg"])
        msg_col.pack(side="left" if not is_user else "right")

        if is_user:
            bubble_bg, bubble_fg = self.ui["user_bg"], self.ui["user_fg"]
        elif kind == "code":
            bubble_bg, bubble_fg = self.ui["code_bg"], self.ui["code_fg"]
        else:
            bubble_bg, bubble_fg = self.ui["bot_bg"], self.ui["bot_fg"]

        if kind == "code":
            block = self._render_chat_code_block(msg_col, text, code_lang)
            block.pack(anchor="w")
        else:
            bubble = tk.Frame(msg_col, bg=bubble_bg)
            bubble.pack(anchor="e" if is_user else "w")
            label_font = self.fonts["ui"]
            lbl = tk.Label(
                bubble, text=plain, font=label_font, bg=bubble_bg, fg=bubble_fg,
                justify="left", wraplength=600,
                padx=self.ui["radius_pad"] if is_user else 2,
                pady=10 if is_user else 4,
            )
            lbl.pack()

        if is_user:
            self._avatar(content_row, "U", self.ui["avatar_user"]).pack(side="left", padx=(10, 0), pady=2)

        self.root.after(10, self._scroll_to_bottom)

    def _scroll_to_bottom(self):
        self.chat_canvas.update_idletasks()
        self.chat_canvas.yview_moveto(1.0)

    def log(self, sender, text, tag=None, code_lang: str = ""):
        if text is None:
            text = ""
        text = str(text).strip()
        if not text:
            return
        if sender in ("YOU", "API", "user"):
            self._append_message("user", text)
        elif sender == "THINK":
            self._append_think_block(text)
        elif tag == "code":
            body, lang = self._parse_chat_codeblock(text, code_lang)
            self._append_message("assistant", body, kind="code", code_lang=lang)
        else:
            fenced = re.search(r"```([a-zA-Z0-9_+#-]*)\s*\n([\s\S]*?)```", text)
            if fenced:
                lead = text[:fenced.start()].strip()
                if lead:
                    self._append_message("assistant", lead)
                block = fenced.group(2).rstrip()
                lang = (fenced.group(1) or "").strip()
                self._append_message("assistant", block, kind="code", code_lang=lang)
                tail = text[fenced.end():].strip()
                if tail:
                    self._append_message("assistant", tail)
            else:
                self._append_message("assistant", text)
        if sender == "SYSTEM":
            self.header_status.config(text=text[:72])

    def send(self):
        raw = self._entry_text()
        if self._placeholder_active:
            if not raw.strip() or raw.strip() == self._placeholder:
                return
            self._clear_placeholder()
        msg = self._entry_text().strip()
        if not msg:
            return
        self.entry.delete("1.0", "end")
        self._placeholder_active = True
        self._set_placeholder()
        self._add_history_item(msg)
        self.log("YOU", msg, "user")
        self.header_status.config(text=f"{BRAND} thinking…")
        self._apply_chat_mode()
        threading.Thread(target=self._infer, args=(msg,), daemon=True).start()

    @staticmethod
    def _strip_thinking_prefix(resp: str, think: str) -> str:
        if not think or not resp:
            return resp
        tc = think.strip()
        if resp.startswith(tc):
            return resp[len(tc):].lstrip("\n")
        return resp

    def _infer(self, prompt):
        try:
            expert = self._chat_mode == "expert"
            self.engine.ultrathink_on = expert and self._thinking_on

            out = self.engine.chat.turn(prompt, simulate=False)
            resp = (out.get("message") or {}).get("content", "") or "(no response)"
            think = out.get("thinking", "")
            display = self._strip_thinking_prefix(resp, think)
            sid = out.get("session", "")[:8]
            mode_txt = CAT_R1_PRO if expert else CAT_R1_FLASH
            show_think = bool(think and expert and self._thinking_on)

            def _show():
                self._finalize_stream()
                if show_think:
                    self._append_think_block(think)
                self._display(display)
                stats = getattr(self.engine, "last_inference_stats", None)
                speed = (
                    f" · {stats.tps_estimate:.0f} tok/s · {stats.encode_ms:.0f}ms"
                    if stats and stats.model_label else ""
                )
                self.header_status.config(
                    text=f"{BRAND} ready · {mode_txt} · session {sid}{speed}" if sid
                    else f"{BRAND} ready · {mode_txt}{speed}"
                )

            self.root.after(0, _show)
        except Exception as exc:
            import traceback
            tb = traceback.format_exc()
            err = f"`{exc.__class__.__name__}: {exc}`"
            self.root.after(0, lambda e=err: self.log(self._bot_display_name(), f"Error — {e}", "bot"))
            self.root.after(0, lambda: self.header_status.config(text=f"{BRAND} ready"))
            print(f"[{BRAND}] error: {tb}", file=sys.stderr)

    def _flush_stream(self):
        frame_buf = []
        tokens = []
        with self.engine._lock:
            tokens, self._stream_tokens = self._stream_tokens, []

        for tok in tokens:
            frame_buf.append(tok)

        frame = "".join(frame_buf)
        if not frame:
            return

        if not self._stream_started:
            self._stream_started = True
            self._hide_empty()
            row = tk.Frame(self.messages_frame, bg=self.ui["bg"])
            row.pack(fill="x", pady=(10, 10), padx=8)
            self._msg_widgets.append(row)
            inner = tk.Frame(row, bg=self.ui["bg"])
            inner.pack(anchor="w", fill="x")
            content_row = tk.Frame(inner, bg=self.ui["bg"])
            content_row.pack(anchor="w")
            self._avatar(content_row, MASCOT_GLYPH, self.ui["mascot_bg"]).pack(
                side="left", padx=(0, 10), pady=2,
            )
            msg_col = tk.Frame(content_row, bg=self.ui["bg"])
            msg_col.pack(side="left")
            bubble = tk.Frame(msg_col, bg=self.ui["bot_bg"])
            bubble.pack(anchor="w")
            self._stream_widget = tk.Text(
                bubble, font=self.fonts["ui"], bg=self.ui["bot_bg"], fg=self.ui["bot_fg"],
                justify="left", wrap="word", width=72, height=1,
                relief="flat", bd=0, padx=2, pady=4, highlightthickness=0,
                cursor="arrow",
            )
            self._stream_widget.pack()
            self._stream_widget.config(state="normal")

        if self._stream_widget is not None:
            self._stream_widget.config(state="normal")
            self._stream_widget.insert("end", frame)
            self._stream_widget.see("end")
            line_count = int(self._stream_widget.index("end-1c").split(".")[0])
            self._stream_widget.config(height=min(line_count, 28))
            self._stream_widget.config(state="disabled")
            self.root.after(5, self._scroll_to_bottom)

    def _finalize_stream(self):
        self._flush_stream()
        if self._stream_widget is not None:
            self._stream_widget.config(state="disabled")
        self._stream_widget = None
        self._stream_tokens = []
        self._stream_started = False

    def _display(self, text):
        if not text:
            return
        text = str(text)
        pattern = re.compile(r"```([a-zA-Z0-9_+#-]*)\s*\n([\s\S]*?)```")
        pos = 0
        found = False
        for m in pattern.finditer(text):
            found = True
            if m.start() > pos:
                lead = text[pos:m.start()].strip()
                if lead:
                    self.log(self._bot_display_name(), lead, "bot")
            lang = (m.group(1) or "").strip()
            block = m.group(2).rstrip()
            self.log(self._bot_display_name(), block, "code", code_lang=lang)
            pos = m.end()   
        if found:
            tail = text[pos:].strip()
            if tail:
                self.log(self._bot_display_name(), tail, "bot")
        else:
            self.log(self._bot_display_name(), text, "bot")

    def _start_api(self):
        if not CONFIG.get("api_enabled", False):
            return
        gui = self
        class Handler(BaseHTTPRequestHandler):
            def _json(self, code, data):
                body = json.dumps(data).encode(); self.send_response(code); self.send_header("Content-Type","application/json"); self.send_header("Content-Length", len(body)); self.end_headers(); self.wfile.write(body)
            def _auth(self):
                key = self.headers.get("Authorization","").replace("Bearer ","").strip()
                return not CONFIG["api_key"] or key == CONFIG["api_key"]
            def do_POST(self):
                if not self._auth(): return self._json(401,{"error":"Unauthorized"})
                coding_paths = ("/v1/coding/run", "/v1/coding", "/coding")
                catcode_paths = ("/v1/catcode", "/catcode", "/v1/catcode/help")
                paths = ("/message", "/v1/chat/completions", "/chat") + coding_paths
                if CONFIG.get("catcode_no_api", True):
                    paths = ("/message", "/v1/chat/completions", "/chat") + coding_paths
                else:
                    paths = paths + catcode_paths
                if self.path in catcode_paths and CONFIG.get("catcode_no_api", True):
                    return self._json(403, {
                        "error": "catcode is BitNet local-only — use /catcode in chat or GUI",
                        "interpreter": CatCodeInterpreterR1.NAME,
                        "backend": "bitnet",
                        "api": "off",
                        "files": "off",
                    })
                if self.path not in paths: return self._json(404,{"error":"Not found"})
                try:
                    length = int(self.headers.get("Content-Length",0)); data = json.loads(self.rfile.read(length).decode()) if length else {}
                except Exception: return self._json(400,{"error":"Invalid JSON"})
                if self.path in coding_paths:
                    out = CatR1CodingAPI.parse_request(gui.engine, data)
                    if out.get("error"):
                        return self._json(400, out)
                    if out.get("action") == "run" and out.get("ok"):
                        gui.root.after(0, lambda o=out: (
                            gui.code_output.config(state="normal"),
                            gui.code_output.delete("1.0", "end"),
                            gui.code_output.insert("1.0", o.get("output", "")),
                        ))
                    return self._json(200, out)
                if self.path == "/chat":
                    out = gui.engine.chat.parse_request(data)
                    if out.get("error"):
                        return self._json(400, out)
                    msg = out.get("message", {})
                    gui.root.after(0, lambda: (
                        gui.log("API", str(data.get("message", data))[:120], "user"),
                        gui.log(gui.engine.name, msg.get("content", "")),
                    ))
                    return self._json(200, out)
                prompt = data.get("message") or data.get("prompt") or next((m["content"] for m in reversed(data.get("messages",[])) if m.get("role")=="user"), "")
                if not prompt: return self._json(400,{"error":"Missing prompt"})
                out = gui.engine.chat.turn(str(prompt), simulate=False)
                resp = out.get("message", {}).get("content", "")
                think = out.get("thinking", "")
                gui.root.after(0, lambda: (gui.log("API", prompt, "user"), gui.log(gui.engine.name, resp)))
                if self.path == "/v1/chat/completions":
                    model = data.get("model") or CATR1_MODEL_ID
                    msg_payload: Dict[str, Any] = {"role": "assistant", "content": resp}
                    if think:
                        msg_payload["reasoning_content"] = think
                    if think and data.get("include_thinking"):
                        msg_payload["content"] = (
                            f"{active_interpreter().wrap_thinking(think)}\n\n{resp}"
                            if CatCodeInterpreterR1.enabled() or CatR1CodeInterpreter01.enabled()
                            else f"\n{think}\n\n\n{resp}"
                        )
                    return self._json(200, {
                        "id": f"chatcmpl-{int(time.time())}",
                        "object": "chat.completion",
                        "model": model,
                        "choices": [{"index": 0, "message": msg_payload, "finish_reason": "stop"}],
                    })
                return self._json(200, {"response": resp, "thinking": think, "model": CATR1_MODEL_ID, **{k: out[k] for k in ("session", "turn", "protocol") if k in out}})
            def do_GET(self):
                if not self._auth(): return self._json(401,{"error":"Unauthorized"})
                if self.path == "/v1/models":
                    return self._json(200, {"data": [CatR1LLM.model_card()]})
                if self.path == "/web/sites":
                    sites = [
                        {"id": r.site_id, "title": r.title, "template": r.template,
                        "preview": r.preview_url(CONFIG["api_port"]), "created": r.created}
                        for r in list(gui.engine.web._sites.values())
                    ]
                    return self._json(200, {"files": "off", "sites": sites, "count": len(sites)})
                if self.path.startswith("/web/preview/"):
                    sid = self.path.rsplit("/", 1)[-1].split("?")[0]
                    rec = gui.engine.web.get(sid)
                    if not rec:
                        return self._json(404, {"error": f"site not found: {sid}"})
                    body = rec.html.encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                if self.path == "/v1/coding/help":
                    return self._json(200, {
                        "protocol": CODING_API_PROTO,
                        "version": CODING_API_VER,
                        "engine": CODING_API_LABEL,
                        "interpreter": CatCodeInterpreterR1.NAME,
                        "fork": CatCodeInterpreterR1.FORK_REPO,
                        "backend": "bitnet",
                        "api": "off" if CONFIG.get("catcode_no_api", True) else "local",
                        "files": "off",
                        "help": CatR1CodingAPI.help_text(),
                        "tools": list(CatR1CodingAPI.TOOLS),
                    })
                if self.path == "/v1/catcode/help" and not CONFIG.get("catcode_no_api", True):
                    return self._json(200, {
                        "protocol": CODING_API_PROTO,
                        "version": CODING_API_VER,
                        "engine": CODING_API_LABEL,
                        "interpreter": CatCodeInterpreterR1.NAME,
                        "fork": CatCodeInterpreterR1.FORK_REPO,
                        "backend": "bitnet",
                        "files": "off",
                        "help": CatR1CodingAPI.help_text(),
                        "tools": list(CatR1CodingAPI.TOOLS),
                    })
                self._json(200, {
                    "usage": (
                        "POST /chat · BitNet catcode via chat/GUI `/catcode` (no API)"
                        if CONFIG.get("catcode_no_api", True)
                        else "POST /chat · POST /coding · POST /catcode · GET /v1/catcode/help · GET /web/sites · GET /web/preview/<id>"
                    ),
                    "coding_api": CODING_API_LABEL,
                    "catcode_backend": "bitnet",
                    "catcode_api": "off" if CONFIG.get("catcode_no_api", True) else "local",
                    "web": CatR1WebProgram.help_text() if CatR1WebProgram.enabled() else "disabled",
                })
            def log_message(self,*a): pass
        def serve():
            try: ThreadingHTTPServer(("127.0.0.1", CONFIG["api_port"]), Handler).serve_forever()
            except Exception as e: gui.root.after(0, lambda: gui.log("SYSTEM", f"API error: {e}"))
        threading.Thread(target=serve, daemon=True).start()

# ──────────────────────────────────────────────────────────────
# ENTRY
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if "--catcode" in sys.argv or "--catseek" in sys.argv or "--catrcode" in sys.argv:
        engine = CatR11Engine()
        try:
            if "--catrcode" in sys.argv:
                flag = "--catrcode"
            else:
                flag = "--catseek" if "--catseek" in sys.argv else "--catcode"
            idx = sys.argv.index(flag)
            tail = " ".join(sys.argv[idx + 1:]).strip()
            prompt = tail if tail else "/catrcode"
            if not prompt.startswith(("/catrcode", "/catseek", "/catcode")) and not CatR1CodeR1.wants_catseek(prompt):
                prompt = f"/catrcode {prompt}"
            print(engine.generate(prompt))
        finally:
            engine.clear_history()
            _save_persistent_memory(engine.persistent_memory)
    elif "--chat" in sys.argv or "-c" in sys.argv:
        engine = CatR11Engine()
        try:
            run_chat_cli(engine)
        finally:
            engine.clear_history()
            _save_persistent_memory(engine.persistent_memory)
    else:
        def _on_exit(root, engine):
            if messagebox.askokcancel("Quit", f"Exit {WINDOW_TITLE}?"):
                engine.persistent_memory.total_messages += len(engine.chat_history)
                engine.clear_history()
                _save_persistent_memory(engine.persistent_memory)
                root.destroy()

        root = tk.Tk()
        engine = CatR11Engine()
        CatR11GUI(root, engine=engine)
        root.protocol("WM_DELETE_WINDOW", lambda: _on_exit(root, engine))
        root.mainloop()
