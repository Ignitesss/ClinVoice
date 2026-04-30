# -*- coding: utf-8 -*-
"""Каталоги кэша HF / torch / openai-whisper для ASR (FastAPI и CLI)."""

from __future__ import annotations

import os


def resolve_app_cache_root() -> str:
    root = (os.environ.get("CLINVOICE_CACHE_DIR") or "").strip()
    if not root:
        root = os.path.join(os.path.expanduser("~"), ".cache", "clinvoice")
    os.makedirs(root, exist_ok=True)
    return root


def apply_disk_cache_layout(cache_root: str) -> None:
    hf_default = os.path.join(cache_root, "huggingface")
    openai_dir = os.path.join(cache_root, "openai-whisper")
    torch_dir = os.path.join(cache_root, "torch")
    for d in (hf_default, openai_dir, torch_dir):
        os.makedirs(d, exist_ok=True)
    os.environ.setdefault("HF_HOME", hf_default)
    hf_home = os.environ["HF_HOME"]
    os.makedirs(hf_home, exist_ok=True)
    hub = os.path.join(hf_home, "hub")
    os.environ.setdefault("HF_HUB_CACHE", hub)
    os.makedirs(os.environ["HF_HUB_CACHE"], exist_ok=True)
    os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(hf_home, "transformers"))
    os.makedirs(os.environ["TRANSFORMERS_CACHE"], exist_ok=True)
    os.environ.setdefault("TORCH_HOME", torch_dir)
    os.environ["_CLINVOICE_OPENAI_WHISPER_DIR"] = openai_dir
