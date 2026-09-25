"""Catálogo free + IAlocal embebidos para import directo desde mainLCSTM."""
from __future__ import annotations
from typing import Any, Dict, Final, List

MODELOS_FREE_DEFAULT: Final[List[Dict[str, Any]]] = [
    {"id": "openrouter/free", "nombre": "OpenRouter Free Router", "contexto": 200000},
    {"id": "qwen/qwen3.8-27b:free", "nombre": "Qwen 3.8 27B Free", "contexto": 262144},
    {"id": "google/gemma-4-31b-it:free", "nombre": "Gemma 4 31B Free", "contexto": 262144},
    {"id": "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free", "nombre": "Nemotron Nano Reasoning Free", "contexto": 256000},
    {"id": "cohere/north-mini-code:free", "nombre": "North Mini Code Free", "contexto": 256000},
    {"id": "z-ai/glm-5.2:free", "nombre": "GLM 5.2 Free", "contexto": 32768},
]

IALOCAL_EMBEBIDO: Final[Dict[str, Any]] = {
    "provider": "ollama",
    "endpoint": "http://localhost:11434",
    "default_model": "cogito:3b",
    "stream": True,
    "temperature": 0.7,
    "top_p": 0.9,
    "max_tokens": 2048,
    "models_available": [
        "cogito:3b", "qwen3:1.7b", "gemma2:2b", "llama3.2:1b",
        "qwen2-math:1.5b", "qwen2.5-coder:7b", "qwen2.5-coder:3b",
        "qwen2.5:7b", "qwen2.5-coder:latest", "qwen3.5:latest",
    ],
}

FREE_SUFFIX: Final[str] = ":free"


def solo_free(modelos: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [m for m in modelos if str(m.get("id", "")).endswith(FREE_SUFFIX) or m.get("id") == "openrouter/free"]


def es_free(model_id: str) -> bool:
    m = (model_id or "").strip()
    return m == "openrouter/free" or m.endswith(FREE_SUFFIX)
