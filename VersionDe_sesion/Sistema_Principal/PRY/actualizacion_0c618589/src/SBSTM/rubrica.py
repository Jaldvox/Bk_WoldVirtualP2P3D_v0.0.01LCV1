"""Rubrica de calidad creativa para respuestas de LucIA (SBSTM).

Metricas sin dependencias: diversidad lexica, longitud util,
coherencia con el tono y presencia de estructura.
Rango de cada metrica: [0, 1]. Puntuacion global: media ponderada.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List

_PESOS = {
    "diversidad_lexica": 0.30,
    "longitud_util": 0.20,
    "coherencia_tono": 0.25,
    "estructura": 0.15,
    "cierre": 0.10,
}

_TONO_MARCAS = {
    "reflexivo": ("porque", "quizas", "significa", "profundo", "sentido"),
    "analitico": ("datos", "porque", "analisis", "primero", "segundo"),
    "creativo": ("imagina", "como si", "nuevo", "posible", "sueño"),
    "pragmatico": ("paso", "haz", "prueba", "concreto", "ahora"),
}

_CIERRES = ("?", "!", ".", ":", "...")


def _toks(texto: str) -> List[str]:
    return re.findall(r"[a-záéíóúñü0-9]+", (texto or "").lower())


def evaluar_calidad(respuesta: str, tono: str = "") -> Dict[str, Any]:
    """Evalua la calidad creativa de una respuesta. Nunca lanza excepcion."""
    try:
        return _evaluar(respuesta or "", (tono or "").lower())
    except Exception:
        return {"puntuacion": 0.0, "metricas": {}, "veredicto": "error"}


def _evaluar(respuesta: str, tono: str) -> Dict[str, Any]:
    toks = _toks(respuesta)
    n = len(toks)
    if n == 0:
        return {"puntuacion": 0.0, "metricas": {}, "veredicto": "vacia"}
    diversidad = round(len(set(toks)) / n, 3)
    longitud = round(min(1.0, n / 120.0), 3)
    marcas = _TONO_MARCAS.get(tono, ())
    coherencia = round(
        min(1.0, sum(1 for m in marcas if m in respuesta.lower()) / 2.0), 3
    ) if marcas else 0.5
    estructura = round(min(1.0, respuesta.count("\n") / 3.0 + 0.3), 3)
    cierre = 1.0 if respuesta.rstrip().endswith(_CIERRES) else 0.0
    metricas = {
        "diversidad_lexica": diversidad,
        "longitud_util": longitud,
        "coherencia_tono": coherencia,
        "estructura": estructura,
        "cierre": cierre,
    }
    puntuacion = round(sum(metricas[k] * _PESOS[k] for k in _PESOS), 3)
    veredicto = (
        "excelente" if puntuacion >= 0.75
        else "buena" if puntuacion >= 0.5
        else "aceptable" if puntuacion >= 0.3
        else "pobre"
    )
    return {"puntuacion": puntuacion, "metricas": metricas, "veredicto": veredicto}