"""Clasificador afectivo por lexico para LucIA (SBSTM).

Sin dependencias externas. Salida normalizada en [-1, 1].
Valencia > 0.15 positiva, < -0.15 negativa, resto neutral.
"""
from __future__ import annotations

import re
from typing import Any, Dict, Tuple

POSITIVAS = frozenset(
    "alegre feliz contento genial excelente maravilloso amor encanta "
    "gracias bien bueno mejor increible fantastico bravo exito logro "
    "hermoso bonito divertido risa sonrisa esperanza animo"
    .split()
)

NEGATIVAS = frozenset(
    "triste mal malo peor horrible terrible odio enfado rabia miedo "
    "ansiedad preocupado dolor lloro llorar solo soledad cansado "
    "estres frustrado fracaso error problema dificil imposible "
    "odioso decepcion"
    .split()
)

INTENSIFICADORES = frozenset("muy super tan demasiado realmente totalmente".split())
NEGADORES = frozenset("no ni nunca jamas tampoco".split())


def _tokens(texto: str) -> list:
    return re.findall(r"[a-záéíóúñü]+", (texto or "").lower())


def analizar_emocion(texto: str) -> Dict[str, Any]:
    """Clasifica el afecto del texto. Latencia objetivo < 5 ms."""
    toks = _tokens(texto)
    if not toks:
        return {"valencia": 0.0, "etiqueta": "neutral", "confianza": 0.0}
    pos = neg = 0.0
    negar = False
    for i, t in enumerate(toks):
        if t in NEGADORES:
            negar = True
            continue
        peso = 1.5 if (i > 0 and toks[i - 1] in INTENSIFICADORES) else 1.0
        v = 0.0
        if t in POSITIVAS:
            v = peso
        elif t in NEGATIVAS:
            v = -peso
        if v:
            if negar:
                v = -v * 0.7
                negar = False
            if v > 0:
                pos += v
            else:
                neg += -v
    total = pos + neg
    if total == 0:
        return {"valencia": 0.0, "etiqueta": "neutral", "confianza": 0.0}
    val = round((pos - neg) / max(len(toks), 1) * 2.0, 3)
    val = max(-1.0, min(1.0, val))
    etiqueta = "positiva" if val > 0.15 else ("negativa" if val < -0.15 else "neutral")
    conf = round(min(1.0, total / max(len(toks), 1)), 3)
    return {"valencia": val, "etiqueta": etiqueta, "confianza": conf}


def modular_prosodia(valencia: float, emocion_base: float = 0.0) -> Tuple[float, str]:
    """Combina la emocion del usuario con la base del sistema.

    Devuelve (emocion_resultante, consejo_de_tono).
    """
    v = max(-1.0, min(1.0, float(valencia)))
    if v < -0.4:
        return round((emocion_base + v * 0.3) / 2.0, 3), "empatico"
    if v > 0.4:
        return round((emocion_base + v * 0.5) / 2.0, 3), "celebratorio"
    return round(emocion_base, 3), "neutral"