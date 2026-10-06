"""Base de conocimiento especifico para LucIA (STM_IA).

Documentos .md/.json en STM_IA/conocimiento/.
Busqueda por palabras clave con puntuacion simple, sin dependencias.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List

CONOCIMIENTO_DIR: Path = Path(__file__).resolve().parent / "conocimiento"

_STOP = frozenset(
    "el la los las un una de del en y o que se su con por para como mas "
    "pero sus este esta estos estas ese esa eso aqui hay son fue eran "
    "the and for with from".split()
)


def _tokens(texto: str) -> List[str]:
    toks = re.findall(r"[a-záéíóúñü0-9]{3,}", (texto or "").lower())
    return [t for t in toks if t not in _STOP]


def palabras_clave(texto: str, top_n: int = 5) -> List[str]:
    """Palabras clave de un texto, ordenadas por frecuencia."""
    conteo: Dict[str, int] = {}
    for t in _tokens(texto):
        conteo[t] = conteo.get(t, 0) + 1
    return sorted(conteo, key=lambda k: conteo[k], reverse=True)[: max(1, top_n)]


def _cargar_documentos() -> List[Dict[str, Any]]:
    docs: List[Dict[str, Any]] = []
    if not CONOCIMIENTO_DIR.exists():
        return docs
    for p in sorted(CONOCIMIENTO_DIR.rglob("*")):
        if not p.is_file() or p.suffix not in (".md", ".json"):
            continue
        try:
            if p.suffix == ".json":
                data = json.loads(p.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    docs.append({
                        "fuente": str(p.relative_to(CONOCIMIENTO_DIR)),
                        "titulo": str(data.get("titulo", p.stem)),
                        "texto": str(data.get("texto", "")),
                    })
                elif isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict) and item.get("texto"):
                            docs.append({
                                "fuente": str(p.relative_to(CONOCIMIENTO_DIR)),
                                "titulo": str(item.get("titulo", p.stem)),
                                "texto": str(item.get("texto", "")),
                            })
            else:
                texto = p.read_text(encoding="utf-8", errors="replace")
                titulo = p.stem
                for linea in texto.splitlines()[:5]:
                    if linea.startswith("#"):
                        titulo = linea.lstrip("# ").strip() or titulo
                        break
                docs.append({
                    "fuente": str(p.relative_to(CONOCIMIENTO_DIR)),
                    "titulo": titulo,
                    "texto": texto,
                })
        except Exception:
            continue
    return docs


def buscar(consulta: str, top_k: int = 3) -> List[Dict[str, Any]]:
    """Devuelve los top_k fragmentos mas relevantes con fuente y puntuacion."""
    q = _tokens(consulta)
    if not q:
        return []
    resultados: List[Dict[str, Any]] = []
    for doc in _cargar_documentos():
        toks = _tokens(doc["texto"])
        if not toks:
            continue
        coincid = sum(1 for t in q if t in toks)
        if not coincid:
            continue
        resultados.append({
            "fuente": doc["fuente"],
            "titulo": doc["titulo"],
            "fragmento": doc["texto"][:500],
            "puntuacion": round(coincid / len(set(q)), 3),
        })
    resultados.sort(key=lambda r: r["puntuacion"], reverse=True)
    return resultados[: max(1, top_k)]


def contexto_para_prompt(consulta: str, top_k: int = 2) -> str:
    """Texto listo para anteponer al prompt del modelo, con fuentes."""
    res = buscar(consulta, top_k=top_k)
    if not res:
        return ""
    partes = ["Conocimiento relevante:"]
    for r in res:
        partes.append(f"- [{r['titulo']} | {r['fuente']}]: {r['fragmento'][:300]}")
    return "\n".join(partes)


def listar_fuentes() -> List[str]:
    return sorted({d["fuente"] for d in _cargar_documentos()})
