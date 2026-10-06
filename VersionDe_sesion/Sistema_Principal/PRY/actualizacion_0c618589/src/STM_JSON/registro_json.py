"""Registro central de todos los JSON del sistema (STM_JSON).

Canónicos persistentes viven en STM_JSON/.
Volátiles de ejecución viven en STM_CH/ y se borran al cerrar.
"""
from __future__ import annotations
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

STM_JSON_DIR: Path = Path(__file__).resolve().parent


def _stm_ch(nombre: str) -> Path:
    try:
        from STM_CH.rutas import STM_CH_DIR
        return STM_CH_DIR / nombre
    except ImportError:
        return STM_JSON_DIR.parent / "STM_CH" / nombre


REGISTRO: Dict[str, Dict[str, Any]] = {
    "ialocal": {
        "ruta": STM_JSON_DIR / "IAlocal.json",
        "descripcion": "Configuración Ollama local (modelos, endpoint, sampling)",
        "volatil": False,
    },
    "ledger_blockchain": {
        "ruta": _stm_ch("blockchain_ledger.json"),
        "descripcion": "Ledger BKSVCB en ejecución (se borra al cerrar)",
        "volatil": True,
    },
    "cache_iafree": {
        "ruta": _stm_ch("openrouter_free_cache.json"),
        "descripcion": "Cache de modelos OpenRouter free (se borra al cerrar)",
        "volatil": True,
    },
    "ledger_historico": {
        "ruta": STM_JSON_DIR / "nonexistent_ledger.json",
        "descripcion": "Ledger de ejemplo/histórico (pruebas 2026-09-22)",
        "volatil": False,
    },
    "informe_purga": {
        "ruta": STM_JSON_DIR / "purgador_informe.json",
        "descripcion": "Informe histórico de purga (2026-09-22)",
        "volatil": False,
    },
}


def listar() -> List[Dict[str, Any]]:
    return [{"nombre": k, **{kk: (str(vv) if kk == "ruta" else vv) for kk, vv in v.items()}}
            for k, v in REGISTRO.items()]


def ruta_de(nombre: str) -> Path:
    if nombre not in REGISTRO:
        raise KeyError(f"JSON '{nombre}' no registrado en STM_JSON")
    return Path(REGISTRO[nombre]["ruta"])


def es_volatil(nombre: str) -> bool:
    return bool(REGISTRO.get(nombre, {}).get("volatil", False))


def leer(nombre: str, defecto: Any = None) -> Any:
    try:
        return json.loads(ruta_de(nombre).read_text(encoding="utf-8"))
    except Exception:
        return defecto


def escribir(nombre: str, datos: Any) -> Path:
    destino = ruta_de(nombre)
    destino.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(destino.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(datos, f, indent=2, ensure_ascii=False)
        Path(tmp).replace(destino)
    except Exception:
        try:
            Path(tmp).unlink(missing_ok=True)
        except Exception:
            pass
        raise
    return destino


class RegistroJSON:
    """Acceso instanciable al registro (uno por orquestador)."""

    def __init__(self) -> None:
        self._nombres = list(REGISTRO)

    def nombres(self) -> List[str]:
        return list(self._nombres)

    def leer(self, nombre: str, defecto: Any = None) -> Any:
        return leer(nombre, defecto)

    def escribir(self, nombre: str, datos: Any) -> Path:
        return escribir(nombre, datos)

    def config_ia_local(self) -> Dict[str, Any]:
        return leer("ialocal", {}) or {}

    def resumen(self) -> List[Dict[str, Any]]:
        out = []
        for item in listar():
            p = Path(item["ruta"])
            out.append({**item, "existe": p.exists(),
                        "bytes": p.stat().st_size if p.exists() else 0})
        return out


_instancia: Optional[RegistroJSON] = None


def get_registro() -> RegistroJSON:
    global _instancia
    if _instancia is None:
        _instancia = RegistroJSON()
    return _instancia
