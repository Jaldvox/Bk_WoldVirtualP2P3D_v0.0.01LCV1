"""Registro extensible de comandos de consola para LucIA (SBSTM).

Sustituye los diccionarios fijos de comandos por un registro
nombre -> funcion con ayuda integrada.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Final, List, Optional

logger: logging.Logger = logging.getLogger("WoldVirtualP2P3D.CMFG.SBSTM")

ComandoFn = Callable[..., Optional[str]]

_REGISTRO: Dict[str, Dict[str, Any]] = {}


def registrar_comando(
    nombre: str,
    funcion: ComandoFn,
    ayuda: str = "",
    alias: Optional[List[str]] = None,
) -> None:
    """Registra un comando. Los alias apuntan a la misma funcion."""
    entrada = {"funcion": funcion, "ayuda": ayuda or "(sin ayuda)"}
    _REGISTRO[nombre.lower()] = entrada
    for a in alias or []:
        _REGISTRO[a.lower()] = entrada


def _buscar_entrada(texto: str) -> Tuple[Optional[Dict[str, Any]], str, str]:
    t = (texto or "").strip().lower()
    if not t:
        return None, "", ""
    if t in _REGISTRO:
        return _REGISTRO[t], t, ""
    for cmd in sorted(_REGISTRO.keys(), key=len, reverse=True):
        if t == cmd or t.startswith(cmd + " "):
            args = (texto or "").strip()[len(cmd):].strip()
            return _REGISTRO[cmd], cmd, args
    partes = (texto or "").strip().split(None, 1)
    if partes and partes[0].lower() in _REGISTRO:
        args = partes[1] if len(partes) > 1 else ""
        return _REGISTRO[partes[0].lower()], partes[0].lower(), args
    return None, "", ""


def despachar(texto: str, contexto: Any = None) -> Optional[str]:
    """Ejecuta el comando si el texto coincide con un nombre registrado.

    Devuelve el resultado de la funcion, o None si no es un comando.
    Nunca lanza excepcion.
    """
    entrada, cmd, args = _buscar_entrada(texto)
    if not entrada:
        return None
    try:
        return entrada["funcion"](args, contexto)
    except Exception as exc:
        logger.warning("[comandos] '%s' fallo: %s", cmd, exc)
        return f"[comando] error: {exc}"


def es_comando(texto: str) -> bool:
    entrada, _, _ = _buscar_entrada(texto)
    return entrada is not None


def listar_comandos() -> List[Dict[str, str]]:
    vistos: Dict[str, str] = {}
    for nombre, entrada in _REGISTRO.items():
        clave = entrada["ayuda"]
        vistos.setdefault(clave, nombre)
    return [
        {"comando": nombre, "ayuda": ayuda}
        for ayuda, nombre in sorted(vistos.items(), key=lambda kv: kv[1])
    ]


def ayuda_texto() -> str:
    lineas = ["Comandos registrados:"]
    for c in listar_comandos():
        lineas.append(f"  {c['comando']:<18} {c['ayuda']}")
    return "\n".join(lineas)


__all__: Final[List[str]] = [
    "registrar_comando",
    "despachar",
    "es_comando",
    "listar_comandos",
    "ayuda_texto",
]