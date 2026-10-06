"""Gestor central de errores del sistema SBSTM."""
from __future__ import annotations

import sys
import threading
import time
import traceback
from typing import Any, Dict, List, Optional


class GestorErroresSistema:
    _instancia: Optional["GestorErroresSistema"] = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls) -> "GestorErroresSistema":
        with cls._lock:
            if cls._instancia is None:
                cls._instancia = super().__new__(cls)
                cls._instancia._inicializar()
            return cls._instancia

    def _inicializar(self) -> None:
        self._errores: List[Dict[str, Any]] = []
        self._conteo: int = 0

    def capturar(
        self,
        excepcion: Optional[BaseException] = None,
        contexto: str = "",
        severidad: str = "ERROR",
    ) -> Dict[str, Any]:
        with self._lock:
            self._conteo += 1
            entrada = {
                "id": self._conteo,
                "timestamp": time.time(),
                "tipo": type(excepcion).__name__ if excepcion else "N/A",
                "mensaje": str(excepcion) if excepcion else "",
                "contexto": contexto,
                "severidad": severidad,
                "traceback": (
                    "".join(traceback.format_exception(type(excepcion), excepcion, excepcion.__traceback__))
                    if excepcion
                    else ""
                ),
            }
            self._errores.append(entrada)
            return entrada

    def resumen_errores(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "total": self._conteo,
                "errores": list(self._errores),
            }

    def limpiar(self) -> None:
        with self._lock:
            self._errores.clear()
            self._conteo = 0


def obtener_gestor_errores() -> GestorErroresSistema:
    return GestorErroresSistema()
