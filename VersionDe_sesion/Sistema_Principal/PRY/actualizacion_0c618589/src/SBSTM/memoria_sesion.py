"""Memoria contextual de sesion para LucIA (SBSTM).

Ventana de ultimos turnos + resumen comprimido por turno.
Persistencia en STM_CH/PSNRL/memoria_<sesion_id>.json.
Sin dependencias externas.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

VENTANA_DEFECTO: int = 8
MAX_CHARS_TURNO: int = 400


def _dir_psnrl() -> Path:
    try:
        from STM_CH.rutas import PSNRL_DIR
        d = Path(PSNRL_DIR)
    except ImportError:
        d = Path(__file__).resolve().parents[1] / "STM_CH" / "PSNRL"
    d.mkdir(parents=True, exist_ok=True)
    return d


class MemoriaSesion:
    """Ventana deslizante de turnos con resumen inyectable en el prompt."""

    def __init__(self, sesion_id: str = "", ventana: int = VENTANA_DEFECTO) -> None:
        self.sesion_id: str = sesion_id or time.strftime("%Y%m%d_%H%M%S")
        self.ventana: int = max(1, int(ventana))
        self._turnos: List[Dict[str, Any]] = []
        self._ruta: Path = _dir_psnrl() / f"memoria_{self.sesion_id}.json"
        self.cargar()

    # ── REGISTRO ──────────────────────────────────────────────────────────
    def registrar_turno(
        self,
        prompt: str,
        respuesta: str,
        tono: str = "",
        valencia: float = 0.0,
    ) -> Dict[str, Any]:
        turno = {
            "n": len(self._turnos) + 1,
            "ts": time.time(),
            "prompt": (prompt or "")[:MAX_CHARS_TURNO],
            "respuesta": (respuesta or "")[:MAX_CHARS_TURNO],
            "tono": tono,
            "valencia": round(float(valencia), 3),
        }
        self._turnos.append(turno)
        self._turnos = self._turnos[-self.ventana:]
        self.guardar()
        return turno

    # ── CONSULTA ──────────────────────────────────────────────────────────
    def resumen_contexto(self) -> str:
        """Texto compacto para anteponer al prompt del modelo."""
        if not self._turnos:
            return ""
        lineas = []
        for t in self._turnos:
            lineas.append(
                f"[T{t['n']}] U:{t['prompt'][:120]} | L:{t['respuesta'][:120]}"
            )
        return "Contexto previo:\n" + "\n".join(lineas)

    def ultimos(self, n: int = 3) -> List[Dict[str, Any]]:
        return list(self._turnos[-max(1, n):])

    def total(self) -> int:
        return len(self._turnos)

    # ── PERSISTENCIA ──────────────────────────────────────────────────────
    def guardar(self) -> Path:
        self._ruta.write_text(
            json.dumps(
                {
                    "sesion_id": self.sesion_id,
                    "ventana": self.ventana,
                    "turnos": self._turnos,
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        return self._ruta

    def cargar(self) -> bool:
        if not self._ruta.exists():
            return False
        try:
            data = json.loads(self._ruta.read_text(encoding="utf-8"))
            self._turnos = list(data.get("turnos", []))[-self.ventana:]
            return True
        except Exception:
            return False

    def limpiar(self) -> None:
        self._turnos = []
        try:
            self._ruta.unlink(missing_ok=True)
        except Exception:
            pass


_memoria_inst: Optional[MemoriaSesion] = None


def get_memoria(sesion_id: str = "") -> MemoriaSesion:
    global _memoria_inst
    if _memoria_inst is None or (sesion_id and _memoria_inst.sesion_id != sesion_id):
        _memoria_inst = MemoriaSesion(sesion_id=sesion_id)
    return _memoria_inst