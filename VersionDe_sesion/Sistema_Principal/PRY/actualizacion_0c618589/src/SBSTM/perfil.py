"""Perfil de usuario persistente para LucIA (SBSTM).

Aprende tono preferido, valencia media, temas y modelos usados.
Persistencia en STM_CH/PSNRL/perfil_usuario.json.
Sin dependencias externas.
"""
from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional


def _ruta_perfil() -> Path:
    try:
        from STM_CH.rutas import PSNRL_DIR
        d = Path(PSNRL_DIR)
    except ImportError:
        d = Path(__file__).resolve().parents[1] / "STM_CH" / "PSNRL"
    d.mkdir(parents=True, exist_ok=True)
    return d / "perfil_usuario.json"


class PerfilUsuario:
    """Perfil agregado y persistente del usuario de LucIA."""

    def __init__(self) -> None:
        self._ruta: Path = _ruta_perfil()
        self.tonos: Counter = Counter()
        self.valencias: List[float] = []
        self.modelos: Counter = Counter()
        self.temas: Counter = Counter()
        self.turnos_totales: int = 0
        self.actualizado_ts: float = 0.0
        self.cargar()

    # ── APRENDIZAJE ───────────────────────────────────────────────────────
    def aprender_turno(
        self,
        tono: str = "",
        valencia: float = 0.0,
        modelo: str = "",
        palabras_clave: Optional[List[str]] = None,
    ) -> None:
        if tono:
            self.tonos[tono] += 1
        try:
            self.valencias.append(max(-1.0, min(1.0, float(valencia))))
            self.valencias = self.valencias[-200:]
        except (TypeError, ValueError):
            pass
        if modelo:
            self.modelos[modelo] += 1
        for p in palabras_clave or []:
            if p:
                self.temas[p.lower()] += 1
        self.turnos_totales += 1
        self.actualizado_ts = time.time()
        self.guardar()

    # ── CONSULTA ──────────────────────────────────────────────────────────
    def tono_preferido(self) -> str:
        return self.tonos.most_common(1)[0][0] if self.tonos else ""

    def valencia_media(self) -> float:
        if not self.valencias:
            return 0.0
        return round(sum(self.valencias) / len(self.valencias), 3)

    def modelo_preferido(self) -> str:
        return self.modelos.most_common(1)[0][0] if self.modelos else ""

    def temas_top(self, n: int = 5) -> List[str]:
        return [t for t, _ in self.temas.most_common(max(1, n))]

    def resumen(self) -> Dict[str, Any]:
        return {
            "turnos_totales": self.turnos_totales,
            "tono_preferido": self.tono_preferido(),
            "valencia_media": self.valencia_media(),
            "modelo_preferido": self.modelo_preferido(),
            "temas_top": self.temas_top(),
            "actualizado_ts": self.actualizado_ts,
        }

    # ── PERSISTENCIA ──────────────────────────────────────────────────────
    def guardar(self) -> Path:
        self._ruta.write_text(
            json.dumps(
                {
                    "tonos": dict(self.tonos),
                    "valencias": self.valencias[-200:],
                    "modelos": dict(self.modelos),
                    "temas": dict(self.temas),
                    "turnos_totales": self.turnos_totales,
                    "actualizado_ts": self.actualizado_ts,
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
            self.tonos = Counter(data.get("tonos", {}))
            self.valencias = list(data.get("valencias", []))[-200:]
            self.modelos = Counter(data.get("modelos", {}))
            self.temas = Counter(data.get("temas", {}))
            self.turnos_totales = int(data.get("turnos_totales", 0))
            self.actualizado_ts = float(data.get("actualizado_ts", 0.0))
            return True
        except Exception:
            return False

    def reiniciar(self) -> None:
        self.tonos.clear()
        self.valencias = []
        self.modelos.clear()
        self.temas.clear()
        self.turnos_totales = 0
        self.actualizado_ts = 0.0
        try:
            self._ruta.unlink(missing_ok=True)
        except Exception:
            pass


_perfil_inst: Optional[PerfilUsuario] = None


def get_perfil() -> PerfilUsuario:
    global _perfil_inst
    if _perfil_inst is None:
        _perfil_inst = PerfilUsuario()
    return _perfil_inst
