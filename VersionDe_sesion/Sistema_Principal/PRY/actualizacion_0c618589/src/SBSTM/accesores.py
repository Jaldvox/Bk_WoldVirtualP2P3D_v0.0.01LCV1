"""Accesores SBSTM: RPLC, voz, IAFREE, STYLOS, JSON, HRTS."""
from __future__ import annotations

import logging
from typing import Any, Dict, Final, List, Optional

logger: logging.Logger = logging.getLogger("WoldVirtualP2P3D.CMFG.SBSTM")

# ─── INTEGRACIÓN SUBSISTEMA RPLC (Reprocesador Lingüístico-Cognitivo) ──────────
def obtener_procesador_rplc() -> Optional[Any]:
    """
    Devuelve la instancia singleton de ProcesadorRPLC.
    RPLC convierte la respuesta bruta de OpenRouter en la voz propia de LucIA
    pasándola por 3 capas: vectorización → transformación sináptica → reformulación.
    """
    try:
        from SBSTM.RPLC import get_procesador_rplc
        return get_procesador_rplc()
    except Exception as _err:
        logger.warning("[SBSTM] RPLC no disponible: %s", _err)
        return None


def reprocesar_respuesta_lucia(texto_bruto: str, contexto_neuronal: Optional[Dict[str, Any]] = None) -> str:
    """
    Conveniencia: pasa el texto crudo de OpenRouter por el pipeline RPLC
    y devuelve la respuesta reformulada en la voz propia de LucIA.
    Si RPLC no está disponible, devuelve el texto original sin modificar.
    """
    try:
        from SBSTM.RPLC import reprocesar_para_lucia
        return reprocesar_para_lucia(texto_bruto, contexto_neuronal)
    except Exception:
        return texto_bruto


# ─── INTEGRACIÓN SUBSISTEMA IAFREE (MODELOS GRATUITOS OPENROUTER) ──────────────
def obtener_servicio_iafree() -> Optional[Any]:
    """Devuelve el cliente singleton de inferencia gratuita OpenRouter."""
    try:
        from SBSTM.IAFREE import get_cliente_iafree
        return get_cliente_iafree()
    except Exception as _err:
        logger.warning("[SBSTM] IAFREE no disponible: %s", _err)
        return None


# ─── INTEGRACIÓN SUBSISTEMA STYLOS (ESTILOS Y TARJETAS DE TERMINAL) ─────────────
def obtener_motor_stylos() -> Optional[Any]:
    """Devuelve la clase EstiloTerminalLucIA del motor de estilos visuales."""
    try:
        from SBSTM.STYLOS import EstiloTerminalLucIA
        return EstiloTerminalLucIA
    except Exception as _err:
        logger.warning("[SBSTM] STYLOS no disponible: %s", _err)
        return None


# ─── VOICE ENGINE (VOZ JUVENIL DE LUCIA) ──────────────────────────────────────
def obtener_motor_voz() -> Optional[Any]:
    """Retorna la instancia global del motor de voz de LucIA."""
    try:
        from SBSTM.voice_engine import get_voice_engine
        return get_voice_engine()
    except Exception as _err:
        logger.warning("[SBSTM] Voice engine no disponible: %s", _err)
        return None


def reproducir_voz_lucia(texto: str, esperar: bool = False, emocion: Optional[float] = None) -> None:
    """Reproduce texto con la voz juvenil oficial de LucIA (es-ES-ElviraNeural)."""
    try:
        from SBSTM.voice_engine import speak
        speak(texto, esperar=esperar, emocion=emocion)
    except Exception as _err:
        logger.debug("[SBSTM] Reproduccion de voz omitida: %s", _err)


def cancelar_voz_lucia() -> None:
    """Interrumpe inmediatamente cualquier emision sonora de LucIA."""
    try:
        from SBSTM.voice_engine import cancel_speech
        cancel_speech()
    except Exception as _err:
        logger.debug("[SBSTM] Cancelacion de voz omitida: %s", _err)


# ─── PUERTAS A REGISTROS CENTRALIZADOS ─────────────────────────────────────────
class GestorJSONLucIA:
    """Puerta del orquestador al registro central STM_JSON."""
    def __init__(self) -> None:
        try:
            from STM_JSON.registro_json import get_registro
            self.registro = get_registro()
        except Exception:
            self.registro = None

    def config_ia(self) -> Dict[str, Any]:
        try:
            return self.registro.config_ia_local() if self.registro else {}
        except Exception:
            return {}

    def linea_estado(self) -> str:
        if not self.registro:
            return "STM_JSON no disponible"
        try:
            res = self.registro.resumen()
            ok = sum(1 for r in res if r["existe"])
            return f"STM_JSON: {ok}/{len(res)} registrados"
        except Exception:
            return "STM_JSON error"


class GestorHRTLucIA:
    """Puerta del orquestador al manifiesto STM_HRTS/pyproject.toml."""
    def __init__(self) -> None:
        try:
            from STM_HRTS.registro_hrts import get_registro_hrts
            self.registro = get_registro_hrts()
        except Exception:
            self.registro = None

    def linea_estado(self) -> str:
        try:
            return self.registro.linea_estado() if self.registro else "STM_HRTS no disponible"
        except Exception:
            return "STM_HRTS error"


# ─── EXPORTACIONES PUBLICAS OFICIALES DEL PAQUETE ────────────────────────────
__all__: Final[List[str]] = [
    # RPLC
    "obtener_procesador_rplc",
    "reprocesar_respuesta_lucia",
    # IAFREE / STYLOS
    "obtener_servicio_iafree",
    "obtener_motor_stylos",
    # VOICE ENGINE
    "obtener_motor_voz",
    "reproducir_voz_lucia",
    "cancelar_voz_lucia",
    # REGISTROS
    "GestorJSONLucIA",
    "GestorHRTLucIA",
]