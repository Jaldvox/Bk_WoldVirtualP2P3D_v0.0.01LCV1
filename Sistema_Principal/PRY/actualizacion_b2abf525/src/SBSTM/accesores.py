"""Accesores SBSTM extraidos de init: RPLC, voz, IAFREE, STYLOS."""
from __future__ import annotations
# ─── INTEGRACIÓN SUBSISTEMA RPLC (Reprocesador Lingüístico-Cognitivo) ──────────
def obtener_procesador_rplc():
    """
    Devuelve la instancia singleton de ProcesadorRPLC.
    RPLC convierte la respuesta bruta de OpenRouter en la voz propia de LucIA
    pasándola por 3 capas: vectorización → transformación sináptica → reformulación.
    """
    try:
        from LC.celebro.CMFG.SBSTM.RPLC import get_procesador_rplc  # pyrefly: ignore[missing-import]
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
        from LC.celebro.CMFG.SBSTM.RPLC import reprocesar_para_lucia  # pyrefly: ignore[missing-import]
        return reprocesar_para_lucia(texto_bruto, contexto_neuronal)
    except Exception:
        return texto_bruto


# ─── EXPORTACIONES PUBLICAS OFICIALES DEL PAQUETE ────────────────────────────
__all__: Final[List[str]] = [
    "TerminalColors",
    "SBSTMSessionContext",
    "obtener_directorio_sbstm",
    "obtener_estado_subsistema",
    "registrar_gancho_post_turno",
    "verificar_integridad_sbstm",
    "obtener_clase_sesion",
    "crear_sesion_neuronal",
    "ejecutar_orquestador_interactivo",
    "consultar_modelos_disponibles",
    "recopilar_estadisticas_sesion",
    "notificar_turno_completado",
    "verificar_cadena_bloques_activa",
    "obtener_ultimo_hash_bloque",
    "forzar_checkpoint_psnrl",
    "purgar_archivos_temporales",
    "obtener_servicio_iafree",
    "obtener_motor_stylos",
    # RPLC
    "obtener_procesador_rplc",
    "reprocesar_respuesta_lucia",
    # VOICE ENGINE
    "obtener_motor_voz",
    "reproducir_voz_lucia",
    "cancelar_voz_lucia",
]


def obtener_motor_voz() -> Any:
    """Retorna la instancia global del motor de voz de LucIA."""
    try:
        from LC.celebro.CMFG.SBSTM.voice_engine import get_voice_engine  # pyrefly: ignore[missing-import]
        return get_voice_engine()
    except Exception:
        return None


def reproducir_voz_lucia(texto: str, esperar: bool = False, emocion: Optional[float] = None) -> None:
    """Reproduce texto con la voz juvenil oficial de LucIA (es-ES-ElviraNeural)."""
    try:
        from LC.celebro.CMFG.SBSTM.voice_engine import speak  # pyrefly: ignore[missing-import]
        speak(texto, esperar=esperar, emocion=emocion)
    except Exception:
        pass


def cancelar_voz_lucia() -> None:
    """Interrumpe inmediatamente cualquier emision sonora de LucIA."""
    try:
        from LC.celebro.CMFG.SBSTM.voice_engine import cancel_speech  # pyrefly: ignore[missing-import]
        cancel_speech()
    except Exception:
        pass


class GestorJSONLucIA:
    """Puerta del orquestador al registro central STM_JSON."""
    def __init__(self) -> None:
        try:
            from Sistema_Principal.PRY.actualizacion_b2abf525.src.STM_JSON.registro_json import get_registro
            self.registro = get_registro()
        except Exception:
            self.registro = None
    def config_ia(self):
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
            from Sistema_Principal.PRY.actualizacion_b2abf525.src.STM_HRTS.registro_hrts import get_registro_hrts
            self.registro = get_registro_hrts()
        except Exception:
            self.registro = None
    def linea_estado(self) -> str:
        try:
            return self.registro.linea_estado() if self.registro else "STM_HRTS no disponible"
        except Exception:
            return "STM_HRTS error"
