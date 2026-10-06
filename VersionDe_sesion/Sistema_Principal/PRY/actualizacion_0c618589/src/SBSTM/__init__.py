"""
SBSTM - SubSistema de Sesion y Transito Neuronal (Arquitectura WoldVirtualP2P3D 2026)
====================================================================================
Subpaquete especializado en la orquestacion de sesiones interactivas, sincronizacion
de bloques transaccionales y pasarela conversacional neuronal P2P:
  - Gestion de ciclo de vida de sesion neuronal (SesionNeuronalP2P / SNSBSTNPRB).
  - Enlace dinamico con la cadena criptografica BKSVCB y transductor PSNRCV.
  - Streaming con modelos de lenguaje locales (Ollama: cogito:3b, qwen2.5:7b, etc.).
  - Transduccion de tokens en transacciones sinapticas y minado por turnos.
  - Control de telemetria en tiempo real: GSNR, deriva Muon y entropia de Shannon.
  - Checkpoint y persistencia automatica en PSNRL e IPFS mediante ganchos atexit/signal.
"""
from __future__ import annotations

import atexit
import json
import logging
import os
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, Final, Generator, List, Optional, Tuple, Union

__version__: Final[str] = "2026.3.1"
__author__: Final[str] = "Equipo Celebro WoldVirtualP2P3D"
__status__: Final[str] = "Production / Optimized"
__package_name__: Final[str] = "SBSTM"

logger: logging.Logger = logging.getLogger("WoldVirtualP2P3D.CMFG.SBSTM")

# ─── JERARQUIA DE DIRECTORIOS Y RUTA RAIS ────────────────────────────────────
PACKAGE_ROOT: Final[Path] = Path(__file__).parent.resolve()
CMFG_DIR: Final[Path] = PACKAGE_ROOT.parent.resolve()
CELEBRO_DIR: Final[Path] = CMFG_DIR.parent.resolve()
try:
    from STM_CH.rutas import PSNRL_DIR as PSNRL_DIR, LEDGER_PATH as LEDGER_PATH
except ImportError:
    PSNRL_DIR: Final[Path] = CELEBRO_DIR / "PSNRL"
    LEDGER_PATH: Final[Path] = CELEBRO_DIR / "blockchain_ledger.json"

# ─── ESTADO GLOBAL DEL SUBSISTEMA SBSTM ──────────────────────────────────────
_SBSTM_STATE: Dict[str, Any] = {
    "session_active": False,
    "active_instance": None,
    "total_sessions_launched": 0,
    "total_dialog_turns": 0,
    "last_checkpoint_timestamp": 0.0,
    "last_mined_block_index": 0,
    "ipfs_last_cid": None,
}

_HOOKS_SESION: List[Callable[[Dict[str, Any]], Any]] = []
_LOCK: threading.Lock = threading.Lock()


# ─── PALETA VISUAL ANSI Y FORMATO DE LOG ─────────────────────────────────────
class TerminalColors:
    """Paleta ANSI optimizada para consolas modernas de telemetria neuronal."""
    R: Final[str] = "\033[0m"
    B: Final[str] = "\033[1m"
    DIM: Final[str] = "\033[2m"
    CY: Final[str] = "\033[96m"
    BL: Final[str] = "\033[94m"
    GR: Final[str] = "\033[92m"
    YL: Final[str] = "\033[93m"
    MG: Final[str] = "\033[95m"
    RD: Final[str] = "\033[91m"
    WH: Final[str] = "\033[97m"

    @classmethod
    def colorize(cls, color: str, text: str) -> str:
        """Aplica estilo ANSI a una cadena de texto asegurando el reseteo final."""
        return f"{color}{text}{cls.R}"


# ─── FUNCIONES DE DIAGNOSTICO Y ACCESO A ENTORNOS ───────────────────────────
def obtener_directorio_sbstm() -> Path:
    """Devuelve la ruta absoluta al subpaquete SBSTM."""
    return PACKAGE_ROOT


def obtener_estado_subsistema() -> Dict[str, Any]:
    """Retorna una instantanea inmutable del estado del orquestador SBSTM."""
    with _LOCK:
        return {
            "session_active": bool(_SBSTM_STATE["session_active"]),
            "total_sessions_launched": int(_SBSTM_STATE["total_sessions_launched"]),
            "total_dialog_turns": int(_SBSTM_STATE["total_dialog_turns"]),
            "last_checkpoint_timestamp": float(_SBSTM_STATE["last_checkpoint_timestamp"]),
            "last_mined_block_index": int(_SBSTM_STATE["last_mined_block_index"]),
            "ipfs_last_cid": _SBSTM_STATE["ipfs_last_cid"],
            "psnrl_dir_exists": PSNRL_DIR.exists(),
            "ledger_exists": LEDGER_PATH.exists(),
        }


def registrar_gancho_post_turno(callback: Callable[[Dict[str, Any]], Any]) -> None:
    """Registra una funcion callback invocada tras procesar cada turno de sesion."""
    with _LOCK:
        if callback not in _HOOKS_SESION:
            _HOOKS_SESION.append(callback)
            logger.debug("Gancho post-turno registrado satisfactoriamente.")


# ─── VERIFICADOR DE INTEGRIDAD DE SUBSISTEMA ─────────────────────────────────
def verificar_integridad_sbstm() -> Dict[str, Any]:
    """
    Verifica que el entorno SBSTM cuente con todos los componentes requeridos:
      - Script orquestador SNSBSTNPRB.PY.
      - Acceso al conversor de pesos PSNRCV.
      - Acceso al servidor de bloques BKSVCB.
      - Acceso a la capa de persistencia IPFSManager.
    """
    script_prb = PACKAGE_ROOT / "SNSBSTNPRB.py"
    if not script_prb.exists():
        script_prb = PACKAGE_ROOT / "SNSBSTNPRB.PY"
    psnrcv_py = CMFG_DIR / "PSNRCV.py"
    bksvcb_py = CELEBRO_DIR / "BKSVCB.py"
    ipfs_py = CMFG_DIR / "ipfs_manager.py"

    diagnostico = {
        "orquestador_snsbstnprb": script_prb.exists(),
        "script_prb_ruta": str(script_prb),
        "conversor_psnrcv": psnrcv_py.exists(),
        "blockchain_bksvcb": bksvcb_py.exists(),
        "ipfs_manager": ipfs_py.exists(),
        "psnrl_directorio": PSNRL_DIR.exists(),
        "ledger_inmutable": LEDGER_PATH.exists(),
        "timestamp_evaluacion": time.time(),
    }
    diagnostico["operativo"] = bool(
        diagnostico["orquestador_snsbstnprb"]
        and diagnostico["conversor_psnrcv"]
        and diagnostico["blockchain_bksvcb"]
    )
    return diagnostico


# ─── IMPORTACION CONTROLADA DE LA CLASE DE SESION PRINCIPAL ──────────────────
def obtener_clase_sesion() -> Any:
    """
    Carga de forma diferida la clase SesionNeuronalP2P desde SNSBSTNPRB.
    Garantiza aislamiento ante inicializaciones circulares del paquete Celebro.
    """
    from SBSTM.SNSBSTNPRB import SesionNeuronalP2P
    return SesionNeuronalP2P



def crear_sesion_neuronal(
    modelo: Optional[str] = None,
    puerto_blockchain: int = 8545,
    auto_minado: bool = True,
) -> Any:
    """
    Crea e inicializa una instancia controlada de SesionNeuronalP2P.
    Actualiza el registro central de sesiones activas en SBSTM.
    """
    cls_sesion = obtener_clase_sesion()
    instancia = cls_sesion(modelo_inicial=modelo, puerto_bksvcb=puerto_blockchain)
    with _LOCK:
        _SBSTM_STATE["session_active"] = True
        _SBSTM_STATE["active_instance"] = instancia
        _SBSTM_STATE["total_sessions_launched"] += 1
    return instancia


def ejecutar_orquestador_interactivo() -> int:
    """
    Punto de entrada de alta conveniencia para iniciar el bucle interactivo completo
    de la sesion neuronal con visualizacion de bloques, hashes e inferencia LLM.
    """
    try:
        sesion = crear_sesion_neuronal()
        sesion.iniciar_consola_interactiva()
        return 0
    except KeyboardInterrupt:
        logger.info("Sesion SBSTM finalizada por interrupcion del usuario.")
        return 130
    except Exception as exc:
        logger.critical("Error critico en la ejecucion del orquestador SBSTM: %s", exc, exc_info=True)
        return 1
    finally:
        with _LOCK:
            _SBSTM_STATE["session_active"] = False
            _SBSTM_STATE["active_instance"] = None


# ─── GESTOR DE CONTEXTO PARA PIPELINES DE EVALUACION ────────────────────────
class SBSTMSessionContext:
    """
    Context Manager profesional para encapsular ejecuciones por lotes o pruebas
    de red dentro del ciclo de vida controlado de una sesion neuronal P2P.
    """

    def __init__(self, modelo: Optional[str] = None, puerto_blockchain: int = 8545) -> None:
        self.modelo = modelo
        self.puerto_blockchain = puerto_blockchain
        self.sesion: Optional[Any] = None

    def __enter__(self) -> Any:
        self.sesion = crear_sesion_neuronal(
            modelo=self.modelo,
            puerto_blockchain=self.puerto_blockchain,
        )
        return self.sesion

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        if self.sesion is not None and hasattr(self.sesion, "cerrar"):
            try:
                self.sesion.cerrar()
            except Exception as e:
                logger.warning("Error durante cierre automatico de sesion SBSTM: %s", e)
        with _LOCK:
            _SBSTM_STATE["session_active"] = False
            _SBSTM_STATE["active_instance"] = None
        return False


# ─── SERVICIO DE DISCOVERY DE MODELOS OLLAMA LOCALES ────────────────────────
def consultar_modelos_disponibles(host_url: str = "http://127.0.0.1:11434") -> List[str]:
    """
    Consulta la pasarela local de inferencia Ollama y devuelve la lista de tags
    de modelos instalados y listos para streaming en sesion.
    """
    import urllib.error
    import urllib.request

    url = f"{host_url.rstrip('/')}/api/tags"
    req = urllib.request.Request(url, headers={"User-Agent": "Celebro-SBSTM/2026"})
    try:
        with urllib.request.urlopen(req, timeout=2.5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return [m.get("name", "") for m in data.get("models", []) if m.get("name")]
    except Exception as exc:
        logger.debug("No se logro comunicar con Ollama en %s: %s", url, exc)
        return []


# ─── METRICAS Y EXPORTACION DE TELEMETRIA ───────────────────────────────────
def recopilar_estadisticas_sesion() -> Dict[str, Any]:
    """Recopila un resumen consolidado de telemetria para inspeccion y monitoreo."""
    with _LOCK:
        active = _SBSTM_STATE.get("active_instance")
        inst_telemetria = {}
        if active is not None and hasattr(active, "obtener_telemetria"):
            try:
                inst_telemetria = active.obtener_telemetria()
            except Exception:
                pass

        return {
            "estado_general": dict(_SBSTM_STATE),
            "telemetria_instancia": inst_telemetria,
            "entorno_integridad": verificar_integridad_sbstm(),
            "timestamp": time.time(),
        }


# ─── DISPARADOR DE NOTIFICACIONES POST-TURNO ────────────────────────────────
def notificar_turno_completado(datos_turno: Dict[str, Any]) -> None:
    """
    Dispara la ejecucion segura de todos los ganchos registrados tras
    la finalizacion de un turno interactivo o transaccional.
    """
    with _LOCK:
        _SBSTM_STATE["total_dialog_turns"] += 1
        callbacks = list(_HOOKS_SESION)

    for cb in callbacks:
        try:
            cb(datos_turno)
        except Exception as exc:
            logger.warning("Excepcion atrapada en gancho post-turno SBSTM: %s", exc)


# ─── UTILIDADES DE CONEXION RAPIDA CON LA BLOCKCHAIN ─────────────────────────
def verificar_cadena_bloques_activa(puerto: int = 8545) -> bool:
    """Verifica si el nodo RPC/HTTP de BKSVCB esta respondiendo en el puerto local."""
    import urllib.request

    url = f"http://127.0.0.1:{puerto}/status"
    req = urllib.request.Request(url, headers={"User-Agent": "Celebro-SBSTM-Check/2026"})
    try:
        with urllib.request.urlopen(req, timeout=1.0) as resp:
            return resp.status == 200
    except Exception:
        return False


def obtener_ultimo_hash_bloque(puerto: int = 8545) -> Optional[str]:
    """Obtiene el hash del bloque mas reciente minado en la cadena local."""
    import urllib.request

    url = f"http://127.0.0.1:{puerto}/chain"
    req = urllib.request.Request(url, headers={"User-Agent": "Celebro-SBSTM-Check/2026"})
    try:
        with urllib.request.urlopen(req, timeout=1.5) as resp:
            datos = json.loads(resp.read().decode("utf-8"))
            bloques = datos.get("cadena", [])
            if bloques:
                return bloques[-1].get("hash_bloque")
    except Exception:
        pass
    return None


# ─── SINCRONIZACION MANUAL DE CHECKPOINTS ────────────────────────────────────
def forzar_checkpoint_psnrl() -> Dict[str, Any]:
    """
    Solicita a la sesion activa que guarde un checkpoint inmediato de todos los
    pesos de las 50 neuronas en el directorio PSNRL y los marque para respaldo.
    """
    with _LOCK:
        instancia = _SBSTM_STATE.get("active_instance")

    if instancia is None:
        return {"exito": False, "mensaje": "No hay sesion activa en curso"}

    try:
        if hasattr(instancia, "guardar_checkpoint_psnrl"):
            res = instancia.guardar_checkpoint_psnrl()
            with _LOCK:
                _SBSTM_STATE["last_checkpoint_timestamp"] = time.time()
            return {"exito": True, "detalle": res}
        return {"exito": False, "mensaje": "La instancia activa no implementa checkpoint"}
    except Exception as exc:
        logger.error("Error forzando checkpoint PSNRL desde SBSTM: %s", exc)
        return {"exito": False, "error": str(exc)}


# ─── INTERFAZ DE LIMPIEZA Y PURGA DE RESIDUOS ────────────────────────────────
def purgar_archivos_temporales(max_edad_segundos: float = 86400.0) -> int:
    """
    Limpia archivos de telemetria o volcados temporales residuales en el
    entorno de ejecucion que superen la edad configurada.
    """
    eliminados = 0
    ahora = time.time()
    try:
        for patron in ("*.tmp", "*.lock"):
            for archivo in PACKAGE_ROOT.glob(patron):
                try:
                    if (ahora - archivo.stat().st_mtime) > max_edad_segundos:
                        archivo.unlink(missing_ok=True)
                        eliminados += 1
                except Exception:
                    pass
    except Exception as e:
        logger.debug("Error durante purga preventiva en SBSTM: %s", e)
    return eliminados


# ─── ACCESO DIRECTO AL SUBSISTEMA IAFREE (MODELOS GRATIS) ───────────────────
# Definidos en accesores.py y re-exportados aqui tras el bloque de imports.


# ─── REGISTRO DE SEÑALES DEL SISTEMA OPERATIVO ───────────────────────────────
def _manejador_interrupcion_global(sig: int, frame: Any) -> None:
    """Asegura el cierre limpio y ordenado de la sesion ante señales SIGINT / SIGTERM."""
    logger.info("Señal de terminacion capturada (%s). Ejecutando limpieza SBSTM...", sig)
    with _LOCK:
        instancia = _SBSTM_STATE.get("active_instance")
    if instancia is not None and hasattr(instancia, "cerrar"):
        try:
            instancia.cerrar()
        except Exception:
            pass


try:
    if threading.current_thread() is threading.main_thread():
        signal.signal(signal.SIGINT, _manejador_interrupcion_global)
        signal.signal(signal.SIGTERM, _manejador_interrupcion_global)
except (ValueError, AttributeError):
    pass


# ─── PROTOCOLO DE DESPEDIDA / EXPORTACION ATEXIT ─────────────────────────────
def _finalizacion_modulo_sbstm() -> None:
    """Rutina invoked al descargar el modulo o finalizar el interprete de Python."""
    with _LOCK:
        active = _SBSTM_STATE.get("active_instance")
    if active is not None and hasattr(active, "cerrar"):
        try:
            active.cerrar()
        except Exception:
            pass


atexit.register(_finalizacion_modulo_sbstm)


try:
    from SBSTM.accesores import (
        obtener_procesador_rplc,
        reprocesar_respuesta_lucia,
        obtener_motor_voz,
        reproducir_voz_lucia,
        cancelar_voz_lucia,
        obtener_servicio_iafree,
        obtener_motor_stylos,
    )
except ImportError:
    try:
        from .accesores import (
            obtener_procesador_rplc,
            reprocesar_respuesta_lucia,
            obtener_motor_voz,
            reproducir_voz_lucia,
            cancelar_voz_lucia,
            obtener_servicio_iafree,
            obtener_motor_stylos,
        )
    except ImportError as _err:
        logger.warning("[SBSTM] Accesores no disponibles: %s", _err)


