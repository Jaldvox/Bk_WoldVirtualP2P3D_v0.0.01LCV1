"""
Celebro - Nucleo Cognitivo y Consciencia Neuronal Distribuida (WoldVirtualP2P3D 2026)
======================================================================================
Paquete central de inteligencia, computacion sinaptica y consenso inmutable:
  - BKSVCB: Servidor Blockchain con consenso PoNL (Proof of Neural Learning) y Merkle Root.
  - CMFG: Celebro Model & Feed Gateway (PSNRCV, pesos_vivos, IPFSManager, IPFSonl, SBSTM).
  - Red Neuronal: 50 neuronas activas distribuidas en 5 subpaquetes especializados:
      * ENRN:  10 neuronas recurrentes no lineales de entrada y activacion primaria.
      * RF_EN: 10 neuronas de retroalimentacion y modulacion por refuerzo adaptativo.
      * RF_SL: 10 neuronas de aprendizaje supervisado con compuertas residuales.
      * RNP:   10 neuronas de plasticidad sinaptica y memoria asociativa distribuida.
      * SLRN:  10 optimizadores y sinapsis de convergencia temporal continua.
  - PSNRL: Directorio de persistencia y checkpoint local de pesos sinapticos activos.
  - Persistencia descentralizada sobre protocolo IPFS Kubo con retencion de CIDs.
"""
from __future__ import annotations

import atexit
import hashlib
import json
import logging
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, Final, Generator, List, Optional, Tuple, Union

__version__: Final[str] = "2026.3.1"
__author__: Final[str] = "Equipo Celebro WoldVirtualP2P3D"
__status__: Final[str] = "Production / Optimized"
__package_name__: Final[str] = "LC.celebro"

logger: logging.Logger = logging.getLogger("WoldVirtualP2P3D.Celebro")

# ─── ENRUTAMIENTO Y RUTAS CRITICAS DEL ARBOL COGNITIVO ───────────────────────
PACKAGE_ROOT: Final[Path] = Path(__file__).parent.resolve()
LC_ROOT: Final[Path] = PACKAGE_ROOT.parent.resolve()
WORKSPACE_ROOT: Final[Path] = LC_ROOT.parent.resolve()

CMFG_DIR: Final[Path] = PACKAGE_ROOT / "CMFG"
SBSTM_DIR: Final[Path] = CMFG_DIR / "SBSTM"
PSNRL_DIR: Final[Path] = PACKAGE_ROOT / "PSNRL"
RED_NEURONAL_DIR: Final[Path] = PACKAGE_ROOT / "red_neuronal"
LEDGER_PATH: Final[Path] = PACKAGE_ROOT / "blockchain_ledger.json"

# Asegurar que el Workspace Root este al frente del sys.path
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

# ─── REGISTRO Y CONFIGURACION DE LAS 5 FAMILIAS DE NEURONAS ──────────────────
NEURONAL_FAMILIES: Final[Dict[str, Dict[str, Any]]] = {
    "ENRN": {
        "modulo": "LC.celebro.red_neuronal.ENRN",
        "descripcion": "10 Neuronas Recurrentes No Lineales de Entrada",
        "recuento": 10,
        "prefijo": "ENRN",
        "tipo": "entrada_recurrente",
    },
    "RF_EN": {
        "modulo": "LC.celebro.red_neuronal.RF_EN",
        "descripcion": "10 Neuronas de Retroalimentacion y Refuerzo Dinamico",
        "recuento": 10,
        "prefijo": "RFEN",
        "tipo": "refuerzo_adaptativo",
    },
    "RF_SL": {
        "modulo": "LC.celebro.red_neuronal.RF_SL",
        "descripcion": "10 Neuronas de Aprendizaje Supervisado y Residual",
        "recuento": 10,
        "prefijo": "RFSL",
        "tipo": "residual_supervisado",
    },
    "RNP": {
        "modulo": "LC.celebro.red_neuronal.RNP",
        "descripcion": "10 Neuronas de Plasticidad Sinaptica y Memoria Hebbiana",
        "recuento": 10,
        "prefijo": "RNP",
        "tipo": "plasticidad_hebbiana",
    },
    "SLRN": {
        "modulo": "LC.celebro.red_neuronal.SLRN",
        "descripcion": "10 Optimizadores y Sinapsis de Convergencia Supervisada",
        "recuento": 10,
        "prefijo": "SLRN",
        "tipo": "optimizador_continuo",
    },
}

TOTAL_NEURONAS_SISTEMA: Final[int] = 50

# ─── ESTADO CENTRAL DE LA CONSCIENCIA DISTRIBUIDA ───────────────────────────
_CELEBRO_STATE: Dict[str, Any] = {
    "init_timestamp": time.time(),
    "neuronas_activas": 0,
    "blockchain_online": False,
    "servidor_bksvcb_ref": None,
    "conversor_psnrcv_ref": None,
    "ipfs_manager_ref": None,
    "ultimo_bloque_hash": None,
    "sesiones_ejecutadas": 0,
    "ledger_sincronizado": False,
    "operaciones_sinapticas_totales": 0,
}

_HOOKS_CICLO_VIDA: List[Callable[[], Any]] = []
_LOCK: threading.Lock = threading.Lock()


# ─── DISCOVERY Y VALIDACION ESTRUCTURAL DE SUBSISTEMAS ───────────────────────
def inicializar_entorno_celebro() -> Dict[str, bool]:
    """
    Garantiza la presencia de directorios vitales y estructura de persistencia:
      - PSNRL: almacenamiento local de tensores y estados neuronales.
      - red_neuronal: ramas ENRN, RF_EN, RF_SL, RNP, SLRN.
      - CMFG / SBSTM: pasarelas y orquestacion de sesiones.
    """
    estados: Dict[str, bool] = {}
    rutas_vitales = [
        (PSNRL_DIR, "PSNRL"),
        (RED_NEURONAL_DIR, "red_neuronal"),
        (CMFG_DIR, "CMFG"),
        (SBSTM_DIR, "CMFG_SBSTM"),
    ]
    for ruta, nombre in rutas_vitales:
        try:
            ruta.mkdir(parents=True, exist_ok=True)
            estados[nombre] = True
        except Exception as exc:
            logger.error("Error creando infraestructura en %s (%s): %s", nombre, ruta, exc)
            estados[nombre] = False
    return estados


# ─── FABRICA Y GESTION DEL SERVIDOR BLOCKCHAIN BKSVCB ────────────────────────
def obtener_servidor_blockchain() -> Any:
    """
    Carga de forma segura e idempotente la instancia del servidor blockchain
    BKSVCB sin redundancias en ejecucion.
    """
    with _LOCK:
        if _CELEBRO_STATE["servidor_bksvcb_ref"] is not None:
            return _CELEBRO_STATE["servidor_bksvcb_ref"]

    try:
        from LC.celebro.BKSVCB import get_blockchain_server
        instancia = get_blockchain_server()
        with _LOCK:
            _CELEBRO_STATE["servidor_bksvcb_ref"] = instancia
            _CELEBRO_STATE["blockchain_online"] = True
        return instancia
    except Exception as exc:
        logger.error("Error inicializando Servidor Blockchain BKSVCB: %s", exc)
        raise RuntimeError(f"Fallo al invocar BKSVCB: {exc}") from exc


def iniciar_blockchain_en_hilo(puerto: int = 8545) -> Any:
    """
    Arranca el servidor HTTP JSON-RPC de BKSVCB en un hilo daemon background
    desplegando los hashes unicos de cada bloque en consola.
    """
    from LC.celebro.BKSVCB import iniciar_servidor_blockchain
    return iniciar_servidor_blockchain(puerto=puerto)



# ─── ACCESO AL CONVERSOR Y TRANSDUCTOR DE 50 NEURONAS (PSNRCV) ──────────────
def obtener_conversor_neuronal() -> Any:
    """
    Retorna el singleton del ConversorRespuestaPesos encargado de mapear
    los bloques y tokens transaccionales a tensores de las 50 neuronas.
    """
    with _LOCK:
        if _CELEBRO_STATE["conversor_psnrcv_ref"] is not None:
            return _CELEBRO_STATE["conversor_psnrcv_ref"]

    try:
        from LC.celebro.CMFG.PSNRCV import get_conversor_pesos
        conversor = get_conversor_pesos()
        with _LOCK:
            _CELEBRO_STATE["conversor_psnrcv_ref"] = conversor
            _CELEBRO_STATE["neuronas_activas"] = TOTAL_NEURONAS_SISTEMA
        return conversor
    except Exception as exc:
        logger.error("Error cargando el conversor transductor PSNRCV: %s", exc)
        raise RuntimeError(f"Fallo en PSNRCV: {exc}") from exc


# ─── ACCESO AL GESTOR DESCENTRALIZADO IPFS ──────────────────────────────────
def obtener_gestor_ipfs() -> Any:
    """
    Obtiene la instancia unica del gestor IPFSManager con soporte para Kubo local,
    generacion de CIDv1 criptograficos y purga atomica de residuos en PSNRL.
    """
    with _LOCK:
        if _CELEBRO_STATE["ipfs_manager_ref"] is not None:
            return _CELEBRO_STATE["ipfs_manager_ref"]

    try:
        from LC.celebro.CMFG.ipfs_manager import get_ipfs_manager
        gestor = get_ipfs_manager()
        with _LOCK:
            _CELEBRO_STATE["ipfs_manager_ref"] = gestor
        return gestor
    except Exception as exc:
        logger.error("Error al obtener IPFSManager: %s", exc)
        raise RuntimeError(f"Fallo en IPFSManager: {exc}") from exc


# ─── INICIALIZACION Y CONTEO DE LAS 50 NEURONAS ──────────────────────────────
def verificar_red_50_neuronas() -> Dict[str, Any]:
    """
    Inspecciona y valida la presencia operativa de los 5 modulos neuronales
    verificando la conectividad de las 50 neuronas activas del sistema.
    """
    reporte: Dict[str, Any] = {
        "familias_evaluadas": {},
        "total_neuronas_esperadas": TOTAL_NEURONAS_SISTEMA,
        "total_neuronas_activas": 0,
        "integridad_global": False,
    }
    neuronas_totales = 0

    for nombre_fam, info in NEURONAL_FAMILIES.items():
        try:
            mod = __import__(info["modulo"], fromlist=[nombre_fam])
            fam_ok = True
            recuento = info["recuento"]
            neuronas_totales += recuento
        except Exception as exc:
            fam_ok = False
            recuento = 0
            logger.warning("Falla al cargar familia neuronal %s: %s", nombre_fam, exc)

        reporte["familias_evaluadas"][nombre_fam] = {
            "cargado": fam_ok,
            "neuronas": recuento,
            "tipo": info.get("tipo", "generico"),
            "descripcion": info["descripcion"],
        }

    reporte["total_neuronas_activas"] = neuronas_totales
    reporte["integridad_global"] = bool(neuronas_totales == TOTAL_NEURONAS_SISTEMA)
    with _LOCK:
        _CELEBRO_STATE["neuronas_activas"] = neuronas_totales

    return reporte


def enumerar_neuronas_activas() -> List[str]:
    """
    Devuelve la nomenclatura identificadora de las 50 neuronas del sistema
    (ENRN1..10, RFEN1..10, RFSL1..10, RNP1..10, SLRN1..10).
    """
    nombres: List[str] = []
    for info in NEURONAL_FAMILIES.values():
        prefijo = info["prefijo"]
        for idx in range(1, info["recuento"] + 1):
            nombres.append(f"{prefijo}{idx}")
    return nombres


# ─── TRANSVERSAL: SESION NEURONAL P2P (SBSTM / SNSBSTNPRB) ──────────────────
def crear_sesion_p2p(modelo: Optional[str] = None, puerto: int = 8545) -> Any:
    """
    Crea una sesion orquestada completa a traves del subsistema SBSTM
    con hashing de bloques en terminal, streaming y calibracion neuronal.
    """
    from LC.celebro.CMFG.SBSTM import crear_sesion_neuronal
    with _LOCK:
        _CELEBRO_STATE["sesiones_ejecutadas"] += 1
    return crear_sesion_neuronal(modelo=modelo, puerto_blockchain=puerto)


def lanzar_consola_neuronal() -> int:
    """
    Inicia la interfaz de consola interactiva directa de Celebro mediante SBSTM.
    """
    from LC.celebro.CMFG.SBSTM import ejecutar_orquestador_interactivo
    return ejecutar_orquestador_interactivo()


# ─── AUDITORIA DE CRIPTOGRAFIA Y ESTADO DEL LEDGER ──────────────────────────
def auditar_ledger_blockchain() -> Dict[str, Any]:
    """
    Realiza una auditoria de consistencia criptografica sobre blockchain_ledger.json:
      - Existencia y peso en disco.
      - Total de bloques almacenados.
      - Hash SHA-256 de la raiz del fichero para verificacion de integridad.
    """
    if not LEDGER_PATH.exists():
        return {
            "existe": False,
            "ruta": str(LEDGER_PATH),
            "bloques": 0,
            "tamano_bytes": 0,
            "sha256": None,
        }

    try:
        contenido = LEDGER_PATH.read_bytes()
        sha_ledger = hashlib.sha256(contenido).hexdigest()
        tamano = len(contenido)
        datos = json.loads(contenido.decode("utf-8"))
        bloques = datos.get("cadena", []) if isinstance(datos, dict) else []
        return {
            "existe": True,
            "ruta": str(LEDGER_PATH),
            "bloques": len(bloques),
            "tamano_bytes": tamano,
            "sha256": sha_ledger,
            "ultimo_hash": bloques[-1].get("hash_bloque") if bloques else None,
        }
    except Exception as exc:
        logger.error("Error auditando archivo ledger: %s", exc)
        return {"existe": True, "error": str(exc)}


def validar_integridad_bloque(bloque: Dict[str, Any]) -> bool:
    """
    Verifica de forma estricta el hash doble SHA-256 de un bloque individual
    garantizando su inmutabilidad frente a manipulaciones.
    """
    try:
        idx = bloque.get("indice", 0)
        h_prev = bloque.get("hash_previo", "")
        txs = bloque.get("transacciones", [])
        mr = bloque.get("merkle_root", "")
        diff = bloque.get("dificultad", 2)
        nonce = bloque.get("nonce", 0)
        h_decl = bloque.get("hash_bloque", "")

        tx_str = json.dumps(txs, sort_keys=True)
        raw = f"{idx}:{h_prev}:{mr}:{tx_str}:{diff}:{nonce}".encode("utf-8")
        h1 = hashlib.sha256(raw).digest()
        h2 = hashlib.sha256(h1).hexdigest()
        return h2 == h_decl
    except Exception:
        return False


# ─── CARGADOR Y VERIFICADOR DE TENSORES EN PSNRL ────────────────────────────
def listar_archivos_psnrl() -> List[Dict[str, Any]]:
    """
    Devuelve un inventario detallado de todos los archivos de pesos y metadatos
    almacenados en el directorio PSNRL (.npz, .json, .pt).
    """
    if not PSNRL_DIR.exists():
        return []

    archivos = []
    for elem in PSNRL_DIR.iterdir():
        if elem.is_file():
            archivos.append({
                "nombre": elem.name,
                "ruta": str(elem),
                "tamano_bytes": elem.stat().st_size,
                "modificado": elem.stat().st_mtime,
                "extension": elem.suffix,
            })
    archivos.sort(key=lambda x: x["modificado"], reverse=True)
    return archivos


# ─── REGISTRO DE GANCHOS DE CICLO DE VIDA (LIFECYCLE HOOKS) ─────────────────
def registrar_hook_terminacion(hook: Callable[[], Any]) -> None:
    """Registra una accion de persistencia o cierre en la cola de salida de Celebro."""
    with _LOCK:
        if hook not in _HOOKS_CICLO_VIDA:
            _HOOKS_CICLO_VIDA.append(hook)
            logger.debug("Hook de terminacion registrado en Celebro.")


# ─── RESUMEN DEL SISTEMA COMPLETO Y TELEMETRIA GLOBAL ───────────────────────
def obtener_diagnostico_global() -> Dict[str, Any]:
    """
    Genera un informe exhaustivo del estado integral de la arquitectura Celebro:
      - Estado de las 50 neuronas activas.
      - Integridad del libro mayor blockchain.
      - Disponibilidad del conversor transductor y pasarela IPFS.
    """
    with _LOCK:
        copia_estado = dict(_CELEBRO_STATE)

    return {
        "version": __version__,
        "paquete": __package_name__,
        "directorio_base": str(PACKAGE_ROOT),
        "estado_sistema": copia_estado,
        "diagnostico_neuronal": verificar_red_50_neuronas(),
        "nombres_neuronas": enumerar_neuronas_activas(),
        "auditoria_blockchain": auditar_ledger_blockchain(),
        "entorno_directorios": inicializar_entorno_celebro(),
        "inventario_psnrl": listar_archivos_psnrl(),
        "uptime_segundos": time.time() - float(copia_estado["init_timestamp"]),
    }


# ─── CIERRE ORDENADO Y RECURSOS ATEXIT ───────────────────────────────────────
def _rutina_cierre_celebro() -> None:
    """Ejecuta todos los ganchos de limpieza al finalizar la ejecucion de Python."""
    with _LOCK:
        hooks = list(_HOOKS_CICLO_VIDA)
        bksvcb = _CELEBRO_STATE.get("servidor_bksvcb_ref")

    for h in hooks:
        try:
            h()
        except Exception as err:
            logger.warning("Error ejecutando hook de cierre Celebro: %s", err)

    if bksvcb is not None and hasattr(bksvcb, "cerrar_servidor"):
        try:
            bksvcb.cerrar_servidor()
        except Exception:
            pass


atexit.register(_rutina_cierre_celebro)


# ─── EXPORTACIONES OFICIALES Y API PUBLICA DE CELEBRO ───────────────────────
__all__: Final[List[str]] = [
    "TOTAL_NEURONAS_SISTEMA",
    "NEURONAL_FAMILIES",
    "PACKAGE_ROOT",
    "PSNRL_DIR",
    "LEDGER_PATH",
    "RED_NEURONAL_DIR",
    "inicializar_entorno_celebro",
    "obtener_servidor_blockchain",
    "iniciar_blockchain_en_hilo",
    "obtener_conversor_neuronal",
    "obtener_gestor_ipfs",
    "verificar_red_50_neuronas",
    "enumerar_neuronas_activas",
    "crear_sesion_p2p",
    "lanzar_consola_neuronal",
    "auditar_ledger_blockchain",
    "validar_integridad_bloque",
    "listar_archivos_psnrl",
    "registrar_hook_terminacion",
    "obtener_diagnostico_global",
]
