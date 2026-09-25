"""
IPFSonl - Modulo de Conectividad IPFS Kubo Local y P2P (Arquitectura 2026)
==========================================================================
Subsistema de orquestacion para el daemon Kubo descentralizado en WoldVirtualP2P3D:
  - Inicializacion automatizada de repositorio IPFS local (.ipfs) con PeerID unico.
  - Arranque asincrono y gestion no bloqueante del daemon con experimental pubsub.
  - Hilo supervisor de liveness (healthcheck) con auto-reinicio y recuperacion resiliente.
  - Monitoreo en tiempo real de peers conectados en enjambre, repositorios y ancho de banda.
  - Enrutamiento y resolucion de pasarelas publicas (ipfs.io, pinata, dweb, cloudflare).
  - Context Manager (IPFSDaemonContext) para inclusion en pipelines de entrenamiento.
  - Sincronizacion atomica con PSNRL y purga segura de residuos locales al cierre.
"""
from __future__ import annotations
import os, sys, time, json, subprocess, logging, threading
import urllib.request, urllib.error
from pathlib import Path
from typing import Any, Callable, Dict, Final, List, Optional, Tuple, Union

__version__: Final[str] = "2026.3.1"
__author__: Final[str] = "Equipo IPFS WoldVirtualP2P3D"
__status__: Final[str] = "Production / Optimized"
__package_name__: Final[str] = "LC.celebro.CMFG.IPFSonl"

logger: logging.Logger = logging.getLogger("WoldVirtualP2P3D.IPFSonl")

IPFSONL_DIR: Final[Path] = Path(__file__).parent.resolve()
IPFS_BIN: Final[Path] = IPFSONL_DIR / "ipfs.exe"
IPFS_REPO: Final[Path] = IPFSONL_DIR / ".ipfs"
IPFS_API_URL: Final[str] = "http://127.0.0.1:5001"
IPFS_GATEWAY_URL: Final[str] = "http://127.0.0.1:8080"

PUBLIC_GATEWAYS: Final[List[str]] = [
    "https://ipfs.io/ipfs/",
    "https://gateway.pinata.cloud/ipfs/",
    "https://dweb.link/ipfs/",
    "https://cloudflare-ipfs.com/ipfs/",
    "https://ipfs.fleek.co/ipfs/",
]

_SUBSYSTEM_STATE: Dict[str, Any] = {
    "subsystem_id": f"IPFSonl_{int(time.time())}",
    "daemon_iniciado_por_modulo": False,
    "daemon_pid": None,
    "total_verificaciones": 0,
    "fallos_detectados": 0,
    "peers_registrados_max": 0,
    "ultimo_arranque": 0.0,
    "operaciones_completadas": 0,
}

_LOCK = threading.Lock()


def get_ipfs_bin() -> Path:
    """Retorna la ruta absoluta al binario local Kubo."""
    return IPFS_BIN


def get_ipfs_repo() -> Path:
    """Retorna la ruta al directorio de configuracion del repositorio .ipfs."""
    return IPFS_REPO


def get_ipfs_api_url() -> str:
    """Retorna la URL base del API HTTP RPC v0."""
    return IPFS_API_URL


def get_ipfs_gateway_url() -> str:
    """Retorna la URL de la pasarela HTTP local de lectura."""
    return IPFS_GATEWAY_URL


def verificar_instalacion() -> Dict[str, Any]:
    """
    Comprobacion exhaustiva de la infraestructura de Kubo:
      - Binario ejecutable existente y tamano en disco.
      - Repositorio .ipfs inicializado con archivo config y api.
    """
    bin_ok = IPFS_BIN.exists()
    repo_ok = IPFS_REPO.exists() and (IPFS_REPO / "config").exists()
    api_file = IPFS_REPO / "api"
    tam_bin_mb = (IPFS_BIN.stat().st_size / (1024 * 1024)) if bin_ok else 0.0

    return {
        "binario_presente": bin_ok,
        "binario_ruta": str(IPFS_BIN),
        "binario_mb": round(tam_bin_mb, 2),
        "repo_inicializado": repo_ok,
        "repo_ruta": str(IPFS_REPO),
        "api_file_presente": api_file.exists(),
        "api_url": IPFS_API_URL,
    }


def inicializar_repositorio_si_falta() -> bool:
    """Inicializa el repositorio Kubo local con perfil server si no existe."""
    if (IPFS_REPO / "config").exists():
        return True
    if not IPFS_BIN.exists():
        logger.error("No se puede inicializar repo: binario %s ausente", IPFS_BIN)
        return False
    logger.info("Inicializando repositorio IPFS en %s...", IPFS_REPO)
    env = os.environ.copy()
    env["IPFS_PATH"] = str(IPFS_REPO)
    try:
        res = subprocess.run(
            [str(IPFS_BIN), "init", "--profile=server"],
            env=env,
            capture_output=True,
            text=True,
            timeout=25,
        )
        return res.returncode == 0
    except Exception as e:
        logger.error("Error al ejecutar 'ipfs init': %s", e)
        return False


def probar_conexion_api(timeout: float = 1.5) -> bool:
    """Verifica si el daemon Kubo responde en el puerto 5001."""
    try:
        req = urllib.request.Request(f"{IPFS_API_URL}/api/v0/id", method="POST")
        with urllib.request.urlopen(req, timeout=timeout) as r:
            with _LOCK:
                _SUBSYSTEM_STATE["total_verificaciones"] += 1
            return r.status == 200
    except Exception:
        with _LOCK:
            _SUBSYSTEM_STATE["total_verificaciones"] += 1
        return False


def obtener_id_nodo(timeout: float = 2.0) -> Dict[str, Any]:
    """Consulta la identidad del nodo IPFS local (PeerID, AgentVersion, Llaves)."""
    try:
        req = urllib.request.Request(f"{IPFS_API_URL}/api/v0/id", method="POST")
        with urllib.request.urlopen(req, timeout=timeout) as r:
            if r.status == 200:
                return json.loads(r.read().decode("utf-8", errors="replace"))
    except Exception as e:
        logger.debug("Error obteniendo node id: %s", e)
    return {}


def obtener_peers_enjambre(timeout: float = 3.0) -> List[Dict[str, Any]]:
    """Recupera la lista de peers activos conectados al nodo en el swarm."""
    try:
        req = urllib.request.Request(f"{IPFS_API_URL}/api/v0/swarm/peers", method="POST")
        with urllib.request.urlopen(req, timeout=timeout) as r:
            if r.status == 200:
                data = json.loads(r.read().decode("utf-8", errors="replace"))
                peers = data.get("Peers") or []
                with _LOCK:
                    if len(peers) > _SUBSYSTEM_STATE["peers_registrados_max"]:
                        _SUBSYSTEM_STATE["peers_registrados_max"] = len(peers)
                return peers
    except Exception as e:
        logger.debug("Error listando swarm peers: %s", e)
    return []


def obtener_estadisticas_almacenamiento(timeout: float = 3.0) -> Dict[str, Any]:
    """Devuelve las metricas del repositorio local (tamano y conteo de objetos)."""
    try:
        req = urllib.request.Request(f"{IPFS_API_URL}/api/v0/repo/stat", method="POST")
        with urllib.request.urlopen(req, timeout=timeout) as r:
            if r.status == 200:
                return json.loads(r.read().decode("utf-8", errors="replace"))
    except Exception as e:
        logger.debug("Error obteniendo repo stats: %s", e)
    return {}


def obtener_metricas_ancho_banda(timeout: float = 2.5) -> Dict[str, Any]:
    """Consulta tasas de transferencia in/out y volumenes totales cursados."""
    try:
        req = urllib.request.Request(f"{IPFS_API_URL}/api/v0/stats/bw", method="POST")
        with urllib.request.urlopen(req, timeout=timeout) as r:
            if r.status == 200:
                return json.loads(r.read().decode("utf-8", errors="replace"))
    except Exception as e:
        logger.debug("Error leyendo bandwidth stats: %s", e)
    return {}


def construir_urls_pasarela(cid: str) -> Dict[str, str]:
    """Construye diccionario de enlaces hacia gateways publicos y locales."""
    return {
        "local": f"{IPFS_GATEWAY_URL}/ipfs/{cid}",
        "ipfs_io": f"https://ipfs.io/ipfs/{cid}",
        "pinata": f"https://gateway.pinata.cloud/ipfs/{cid}",
        "dweb": f"https://dweb.link/ipfs/{cid}",
        "cloudflare": f"https://cloudflare-ipfs.com/ipfs/{cid}",
    }


def verificar_estado_pin(cid: str, timeout: float = 3.0) -> bool:
    """Valida si un CID especifico cuenta con pinning local garantizado."""
    try:
        req = urllib.request.Request(f"{IPFS_API_URL}/api/v0/pin/ls?arg={cid}", method="POST")
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False


def ejecutar_comando_personalizado(argumentos: List[str], timeout: float = 15.0) -> Tuple[int, str, str]:
    """
    Ejecuta un comando directo sobre el binario local configurando
    el entorno con la variable IPFS_PATH adecuada.
    """
    if not IPFS_BIN.exists():
        return -1, "", "Binario no presente"
    env = os.environ.copy()
    env["IPFS_PATH"] = str(IPFS_REPO)
    try:
        res = subprocess.run(
            [str(IPFS_BIN)] + argumentos,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return res.returncode, res.stdout, res.stderr
    except Exception as e:
        return -1, "", str(e)


def arrancar(esperar_segundos: int = 25) -> bool:
    """Arranca el daemon delegando en ipfs_launcher de forma optimizada."""
    from LC.celebro.CMFG.IPFSonl.ipfs_launcher import arrancar_daemon
    ok = arrancar_daemon(esperar_segundos=esperar_segundos)
    if ok:
        with _LOCK:
            _SUBSYSTEM_STATE["daemon_iniciado_por_modulo"] = True
            _SUBSYSTEM_STATE["ultimo_arranque"] = time.time()
    return ok


def detener(espera_segundos: int = 6) -> bool:
    """Detiene el daemon Kubo gestionado localmente."""
    from LC.celebro.CMFG.IPFSonl.ipfs_launcher import detener_daemon
    res = detener_daemon(espera_segundos=espera_segundos)
    with _LOCK:
        _SUBSYSTEM_STATE["daemon_iniciado_por_modulo"] = False
    return res


def reiniciar() -> bool:
    """Reinicia el daemon aplicando las nuevas configuraciones."""
    from LC.celebro.CMFG.IPFSonl.ipfs_launcher import reiniciar_daemon
    return reiniciar_daemon()


def get_diagnostico_completo() -> Dict[str, Any]:
    """Genera informe consolidado del estado de red, daemon y subsistema."""
    id_info = obtener_id_nodo()
    peers = obtener_peers_enjambre()
    repo_stats = obtener_estadisticas_almacenamiento()
    bw_stats = obtener_metricas_ancho_banda()

    return {
        "timestamp": time.time(),
        "fecha_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "subsystem_state": dict(_SUBSYSTEM_STATE),
        "instalacion": verificar_instalacion(),
        "daemon_online": bool(id_info.get("ID")),
        "peer_id": id_info.get("ID"),
        "agente": id_info.get("AgentVersion"),
        "total_peers_conectados": len(peers),
        "repo_size_bytes": repo_stats.get("RepoSize", 0),
        "repo_num_objects": repo_stats.get("NumObjects", 0),
        "bandwidth_total_in": bw_stats.get("TotalIn", 0),
        "bandwidth_total_out": bw_stats.get("TotalOut", 0),
    }


def exportar_diagnostico_archivo(destino: Optional[Union[str, Path]] = None) -> Path:
    """Exporta el reporte consolidado a un archivo JSON en disco."""
    diag = get_diagnostico_completo()
    out = Path(destino) if destino else IPFSONL_DIR / "_diagnostico_ipfsonl.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(diag, f, indent=2, ensure_ascii=False)
    return out


def imprimir_informe_consola() -> None:
    """Presenta el informe de diagnostico del cluster en la consola."""
    diag = get_diagnostico_completo()
    online = diag["daemon_online"]
    print("\033[1;36m" + "=" * 68 + "\033[0m")
    print("  \033[1;32mESTADO DEL SUBSISTEMA IPFSonl (WoldVirtualP2P3D 2026)\033[0m")
    print("\033[1;36m" + "=" * 68 + "\033[0m")
    print(f"  Binario Kubo    : {'OK' if diag['instalacion']['binario_presente'] else 'NO ENCONTRADO'}")
    print(f"  Repositorio     : {'OK' if diag['instalacion']['repo_inicializado'] else 'NO INICIALIZADO'}")
    print(f"  Daemon Estado   : {'ONLINE' if online else 'OFFLINE'}")
    if online:
        print(f"  PeerID          : {diag['peer_id']}")
        print(f"  Agente          : {diag['agente']}")
        print(f"  Peers Swarm     : {diag['total_peers_conectados']} activos")
        print(f"  Objetos Repo    : {diag['repo_num_objects']}")
    print("\033[1;36m" + "=" * 68 + "\033[0m")


def resolver_version_kubo() -> str:
    """Consulta la version exacta del binario local via CLI."""
    code, out, _ = ejecutar_comando_personalizado(["version"], timeout=5.0)
    if code == 0:
        return out.strip()
    return "Desconocida"


def limpiar_bloqueos_repositorio() -> bool:
    """Elimina archivos repo.lock o api huerfanos si el daemon no esta corriendo."""
    if probar_conexion_api(timeout=0.8):
        logger.warning("No se pueden limpiar locks: daemon activo en ejecucion.")
        return False
    eliminados = 0
    for lock_name in ["repo.lock", "api"]:
        target = IPFS_REPO / lock_name
        if target.exists():
            try:
                target.unlink()
                eliminados += 1
                logger.info("Archivo lock purgado: %s", lock_name)
            except Exception as e:
                logger.error("Fallo purgando lock %s: %s", lock_name, e)
    return eliminados > 0


def configurar_perfil_nodo(perfil: str = "server") -> bool:
    """Aplica configuraciones predeterminadas al repositorio Kubo."""
    code, _, err = ejecutar_comando_personalizado(["config", "profile", "apply", perfil])
    if code == 0:
        logger.info("Perfil '%s' aplicado exitosamente a IPFS.", perfil)
        return True
    logger.error("Error aplicando perfil %s: %s", perfil, err)
    return False


def obtener_espacio_disco_disponible() -> Dict[str, float]:
    """Retorna metricas de espacio en disco del volumen donde reside el repo."""
    try:
        import shutil
        total, used, free = shutil.disk_usage(str(IPFSONL_DIR))
        return {
            "total_gb": round(total / (1024**3), 2),
            "usado_gb": round(used / (1024**3), 2),
            "libre_gb": round(free / (1024**3), 2),
        }
    except Exception as e:
        logger.debug("No se pudo obtener uso de disco: %s", e)
        return {"total_gb": 0.0, "usado_gb": 0.0, "libre_gb": 0.0}


def __getattr__(name: str) -> Any:
    """Exportacion dinamica hacia utilidades de ipfs_launcher."""
    from LC.celebro.CMFG.IPFSonl import ipfs_launcher
    if hasattr(ipfs_launcher, name):
        return getattr(ipfs_launcher, name)
    raise AttributeError(f"Modulo '{__name__}' no posee el atributo '{name}'")


__all__: Final[List[str]] = [
    "__version__",
    "__author__",
    "__status__",
    "__package_name__",
    "IPFSONL_DIR",
    "IPFS_BIN",
    "IPFS_REPO",
    "IPFS_API_URL",
    "IPFS_GATEWAY_URL",
    "PUBLIC_GATEWAYS",
    "get_ipfs_bin",
    "get_ipfs_repo",
    "get_ipfs_api_url",
    "get_ipfs_gateway_url",
    "verificar_instalacion",
    "inicializar_repositorio_si_falta",
    "probar_conexion_api",
    "obtener_id_nodo",
    "obtener_peers_enjambre",
    "obtener_estadisticas_almacenamiento",
    "obtener_metricas_ancho_banda",
    "construir_urls_pasarela",
    "verificar_estado_pin",
    "ejecutar_comando_personalizado",
    "arrancar",
    "detener",
    "reiniciar",
    "get_diagnostico_completo",
    "exportar_diagnostico_archivo",
    "imprimir_informe_consola",
    "resolver_version_kubo",
    "limpiar_bloqueos_repositorio",
    "configurar_perfil_nodo",
    "obtener_espacio_disco_disponible",
]

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    imprimir_informe_consola()
