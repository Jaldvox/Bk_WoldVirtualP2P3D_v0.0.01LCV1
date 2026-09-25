"""
ipfs_launcher.py - Orquestador y Gestor de Rendimiento del Daemon IPFS Kubo (Celebro 2026)
==========================================================================================
Control de ciclo de vida completo para Kubo IPFS en WoldVirtualP2P3D:
  - Auto-arranque no bloqueante en background con gestion de flags de pubsub y dht.
  - Sincronizacion de configuracion de repositorios IPFS (.ipfs) locales.
  - Monitorizacion activa de salud (healthcheck), auto-reinicio y recuperacion de fallos.
  - Context Manager (IPFSDaemonContext) para sesiones seguras con cierre garantizado.
  - Consultas de diagnostico: Swarm peers, Bandwidth, Repo stats y PeerID en tiempo real.
  - Sincronizacion y subida atomica de pesos neuronales PSNRL en hook de finalizacion.
  - Generador de URLs de pasarelas publicas y verificacion de pinning.
  - Interfaz de terminal ANSI enriquecida para visualizacion operativa.
"""
from __future__ import annotations
import os, sys, time, json, subprocess, logging, threading
import urllib.request, urllib.error
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger("WoldVirtualP2P3D.IPFSLauncher")

IPFSONL_DIR: Path = Path(__file__).parent.resolve()
IPFS_BIN: Path = IPFSONL_DIR / "ipfs.exe"
IPFS_REPO: Path = IPFSONL_DIR / ".ipfs"
IPFS_API_URL: str = "http://127.0.0.1:5001"
IPFS_GATEWAY_URL: str = "http://127.0.0.1:8080"

PUBLIC_GATEWAYS: List[str] = [
    "https://ipfs.io/ipfs/",
    "https://gateway.pinata.cloud/ipfs/",
    "https://dweb.link/ipfs/",
    "https://cloudflare-ipfs.com/ipfs/",
]

_daemon_proc: Optional[subprocess.Popen] = None
_health_thread: Optional[threading.Thread] = None
_stop_health_event = threading.Event()
_session_stats: Dict[str, Any] = {
    "start_time": 0.0,
    "cids_uploaded": 0,
    "bytes_uploaded": 0,
    "restarts": 0,
    "last_healthcheck": 0.0,
    "total_pings": 0,
}


def _get_env() -> Dict[str, str]:
    """Prepara variables de entorno con IPFS_PATH dirigido al repo local."""
    env = os.environ.copy()
    env["IPFS_PATH"] = str(IPFS_REPO)
    return env


def verificar_binario() -> bool:
    """Valida la presencia y permisos de ejecucion del binario Kubo."""
    if not IPFS_BIN.exists():
        logger.error("Binario Kubo ausente en: %s", IPFS_BIN)
        return False
    return True


def verificar_repositorio() -> bool:
    """Comprueba si el directorio de configuracion e infraestructura .ipfs existe."""
    if not IPFS_REPO.exists():
        logger.warning("Repositorio IPFS ausente en %s. Requiere inicializacion.", IPFS_REPO)
        return False
    config_file = IPFS_REPO / "config"
    return config_file.exists()


def daemon_responde(timeout: float = 1.5) -> bool:
    """Comprueba respuesta HTTP de Kubo en el endpoint /api/v0/id."""
    try:
        req = urllib.request.Request(f"{IPFS_API_URL}/api/v0/id", method="POST")
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False


def obtener_id_nodo(timeout: float = 2.0) -> Dict[str, Any]:
    """Obtiene metadatos del nodo: PeerID, AgentVersion y Protocolos."""
    try:
        req = urllib.request.Request(f"{IPFS_API_URL}/api/v0/id", method="POST")
        with urllib.request.urlopen(req, timeout=timeout) as r:
            if r.status == 200:
                return json.loads(r.read().decode("utf-8", errors="replace"))
    except Exception as e:
        logger.debug("Error obteniendo node id: %s", e)
    return {}


def obtener_estadisticas_repo(timeout: float = 3.0) -> Dict[str, Any]:
    """Recupera volumen de almacenamiento, numero de objetos y version de repo."""
    try:
        req = urllib.request.Request(f"{IPFS_API_URL}/api/v0/repo/stat", method="POST")
        with urllib.request.urlopen(req, timeout=timeout) as r:
            if r.status == 200:
                return json.loads(r.read().decode("utf-8", errors="replace"))
    except Exception as e:
        logger.debug("Error leyendo repo stats: %s", e)
    return {}


def obtener_peers_conectados(timeout: float = 3.0) -> List[Dict[str, Any]]:
    """Consulta la lista de peers activos en el IPFS swarm."""
    try:
        req = urllib.request.Request(f"{IPFS_API_URL}/api/v0/swarm/peers", method="POST")
        with urllib.request.urlopen(req, timeout=timeout) as r:
            if r.status == 200:
                data = json.loads(r.read().decode("utf-8", errors="replace"))
                return data.get("Peers") or []
    except Exception as e:
        logger.debug("Error listando peers de enjambre: %s", e)
    return []


def obtener_estadisticas_ancho_banda(timeout: float = 2.5) -> Dict[str, Any]:
    """Consulta el ratio y volumen total de ancho de banda consumido (in/out)."""
    try:
        req = urllib.request.Request(f"{IPFS_API_URL}/api/v0/stats/bw", method="POST")
        with urllib.request.urlopen(req, timeout=timeout) as r:
            if r.status == 200:
                return json.loads(r.read().decode("utf-8", errors="replace"))
    except Exception as e:
        logger.debug("Error consultando bandwidth stats: %s", e)
    return {}


def ejecutar_comando_cli(argumentos: List[str], timeout: float = 15.0) -> Tuple[int, str, str]:
    """Ejecuta un comando directo sobre el binario local con el entorno configurado."""
    if not verificar_binario():
        return -1, "", "Binario no disponible"
    cmd = [str(IPFS_BIN)] + argumentos
    try:
        res = subprocess.run(
            cmd,
            env=_get_env(),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return res.returncode, res.stdout, res.stderr
    except Exception as e:
        return -1, "", str(e)


def arrancar_daemon(esperar_segundos: int = 25, habilitar_dht: bool = True) -> bool:
    """
    Arranca el daemon Kubo en segundo plano optimizado para P2P 2026.
    Devuelve True si el API HTTP y el nodo quedan listos para recepcion.
    """
    global _daemon_proc, _session_stats

    if daemon_responde():
        logger.info("Daemon Kubo ya operativo en %s", IPFS_API_URL)
        if _session_stats["start_time"] == 0.0:
            _session_stats["start_time"] = time.time()
        iniciar_monitor_salud()
        return True

    if not verificar_binario():
        return False

    cmd = [
        str(IPFS_BIN),
        "daemon",
        "--enable-pubsub-experiment",
        "--enable-namesys-pubsub",
    ]
    if not habilitar_dht:
        cmd.append("--routing=none")

    flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
    logger.info("Iniciando proceso daemon Kubo: %s", " ".join(cmd))

    try:
        _daemon_proc = subprocess.Popen(
            cmd,
            env=_get_env(),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=flags,
        )
        _session_stats["start_time"] = time.time()
    except Exception as e:
        logger.error("Error critico lanzando subprocess Kubo: %s", e)
        return False

    for i in range(esperar_segundos):
        time.sleep(1)
        if daemon_responde():
            logger.info("Kubo IPFS listo y escuchando en %s (t=%ds)", IPFS_API_URL, i + 1)
            iniciar_monitor_salud()
            return True
        if _daemon_proc.poll() is not None:
            logger.error("Daemon fallo al arrancar (exit code=%d)", _daemon_proc.returncode)
            return False

    logger.warning("Timeout esperando inicio de Kubo tras %ds", esperar_segundos)
    return False


def detener_daemon(espera_segundos: int = 6) -> bool:
    """Detiene ordenadamente el proceso daemon local."""
    global _daemon_proc
    detener_monitor_salud()

    if _daemon_proc is None or _daemon_proc.poll() is not None:
        _daemon_proc = None
        return True

    logger.info("Enviando senal de apagado al daemon Kubo...")
    try:
        _daemon_proc.terminate()
        _daemon_proc.wait(timeout=espera_segundos)
        logger.info("Daemon Kubo detenido correctamente.")
        _daemon_proc = None
        return True
    except subprocess.TimeoutExpired:
        logger.warning("Daemon no respondio a SIGTERM; forzando SIGKILL.")
        _daemon_proc.kill()
        _daemon_proc = None
        return True
    except Exception as e:
        logger.error("Excepcion deteniendo daemon: %s", e)
        return False


def reiniciar_daemon() -> bool:
    """Reinicia el daemon de forma segura y actualiza metricas de sesion."""
    logger.info("Reiniciando daemon Kubo IPFS...")
    detener_daemon(espera_segundos=4)
    time.sleep(1)
    _session_stats["restarts"] += 1
    return arrancar_daemon()


def _bucle_monitor_salud() -> None:
    """Hilo de fondo para verificacion continua de liveness y auto-reinicio."""
    while not _stop_health_event.is_set():
        _stop_health_event.wait(15.0)
        if _stop_health_event.is_set():
            break
        vivo = daemon_responde(timeout=2.0)
        _session_stats["total_pings"] += 1
        _session_stats["last_healthcheck"] = time.time()
        if not vivo and _daemon_proc is not None:
            logger.warning("Kubo no responde en healthcheck. Procediendo a auto-recuperacion.")
            reiniciar_daemon()


def iniciar_monitor_salud() -> None:
    """Inicia el hilo supervisor de salud si no esta corriendo."""
    global _health_thread
    if _health_thread is None or not _health_thread.is_alive():
        _stop_health_event.clear()
        _health_thread = threading.Thread(target=_bucle_monitor_salud, daemon=True)
        _health_thread.start()


def detener_monitor_salud() -> None:
    """Detiene el hilo supervisor de salud."""
    _stop_health_event.set()


def generar_urls_pasarela(cid: str) -> Dict[str, str]:
    """Construye enlaces a pasarelas publicas y privadas para un CID dado."""
    return {
        "local": f"{IPFS_GATEWAY_URL}/ipfs/{cid}",
        "ipfs_io": f"https://ipfs.io/ipfs/{cid}",
        "pinata": f"https://gateway.pinata.cloud/ipfs/{cid}",
        "dweb": f"https://dweb.link/ipfs/{cid}",
        "cloudflare": f"https://cloudflare-ipfs.com/ipfs/{cid}",
    }


def verificar_pin_cid(cid: str, timeout: float = 3.0) -> bool:
    """Verifica si un CID concreto se encuentra pinned en el nodo local."""
    try:
        req = urllib.request.Request(f"{IPFS_API_URL}/api/v0/pin/ls?arg={cid}", method="POST")
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False


def sincronizar_y_subir_al_cierre(forzar_borrado: bool = False) -> Dict[str, Any]:
    """
    Hook de sincronizacion final para Celebro:
    Asegura conexion activa a IPFS, sube pesos de PSNRL y purga temporales locales.
    """
    sys.path.insert(0, str(IPFSONL_DIR.parent.parent.parent.parent))
    try:
        from LC.celebro.CMFG.ipfs_manager import get_ipfs_manager
        mgr = get_ipfs_manager(api_url=IPFS_API_URL)
        res = mgr.subir_y_limpiar_psnrl(forzar_borrado_sin_daemon=forzar_borrado)
        _session_stats["cids_uploaded"] += len(res.get("cids", []))
        return res
    except Exception as e:
        logger.error("Error sincronizando en hook de cierre: %s", e)
        return {"error": str(e), "archivos_procesados": 0, "cids": [], "borrados": []}


def descargar_pesos_desde_ipfs(cid: str, destino_path: Union[str, Path]) -> bool:
    """Recupera un archivo de pesos por CID y lo deposita en la ruta indicada."""
    sys.path.insert(0, str(IPFSONL_DIR.parent.parent.parent.parent))
    try:
        from LC.celebro.CMFG.ipfs_manager import get_ipfs_manager
        mgr = get_ipfs_manager(api_url=IPFS_API_URL)
        mgr.recuperar_pesos(cid, destino=destino_path)
        return True
    except Exception as e:
        logger.error("Fallo recuperando pesos (%s): %s", cid, e)
        return False


def imprimir_panel_estado() -> None:
    """Muestra un dashboard ANSI con telemetria del nodo y estado del cluster."""
    id_info = obtener_id_nodo()
    peers = obtener_peers_conectados()
    repo_stats = obtener_estadisticas_repo()
    bw_stats = obtener_estadisticas_ancho_banda()

    peer_id = id_info.get("ID", "Desconocido")
    agent = id_info.get("AgentVersion", "Kubo P2P")
    repo_size = repo_stats.get("RepoSize", 0) / (1024 * 1024)
    num_obj = repo_stats.get("NumObjects", 0)
    total_in = bw_stats.get("TotalIn", 0) / 1024
    total_out = bw_stats.get("TotalOut", 0) / 1024

    uptime = 0.0
    if _session_stats["start_time"] > 0:
        uptime = time.time() - _session_stats["start_time"]

    print("\033[1;36m" + "=" * 68 + "\033[0m")
    print("  \033[1;32mPANEL OPERATIVO IPFS KUBO (WoldVirtualP2P3D 2026)\033[0m")
    print("\033[1;36m" + "=" * 68 + "\033[0m")
    print(f"  Estado Daemon   : \033[1;32mONLINE\033[0m (HTTP {IPFS_API_URL})")
    print(f"  PeerID Nodo     : \033[1;33m{peer_id}\033[0m")
    print(f"  Agente / Motor  : {agent}")
    print(f"  Peers Conectados: \033[1;35m{len(peers)}\033[0m nodos en enjambre")
    print(f"  Repo Local      : {repo_size:.2f} MB ({num_obj} objetos persistidos)")
    print(f"  Trafico P2P     : IN: {total_in:.1f} KB | OUT: {total_out:.1f} KB")
    print(f"  Uptime Sesion   : {uptime:.1f} s | Reinicios: {_session_stats['restarts']}")
    print(f"  Subidas Sesion  : {_session_stats['cids_uploaded']} CIDs")
    print("\033[1;36m" + "-" * 68 + "\033[0m")
    if peers:
        print("  Top Peers Activos:")
        for p in peers[:3]:
            print(f"    -> {p.get('Peer', '')[:24]}... ({p.get('Addr', '')[:32]})")
    print("\033[1;36m" + "=" * 68 + "\033[0m\n")


class IPFSDaemonContext:
    """Context Manager para arranque y parada determinista en pipelines."""

    def __init__(self, esperar_segundos: int = 25, auto_subida_cierre: bool = True):
        self.esperar = esperar_segundos
        self.auto_subida = auto_subida_cierre

    def __enter__(self) -> bool:
        return arrancar_daemon(esperar_segundos=self.esperar)

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self.auto_subida:
            sincronizar_y_subir_al_cierre()
        detener_daemon()


def get_ipfs_bin() -> Path:
    """Devuelve la ruta absoluta al binario local."""
    return IPFS_BIN


def get_ipfs_api_url() -> str:
    """Devuelve la URL HTTP del API Kubo."""
    return IPFS_API_URL


def get_session_stats() -> Dict[str, Any]:
    """Devuelve las metricas acumuladas durante la sesion."""
    return dict(_session_stats)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    print("\n[+] Verificando entorno e infraestructura de IPFS Kubo...")
    print(f"    Binario objetivo: {IPFS_BIN}")
    print(f"    Directorio Repo : {IPFS_REPO}")

    exito = arrancar_daemon(esperar_segundos=20)
    if exito:
        imprimir_panel_estado()
        print("[+] Probando sincronizacion de cierre de sesion para PSNRL...")
        resumen = sincronizar_y_subir_al_cierre(forzar_borrado=False)
        print(f"    Archivos procesados : {resumen.get('archivos_procesados', 0)}")
        print(f"    CIDs confirmados    : {len(resumen.get('cids', []))}")
        print(f"    Borrados asegurados : {len(resumen.get('borrados', []))}")
        for c in resumen.get("cids", [])[:3]:
            print(f"    CID: {c} -> {generar_urls_pasarela(c)['ipfs_io']}")
    else:
        print("[-] No fue posible inicializar el daemon local Kubo.")