"""Servidor HTTP y daemon BKSVCB extraidos para limite 450."""
from __future__ import annotations
import atexit
import json
import threading
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from typing import Any, Dict, Optional

__server_name__ = "CelebroBlockchainServer-BKSVCB"


def _clase_blockchain():
    from STM_BKCH.BKSVCB import CelebroBlockchain
    return CelebroBlockchain
class BlockchainHTTPHandler(BaseHTTPRequestHandler):
    """Manejador HTTP REST y JSON-RPC para el servidor blockchain."""

    def _enviar_json(self, status: int, payload: Dict[str, Any]) -> None:
        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers(); self.wfile.write(raw)

    def do_GET(self) -> None:
        bks = get_blockchain_server()
        if self.path in ("/", "/status"):
            valida, err = bks.validar_cadena()
            self._enviar_json(200, {
                "servidor": __server_name__, "total_bloques": len(bks.cadena),
                "cadena_valida": valida, "error_validacion": err, "ultimo_bloque": bks.obtener_ultimo_bloque().to_dict(),
            })
        elif self.path == "/blocks": self._enviar_json(200, {"bloques": [b.to_dict() for b in bks.cadena]})
        elif self.path == "/transform": self._enviar_json(200, bks.transformar_ledger_a_pesos_neuronales())
        elif self.path == "/mine":
            bloque = bks.minar_transacciones_pendientes()
            if bloque: self._enviar_json(200, {"mensaje": "Bloque minado", "bloque": bloque.to_dict()})
            else: self._enviar_json(200, {"mensaje": "Sin transacciones pendientes para minar"})
        else: self._enviar_json(404, {"error": "Ruta no encontrada"})

    def do_POST(self) -> None:
        bks = get_blockchain_server()
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length).decode("utf-8", errors="replace") if length > 0 else "{}"
        try: params = json.loads(body)
        except Exception: params = {}
        if self.path == "/transaction":
            idx = bks.agregar_transaccion(
                tipo=params.get("tipo", "TRANSFERENCIA_COGNITIVA"), emisor=params.get("emisor", "EXTERNO"),
                receptor=params.get("receptor", "CELEBRO"), datos=params.get("datos", {}), cid_ipfs=params.get("cid_ipfs"),
            )
            self._enviar_json(201, {"mensaje": "Transaccion encolada", "bloque_objetivo": idx})
        elif self.path == "/learn":
            res = bks.registrar_aprendizaje_neural(
                prompt=params.get("prompt", ""), respuesta=params.get("respuesta", ""), modelo=params.get("modelo", "qwen2.5:7b"),
            )
            self._enviar_json(200, res)
        else: self._enviar_json(404, {"error": "Endpoint POST no reconocido"})

    def log_message(self, format: str, *args: Any) -> None: pass


class BlockchainServerDaemon:
    """Contenedor de ejecucion asincrona para el servicio HTTP Blockchain."""

    def __init__(self, host: str = "127.0.0.1", puerto: int = 8545) -> None:
        self.host, self.puerto = host, puerto
        self.servidor: Optional[ThreadingHTTPServer] = None
        self.hilo: Optional[threading.Thread] = None

    def iniciar(self) -> bool:
        """Arranca el servidor HTTP en un hilo demonio desacoplado."""
        try:
            self.servidor = ThreadingHTTPServer((self.host, self.puerto), BlockchainHTTPHandler)
            self.hilo = threading.Thread(target=self.servidor.serve_forever, daemon=True)
            self.hilo.start()
            return True
        except Exception: return False

    def detener(self) -> None:
        """Detiene ordenadamente el servidor HTTP."""
        if self.servidor:
            self.servidor.shutdown(); self.servidor.server_close(); self.servidor = None


_blockchain_instancia: Optional[CelebroBlockchain] = None
_daemon_http_instancia: Optional[BlockchainServerDaemon] = None


def get_blockchain_server():
    """Acceso singleton al motor de la blockchain neuronal."""
    global _blockchain_instancia
    if _blockchain_instancia is None: _blockchain_instancia = _clase_blockchain()()
    return _blockchain_instancia


def iniciar_servidor_blockchain(puerto: int = 8545) -> BlockchainServerDaemon:
    """Arranca el servicio de API HTTP del servidor de bloques."""
    global _daemon_http_instancia
    if _daemon_http_instancia is None:
        _daemon_http_instancia = BlockchainServerDaemon(puerto=puerto)
        _daemon_http_instancia.iniciar()
    return _daemon_http_instancia


def _hook_salida_sistema() -> None:
    """Ejecutado al cerrar la sesion de terminal: sube ledger y pesos a IPFS."""
    if _blockchain_instancia:
        _blockchain_instancia.cerrar_sesion_y_subir_ipfs()


atexit.register(_hook_salida_sistema)


if __name__ == "__main__":
    bks = get_blockchain_server()
    print("\n\033[1;36m" + "=" * 74 + "\033[0m")
    print(f"  \033[1;32mBLOCKCHAIN SERVER & CONSENSO NEURAL: {__server_name__}\033[0m")
    print(f"  Neuronas Activas: \033[1;35m{len(bks.conversor.neuronas)}\033[0m | Dificultad: \033[1;37m{bks.dificultad}\033[0m | Bloques: \033[1;37m{len(bks.cadena)}\033[0m")
    print("\033[1;36m" + "=" * 74 + "\033[0m")

    # Muestra los hashes unicos inmediatamente
    bks.mostrar_cadena_hashes_terminal()

    # 1. Transformacion del ledger en pesos neuronales
    print("[+] Transformando blockchain_ledger.json en pesos neuronales (50 neuronas)...")
    res_t = bks.transformar_ledger_a_pesos_neuronales()
    print(f"    Bloques convertidos : \033[1;33m{res_t['total_bloques']}\033[0m")
    print(f"    Pesos en PSNRL      : \033[1;32m{res_t['npz_pesos']}\033[0m (Norma: {res_t['norma_acumulada']})")

    # 2. Arranque del actualizador continuo de la red neuronal durante la sesion
    bks.arrancar_actualizador_red()
    print("    Actualizador neuronal: \033[1;32mACTIVO EN SEGUNDO PLANO DURANTE LA SESION\033[0m")

    # 3. Transaccion y minado en vivo
    print("\n[+] Nueva transaccion cognitiva en blockchain...")
    res = bks.registrar_aprendizaje_neural(
        prompt="¿Cual es el rol de las 50 neuronas en la blockchain de Celebro?",
        respuesta="Registrar cada gradiente Hebbiano y estado emocional en bloques inmutables con hashes unicos.",
        modelo="qwen2.5:7b",
    )
    print(f"    CID IPFS vinculado   : \033[1;36m{res['ipfs_cid']}\033[0m")

    bloque = bks.minar_transacciones_pendientes()
    if bloque:
        print(f"    \033[1;32mBloque #{bloque.indice} consolidado con exito!\033[0m")
        print(f"    Hash unico del bloque: \033[1;33m{bloque.hash_bloque}\033[0m")
        print(f"    Raiz de Merkle       : \033[1;35m{bloque.merkle_root}\033[0m")

    valida, err = bks.validar_cadena()
    estado_str = "\033[1;32mINTEGRA Y VALIDA\033[0m" if valida else f"\033[1;31mERROR: {err}\033[0m"
    print(f"\n[+] Estado de la cadena: {estado_str}")
    print("    Persistencia IPFS al cierre: \033[1;36mREGISTRADA (atexit hook activo)\033[0m")
    print("\033[1;36m" + "=" * 74 + "\033[0m\n")
