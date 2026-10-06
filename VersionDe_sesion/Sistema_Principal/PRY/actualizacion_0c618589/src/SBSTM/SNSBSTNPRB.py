"""
SNSBSTNPRB.PY - SubSistema de Sesion Neuronal P2P de Celebro (Arquitectura 2026)
==================================================================================
Orquestador de sesion completa para WoldVirtualP2P3D:
  - Arranca la blockchain (BKSVCB) y despliega hashes unicos en terminal.
  - Transforma el ledger existente en pesos neuronales sobre las 50 sinapsis.
  - Mantiene conversacion real con modelos Ollama locales (cogito:3b, qwen2.5:7b).
  - Cada respuesta del modelo genera transaccion sinaptica en blockchain.
  - Monitor de pesos vivos (GSNR, deriva Muon, entropia Shannon) por turno.
  - Minado automatico cada 3 turnos; checkpoint PSNRL + IPFS al cerrar sesion.
"""
from __future__ import annotations
import os, sys, json, time, urllib.request, urllib.error, atexit, signal
from pathlib import Path
from typing import Any, Dict, List, Optional

# ─── RUTAS BASE ─────────────────────────────────────────────────────────────
SBSTM_DIR   = Path(__file__).resolve().parent
CMFG_DIR    = SBSTM_DIR.parent
CELEBRO_DIR = CMFG_DIR.parent
ROOT_DIR    = CELEBRO_DIR.parent.parent
try:
    from STM_CH.rutas import PSNRL_DIR as PSNRL_DIR
except ImportError:
    PSNRL_DIR = CELEBRO_DIR / "PSNRL"
PSNRL_DIR.mkdir(parents=True, exist_ok=True)

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import logging
logging.basicConfig(level=logging.CRITICAL)
for _lg in ("", "WoldVirtualP2P3D", "LC", "urllib3", "ENRN", "SLRN", "RNP", "httpx"):
    logging.getLogger(_lg).setLevel(logging.CRITICAL)

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

# ─── PALETA ANSI PREMIUM ─────────────────────────────────────────────────────
class C:
    R = "\033[0m";  B = "\033[1m";  DIM = "\033[2m"
    CY = "\033[96m"; BL = "\033[94m"; GR = "\033[92m"
    YL = "\033[93m"; MG = "\033[95m"; RD = "\033[91m"
    WH = "\033[97m"

    @staticmethod
    def c(color: str, txt: str) -> str:
        return f"{color}{txt}{C.R}"


# ─── CARGA DIFERIDA DEL NUCLEO ───────────────────────────────────────────────
def _importar_nucleo():
    """Carga los modulos core desde STM_BKCH."""
    from STM_BKCH.BKSVCB import get_blockchain_server, iniciar_servidor_blockchain
    from STM_BKCH.compat_local import get_conversor_pesos
    get_gestor_pesos_vivos = lambda: None
    return get_blockchain_server, iniciar_servidor_blockchain, get_conversor_pesos, get_gestor_pesos_vivos


# ─── CLIENTE OLLAMA ──────────────────────────────────────────────────────────
OLLAMA_URL  = "http://localhost:11434"
try:
    from STM_JSON.registro_json import ruta_de as _ruta_json
    IA_CFG_PATH = _ruta_json("ialocal")
except Exception:
    IA_CFG_PATH = ROOT_DIR / "LC" / "modelosIAlocal" / "IAlocal.json"


def _cargar_config_ia() -> Dict[str, Any]:
    """Lee IAlocal.json o retorna defaults seguros."""
    if IA_CFG_PATH.exists():
        try:
            with open(IA_CFG_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"default_model": "cogito:3b", "temperature": 0.7, "max_tokens": 2048}


def _ollama_disponible() -> bool:
    """Comprueba si Ollama esta corriendo."""
    try:
        with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=2) as r:
            return r.status == 200
    except Exception:
        return False


def _listar_modelos_ollama() -> List[str]:
    """Retorna modelos disponibles en Ollama."""
    try:
        with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=3) as r:
            data = json.loads(r.read().decode())
            return [m["name"] for m in data.get("models", [])]
    except Exception:
        return []


def consultar_ollama_stream(prompt: str, modelo: str,
                            temperatura: float = 0.7, max_tokens: int = 2048) -> str:
    """Envia prompt a Ollama con streaming real; imprime tokens en vivo."""
    payload = json.dumps({
        "model": modelo, "prompt": prompt,
        "stream": True,
        "options": {"temperature": temperatura, "num_predict": max_tokens},
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/generate", data=payload,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    tokens: List[str] = []
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            print(f"\n{C.c(C.GR, chr(9654))} ", end="", flush=True)
            for linea in resp:
                try:
                    chunk = json.loads(linea.decode("utf-8"))
                    tok = chunk.get("response", "")
                    if tok:
                        print(tok, end="", flush=True)
                        tokens.append(tok)
                    if chunk.get("done", False):
                        break
                except Exception:
                    continue
            print()
    except urllib.error.URLError as e:
        print(C.c(C.RD, f"\n[Ollama] Error: {e}"))
    return "".join(tokens)


# ─── TELEMETRIA COMPACTA DE PESOS ────────────────────────────────────────────
# _mini_pesos vive en snsbstnprb_utils.py para respetar el limite de 450 lineas.
from SBSTM.snsbstnprb_utils import mini_pesos as _mini_pesos  # noqa: E402,F401


# ─── ORQUESTADOR PRINCIPAL ───────────────────────────────────────────────────
class SesionNeuronalP2P:
    """
    SubSistema de Sesion Neuronal P2P (SNSBSTNPRB).
    Gestiona el ciclo completo: arranque -> conversacion -> cierre IPFS.
    """

    def __init__(
        self,
        modelo_inicial: Optional[str] = None,
        puerto_bksvcb: int = 8545,
    ) -> None:
        self._nucleo    = _importar_nucleo()
        self.cfg_ia     = _cargar_config_ia()
        self.modelo     = modelo_inicial or self.cfg_ia.get("default_model", "cogito:3b")
        self.temp       = float(self.cfg_ia.get("temperature", 0.7))
        self.max_tok    = int(self.cfg_ia.get("max_tokens", 2048))
        self.puerto_bksvcb = puerto_bksvcb
        self.bks        = None
        self.daemon     = None
        self.conversor  = None
        self.gestor_pv  = None
        self.activa     = False
        self._turno     = 0
        self._sesion_id = time.strftime("%Y%m%d_%H%M%S")
        self._cerrado   = False
        atexit.register(self._cerrar)

    # ── ARRANQUE ──────────────────────────────────────────────────────────────
    def arrancar(self) -> None:
        """Inicializa todos los subsistemas en orden y despliega la bienvenida."""
        get_bks, iniciar_daemon, get_conv, get_gpv = self._nucleo

        print(C.c(C.CY, "\n" + "=" * 74))
        print(C.c(C.B,  "  WOLDVIRTUALP2P3D -- SUBSISTEMA DE SESION NEURONAL 2026"))
        print(C.c(C.CY, "=" * 74))

        # 1. Motor blockchain
        print(C.c(C.DIM, "  [1/5] Iniciando motor blockchain..."), end="\r")
        self.bks = get_bks()
        print(f"  [1/5] Blockchain: {C.c(C.GR,'ACTIVA')} | "
              f"Bloques: {C.c(C.YL, str(len(self.bks.cadena)))} | "
              f"Dificultad: {C.c(C.WH, str(self.bks.dificultad))}")

        # Hashes de cada bloque inmediatamente
        self.bks.mostrar_cadena_hashes_terminal()

        # 2. Conversor neuronal
        print(C.c(C.DIM, "  [2/5] Cargando conversor neuronal..."), end="\r")
        self.conversor = get_conv()
        print(f"  [2/5] Conversor PSNRCV: {C.c(C.GR,'OK')} | "
              f"Neuronas: {C.c(C.MG, str(len(self.conversor.neuronas)))}")

        # 3. Ledger -> pesos
        print(C.c(C.DIM, "  [3/5] Transdificando ledger a pesos neuronales..."), end="\r")
        res_t = self.bks.transformar_ledger_a_pesos_neuronales()
        print(f"  [3/5] Ledger->Pesos: {C.c(C.GR, str(res_t['total_bloques']))} bloques | "
              f"Norma: {C.c(C.YL, str(res_t['norma_acumulada']))}")

        # 4. Actualizador continuo
        print(C.c(C.DIM, "  [4/5] Arrancando actualizador neuronal..."), end="\r")
        self.bks.arrancar_actualizador_red()
        print(f"  [4/5] Actualizador: {C.c(C.GR,'ACTIVO')} {C.c(C.DIM,'(latido cada 12s)')}")

        # 5. Servidor HTTP REST
        print(C.c(C.DIM, f"  [5/5] Levantando servidor HTTP :{self.puerto_bksvcb}..."), end="\r")
        self.daemon = iniciar_daemon(puerto=self.puerto_bksvcb)
        print(f"  [5/5] REST API: {C.c(C.GR,f'http://127.0.0.1:{self.puerto_bksvcb}')} "
              f"{C.c(C.DIM,'(/status /blocks /mine /transform)')}")

        self.gestor_pv = get_gpv()

        # Ollama
        if _ollama_disponible():
            modelos = _listar_modelos_ollama()
            disponibles = self.cfg_ia.get("models_available", [])
            activo = next((m for m in disponibles if any(m in ml for ml in modelos)), self.modelo)
            self.modelo = activo
            print(f"\n  Ollama: {C.c(C.GR,'CONECTADO')} | Modelo: {C.c(C.CY, self.modelo)}")
        else:
            print(f"\n  Ollama: {C.c(C.YL,'OFFLINE')} -- modo reflejo neuronal autonomo activo")

        print(C.c(C.CY, "\n" + "=" * 74))
        print(C.c(C.DIM, "  Sesion ID: ") + C.c(C.WH, self._sesion_id))
        print(C.c(C.GR,  "  Comandos: estado | modelo <nombre> | saber <tema> | escuchar | salir"))
        print(C.c(C.CY, "=" * 74 + "\n"))
        self.activa = True

    # ── BUCLE DE CONVERSACION ─────────────────────────────────────────────────
    def bucle_conversacion(self) -> None:
        """Loop principal: prompt -> modelo -> pesos -> blockchain."""
        while self.activa:
            try:
                prompt = input(C.c(C.CY, "\n> Tu: ")).strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break

            if not prompt:
                continue
            if prompt.lower() in ("salir", "exit", "quit"):
                print(C.c(C.YL, "\n  Cerrando sesion..."))
                break
            from SBSTM.snsbstnprb_utils import gestionar_comando
            prompt = gestionar_comando(self, prompt)
            if not prompt:
                continue

            self._turno += 1
            t0 = time.perf_counter()
            print(C.c(C.DIM, f"\n  [{self.modelo}] Procesando..."), end="\r")

            ctx = ""
            try:
                from SBSTM.memoria_sesion import get_memoria
                ctx = get_memoria(self._sesion_id).resumen_contexto()
            except Exception:
                pass
            emo: Dict[str, Any] = {"valencia": 0.0, "etiqueta": "neutral"}
            try:
                from SBSTM.emocion import analizar_emocion
                emo = analizar_emocion(prompt)
            except Exception:
                pass
            prompt_ctx = f"{ctx}\n\nPregunta actual: {prompt}" if ctx else prompt

            if _ollama_disponible():
                respuesta = consultar_ollama_stream(prompt_ctx, self.modelo, self.temp, self.max_tok)
            else:
                respuesta = self._reflejo_neuronal(prompt_ctx)

            dt = (time.perf_counter() - t0) * 1000.0

            if not respuesta:
                print(C.c(C.RD, "  [Sin respuesta]"))
                continue

            # Registrar en blockchain
            cid = "--"; info_neural: Dict[str, Any] = {}
            try:
                res_chain = self.bks.registrar_aprendizaje_neural(
                    prompt=prompt, respuesta=respuesta, modelo=self.modelo,
                )
                cid = res_chain.get("ipfs_cid", "--")
                info_neural = res_chain.get("neural", {})
            except Exception as ex:
                print(C.c(C.YL, f"  [blockchain] {ex}"))

            # Minado automatico cada 3 turnos
            if self._turno % 3 == 0:
                blk = self.bks.minar_transacciones_pendientes()
                if blk:
                    print(C.c(C.GR,
                          f"  [Bloque #{blk.indice}] Hash: {blk.hash_bloque[:28]}..."))

            # Post-procesado del turno: memoria, perfil, voz, dashboard, rubrica
            from SBSTM.snsbstnprb_utils import postprocesar_turno
            postprocesar_turno(
                sesion=self, prompt=prompt, respuesta=respuesta,
                info_neural=info_neural, emo=emo, dt=dt, cid=cid,
            )

        self.activa = False

    # ─── API PUBLICA (consumida por SBSTM/__init__.py) ─────────────────────────
    def iniciar_consola_interactiva(self) -> None:
        """Arranca los subsistemas y entra en el bucle de conversacion."""
        self.arrancar()
        self.bucle_conversacion()

    def cerrar(self) -> None:
        """Alias publico de _cerrar(), idempotente."""
        if not getattr(self, "_cerrado", False):
            self._cerrado = True
            self._cerrar()

    def obtener_telemetria(self) -> Dict[str, Any]:
        """Instantanea de telemetria y estado para el orquestador."""
        valida, err = self.bks.validar_cadena() if self.bks else (False, "sin blockchain")
        tel: Dict[str, Any] = {}
        if self.gestor_pv:
            try:
                tel = self.gestor_pv.obtener_telemetria_resumen() or {}
            except Exception:
                tel = {}
        return {
            "sesion_id": self._sesion_id,
            "activa": self.activa,
            "modelo": self.modelo,
            "turnos": self._turno,
            "cadena_valida": bool(valida),
            "error_cadena": err,
            "bloques": len(self.bks.cadena) if self.bks else 0,
            "txs_pendientes": len(self.bks.transacciones_pendientes) if self.bks else 0,
            "neuronas": len(self.conversor.neuronas) if self.conversor else 0,
            "deriva_total": float(tel.get("deriva_total", 0.0)),
            "gsnr_medio": float(tel.get("gsnr_medio", 0.0)),
            "puerto_bksvcb": self.puerto_bksvcb,
        }

    def guardar_checkpoint_psnrl(self) -> List[Any]:
        """Fuerza un checkpoint de pesos en PSNRL. Devuelve los artefactos."""
        if not self.conversor:
            _get_bks, _iniciar, get_conv, _get_gpv = self._nucleo
            self.conversor = get_conv()
        return self.conversor.persistir_pesos_en_psnrl(etiqueta="checkpoint_man")

    # ── REFLEJO NEURONAL (SIN OLLAMA) ─────────────────────────────────────────
    def _reflejo_neuronal(self, prompt: str) -> str:
        """Respuesta interna de la red neuronal cuando Ollama no esta disponible."""
        from SBSTM.snsbstnprb_utils import reflejo_neuronal
        return reflejo_neuronal(self.conversor, prompt)

    # ── COMANDOS AUXILIARES ───────────────────────────────────────────────────
    def _mostrar_estado(self) -> None:
        """Panel de estado compacto del sistema en tiempo real."""
        from SBSTM.snsbstnprb_utils import mostrar_estado
        mostrar_estado(self.bks, self.conversor, self.gestor_pv,
                       self._turno, self.modelo)

    def _cambiar_modelo(self, nombre: str) -> None:
        """Cambia el modelo Ollama activo durante la sesion."""
        from SBSTM.snsbstnprb_utils import resolver_modelo
        nuevo = resolver_modelo(nombre, _listar_modelos_ollama())
        if nuevo:
            self.modelo = nuevo
            print(C.c(C.GR, f"  Modelo cambiado a: {self.modelo}"))
        else:
            print(C.c(C.YL, f"  '{nombre}' no disponible."))

    # ── CIERRE Y PERSISTENCIA ─────────────────────────────────────────────────
    def _cerrar(self) -> None:
        """Hook atexit: mina pendientes finales, checkpoint PSNRL, sube a IPFS."""
        if not self.bks or getattr(self, "_cerrado", False):
            return
        self._cerrado = True
        from SBSTM.snsbstnprb_utils import cerrar_sesion
        cerrar_sesion(self.bks, self.conversor, self.gestor_pv,
                      self._sesion_id, self._turno)


# ─── MANEJADOR DE SEÑAL (Ctrl+C limpio) ─────────────────────────────────────
def _manejador_senal(sig, frame) -> None:  # noqa: ARG001
    print(C.c(C.YL, "\n  [SIGINT] Cerrando sesion neural..."))
    sys.exit(0)


signal.signal(signal.SIGINT, _manejador_senal)


# ─── PUNTO DE ENTRADA ────────────────────────────────────────────────────────
if __name__ == "__main__":
    sesion = SesionNeuronalP2P()
    sesion.arrancar()
    sesion.bucle_conversacion()