'\nmainLCSTM.py - Orquestador Central del Sistema Cognitivo LucIA (WoldVirtualP2P3D 2026)\n=====================================================================================\nCoordinacion maestro: Blockchain BKSVCB, Sesion P2P, IAFREE ($0.00), STYLOS y RPLC.\n'
from __future__ import annotations
import atexit
import json
import logging
import os
import signal
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, Final, List, Optional, Tuple, Union

CURRENT_FILE: Final[Path] = Path(__file__).resolve()
GNRL_DIR: Final[Path] = CURRENT_FILE.parent
LC_DIR: Final[Path] = GNRL_DIR.parent
SRC_DIR: Final[Path] = GNRL_DIR.parent
ROOT_DIR: Final[Path] = SRC_DIR.parent
CONTRRF_DIR: Final[Path] = CURRENT_FILE.parents[3]
ENV_FILE: Final[Path] = next(
    (c for c in (GNRL_DIR / ".env", SRC_DIR / ".env", ROOT_DIR / ".env")
     if c.exists()),
    SRC_DIR / ".env",
)
CELEBRO_DIR: Final[Path] = SRC_DIR / "STM_CH"
PSNRL_DIR: Final[Path] = SRC_DIR / "STM_CH" / "PSNRL"
CMFG_DIR: Final[Path] = SRC_DIR
SBSTM_DIR: Final[Path] = SRC_DIR / "SBSTM"
STM_RFC_DIR: Final[Path] = SRC_DIR.parent / "STM_RFC"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if STM_RFC_DIR.is_dir() and str(STM_RFC_DIR) not in sys.path:
    sys.path.insert(0, str(STM_RFC_DIR))
for _p in (LC_DIR, ROOT_DIR, CONTRRF_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
try:
    from STM_CH.rutas import inicializar as _stmch_init
    _stmch_init()
except Exception:
    pass
try:
    import RFPRMN as _RFPRMN  # pyrefly: ignore[missing-import]
    _RFPRMN_DISPONIBLE = True
except Exception:
    _RFPRMN_DISPONIBLE = False
    _RFPRMN = None
_COMANDOS_RFPRMN: Final[Dict[str, str]] = { "crear carpeta": "comando_crear_carpeta", "crear archivo": "comando_crear_archivo", "refactorizar": "comando_refactorizar", "actualizar": "actualizar", "enviar": "comando_enviar", "ia local": "comando_ia_local", "cerrar": "cerrar", }
try:
    from LC.celebro.CMFG.SBSTM.IAFREE import ClienteIAFree, get_cliente_iafree  # pyrefly: ignore[missing-import]
except Exception:
    try:
        from SBSTM.IAFREE import ClienteIAFree, get_cliente_iafree
    except Exception:
        get_cliente_iafree = None
        ClienteIAFree = None
try:
    from LC.celebro.CMFG.SBSTM.STYLOS import ( ColoresLucIA, EstiloTerminalLucIA, Glifos, badge_turno, banner_bienvenida, formatear_respuesta_lucia, panel_ayuda_comandos, )  # pyrefly: ignore[missing-import]
except Exception:
    try:
        from SBSTM.STYLOS import ( ColoresLucIA, EstiloTerminalLucIA, Glifos, badge_turno, banner_bienvenida, formatear_respuesta_lucia, panel_ayuda_comandos, )
    except Exception:
        EstiloTerminalLucIA = None
try:
    from LC.celebro.CMFG.SBSTM.RPLC import ( get_procesador_rplc, reprocesar_con_metricas, )  # pyrefly: ignore[missing-import]
    _RPLC_DISPONIBLE = True
except Exception:
    try:
        from SBSTM.RPLC import ( get_procesador_rplc, reprocesar_con_metricas, )
        _RPLC_DISPONIBLE = True
    except Exception:
        _RPLC_DISPONIBLE = False
        get_procesador_rplc = None  # type: ignore
        reprocesar_con_metricas = None  # type: ignore
try:
    from LC.celebro.CMFG.SBSTM.voice_engine import ( cancel_speech as _cancelar_voz, configurar_prosodia_juvenil as _prosodia_voz, speak as _hablar_voz, )  # pyrefly: ignore[missing-import]
    _VOZ_DISPONIBLE = True
except Exception:
    try:
        from SBSTM.voice_engine import ( cancel_speech as _cancelar_voz, configurar_prosodia_juvenil as _prosodia_voz, speak as _hablar_voz, )
        _VOZ_DISPONIBLE = True
    except Exception:
        _VOZ_DISPONIBLE = False
        _hablar_voz = None  # type: ignore
        _cancelar_voz = None  # type: ignore
        _prosodia_voz = None  # type: ignore
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
logging.basicConfig(level=logging.CRITICAL)
for _log_name in ("", "WoldVirtualP2P3D", "LC", "urllib3", "ENRN", "SLRN", "RNP", "httpx"):
    logging.getLogger(_log_name).setLevel(logging.CRITICAL)
class GestorEntornoSeguro:
    """Carga y gestiona de forma aislada credenciales y rutas desde .env."""
    @staticmethod
    def cargar_variables(ruta_env: Path = ENV_FILE) -> Dict[str, str]:
        variables: Dict[str, str] = {}
        if not ruta_env.exists():
            return variables
        try:
            contenido = ruta_env.read_text(encoding="utf-8-sig", errors="replace")
            for linea in contenido.splitlines():
                linea_limpia = linea.strip()
                if not linea_limpia or linea_limpia.startswith("#"):
                    continue
                if "=" in linea_limpia:
                    clave, _, valor = linea_limpia.partition("=")
                    clave = clave.strip().lstrip("\ufeff")
                    valor = valor.strip().strip("'\"")
                    variables[clave] = valor
                    os.environ[clave] = valor
        except Exception as err:
            sys.stderr.write(f"[mainLCSTM] Advertencia leyendo .env: {err}\n")
        return variables
    @classmethod
    def obtener_openrouter_key(cls) -> str:
        try:
            from STM_SGR.vault_openrouter import obtener_key
            key = obtener_key()
            if key:
                return key
        except Exception:
            pass
        vars_env = cls.cargar_variables()
        return vars_env.get("OPENROUTER_API_KEY", os.getenv("OPENROUTER_API_KEY", "")).strip()
try:
    from SBSTM.selector_modelos import SelectorModelos
except ImportError:
    try:
        from src.SBSTM.selector_modelos import SelectorModelos
    except ImportError:
        SelectorModelos = object
try:
    from SBSTM.accesores import GestorHRTLucIA, GestorJSONLucIA
except ImportError:
    GestorJSONLucIA = None  # type: ignore
    GestorHRTLucIA = None  # type: ignore
try:
    from STM_IA.modelosIAlocal.IALOCAL import get_gestor_ialocal
except ImportError:
    get_gestor_ialocal = None  # type: ignore
class OrquestadorSistemaLucIA(SelectorModelos):
    '\n    Orquestador maestro que integra BKSVCB, SNSBSTNPRB, IAFREE, STYLOS y las 50 neuronas.\n    '
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.servidor_bks = None
        self.conversor_psn = None
        self.sesion_p2p = None
        self.cliente_iafree = None
        self.activa = False
        self.turno_actual = 0
        self.sesion_id = f"LUCIA_{time.strftime('%Y%m%d_%H%M%S')}"
        atexit.register(self.cerrar_sistema)
    def inicializar_subsistemas(self) -> bool:
        """Inicializa en secuencia la blockchain, transductor, IAFREE y credenciales."""
        GestorEntornoSeguro.cargar_variables()
        try:
            from STM_SGR.vault_openrouter import confirmar_conexion
            confirmar_conexion()
        except Exception:
            pass
        if get_cliente_iafree is not None:
            self.cliente_iafree = get_cliente_iafree()
            modelo_ini = self.cliente_iafree.gestor.obtener_modelo_activo()["id"]
        else:
            modelo_ini = "openrouter/free"
        if EstiloTerminalLucIA:
            banner_bienvenida(self.sesion_id, modelo_ini)
        else:
            print(f"\n[+] WoldVirtualP2P3D -- LucIA Console 2026 | Mod: {modelo_ini}")
        try:
            from STM_BKCH.BKSVCB import get_blockchain_server, iniciar_servidor_blockchain
            self.servidor_bks = get_blockchain_server()
            print(f"  [1/4] Blockchain BKSVCB    : \033[38;5;48mACTIVA\033[0m | Bloques: \033[38;5;220m{len(self.servidor_bks.cadena)}\033[0m")
            self.servidor_bks.mostrar_cadena_hashes_terminal()
            iniciar_servidor_blockchain(puerto=8545)
        except Exception as e_bks:
            print(f"  [1/4] Blockchain BKSVCB    : \033[38;5;203mERROR ({e_bks})\033[0m")
            return False
        try:
            from STM_BKCH.compat_local import get_conversor_pesos
            self.conversor_psn = get_conversor_pesos()
            num_neu = len(self.conversor_psn.neuronas)
            print(f"  [2/4] Conversor PSNRCV     : \033[38;5;48mOK\033[0m | Sinapsis: \033[38;5;141m{num_neu} Activas\033[0m")
        except Exception as e_psn:
            print(f"  [2/4] Conversor PSNRCV     : \033[38;5;203mERROR ({e_psn})\033[0m")
            return False
        try:
            res_t = self.servidor_bks.transformar_ledger_a_pesos_neuronales()
            self.servidor_bks.arrancar_actualizador_red()
            print(f"  [3/4] Ledger -> Pesos PSNRL: \033[38;5;48m{res_t.get('total_bloques', 0)} bloques\033[0m | Norma: \033[38;5;214m{res_t.get('norma_acumulada', 0.0)}\033[0m")
        except Exception as e_led:
            print(f"  [3/4] Ledger -> Pesos      : \033[38;5;214mAVISO ({e_led})\033[0m")
        try:
            from SBSTM import obtener_clase_sesion
            cls_sesion = obtener_clase_sesion()
            self.sesion_p2p = cls_sesion()
            print(f"  [4/4] Sesion SNSBSTNPRB    : \033[38;5;48mACOPLADA\033[0m | Motor STYLOS: \033[38;5;51mACTIVO\033[0m\n")
        except Exception as e_sbs:
            print(f"  [4/4] Sesion SNSBSTNPRB    : \033[38;5;214mAVISO ({e_sbs})\033[0m\n")
        try:
            self.gestor_json = GestorJSONLucIA() if GestorJSONLucIA else None
            print(f"  [JSON] {self.gestor_json.linea_estado() if self.gestor_json else 'no disponible'}")
        except Exception:
            pass
        try:
            self.gestor_hrts = GestorHRTLucIA() if GestorHRTLucIA else None
            print(f"  [HRTS] {self.gestor_hrts.linea_estado() if self.gestor_hrts else 'no disponible'}")
        except Exception:
            pass
        self.activa = True
        return True
    def procesar_turno_dialogo(self, prompt: str) -> None:
        '\n        Ejecuta el pipeline cognitivo completo por turno a coste cero con renderizado STYLOS:\n        1. Pre-procesamiento por las 50 neuronas para tono y estado emocional.\n        2. Inferencia local prioritaria (IALOCAL) y OpenRouter :free opcional.\n        3. Formateo y presentacion de respuesta estilizada (tarjetas redondeadas y markdown).\n        4. Asimilacion de gradientes Hebbianos en tensores de PSNRL.\n        5. Emision de transaccion sinaptica a Blockchain BKSVCB y minado automatico.\n        '
        self.turno_actual += 1
        t_inicio = time.perf_counter()
        estado_previo = self.conversor_psn.procesar_consulta_a_pesos(prompt)
        respuesta = ""
        modelo_usado = "Reflejo-Interno"
        latencia_llm = 0.0
        if get_gestor_ialocal is not None:
            try:
                _rep = reprocesar_con_metricas if _RPLC_DISPONIBLE else None
                respuesta, modelo_usado = get_gestor_ialocal().ciclo(
                    prompt, estado_previo, self.conversor_psn, _rep)
            except Exception as e_local:
                print(f"  [IALocal] {e_local} -> probando OpenRouter")
        if not respuesta and self.cliente_iafree and self.cliente_iafree.esta_autenticado():
            resp_txt, mod_id, lat = self.cliente_iafree.generar_respuesta( prompt=prompt, contexto_neuronal=estado_previo, stream_en_vivo=False, )
            if resp_txt and not resp_txt.startswith("[IAFREE]"):
                respuesta = resp_txt
                modelo_usado = mod_id
                latencia_llm = lat
        if not respuesta:
            respuesta = self._generar_reflejo_interno(prompt, estado_previo)
        elif _RPLC_DISPONIBLE and reprocesar_con_metricas is not None and not modelo_usado.startswith("IALocal"):
            try:
                respuesta, _ = reprocesar_con_metricas(respuesta, estado_previo)
            except Exception:
                pass
        if EstiloTerminalLucIA:
            formatear_respuesta_lucia(respuesta, modelo=modelo_usado)
        else:
            print(f"\nLucIA [{modelo_usado}] ▶ {respuesta}\n")
        if _VOZ_DISPONIBLE and _hablar_voz is not None:
            _hablar_voz(respuesta, esperar=False)
        duracion_ms = (time.perf_counter() - t_inicio) * 1000.0
        cid_ipfs = "--"
        try:
            res_chain = self.servidor_bks.registrar_aprendizaje_neural( prompt=prompt, respuesta=respuesta, modelo=f"LucIA-{modelo_usado}" )
            cid_ipfs = res_chain.get("ipfs_cid", "--")
        except Exception as ex_bks:
            print(f"\033[38;5;203m  [Blockchain Error] {ex_bks}\033[0m")
        if self.turno_actual % 3 == 0:
            bloque_nuevo = self.servidor_bks.minar_transacciones_pendientes()
            if bloque_nuevo:
                if EstiloTerminalLucIA:
                    EstiloTerminalLucIA.renderizar_notificacion_bloque(bloque_nuevo.indice, bloque_nuevo.hash_bloque)
                else:
                    print(f"  [Bloque #{bloque_nuevo.indice} Minado] {bloque_nuevo.hash_bloque[:24]}...")
        self._imprimir_telemetria_turno(duracion_ms, cid_ipfs, modelo_usado)
    def _generar_reflejo_interno(self, prompt: str, info: Dict[str, Any]) -> str:
        tono = info.get("tono_cognitivo", "reflexivo")
        val = info.get("estado_emocional", 0.0)
        norma = info.get("norma_delta_aplicada", 0.0)
        return ( f"Estimulo cognitivo asimilado en red distribuida. Tono: **{tono}** | " f"Valencia afectiva: `{val:+.3f}` | Delta sinaptico: `{norma:.5f}`.\n" "Pesos neuronales sincronizados y activos en persistencia `PSNRL`." )
    def _imprimir_telemetria_turno(self, duracion_ms: float, cid: str, modelo_id: str) -> None:
        deriva = self.conversor_psn.deriva_acumulada
        neuronas = len(self.conversor_psn.neuronas)
        txs_espera = len(self.servidor_bks.transacciones_pendientes)
        if EstiloTerminalLucIA:
            badge_turno( turno=self.turno_actual, dt=duracion_ms, mod=modelo_id, der=deriva, neu=neuronas, cid=cid, txs=txs_espera, )
        else:
            print(f"  Turno #{self.turno_actual} | {duracion_ms:.0f}ms | Mod: {modelo_id} | Deriva: {deriva:.4f} | CID: {cid[:16]}")
    def mostrar_estado_sistema(self) -> None:
        """Despliega en terminal un reporte detallado del estado de todos los componentes."""
        valida, err = self.servidor_bks.validar_cadena()
        cadena_str = "VALIDA E INMUTABLE" if valida else f"ERROR: {err}"
        mod_act = self.cliente_iafree.gestor.obtener_modelo_activo()["id"] if self.cliente_iafree else "N/A"
        metricas = self.cliente_iafree.obtener_metricas_consumo() if self.cliente_iafree else {}
        costo_usd = float(metricas.get("costo_acumulado_usd", 0.0))
        datos = { "Cadena de Bloques": f"{len(self.servidor_bks.cadena)} bloques [{cadena_str}]", "Txs en Espera": str(len(self.servidor_bks.transacciones_pendientes)), "Neuronas Activas": f"{len(self.conversor_psn.neuronas)} (ENRN, RF_EN, RF_SL, RNP, SLRN)", "Deriva Muon": f"{self.conversor_psn.deriva_acumulada:.6f}", "Turnos de Sesion": str(self.turno_actual), "Modelo Activo": mod_act, "Costo Acumulado": f"${costo_usd:.2f} USD (100% Free)", }
        if EstiloTerminalLucIA:
            EstiloTerminalLucIA.renderizar_tarjeta_sistema("ESTADO INTEGRAL DE LUCIA (2026)", datos)
            print()
        else:
            for k, v in datos.items():
                print(f"  {k}: {v}")
    def ejecutar_comando_rfprmn(self, comando: str) -> bool:
        if not _RFPRMN_DISPONIBLE or _RFPRMN is None:
            print(f"RFPRMN no esta disponible en: {STM_RFC_DIR / 'RFPRMN.py'}")
            return True
        if comando == "cerrar":
            _RFPRMN.cerrar()
            print("\n\033[38;5;214mIniciando protocolo de salida y sincronizacion...\033[0m")
            return False
        nombre_funcion = _COMANDOS_RFPRMN.get(comando, "")
        funcion = getattr(_RFPRMN, nombre_funcion, None)
        if not callable(funcion):
            print(f"El comando RFPRMN '{comando}' no esta disponible.")
            return True
        try:
            funcion()
        except Exception as error:
            print(f"Error ejecutando '{comando}': {error}")
        return True
    def ejecutar_bucle_interactivo(self) -> None:
        """Bucle principal de recepcion e interpretacion de comandos de consola."""
        while self.activa:
            try:
                entrada = input("\033[38;5;214mLucIA> Tu:\033[0m ").strip()
            except (KeyboardInterrupt, EOFError):
                print()
                break
            if not entrada:
                continue
            if entrada.lower() in ("salir", "exit", "quit"):
                print("\n\033[38;5;214mIniciando protocolo de salida y sincronizacion...\033[0m")
                break
            if entrada.lower() in ("estado", "status"):
                self.mostrar_estado_sistema()
                continue
            if entrada.lower() in ("ayuda", "help"):
                if EstiloTerminalLucIA:
                    panel_ayuda_comandos()
                continue
            if entrada.lower() in ("modelos", "models", "free"):
                self.listar_modelos_gratuitos()
                continue
            if entrada.lower() in ("minar", "mine"):
                blk = self.servidor_bks.minar_transacciones_pendientes()
                if blk:
                    if EstiloTerminalLucIA:
                        EstiloTerminalLucIA.renderizar_notificacion_bloque(blk.indice, blk.hash_bloque)
                    else:
                        print(f"Bloque #{blk.indice} minado: {blk.hash_bloque}")
                else:
                    print("  No hay transacciones pendientes para minar.")
                continue
            if entrada.lower() in ("api", "apikey", "registrar api"):
                try:
                    from STM_SGR.vault_openrouter import registrar_key_usuario
                    registrar_key_usuario()
                except Exception as e_api:
                    print(f"  No se pudo registrar: {e_api}")
                continue
            if entrada.lower().startswith("modelo "):
                nuevo_mod = entrada[7:].strip()
                if self.cliente_iafree and self.cliente_iafree.gestor.seleccionar_por_id(nuevo_mod):
                    print(f"  \033[38;5;48mModelo gratuito seleccionado: {nuevo_mod}\033[0m")
                else:
                    print(f"  \033[38;5;214mModelo '{nuevo_mod}' no encontrado en catalogo :free.\033[0m")
                continue
            comando_rfprmn = entrada.lower()
            if comando_rfprmn in _COMANDOS_RFPRMN:
                if not self.ejecutar_comando_rfprmn(comando_rfprmn):
                    break
                continue
            self.procesar_turno_dialogo(entrada)
        self.activa = False
    def cerrar_sistema(self) -> None:
        """Cierre ordenado: minado de bloques pendientes y persistencia a IPFS."""
        with self.lock:
            if not self.servidor_bks:
                return
            print("\n\033[38;5;51m" + "=" * 76 + "\033[0m")
            print("  \033[1;37mCONSOLIDANDO ESTADO COGNITIVO Y PERSISTENCIA FINAL...\033[0m")
            try:
                blk = self.servidor_bks.minar_transacciones_pendientes()
                if blk:
                    print(f"  Bloque final consolidado: \033[38;5;220m{blk.hash_bloque[:28]}...\033[0m")
            except Exception:
                pass
            try:
                self.servidor_bks.cerrar_sesion_y_subir_ipfs()
                print("  \033[38;5;48mSincronizacion IPFS de bloques y pesos completada exitosamente.\033[0m")
            except Exception as e_close:
                print(f"  \033[38;5;214mCierre IPFS: {e_close}\033[0m")
            print("\033[38;5;51m" + "=" * 76 + "\033[0m\n")
            self.servidor_bks = None
            try:
                from STM_CH.rutas import limpiar as _stmch_limpiar
                res = _stmch_limpiar()
                print(f"  STM_CH limpiado: {res.get('borrados', 0)} artefactos.")
            except Exception as e_stmch:
                print(f"  STM_CH: {e_stmch}")
class ContextoOrquestadorLucIA:
    """Gestor de contexto para pruebas, evaluacion o invocacion programatica."""
    def __init__(self) -> None:
        self.orquestador = OrquestadorSistemaLucIA()
    def __enter__(self) -> OrquestadorSistemaLucIA:
        if not self.orquestador.inicializar_subsistemas():
            raise RuntimeError("Fallo durante la inicializacion de subsistemas en LucIA")
        return self.orquestador
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        self.orquestador.cerrar_sistema()
        return False
def obtener_diagnostico_orquestador() -> Dict[str, Any]:
    """Genera un reporte estructural para validar la salud de todo el stack."""
    try:
        from LC.celebro.CMFG.SBSTM.IAFREE import get_cliente_iafree  # pyrefly: ignore[missing-import]
    except ImportError:
        from SBSTM.IAFREE import get_cliente_iafree
    cliente_free = get_cliente_iafree()
    return { "orquestador_version": "2026.3.1", "iafree_activo": cliente_free.esta_autenticado(), "modelos_gratuitos_total": len(cliente_free.gestor.listar_modelos()), "modelo_predeterminado": cliente_free.gestor.obtener_modelo_activo()["id"], "costo_acumulado": cliente_free.obtener_metricas_consumo()["costo_acumulado_usd"], "timestamp": time.time(), }
def main() -> int:
    """Punto de arranque del orquestador unificado mainLCSTM."""
    orquestador = OrquestadorSistemaLucIA()
    if not orquestador.inicializar_subsistemas():
        sys.stderr.write("[mainLCSTM] Fallo durante la inicializacion de subsistemas.\n")
        return 1
    try:
        orquestador.ejecutar_bucle_interactivo()
        try:
            from STM_CH.rutas import limpiar as _lf
            _lf()
        except Exception:
            pass
        return 0
    except Exception as err:
        sys.stderr.write(f"[mainLCSTM] Error no controlado: {err}\n")
        return 1
if __name__ == "__main__":
    sys.exit(main())
