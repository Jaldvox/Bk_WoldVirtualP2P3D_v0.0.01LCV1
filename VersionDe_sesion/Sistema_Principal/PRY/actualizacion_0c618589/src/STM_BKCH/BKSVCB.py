"""
BKSVCB.py - Blockchain Server & Consenso Neuronal de Celebro (Arquitectura 2026)
================================================================================
Servidor de bloques inmutable y libro mayor distribuido para WoldVirtualP2P3D:
  - Transforma blockchain_ledger.json en pesos neuronales nada mas arrancar.
  - Mantiene activa la sincronizacion y actualizacion neuronal durante la sesion.
  - Al cerrar sesion, sube atomicamente el estado y pesos de PSNRL a IPFS.
  - Estructura de bloques criptograficos con hashing doble SHA-256 y Merkle Root.
  - Visualizacion de hashes unicos de cada bloque en terminal en tiempo real.
  - Consenso PoNL (Proof of Neural Learning) y servidor HTTP REST/JSON-RPC.
"""
from __future__ import annotations
import os, sys, time, json, hashlib, logging, threading, atexit
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

logging.basicConfig(level=logging.CRITICAL)
for _lg in ("", "WoldVirtualP2P3D", "RFENRN1", "LC", "urllib3", "ENRN", "SLRN", "RNP"):
    logging.getLogger(_lg).setLevel(logging.CRITICAL)

__version__ = "2026.3.1"
__server_name__ = "CelebroBlockchainServer-BKSVCB"

CELEBRO_DIR: Path = Path(__file__).parent.resolve()
ROOT_DIR: Path = CELEBRO_DIR.parent.parent.resolve()
try:
    from STM_CH.rutas import PSNRL_DIR as PSNRL_DIR, LEDGER_PATH as LEDGER_PATH
except ImportError:
    try:
        from STM_CH.rutas import PSNRL_DIR, LEDGER_PATH
    except ImportError:
        PSNRL_DIR: Path = CELEBRO_DIR / "PSNRL"
        LEDGER_PATH: Path = CELEBRO_DIR / "blockchain_ledger.json"

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from STM_BKCH.compat_local import get_conversor_pesos, ConversorRespuestaPesos, get_ipfs_manager, IPFSManager
from SBSTM.IALOCAL import (
    ejecutar_flujo_sesion_local as _ialocal_pre_sesion,
    cerrar_sesion_ialocal as _ialocal_cierre,
    MODELOS_DIR as _IALOCAL_MODELOS_DIR,
)
_IALOCAL_DISPONIBLE = True


class BloqueNeuronal:
    """Representa un bloque inmutable de la cadena de bloques de Celebro."""

    def __init__(
        self, indice: int, hash_previo: str, transacciones: List[Dict[str, Any]],
        estado_neuronal: Optional[Dict[str, Any]] = None, dificultad: int = 2,
        nonce: int = 0, hash_existente: Optional[str] = None,
        merkle_root_existente: Optional[str] = None,
    ) -> None:
        self.indice = indice
        self.hash_previo = hash_previo
        self.transacciones = transacciones
        self.estado_neuronal = estado_neuronal or {}
        self.dificultad = dificultad
        self.nonce = nonce
        self.merkle_root = merkle_root_existente or self.calcular_merkle_root()
        self.hash_bloque = hash_existente or self.calcular_hash()

    def calcular_merkle_root(self) -> str:
        """Calcula la raiz de Merkle de las transacciones incluidas en el bloque."""
        if not self.transacciones:
            return hashlib.sha256(b"empty_block_celebro_2026").hexdigest()
        hashes = [
            hashlib.sha256(json.dumps(tx, sort_keys=True).encode("utf-8")).hexdigest()
            for tx in self.transacciones
        ]
        while len(hashes) > 1:
            if len(hashes) % 2 != 0: hashes.append(hashes[-1])
            nuevos = []
            for i in range(0, len(hashes), 2):
                combinado = (hashes[i] + hashes[i + 1]).encode("utf-8")
                nuevos.append(hashlib.sha256(combinado).hexdigest())
            hashes = nuevos
        return hashes[0]

    def calcular_hash(self) -> str:
        """Calcula el doble hash SHA-256 inmutable de la cabecera."""
        cabecera = {
            "indice": self.indice, "hash_previo": self.hash_previo,
            "merkle_root": self.merkle_root, "dificultad": self.dificultad, "nonce": self.nonce,
        }
        serializado = json.dumps(cabecera, sort_keys=True).encode("utf-8")
        return hashlib.sha256(hashlib.sha256(serializado).digest()).hexdigest()

    def minar_bloque(self, max_iteraciones: int = 500000) -> bool:
        """Realiza el minado por prueba de trabajo/aprendizaje adaptativo."""
        prefijo = "0" * self.dificultad
        self.merkle_root = self.calcular_merkle_root()
        for _ in range(max_iteraciones):
            h = self.calcular_hash()
            if h.startswith(prefijo):
                self.hash_bloque = h
                return True
            self.nonce += 1
        return False

    def to_dict(self) -> Dict[str, Any]:
        """Exporta el bloque completo a un diccionario serializable."""
        return {
            "indice": self.indice, "hash_bloque": self.hash_bloque,
            "hash_previo": self.hash_previo, "merkle_root": self.merkle_root,
            "dificultad": self.dificultad, "nonce": self.nonce,
            "total_transacciones": len(self.transacciones),
            "transacciones": self.transacciones, "estado_neuronal": self.estado_neuronal,
        }


class CelebroBlockchain:
    """Motor central de la Blockchain neuronal con transformacion sinaptica e IPFS."""

    def __init__(self, dificultad: int = 2) -> None:
        self.dificultad = dificultad
        self.cadena: List[BloqueNeuronal] = []
        self.transacciones_pendientes: List[Dict[str, Any]] = []
        self.lock = threading.Lock()
        self.conversor: ConversorRespuestaPesos = get_conversor_pesos()
        self.ipfs_mgr: IPFSManager = get_ipfs_manager()
        self.actualizador_activo = False
        self._hilo_actualizador: Optional[threading.Thread] = None
        self._inicializar_o_cargar_cadena()

    def _inicializar_o_cargar_cadena(self) -> None:
        """Carga el ledger previo desde disco o forja el bloque genesis."""
        if LEDGER_PATH.exists():
            try:
                with open(LEDGER_PATH, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                bloques_cargados = []
                for b in raw.get("cadena", []):
                    bloque = BloqueNeuronal(
                        indice=b["indice"], hash_previo=b["hash_previo"],
                        transacciones=b.get("transacciones", []),
                        estado_neuronal=b.get("estado_neuronal", {}),
                        dificultad=b.get("dificultad", self.dificultad), nonce=b.get("nonce", 0),
                        hash_existente=b.get("hash_bloque"),
                        merkle_root_existente=b.get("merkle_root"),
                    )
                    bloques_cargados.append(bloque)
                if bloques_cargados:
                    self.cadena = bloques_cargados
                    return
            except Exception: pass
        self._crear_bloque_genesis()

    def _crear_bloque_genesis(self) -> None:
        """Forja el bloque 0 inicial con metadatos del cluster Celebro 2026."""
        estado_inicial = {
            "total_neuronas": len(self.conversor.neuronas),
            "capas": ["ENRN", "RF_SL", "RF_EN", "RNP", "SLRN"],
            "version_cluster": __version__, "consenso": "Proof-of-Neural-Learning (PoNL)",
        }
        tx_genesis = {
            "tipo": "GENESIS_COGNITIVO", "emisor": "SYSTEM_CELEBRO", "receptor": "CLUSTER_P2P",
            "descripcion": "Arranque formal de la blockchain neuronal WoldVirtualP2P3D",
            "hash_semilla": hashlib.sha256(b"genesis_seed_2026").hexdigest(),
        }
        genesis = BloqueNeuronal(
            indice=0, hash_previo="0" * 64, transacciones=[tx_genesis],
            estado_neuronal=estado_inicial, dificultad=1,
        )
        genesis.minar_bloque()
        self.cadena = [genesis]
        self.guardar_ledger()

    def obtener_ultimo_bloque(self) -> BloqueNeuronal:
        """Retorna el ultimo bloque confirmado de la cadena."""
        return self.cadena[-1]

    def agregar_transaccion(
        self, tipo: str, emisor: str, receptor: str, datos: Dict[str, Any],
        cid_ipfs: Optional[str] = None,
    ) -> int:
        """Anade una nueva transaccion al pool de espera con validacion criptografica."""
        with self.lock:
            tx = {
                "id_tx": hashlib.sha256(f"{emisor}:{receptor}:{time.time_ns()}".encode()).hexdigest()[:16],
                "tipo": tipo, "emisor": emisor, "receptor": receptor,
                "cid_ipfs": cid_ipfs or "", "datos": datos,
            }
            self.transacciones_pendientes.append(tx)
            return self.obtener_ultimo_bloque().indice + 1

    def registrar_aprendizaje_neural(self, prompt: str, respuesta: str, modelo: str) -> Dict[str, Any]:
        """Procesa consulta a traves de las 50 neuronas activas y emite transaccion."""
        t0 = time.perf_counter()
        info_neural = self.conversor.asimilar_respuestas_y_calcular_sintesis(prompt, respuesta, modelo)
        res_ipfs = self.ipfs_mgr.almacenar_pesos(
            origen=json.dumps(info_neural).encode("utf-8"),
            nombre_modelo=f"tx_neural_{int(time.time())}", eliminar_local=False,
        )
        cid = res_ipfs.get("cid", "")
        datos_tx = {
            "modelo": modelo, "prompt_resumen": prompt[:80],
            "norma_delta": info_neural.get("norma_delta_aplicada", 0.0),
            "estado_emocional": info_neural.get("estado_emocional", 0.0),
            "tono": info_neural.get("tono_cognitivo", "neutral"),
            "duracion_ms": round((time.perf_counter() - t0) * 1000.0, 2),
        }
        self.agregar_transaccion(
            tipo="ACTUALIZACION_SINAPTICA", emisor="CONVERSOR_PSNRCV",
            receptor="CELEBRO_P2P", datos=datos_tx, cid_ipfs=cid,
        )
        return {"neural": info_neural, "ipfs_cid": cid, "transacciones_espera": len(self.transacciones_pendientes)}

    def minar_transacciones_pendientes(self, minero_id: str = "CelebroLocalNode") -> Optional[BloqueNeuronal]:
        """Empaqueta transacciones en espera, acopla estado de las 50 neuronas y mina bloque."""
        with self.lock:
            if not self.transacciones_pendientes: return None
            ultimo = self.obtener_ultimo_bloque()
            resumen_neuronas = {
                "activas": len(self.conversor.neuronas),
                "deriva_acumulada": round(self.conversor.deriva_acumulada, 6),
                "pasos_sesion": self.conversor.pasos_sesion, "minero": minero_id,
            }
            txs = list(self.transacciones_pendientes)
            self.transacciones_pendientes.clear()
            nuevo_bloque = BloqueNeuronal(
                indice=ultimo.indice + 1, hash_previo=ultimo.hash_bloque,
                transacciones=txs, estado_neuronal=resumen_neuronas, dificultad=self.dificultad,
            )
            if nuevo_bloque.minar_bloque():
                self.cadena.append(nuevo_bloque)
                self.guardar_ledger()
                return nuevo_bloque
            return None

    def transformar_ledger_a_pesos_neuronales(self) -> Dict[str, Any]:
        """Transduce todos los bloques y hashes criptograficos a pesos neuronales en PSNRL."""
        if not LEDGER_PATH.exists(): return {"total_bloques": 0, "norma_acumulada": 0.0, "npz_pesos": "", "json_meta": ""}
        with open(LEDGER_PATH, "r", encoding="utf-8") as f: ledger_data = json.load(f)
        bloques = ledger_data.get("cadena", [])
        norma_acum = 0.0
        # Propagar cada bloque directamente sin persistir en cada iteracion
        for b in bloques:
            sintesis = f"Bloque {b['indice']} | Hash: {b['hash_bloque']} | Merkle: {b['merkle_root']}"
            try:
                v = self.conversor.encoder.encode(sintesis)
                act, _ = self.conversor._propagar_todas_las_neuronas(v)
                _, norma_app, _ = self.conversor._actualizar_pesos_en_todas_las_neuronas(v, act, factor=1.0)
                norma_acum += norma_app
            except Exception: pass
        # Persistir UNA sola vez al final de la transformacion completa
        npz_path, json_path = self.conversor.persistir_pesos_en_psnrl(etiqueta="blockchain_ledger")
        print(f"  [PSNRL] {len(bloques)} bloques -> pesos neuronales | Norma total: {round(norma_acum,5)}")
        return {
            "total_bloques": len(bloques), "norma_acumulada": round(norma_acum, 5),
            "npz_pesos": str(npz_path.name), "json_meta": str(json_path.name),
        }

    def _bucle_actualizacion_continua(self) -> None:
        """Hilo de fondo que mantiene actualizando la red neuronal mientras la sesion este activa."""
        while self.actualizador_activo:
            time.sleep(12.0)
            if not self.actualizador_activo: break
            try:
                ult = self.obtener_ultimo_bloque()
                self.conversor.procesar_consulta_a_pesos(f"Latido Bloque #{ult.indice} Hash:{ult.hash_bloque[:16]}")
            except Exception: pass

    def arrancar_actualizador_red(self) -> None:
        """Inicia el sincronizador dinamico de pesos durante la sesion."""
        if not self.actualizador_activo:
            self.actualizador_activo = True
            self._hilo_actualizador = threading.Thread(target=self._bucle_actualizacion_continua, daemon=True)
            self._hilo_actualizador.start()

    def detener_actualizador_red(self) -> None:
        """Detiene el bucle de actualizacion."""
        self.actualizador_activo = False

    def cerrar_sesion_y_subir_ipfs(self) -> Dict[str, Any]:
        """Hook de cierre: persiste pesos, sube ledger + PSNRL a IPFS y borra ledger local."""
        self.detener_actualizador_red()
        # Checkpoint final de pesos antes de cerrar
        self.conversor.persistir_pesos_en_psnrl(etiqueta="cierre_sesion_blockchain")
        cid_ledger = None
        ledger_borrado = False
        if LEDGER_PATH.exists():
            try:
                # eliminar_local=True: borra si hay daemon real confirmado
                res_l = self.ipfs_mgr.almacenar_pesos(
                    origen=LEDGER_PATH, nombre_modelo="blockchain_ledger_final", eliminar_local=True,
                )
                cid_ledger = res_l.get("cid")
                ledger_borrado = res_l.get("borrado_local", False)
            except Exception: pass
            # Borrado garantizado: aunque no haya daemon, eliminar el ledger local
            if not ledger_borrado and LEDGER_PATH.exists():
                try:
                    LEDGER_PATH.unlink()
                    ledger_borrado = True
                except Exception: pass
        # Subir y limpiar PSNRL
        res_psnrl = self.ipfs_mgr.subir_y_limpiar_psnrl(forzar_borrado_sin_daemon=True)
        return {
            "cid_ledger": cid_ledger, "ledger_borrado": ledger_borrado,
            "cids_psnrl": res_psnrl.get("cids", []), "borrados": res_psnrl.get("borrados", []),
        }

    def validar_cadena(self) -> Tuple[bool, Optional[str]]:
        """Verifica la integridad criptografica total de hashes, enlaces y Merkle roots."""
        prefijo = "0" * self.dificultad
        for i in range(1, len(self.cadena)):
            act, prev = self.cadena[i], self.cadena[i - 1]
            if act.hash_previo != prev.hash_bloque: return False, f"Ruptura en #{act.indice}"
            if act.calcular_hash() != act.hash_bloque: return False, f"Hash corrupto en #{act.indice}"
            if act.calcular_merkle_root() != act.merkle_root: return False, f"Merkle invalido en #{act.indice}"
            if not act.hash_bloque.startswith(prefijo) and act.indice > 0:
                return False, f"PoW insuficiente en #{act.indice}"
        return True, None

    def guardar_ledger(self) -> bool:
        """Persiste toda la cadena en el archivo JSON ledger local."""
        data = {
            "version": __version__, "servidor": __server_name__,
            "total_bloques": len(self.cadena), "dificultad": self.dificultad,
            "cadena": [b.to_dict() for b in self.cadena],
        }
        try:
            with open(LEDGER_PATH, "w", encoding="utf-8") as f: json.dump(data, f, indent=2, ensure_ascii=False)
            return True
        except Exception: return False

    def mostrar_cadena_hashes_terminal(self) -> None:
        """Muestra en terminal la cadena con el hash unico de cada bloque sin fechas."""
        print("\033[1;36m" + "=" * 74 + "\033[0m")
        print("  \033[1;32mLIBRO MAYOR DISTRIBUIDO - HASHES UNICOS DE BLOQUES (2026)\033[0m")
        print("\033[1;36m" + "=" * 74 + "\033[0m")
        for b in self.cadena:
            h_prev = b.hash_previo[:18] + "..." if len(b.hash_previo) > 18 else b.hash_previo
            print(f"  [Bloque #{b.indice:<2}] \033[1;33mHASH: {b.hash_bloque}\033[0m")
            print(f"            Prev: {h_prev} | Merkle: {b.merkle_root[:16]}... | Txs: {len(b.transacciones)}")
        print("\033[1;36m" + "-" * 74 + "\033[0m")


try:
    from STM_BKCH.bksvcb_net import (
        BlockchainHTTPHandler,
        BlockchainServerDaemon,
        get_blockchain_server,
        iniciar_servidor_blockchain,
    )
except ImportError:
    try:
        from .bksvcb_net import (
            BlockchainHTTPHandler,
            BlockchainServerDaemon,
            get_blockchain_server,
            iniciar_servidor_blockchain,
        )
    except ImportError:
        pass


