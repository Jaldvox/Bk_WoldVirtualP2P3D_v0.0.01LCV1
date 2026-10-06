"""Shims locales BKSVCB: conversor PSNRCV e IPFSManager mínimos.

Evitan el crash cuando los módulos originales LC.celebro no existen en src/.
El conversor expone la interfaz usada por mainLCSTM y BKSVCB:
neuronas, deriva_acumulada, procesar_consulta_a_pesos().
"""
from __future__ import annotations
import hashlib
import time
from typing import Any, Dict, List, Optional


class ConversorRespuestaPesos:
    """Conversor local de 50 neuronas (5 capas x 10)."""

    def __init__(self) -> None:
        self.neuronas: List[Dict[str, Any]] = [
            {"id": f"{capa}{i}", "peso": 0.0}
            for capa in ("ENRN", "RF_EN", "RF_SL", "RNP", "SLRN")
            for i in range(1, 11)
        ]
        self.deriva_acumulada: float = 0.0
        self.pasos_sesion: int = 0

    def asimilar_respuestas_y_calcular_sintesis(self, prompt: str, respuesta: str,
                                                     modelo: str = "") -> Dict[str, Any]:
        info = self.procesar_consulta_a_pesos(f"{prompt} || {respuesta}")
        info["modelo"] = modelo
        info["sintesis"] = f"Turno asimilado en {len(self.neuronas)} neuronas."
        info["pasos_sesion"] = self.pasos_sesion
        return info

    def persistir_pesos_en_psnrl(self, etiqueta: str = "checkpoint"):
        import json
        from pathlib import Path
        try:
            from STM_CH.rutas import PSNRL_DIR
        except ImportError:
            PSNRL_DIR = Path("STM_CH") / "PSNRL"
        PSNRL_DIR.mkdir(parents=True, exist_ok=True)
        pesos = [n["peso"] for n in self.neuronas]
        npz_p = PSNRL_DIR / f"{etiqueta}.npz"
        js_p = PSNRL_DIR / f"{etiqueta}.json"
        try:
            import numpy as np
            np.savez(str(npz_p), pesos=np.array(pesos, dtype=np.float32))
        except Exception:
            npz_p.write_bytes(repr(pesos).encode("utf-8", "replace"))
        js_p.write_text(json.dumps({"etiqueta": etiqueta, "deriva": self.deriva_acumulada,
                                    "neuronas": len(self.neuronas)}, ensure_ascii=False), encoding="utf-8")
        return (npz_p, js_p)

    def procesar_consulta_a_pesos(self, prompt: str) -> Dict[str, Any]:
        try:
            import numpy as np
            h = int(hashlib.sha256(prompt.encode()).hexdigest()[:8], 16)
            matriz = np.full((5, 10), h, dtype=np.float32)
            val = float(np.mean(matriz % 2000) / 1000.0 - 1.0)
            norma = round(float(np.mean(matriz % 100000) / 100000.0), 5)
        except ImportError:
            h = int(hashlib.sha256(prompt.encode()).hexdigest()[:8], 16)
            val = ((h % 2000) / 1000.0) - 1.0
            norma = round((h % 100000) / 100000.0, 5)
        self.deriva_acumulada = round(self.deriva_acumulada + norma * 0.001, 6)
        self.pasos_sesion += 1
        tonos = ("reflexivo", "analitico", "creativo", "pragmatico")
        return {
            "tono_cognitivo": tonos[h % len(tonos)],
            "estado_emocional": round(val, 3),
            "norma_delta_aplicada": norma,
        }


_conversor_inst: Optional[ConversorRespuestaPesos] = None


def get_conversor_pesos() -> ConversorRespuestaPesos:
    global _conversor_inst
    if _conversor_inst is None:
        _conversor_inst = ConversorRespuestaPesos()
    return _conversor_inst


class IPFSManager:
    """IPFS no-op: devuelve CIDs sintéticos sin requerir daemon."""

    def almacenar_pesos(self, datos: Any = None, origen: Any = None,
                          nombre_modelo: str = "", eliminar_local: bool = False,
                          **kw: Any) -> Dict[str, Any]:
        base = datos if datos is not None else origen
        sello = hashlib.sha256(repr((str(base)[:200], time.time())).encode()).hexdigest()[:16]
        cid = f"QmLocal{sello}"
        return {"cid": cid, "ipfs_cid": cid, "ok": True, "borrado_local": False}

    def subir_y_limpiar_psnrl(self, forzar_borrado_sin_daemon: bool = False) -> Dict[str, Any]:
        borrados: list = []
        try:
            from STM_CH.rutas import PSNRL_DIR
            if PSNRL_DIR.exists():
                for hijo in list(PSNRL_DIR.iterdir()):
                    try:
                        hijo.unlink(missing_ok=True)
                        borrados.append(hijo.name)
                    except Exception:
                        pass
        except Exception:
            pass
        return {"ok": True, "modo": "local-sin-daemon", "cids": [], "borrados": borrados}


_ipfs_inst: Optional[IPFSManager] = None


def get_ipfs_manager() -> IPFSManager:
    global _ipfs_inst
    if _ipfs_inst is None:
        _ipfs_inst = IPFSManager()
    return _ipfs_inst
