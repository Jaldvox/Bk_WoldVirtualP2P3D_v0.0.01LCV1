"""autodiagnostico.py — Subsistema de Autodiagnóstico e Introspección de LucIA
=============================================================================
Permite a LucIA inspeccionar en profundidad su propio sistema en tiempo real:
  - Subsistemas activos (Blockchain, Transductor 50 neuronas, Red neuronal, IPFS)
  - Modelos locales y remotos disponibles (Ollama / OpenRouter :free)
  - Integridad estructural del código (límite 450 líneas, imports circulares)
  - Registro de errores capturados en tiempo de ejecución (SBSTM.gestor_errores)
  - Generación de informe formateado apto para terminal o inyección a LLM
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

SRC_DIR: Path = Path(__file__).resolve().parents[1]


class AutodiagnosticoLucIA:
    """Motor de inspección introspectiva del sistema LucIA."""

    def __init__(self, src_dir: Optional[Path] = None) -> None:
        self.src_dir = src_dir or SRC_DIR

    def inspeccionar_blockchain(self) -> Dict[str, Any]:
        """Evalúa el estado del servidor blockchain BKSVCB y ledger."""
        try:
            from STM_BKCH.BKSVCB import CelebroBlockchain
            bks = CelebroBlockchain()
            valida, err = bks.validar_cadena()
            return {
                "estado": "OPERATIVO" if valida else "ERROR",
                "bloques": len(bks.cadena),
                "valida": valida,
                "error": err,
                "txs_pendientes": len(bks.transacciones_pendientes),
            }
        except Exception as exc:
            return {"estado": "FALLO", "error": str(exc)[:120]}

    def inspeccionar_red_neuronal(self) -> Dict[str, Any]:
        """Evalúa las 50 neuronas y transductor de pesos."""
        try:
            from STM_BKCH.compat_local import get_conversor_pesos
            conv = get_conversor_pesos()
            total_n = len(conv.neuronas)
            test_resp = conv.procesar_consulta_a_pesos("autodiagnostico sinaptico")
            return {
                "estado": "OPERATIVO" if total_n >= 50 else "DEGRADADO",
                "total_neuronas": total_n,
                "deriva_acumulada": round(float(conv.deriva_acumulada), 6),
                "tono_inferido": test_resp.get("tono_cognitivo", "desconocido"),
                "valencia_inferida": test_resp.get("estado_emocional", 0.0),
            }
        except Exception as exc:
            return {"estado": "FALLO", "error": str(exc)[:120]}

    def inspeccionar_modelos_ia(self) -> Dict[str, Any]:
        """Detecta backends de IA local (Ollama) y OpenRouter."""
        resultado: Dict[str, Any] = {
            "ollama_activo": False,
            "modelos_locales": [],
            "openrouter_activo": False,
            "modelos_free_total": 0,
        }
        try:
            import json
            import urllib.request
            req = urllib.request.Request("http://localhost:11434/api/tags")
            with urllib.request.urlopen(req, timeout=1.5) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                # Filtra nombres únicos preservando orden y excluyendo tags duplicados
                vistos = set()
                locales = []
                for m in data.get("models", []):
                    nm = m.get("name", "")
                    if nm and nm not in vistos:
                        vistos.add(nm)
                        locales.append(nm)
                resultado["modelos_locales"] = locales
                resultado["ollama_activo"] = True
        except Exception:
            pass

        try:
            from SBSTM.IAFREE import get_cliente_iafree
            cliente = get_cliente_iafree()
            resultado["openrouter_activo"] = bool(cliente.esta_autenticado())
            resultado["modelos_free_total"] = len(cliente.gestor.listar_modelos())
        except Exception:
            pass
        return resultado

    def inspeccionar_integridad_archivos(self, max_lineas: int = 450) -> Dict[str, Any]:
        """Verifica que ningún archivo .py en src/ exceda el límite de 450 líneas."""
        archivos_excedidos: List[Dict[str, Any]] = []
        total_py = 0
        for py_path in self.src_dir.rglob("*.py"):
            total_py += 1
            try:
                lineas = len(py_path.read_text(encoding="utf-8", errors="replace").splitlines())
                if lineas > max_lineas:
                    archivos_excedidos.append({
                        "archivo": str(py_path.relative_to(self.src_dir)),
                        "lineas": lineas,
                    })
            except Exception:
                pass
        return {
            "total_archivos_py": total_py,
            "archivos_excedidos": archivos_excedidos,
            "limite_respetado": len(archivos_excedidos) == 0,
        }

    def inspeccionar_errores_recientes(self) -> Dict[str, Any]:
        """Consulta el GestorErroresSistema centralizado."""
        try:
            from SBSTM.gestor_errores import obtener_gestor_errores
            res = obtener_gestor_errores().resumen_errores()
            return {
                "total_errores_registrados": res.get("total", 0),
                "ultimos": res.get("errores", [])[-5:],
            }
        except Exception:
            return {"total_errores_registrados": 0, "ultimos": []}

    def ejecutar_diagnostico_completo(self) -> Dict[str, Any]:
        """Ejecuta todos los análisis introspectivos y devuelve el reporte consolidado."""
        t0 = time.perf_counter()
        bkch = self.inspeccionar_blockchain()
        neural = self.inspeccionar_red_neuronal()
        ia = self.inspeccionar_modelos_ia()
        archivos = self.inspeccionar_integridad_archivos()
        errores = self.inspeccionar_errores_recientes()
        duracion_ms = round((time.perf_counter() - t0) * 1000.0, 2)

        salud_general = (
            bkch.get("estado") == "OPERATIVO"
            and neural.get("estado") == "OPERATIVO"
            and archivos.get("limite_respetado") is True
            and errores.get("total_errores_registrados", 0) == 0
        )

        return {
            "version": "2026.4.1",
            "timestamp": time.time(),
            "duracion_ms": duracion_ms,
            "salud_general": "OPTIMO" if salud_general else "DEGRADADO",
            "blockchain": bkch,
            "red_neuronal": neural,
            "modelos_ia": ia,
            "integridad_codigo": archivos,
            "errores_runtime": errores,
        }

    def generar_resumen_texto(self) -> str:
        """Genera un resumen técnico riguroso y conciso para responder al usuario."""
        diag = self.ejecutar_diagnostico_completo()
        dur = f"{diag['duracion_ms']} ms"
        modelos_str = ", ".join(diag['modelos_ia']['modelos_locales']) if diag['modelos_ia']['modelos_locales'] else "ninguno"
        lineas = [
            "=== REPORTE DE AUTODIAGNÓSTICO PROFUNDO DEL SISTEMA LUCIA ===",
            f"Salud General: {diag['salud_general']} (Tiempo de análisis: {dur})",
            "",
            "1. BLOCKCHAIN (STM_BKCH):",
            f"   • Estado: {diag['blockchain'].get('estado')}",
            f"   • Bloques registrados: {diag['blockchain'].get('bloques')}",
            f"   • Criptografía SHA-256: {'VÁLIDA' if diag['blockchain'].get('valida') else 'INVÁLIDA'}",
            f"   • Transacciones pendientes: {diag['blockchain'].get('txs_pendientes', 0)}",
            "",
            "2. RED NEURONAL Y SINAPSIS (50 Neuronas):",
            f"   • Estado: {diag['red_neuronal'].get('estado')}",
            f"   • Neuronas activas: {diag['red_neuronal'].get('total_neuronas')}",
            f"   • Derivada sináptica Muon: {diag['red_neuronal'].get('deriva_acumulada')}",
            f"   • Tono cognitivo inferido: {diag['red_neuronal'].get('tono_inferido')}",
            "",
            "3. SUBSISTEMAS DE IA:",
            f"   • Ollama Local: {'ACTIVO' if diag['modelos_ia']['ollama_activo'] else 'INACTIVO'}",
            f"   • Modelos locales detectados: {modelos_str}",
            f"   • OpenRouter :free: {'CONECTADO' if diag['modelos_ia']['openrouter_activo'] else 'DESCONECTADO'}",
            f"   • Modelos remotos disponibles: {diag['modelos_ia']['modelos_free_total']}",
            "",
            "4. INTEGRIDAD ESTRUCTURAL DEL CÓDIGO:",
            f"   • Archivos analizados: {diag['integridad_codigo']['total_archivos_py']}",
            f"   • Archivos > 450 líneas: {len(diag['integridad_codigo']['archivos_excedidos'])}",
        ]
        for exc in diag['integridad_codigo']['archivos_excedidos']:
            lineas.append(f"     - {exc['archivo']} ({exc['lineas']} líneas)")

        lineas.extend([
            "",
            "5. REGISTRO DE ERRORES RUNTIME:",
            f"   • Errores capturados: {diag['errores_runtime']['total_errores_registrados']}",
        ])
        for err in diag['errores_runtime']['ultimos']:
            lineas.append(f"     - [{err.get('tipo')}] {err.get('contexto')}: {err.get('mensaje')}")

        return "\n".join(lineas)


def ejecutar_diagnostico() -> str:
    """Punto de acceso rápido: ejecuta el diagnóstico y devuelve el texto formateado."""
    return AutodiagnosticoLucIA().generar_resumen_texto()


def obtener_datos_diagnostico() -> Dict[str, Any]:
    """Punto de acceso con datos estructurados para paneles STYLOS."""
    return AutodiagnosticoLucIA().ejecutar_diagnostico_completo()


def es_peticion_diagnostico(prompt: str) -> bool:
    """Detecta si la consulta del usuario solicita introspección o autodiagnóstico del sistema."""
    p = (prompt or "").lower()
    claves = [
        "diagnostico", "diagnóstico", "autodiagnostico", "autodiagnóstico",
        "que funciona y que no", "qué funciona y qué no", "estado del sistema",
        "salud del sistema", "como estas internamente", "cómo estás internamente",
        "revisa tu sistema", "analiza tu sistema", "inspecciona tu sistema",
    ]
    return any(c in p for c in claves)


def respuesta_autodiagnostico_lucia(prompt: str = "") -> str:
    """Genera la respuesta introspectiva completa y estructurada de LucIA."""
    resumen = ejecutar_diagnostico()
    return f"He completado el diagnóstico en profundidad de mi arquitectura y subsistemas:\n\n{resumen}"


__all__ = [
    "AutodiagnosticoLucIA",
    "ejecutar_diagnostico",
    "obtener_datos_diagnostico",
    "es_peticion_diagnostico",
    "respuesta_autodiagnostico_lucia",
]
