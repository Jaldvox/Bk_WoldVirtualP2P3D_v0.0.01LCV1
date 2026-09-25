"""Selector interactivo de modelos free — subsistema extendido SBSTM.

Extraído de mainLCSTM.py para mantener el orquestador bajo 450 líneas.
El orquestador hereda de SelectorModelos y solo necesita `cliente_iafree`.
"""
from __future__ import annotations
import os
import sys
from typing import Any, Dict, List, Optional


class SelectorModelos:
    cliente_iafree: Any = None

    def _leer_tecla_selector_modelos(self) -> Optional[str]:
        try:
            import msvcrt
        except ImportError:
            return None
        try:
            caracter = msvcrt.getwch()
        except (KeyboardInterrupt, EOFError):
            return "cancel"
        if caracter in ("\x00", "\xe0"):
            codigo = msvcrt.getwch()
            return {"H": "up", "P": "down", "K": "left", "M": "right"}.get(codigo)
        return {"\r": "enter", "\n": "enter", "\x1b": "cancel", "\x03": "cancel"}.get(caracter)

    def _dibujar_selector_modelos(self, modelos: List[Dict[str, Any]], indice: int,
                                  id_activo: str, inicio: int, visibles: int,
                                  lineas_previas: int = 0) -> int:
        if lineas_previas:
            sys.stdout.write(f"\033[{lineas_previas}A")
        sys.stdout.write("\033[J")
        try:
            ancho = max(78, min(92, os.get_terminal_size().columns))
        except OSError:
            ancho = 92
        w_id = max(24, min(42, ancho - 50))
        w_nom = max(10, min(20, ancho - w_id - 32))
        sep = "\033[38;5;51m" + "=" * ancho + "\033[0m"
        fin = min(inicio + visibles, len(modelos))
        lineas = ["", f"  \033[1;37mMODELOS GRATUITOS IAFREE\033[0m  ({inicio + 1}-{fin}/{len(modelos)})", sep]
        for pos in range(inicio, fin):
            m = modelos[pos]
            cur = "\033[38;5;214m>\033[0m" if pos == indice else " "
            act = " \033[38;5;48m[ACTIVO]\033[0m" if m["id"] == id_activo else ""
            lineas.append(f" {cur} {pos + 1:>2}. \033[38;5;51m{m['id']:<{w_id}.{w_id}}\033[0m | "
                          f"{m['contexto']} tok | {m['nombre']:<{w_nom}.{w_nom}}{act}")
        lineas.extend([sep, "  \033[38;5;214mArriba/Abajo\033[0m mover  \033[38;5;214mENTER\033[0m elegir  "
                       "\033[38;5;214mESC\033[0m cancelar", ""])
        print("\n".join(lineas))
        sys.stdout.flush()
        return len(lineas)

    @staticmethod
    def _limpiar_selector_modelos(lineas: int) -> None:
        if lineas:
            sys.stdout.write(f"\033[{lineas}A\033[J")
            sys.stdout.flush()

    def _seleccionar_modelo_textual(self, modelos: List[Dict[str, Any]]) -> bool:
        print(f"\n  \033[1;37mMODELOS GRATUITOS IAFREE ({len(modelos)})\033[0m")
        id_activo = self.cliente_iafree.gestor.obtener_modelo_activo()["id"]
        for pos, m in enumerate(modelos, start=1):
            act = " [ACTIVO]" if m["id"] == id_activo else ""
            print(f"  {pos:>2}. {m['id']} | {m['contexto']} tok{act}")
        try:
            sel = input("  Numero o ID del modelo (Enter cancela): ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n  Seleccion cancelada.")
            return False
        if not sel:
            print("  Seleccion cancelada.")
            return False
        modelo_id = modelos[int(sel) - 1]["id"] if sel.isdigit() and 0 < int(sel) <= len(modelos) else sel
        if sel.isdigit() and (int(sel) < 1 or int(sel) > len(modelos)):
            print(f"  La posicion '{sel}' no existe.")
            return False
        if self.cliente_iafree.gestor.seleccionar_por_id(modelo_id):
            print(f"  \033[38;5;48mModelo gratuito seleccionado: {modelo_id}\033[0m")
            return True
        print(f"  \033[38;5;214mModelo '{modelo_id}' no encontrado en catalogo :free.\033[0m")
        return False

    def _seleccionar_modelo_interactivo(self, modelos: List[Dict[str, Any]]) -> bool:
        if not modelos:
            print("No hay modelos gratuitos disponibles.")
            return False
        if os.name != "nt" or not sys.stdin.isatty() or not sys.stdout.isatty():
            return self._seleccionar_modelo_textual(modelos)
        try:
            import msvcrt  # noqa: F401
        except ImportError:
            return self._seleccionar_modelo_textual(modelos)
        gestor = self.cliente_iafree.gestor
        id_activo = gestor.obtener_modelo_activo()["id"]
        indice = next((p for p, m in enumerate(modelos) if m["id"] == id_activo), 0)
        visibles = min(10, len(modelos))
        inicio = max(0, min(indice - visibles // 2, len(modelos) - visibles))
        n = 0
        while True:
            n = self._dibujar_selector_modelos(modelos, indice, id_activo, inicio, visibles, n)
            try:
                tecla = self._leer_tecla_selector_modelos()
            except (KeyboardInterrupt, EOFError):
                tecla = "cancel"
            if tecla == "up":
                indice = (indice - 1) % len(modelos)
            elif tecla == "down":
                indice = (indice + 1) % len(modelos)
            elif tecla == "enter":
                break
            elif tecla == "cancel":
                self._limpiar_selector_modelos(n)
                print("\n  Seleccion cancelada.")
                return False
            if tecla in ("up", "down"):
                if indice < inicio:
                    inicio = indice
                elif indice >= inicio + visibles:
                    inicio = indice - visibles + 1
        self._limpiar_selector_modelos(n)
        modelo_id = modelos[indice]["id"]
        if gestor.seleccionar_por_id(modelo_id):
            print(f"  \033[38;5;48mModelo gratuito seleccionado: {modelo_id}\033[0m")
            return True
        print(f"  \033[38;5;214mNo se pudo seleccionar el modelo '{modelo_id}'.\033[0m")
        return False

    def listar_modelos_gratuitos(self) -> None:
        """Muestra los modelos gratuitos y permite seleccionar uno."""
        if not self.cliente_iafree:
            print("Subsistema IAFREE no disponible.")
            return
        self._seleccionar_modelo_interactivo(self.cliente_iafree.gestor.listar_modelos())
