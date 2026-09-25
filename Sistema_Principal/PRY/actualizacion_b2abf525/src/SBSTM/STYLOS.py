"""
STYLOS.py - Motor de Estilos Visuales y Renderizado de Terminal (Arquitectura 2026)
=====================================================================================
Subsistema visual premium para la presentacion de respuestas de LucIA y telemetria:
  - Paleta Cyber-Bioluminiscente (Electric Cyan, Violeta Neon, Ambar Solar, Esmeralda).
  - Cajas de dialogo delimitadas con bordes redondeados Unicode (╭─╮╰─╯│) y tarjetas.
  - Formateo inteligente de Markdown: negrita, listas, bloques de codigo syntax-highlight,
    tablas ASCII/Unicode alineadas y citas indentadas.
  - Streaming token a token suave con indicador de cursor pulsante.
  - Banners de estado, divisores y badges de telemetria sinaptica y blockchain.
  - Adaptabilidad de ancho de consola con deteccion automatica o fallback a 80 columnas.
"""
from __future__ import annotations

import os
import re
import shutil
import sys
import time
from typing import Any, Callable, Dict, Final, List, Optional, Tuple, Union

# ─── METADATOS DEL SUBSISTEMA ────────────────────────────────────────────────
__version__: Final[str] = "2026.3.1"
__subsystem__: Final[str] = "LucIA-Terminal-Stylos"
__author__: Final[str] = "Equipo WoldVirtualP2P3D"

# ─── PALETA DE COLOR HEX/256 Y ANSI EXTENDIDO ────────────────────────────────
class ColoresLucIA:
    """
    Paleta de colores distintiva estilo OpenCode Cyber-Bioluminiscente.
    Tonos frios de fondo con acentos en violeta cuantico, cian electrico y oro solar.
    """
    RESET: Final[str] = "\033[0m"
    BOLD: Final[str] = "\033[1m"
    DIM: Final[str] = "\033[2m"
    ITALIC: Final[str] = "\033[3m"
    UNDERLINE: Final[str] = "\033[4m"

    # Tonos Principales (256 colores & TrueColor)
    CIAN_ELECTRICO: Final[str] = "\033[38;5;51m"
    CIAN_PROFUNDO: Final[str] = "\033[38;5;38m"
    VIOLETA_NEON: Final[str] = "\033[38;5;141m"
    MAGENTA_CUANTICO: Final[str] = "\033[38;5;198m"
    ESMERALDA_VIVO: Final[str] = "\033[38;5;48m"
    VERDE_SINAPTICO: Final[str] = "\033[38;5;42m"
    AMBAR_SOLAR: Final[str] = "\033[38;5;214m"
    DORADO_BLOCK: Final[str] = "\033[38;5;220m"
    ROJO_ALERTA: Final[str] = "\033[38;5;203m"
    GRIS_METALLIC: Final[str] = "\033[38;5;244m"
    GRIS_OSCURO: Final[str] = "\033[38;5;238m"
    BLANCO_LUMINOSO: Final[str] = "\033[38;5;255m"

    # Fondos
    BG_CARD: Final[str] = "\033[48;5;235m"
    BG_CODE: Final[str] = "\033[48;5;234m"
    BG_HIGHLIGHT: Final[str] = "\033[48;5;236m"


# ─── GLIFOS Y CARACTERES ESTRUCTURALES UNICODE ───────────────────────────────
class Glifos:
    """Simbolos tipograficos y de diseno para bordes de tarjetas y telemetria."""
    ESQ_ARR_IZQ: Final[str] = "╭"
    ESQ_ARR_DER: Final[str] = "╮"
    ESQ_ABA_IZQ: Final[str] = "╰"
    ESQ_ABA_DER: Final[str] = "╯"
    LINEA_H: Final[str] = "─"
    LINEA_V: Final[str] = "│"
    CRUZ_IZQ: Final[str] = "├"
    CRUZ_DER: Final[str] = "┤"
    FLECHA: Final[str] = "◈"
    PUNTO_VIVO: Final[str] = "●"
    SPARKLE: Final[str] = "✦"
    BLOQUE_LLENO: Final[str] = "█"
    BLOQUE_MEDIO: Final[str] = "▒"
    BLOQUE_VACIO: Final[str] = "░"
    RAYO: Final[str] = "⚡"
    CEREBRO: Final[str] = "🧠"


# ─── UTILIDADES DE FORMATO Y ANCHO DE PANTALLA ──────────────────────────────
def obtener_ancho_consola(max_limite: int = 90) -> int:
    """Determina el ancho util de la terminal respetando limites para lectura optima."""
    try:
        columnas, _ = shutil.get_terminal_size((80, 24))
        return min(max_limite, max(60, columnas - 4))
    except Exception:
        return 80


def limpiar_codigos_ansi(texto: str) -> str:
    """Remueve secuencias de escape ANSI para calcular longitudes visuales reales."""
    patron_ansi = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
    return patron_ansi.sub("", texto)


def longitud_visual(texto: str) -> int:
    """Calcula el numero real de caracteres visibles en pantalla."""
    return len(limpiar_codigos_ansi(texto))


# ─── FORMATEADOR Y RENDERIZADOR MARKDOWN DE TERMINAL ─────────────────────────
class RenderizadorMarkdown:
    """Convierte texto plano y Markdown basico en salida coloreada y estructurada."""

    @classmethod
    def colorear_sintaxis(cls, linea: str) -> str:
        """Aplica estilos visuales a negritas, codigos, cursivas y listas."""
        # 1. Bloques de codigo inline `codigo`
        linea = re.sub(
            r"`([^`]+)`",
            rf"{ColoresLucIA.AMBAR_SOLAR}\1{ColoresLucIA.RESET}",
            linea,
        )
        # 2. Negrita **texto**
        linea = re.sub(
            r"\*\*([^*]+)\*\*",
            rf"{ColoresLucIA.BOLD}{ColoresLucIA.BLANCO_LUMINOSO}\1{ColoresLucIA.RESET}",
            linea,
        )
        # 3. Cursiva *texto*
        linea = re.sub(
            r"\*([^*]+)\*",
            rf"{ColoresLucIA.ITALIC}{ColoresLucIA.CIAN_ELECTRICO}\1{ColoresLucIA.RESET}",
            linea,
        )
        # 4. Encabezados ###
        if linea.strip().startswith("###"):
            titulo = linea.strip().lstrip("#").strip()
            return f"\n  {ColoresLucIA.BOLD}{ColoresLucIA.VIOLETA_NEON}◈ {titulo}{ColoresLucIA.RESET}"
        if linea.strip().startswith("##"):
            titulo = linea.strip().lstrip("#").strip()
            return f"\n  {ColoresLucIA.BOLD}{ColoresLucIA.CIAN_ELECTRICO}✦ {titulo}{ColoresLucIA.RESET}"
        if linea.strip().startswith("#"):
            titulo = linea.strip().lstrip("#").strip()
            return f"\n  {ColoresLucIA.BOLD}{ColoresLucIA.DORADO_BLOCK}━ {titulo} ━{ColoresLucIA.RESET}"

        # 5. Listas de vinetas
        if re.match(r"^\s*[-*•]\s+", linea):
            contenido = re.sub(r"^\s*[-*•]\s+", "", linea)
            return f"    {ColoresLucIA.VIOLETA_NEON}•{ColoresLucIA.RESET} {contenido}"

        # 6. Listas numeradas
        m_num = re.match(r"^(\s*)(\d+)\.\s+(.*)$", linea)
        if m_num:
            esp, num, cont = m_num.groups()
            return f"{esp}  {ColoresLucIA.CIAN_ELECTRICO}{num}.{ColoresLucIA.RESET} {cont}"

        return linea

    @classmethod
    def renderizar_tabla(cls, lineas_tabla: List[str], ancho_max: int = 80) -> str:
        """Parsea y genera una tabla Unicode alineada con encabezados destacados."""
        filas = []
        for l in lineas_tabla:
            partes = [c.strip() for c in l.split("|")[1:-1]]
            if partes and not all(set(c) <= {"-", ":", " "} for c in partes):
                filas.append(partes)

        if not filas:
            return ""

        num_cols = max(len(f) for f in filas)
        anchos = [0] * num_cols
        for f in filas:
            for i in range(min(num_cols, len(f))):
                anchos[i] = max(anchos[i], longitud_visual(f[i]))

        lineas_salida = []
        sep_horiz = "  " + Glifos.ESQ_ARR_IZQ + (Glifos.LINEA_H * (sum(anchos) + (num_cols * 3) - 1)) + Glifos.ESQ_ARR_DER
        lineas_salida.append(f"{ColoresLucIA.GRIS_METALLIC}{sep_horiz}{ColoresLucIA.RESET}")

        for idx_f, f in enumerate(filas):
            celdas_fmt = []
            for i in range(num_cols):
                val = f[i] if i < len(f) else ""
                pad = anchos[i] - longitud_visual(val)
                val_col = cls.colorear_sintaxis(val)
                celdas_fmt.append(f"{val_col}{' ' * pad}")

            color_borde = ColoresLucIA.VIOLETA_NEON if idx_f == 0 else ColoresLucIA.GRIS_METALLIC
            linea_str = f"  {color_borde}{Glifos.LINEA_V}{ColoresLucIA.RESET} " + f" {color_borde}{Glifos.LINEA_V}{ColoresLucIA.RESET} ".join(celdas_fmt) + f" {color_borde}{Glifos.LINEA_V}{ColoresLucIA.RESET}"
            lineas_salida.append(linea_str)

            if idx_f == 0:
                sep_div = "  " + Glifos.CRUZ_IZQ + (Glifos.LINEA_H * (sum(anchos) + (num_cols * 3) - 1)) + Glifos.CRUZ_DER
                lineas_salida.append(f"{ColoresLucIA.VIOLETA_NEON}{sep_div}{ColoresLucIA.RESET}")

        sep_fin = "  " + Glifos.ESQ_ABA_IZQ + (Glifos.LINEA_H * (sum(anchos) + (num_cols * 3) - 1)) + Glifos.ESQ_ABA_DER
        lineas_salida.append(f"{ColoresLucIA.GRIS_METALLIC}{sep_fin}{ColoresLucIA.RESET}")
        return "\n".join(lineas_salida)


# ─── MOTOR PRINCIPAL DE TARJETAS Y PRESENTACION ──────────────────────────────
class EstiloTerminalLucIA:
    """Motor de diseno visual para respuestas conversacionales y paneles de control."""

    @staticmethod
    def renderizar_encabezado_sesion(sesion_id: str, modelo_activo: str, total_neuronas: int = 50) -> None:
        """Despliega el banner principal con estetica de consola futurista estilo OpenCode."""
        ancho = obtener_ancho_consola()
        borde_arr = Glifos.ESQ_ARR_IZQ + (Glifos.LINEA_H * (ancho - 2)) + Glifos.ESQ_ARR_DER
        borde_aba = Glifos.ESQ_ABA_IZQ + (Glifos.LINEA_H * (ancho - 2)) + Glifos.ESQ_ABA_DER
        linea_div = Glifos.CRUZ_IZQ + (Glifos.LINEA_H * (ancho - 2)) + Glifos.CRUZ_DER

        print()
        print(f"{ColoresLucIA.CIAN_ELECTRICO}{borde_arr}{ColoresLucIA.RESET}")
        titulo = f" WOLDVIRTUALP2P3D  ✦  LUCIA COGNITIVE CONSOLE 2026 "
        pad_tit = ancho - 2 - len(titulo)
        print(f"{ColoresLucIA.CIAN_ELECTRICO}{Glifos.LINEA_V}{ColoresLucIA.RESET}{ColoresLucIA.BOLD}{ColoresLucIA.BLANCO_LUMINOSO}{titulo}{' ' * max(0, pad_tit)}{ColoresLucIA.CIAN_ELECTRICO}{Glifos.LINEA_V}{ColoresLucIA.RESET}")
        print(f"{ColoresLucIA.GRIS_METALLIC}{linea_div}{ColoresLucIA.RESET}")

        linea_info = f"  ID: {ColoresLucIA.VIOLETA_NEON}{sesion_id}{ColoresLucIA.RESET} | Mod: {ColoresLucIA.AMBAR_SOLAR}{modelo_activo}{ColoresLucIA.RESET} | Sinapsis: {ColoresLucIA.VERDE_SINAPTICO}{total_neuronas} Activas{ColoresLucIA.RESET} | Coste: {ColoresLucIA.ESMERALDA_VIVO}$0.00{ColoresLucIA.RESET}"
        pad_info = ancho - 2 - longitud_visual(linea_info)
        print(f"{ColoresLucIA.CIAN_ELECTRICO}{Glifos.LINEA_V}{ColoresLucIA.RESET}{linea_info}{' ' * max(0, pad_info)}{ColoresLucIA.CIAN_ELECTRICO}{Glifos.LINEA_V}{ColoresLucIA.RESET}")
        print(f"{ColoresLucIA.CIAN_ELECTRICO}{borde_aba}{ColoresLucIA.RESET}\n")

    @staticmethod
    def renderizar_prompt_usuario(prompt: str) -> None:
        """Muestra el estimulo ingresado por el usuario en una tarjeta compacta."""
        ancho = obtener_ancho_consola()
        etiqueta = f" Tu > "
        print(f"\n{ColoresLucIA.BOLD}{ColoresLucIA.AMBAR_SOLAR}╭──{etiqueta}{Glifos.LINEA_H * (ancho - len(etiqueta) - 4)}╮{ColoresLucIA.RESET}")
        print(f"{ColoresLucIA.AMBAR_SOLAR}│{ColoresLucIA.RESET}  {ColoresLucIA.BLANCO_LUMINOSO}{prompt}{ColoresLucIA.RESET}")
        print(f"{ColoresLucIA.AMBAR_SOLAR}╰{'─' * (ancho - 2)}╯{ColoresLucIA.RESET}\n")

    @staticmethod
    def iniciar_tarjeta_respuesta(modelo_id: str = "openrouter/free") -> None:
        """Imprime el encabezado redondeado de la tarjeta antes del streaming de LucIA."""
        ancho = obtener_ancho_consola()
        insignia = f" LucIA  [Free: {modelo_id}] "
        resto = ancho - len(insignia) - 3
        print(f"{ColoresLucIA.VIOLETA_NEON}╭─{ColoresLucIA.BOLD}{ColoresLucIA.CIAN_ELECTRICO}{insignia}{ColoresLucIA.VIOLETA_NEON}{Glifos.LINEA_H * max(2, resto)}╮{ColoresLucIA.RESET}")
        print(f"{ColoresLucIA.VIOLETA_NEON}│{ColoresLucIA.RESET}")

    @staticmethod
    def imprimir_token_stream(token: str) -> None:
        """Escribe los tokens en tiempo real manteniendo el margen visual de la tarjeta."""
        # Se imprime directamente conservando fluidez visual
        sys.stdout.write(f"{ColoresLucIA.BLANCO_LUMINOSO}{token}{ColoresLucIA.RESET}")
        sys.stdout.flush()

    @staticmethod
    def cerrar_tarjeta_respuesta() -> None:
        """Cierra la tarjeta de respuesta de LucIA tras finalizar la emision."""
        ancho = obtener_ancho_consola()
        print(f"\n{ColoresLucIA.VIOLETA_NEON}│{ColoresLucIA.RESET}")
        print(f"{ColoresLucIA.VIOLETA_NEON}╰{Glifos.LINEA_H * (ancho - 2)}╯{ColoresLucIA.RESET}")

    @staticmethod
    def renderizar_respuesta_completa(texto: str, modelo_id: str = "openrouter/free") -> None:
        """Formatea e imprime una respuesta completa con deteccion de Markdown y tablas."""
        ancho = obtener_ancho_consola()
        insignia = f" LucIA  [Free: {modelo_id}] "
        resto = ancho - len(insignia) - 3

        print(f"{ColoresLucIA.VIOLETA_NEON}╭─{ColoresLucIA.BOLD}{ColoresLucIA.CIAN_ELECTRICO}{insignia}{ColoresLucIA.VIOLETA_NEON}{Glifos.LINEA_H * max(2, resto)}╮{ColoresLucIA.RESET}")
        print(f"{ColoresLucIA.VIOLETA_NEON}│{ColoresLucIA.RESET}")

        lineas = texto.splitlines()
        en_tabla = False
        buffer_tabla = []

        for l in lineas:
            # Deteccion de tablas Markdown
            if l.strip().startswith("|") and l.strip().endswith("|"):
                en_tabla = True
                buffer_tabla.append(l)
                continue
            elif en_tabla:
                en_tabla = False
                print(RenderizadorMarkdown.renderizar_tabla(buffer_tabla, ancho))
                buffer_tabla = []

            # Linea normal con formateo de sintaxis
            linea_fmt = RenderizadorMarkdown.colorear_sintaxis(l)
            print(f"  {linea_fmt}")

        if en_tabla and buffer_tabla:
            print(RenderizadorMarkdown.renderizar_tabla(buffer_tabla, ancho))

        print(f"{ColoresLucIA.VIOLETA_NEON}│{ColoresLucIA.RESET}")
        print(f"{ColoresLucIA.VIOLETA_NEON}╰{Glifos.LINEA_H * (ancho - 2)}╯{ColoresLucIA.RESET}")

    @staticmethod
    def renderizar_barra_telemetria(
        turno: int,
        duracion_ms: float,
        modelo_id: str,
        deriva: float,
        neuronas: int,
        cid_ipfs: str,
        txs_espera: int,
    ) -> None:
        """Imprime un badge de telemetria compacto con colores cybernéticos."""
        cid_corte = cid_ipfs[:16] + "..." if len(cid_ipfs) > 16 and cid_ipfs != "--" else cid_ipfs

        badge_turno = f"{ColoresLucIA.VIOLETA_NEON}Turno #{turno}{ColoresLucIA.RESET}"
        badge_ms = f"{ColoresLucIA.CIAN_ELECTRICO}{duracion_ms:.0f}ms{ColoresLucIA.RESET}"
        badge_mod = f"{ColoresLucIA.AMBAR_SOLAR}{modelo_id}{ColoresLucIA.RESET}"
        badge_deriva = f"{ColoresLucIA.MAGENTA_CUANTICO}Deriva: {deriva:.4f}{ColoresLucIA.RESET}"
        badge_sinapsis = f"{ColoresLucIA.VERDE_SINAPTICO}Sinapsis: {neuronas}{ColoresLucIA.RESET}"
        badge_cid = f"{ColoresLucIA.GRIS_METALLIC}IPFS: {cid_corte}{ColoresLucIA.RESET}"
        badge_costo = f"{ColoresLucIA.ESMERALDA_VIVO}$0.00 USD{ColoresLucIA.RESET}"

        print(
            f"  {Glifos.FLECHA} {badge_turno} │ {badge_ms} │ {badge_mod} │ "
            f"{badge_deriva} │ {badge_sinapsis} │ {badge_cid} │ {badge_costo}\n"
        )

    @staticmethod
    def renderizar_notificacion_bloque(indice: int, hash_bloque: str) -> None:
        """Muestra una notificacion destacada al minar un nuevo bloque en la blockchain."""
        h_vis = hash_bloque[:32] + "..."
        print(
            f"  {ColoresLucIA.BOLD}{ColoresLucIA.VERDE_SINAPTICO}◈ BLOQUE #{indice} MINADO PoNL{ColoresLucIA.RESET} "
            f"{ColoresLucIA.DORADO_BLOCK}[Hash: {h_vis}]{ColoresLucIA.RESET}"
        )

    @staticmethod
    def renderizar_panel_ayuda() -> None:
        """Despliega una guia visual de comandos con colores distintivos."""
        ancho = obtener_ancho_consola()
        borde_superior = (
            f"{ColoresLucIA.CIAN_ELECTRICO}╭─{ColoresLucIA.BOLD} COMANDOS DISPONIBLES "
            f"EN CONSOLA {ColoresLucIA.CIAN_ELECTRICO}{Glifos.LINEA_H * max(2, ancho - 37)}╮"
            f"{ColoresLucIA.RESET}"
        )
        print(f"\n{borde_superior}")
        comandos = [
            ("estado / status", "Muestra el estado de la blockchain, 50 neuronas y auditoria de costo."),
            (
                "modelos / free",
                "Lista los modelos gratuitos; use ↑/↓ y Enter para seleccionar.",
            ),
            ("modelo <id>", "Cambia en caliente al modelo :free especificado sin reiniciar."),
            ("minar / mine", "Consolida las transacciones pendientes en un bloque PoNL."),
            ("crear carpeta", "Crea una carpeta mediante RFPRMN."),
            ("crear archivo", "Crea un archivo mediante RFPRMN."),
            ("refactorizar", "Copia el contenido actual de PRY a RFC."),
            ("actualizar", "Reemplaza PRY con una nueva version de RFC."),
            ("IA local", "Analiza un archivo y crea un plan de refactorizacion."),
            ("cerrar", "Cierra LucIA usando el cierre normal del sistema."),
            ("salir / exit", "Guarda el checkpoint sinaptico en PSNRL y sincroniza con IPFS."),
        ]
        for cmd, desc in comandos:
            c_str = (
                f"  {ColoresLucIA.AMBAR_SOLAR}{cmd:<18}{ColoresLucIA.RESET} "
                f"{ColoresLucIA.GRIS_METALLIC}→{ColoresLucIA.RESET} "
                f"{ColoresLucIA.BLANCO_LUMINOSO}{desc}{ColoresLucIA.RESET}"
            )
            relleno = " " * max(0, ancho - 2 - longitud_visual(c_str))
            print(
                f"{ColoresLucIA.CIAN_ELECTRICO}│{ColoresLucIA.RESET}{c_str}{relleno}"
                f"{ColoresLucIA.CIAN_ELECTRICO}│{ColoresLucIA.RESET}"
            )
        print(f"{ColoresLucIA.CIAN_ELECTRICO}╰{Glifos.LINEA_H * (ancho - 2)}╯{ColoresLucIA.RESET}\n")

    @staticmethod
    def renderizar_tarjeta_sistema(titulo: str, datos: Dict[str, Any]) -> None:
        """Despliega una tarjeta estructurada de clave-valor para diagnosticos."""
        ancho = obtener_ancho_consola()
        encabezado = f" {titulo} "
        pad_h = ancho - len(encabezado) - 3
        print(f"{ColoresLucIA.DORADO_BLOCK}╭─{ColoresLucIA.BOLD}{encabezado}{ColoresLucIA.DORADO_BLOCK}{Glifos.LINEA_H * max(2, pad_h)}╮{ColoresLucIA.RESET}")
        for clave, valor in datos.items():
            linea = f"  {ColoresLucIA.CIAN_ELECTRICO}{clave:<24}{ColoresLucIA.RESET} : {ColoresLucIA.BLANCO_LUMINOSO}{valor}{ColoresLucIA.RESET}"
            pad = ancho - 2 - longitud_visual(linea)
            print(f"{ColoresLucIA.DORADO_BLOCK}│{ColoresLucIA.RESET}{linea}{' ' * max(0, pad)}{ColoresLucIA.DORADO_BLOCK}│{ColoresLucIA.RESET}")
        print(f"{ColoresLucIA.DORADO_BLOCK}╰{Glifos.LINEA_H * (ancho - 2)}╯{ColoresLucIA.RESET}")


# ─── ACCESOS DIRECTOS Y EXPORTACIONES ────────────────────────────────────────
def formatear_respuesta_lucia(texto: str, modelo: str = "openrouter/free") -> None:
    """Metodo directo para desplegar respuestas con calidad de terminal moderna."""
    EstiloTerminalLucIA.renderizar_respuesta_completa(texto, modelo_id=modelo)


def banner_bienvenida(sesion_id: str, modelo: str) -> None:
    """Despliega el banner estetico inicial de la sesion."""
    EstiloTerminalLucIA.renderizar_encabezado_sesion(sesion_id, modelo)


def badge_turno(turno: int, dt: float, mod: str, der: float, neu: int, cid: str, txs: int) -> None:
    """Emite la telemetria post-turno formateada."""
    EstiloTerminalLucIA.renderizar_barra_telemetria(turno, dt, mod, der, neu, cid, txs)


def panel_ayuda_comandos() -> None:
    """Despliega la ayuda visual de comandos."""
    EstiloTerminalLucIA.renderizar_panel_ayuda()


def tarjeta_diagnostico(titulo: str, datos: Dict[str, Any]) -> None:
    """Despliega una tarjeta estructurada de clave-valor."""
    EstiloTerminalLucIA.renderizar_tarjeta_sistema(titulo, datos)


# ─── DEMOSTRACION EN CONSOLA Y PRUEBA VISUAL ─────────────────────────────────
if __name__ == "__main__":
    banner_bienvenida("LUCIA_DEV_2026", "openrouter/free")

    demo_prompt = "¿Cual es el rol de las 50 neuronas en WoldVirtualP2P3D?"
    EstiloTerminalLucIA.renderizar_prompt_usuario(demo_prompt)

    demo_respuesta = (
        "Las **50 neuronas activas** se distribuyen en cinco familias especializadas:\n"
        "- `ENRN`: 10 neuronas de activacion y entrada recurrente.\n"
        "- `RF_EN`: 10 neuronas de modulacion por refuerzo adaptativo.\n"
        "- `RF_SL`: 10 neuronas residuales supervisadas.\n"
        "- `RNP`: 10 neuronas de plasticidad Hebbiana y memoria asociativa.\n"
        "- `SLRN`: 10 optimizadores de convergencia continua.\n\n"
        "### Resumen de Arquitectura\n"
        "| Familia | Capacidad | Optimizador |\n"
        "|---|---|---|\n"
        "| ENRN | Entrada 4D a 8D | SOAP Curvature |\n"
        "| RNP | Memoria Distribuida | Hebbian Plasticity |\n"
        "| SLRN | Optimizacion Continua | Muon-NS5 Gradient |\n\n"
        "Cada inferencia ajusta los pesos en `PSNRL` y registra la huella en la blockchain."
    )

    EstiloTerminalLucIA.renderizar_respuesta_completa(demo_respuesta, "openrouter/free")

    badge_turno(
        turno=1,
        dt=1420.0,
        mod="openrouter/free",
        der=12.4589,
        neu=50,
        cid="bafkreihq53...",
        txs=1,
    )
