"""
voice_engine.py - Motor de Sintesis de Voz Neuronal para LucIA (WoldVirtualP2P3D 2026)
=====================================================================================
Modo de expresion: Voz juvenil, espontanea, dinamica y fresca (estilo chica 18 anos).
Voz neuronal oficial: es-ES-ElviraNeural (es-ES / Microsoft Speech Services).
- Modulador prosodico: variaciones naturales de pitch y cadencia segun estado cognitivo.
- Insercion organica de muletillas de expresion conversacional dinamica.
- Diccion fluida: sanitizacion profunda de markdown, simbolos y elementos tecnicos.
- Interrupcion instantanea (cancel_tts) con winmm.dll (MCI) y soporte asincrono en hilo worker.
- Fallback automatico a System.Speech.Synthesis en entornos sin conexion.
"""
from __future__ import annotations

import asyncio
import ctypes
import hashlib
import logging
import os
import queue
import random
import re
import subprocess
import tempfile
import threading
import time
from typing import Any, Dict, Final, List, Optional, Tuple

logger = logging.getLogger("lucIA.Voice")

VOZ_OFICIAL: Final[str] = "es-ES-ElviraNeural"
VERSION_VOICE: Final[str] = "2026.4.2"


class ModuladorExpresivoJuvenil:
    """
    Modula la cadencia, tono emocional y muletillas orales para lograr una
    expresion fresca, viva y espontanea caracteristica de una joven de 18 anos.
    """

    MULETILLAS_APERTURA: Final[List[str]] = [
        "Oye, ",
        "A ver, te cuento: ",
        "Mira, ",
        "Bueno, pues a ver: ",
        "Fijate, ",
        "Pues mira, ",
        "Osea, escucha: ",
    ]

    CONECTORES_ESPONTANEOS: Final[List[Tuple[str, str]]] = [
        (r"\bpor lo tanto\b", "asi que"),
        (r"\bpor consiguiente\b", "con lo cual"),
        (r"\bsin embargo\b", "pero bueno,"),
        (r"\bno obstante\b", "aunque la verdad,"),
        (r"\bes decir\b", "en plan,"),
        (r"\ben efecto\b", "totalmente,"),
        (r"\bcabe destacar que\b", "lo que mola es que"),
        (r"\bes menester\b", "toca"),
        (r"\bciertamente\b", "de verdad,"),
    ]

    CIERRES_FRESCOS: Final[List[str]] = [
        " ¿Tiene sentido, verdad?",
        " ¿Que te parece?",
        " Ya me diras que opinas.",
        " Esta genial, a que si?",
        " Cualquier duda me dices!",
    ]

    def __init__(self) -> None:
        self._contador_turnos: int = 0
        self._lock = threading.Lock()

    def humanizar_texto(self, texto: str, aplicar_muletillas: bool = True) -> str:
        """Transforma formulas formales y rigidas en un estilo oral natural."""
        if not texto:
            return ""
        t = texto
        for pat, rep in self.CONECTORES_ESPONTANEOS:
            t = re.sub(pat, rep, t, flags=re.IGNORECASE)

        if aplicar_muletillas and len(t) > 30 and not t.startswith(("¡", "¿", "Oye", "Mira")):
            with self._lock:
                self._contador_turnos += 1
                if self._contador_turnos % 2 == 1:
                    prefijo = random.choice(self.MULETILLAS_APERTURA)
                    t = prefijo + t[0].lower() + t[1:]

        t = re.sub(r"\s+", " ", t).strip()
        return t

    def calcular_prosodia(self, texto: str, emocion: float = 0.5) -> Tuple[str, str]:
        """Genera rate y pitch adaptativos (+pitch para vivacidad juvenil)."""
        if emocion > 0.6:
            pitch_val = min(12, int(6 + (emocion * 8)))
            rate_val = min(12, int(4 + (emocion * 6)))
        elif emocion < 0.3:
            pitch_val = max(2, int(3 + (emocion * 4)))
            rate_val = max(0, int(emocion * 5))
        else:
            pitch_val = 6
            rate_val = 5

        rate_str = f"+{rate_val}%" if rate_val >= 0 else f"{rate_val}%"
        pitch_str = f"+{pitch_val}Hz" if pitch_val >= 0 else f"{pitch_val}Hz"
        return rate_str, pitch_str


class LimpiadorFoneticoLucIA:
    """Sanitiza y normaliza textos para una fonetica fluida y sin tropiezos."""

    @staticmethod
    def sanitizar(texto: str) -> str:
        if not texto:
            return ""
        t = texto.strip()
        t = re.sub(r"<think>.*?</think>", "", t, flags=re.DOTALL)
        t = re.sub(r"```.*?```", " codigo omitido ", t, flags=re.DOTALL)
        t = re.sub(r"`[^`]*`", "", t)
        t = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", t)
        t = re.sub(r"^\s*#+\s*", "", t, flags=re.MULTILINE)
        t = re.sub(r"^\s*\d+\.\s*", "", t, flags=re.MULTILINE)
        t = re.sub(r"^\s*[-*+•◆]\s*", "", t, flags=re.MULTILINE)
        t = re.sub(r"\*+", "", t)
        t = re.sub(r"[_~>|\\(){}\[\]]", " ", t)
        t = "".join(c for c in t if ord(c) < 0x10000)
        t = re.sub(r"[\u2600-\u27BF]", "", t)
        t = re.sub(r"[ \t]+", " ", t)
        t = re.sub(r"\n\s*\n+", ". ", t)
        t = re.sub(r"\n", ". ", t)
        t = re.sub(r"\.{2,}", ".", t)
        t = t.strip()
        if t and t[-1] not in ".!?":
            t += "."
        return t

    @staticmethod
    def segmentar_frases(texto: str, max_chars: int = 420) -> List[str]:
        if len(texto) <= max_chars:
            return [texto]
        bloques: List[str] = []
        inicio = 0
        while inicio < len(texto):
            resto = len(texto) - inicio
            if resto <= max_chars:
                bloques.append(texto[inicio:].strip())
                break
            fin = inicio + max_chars
            corte = -1
            for sep in (". ", "! ", "? ", "; "):
                pos = texto.rfind(sep, inicio, fin)
                if pos > corte:
                    corte = pos + len(sep)
            if corte > inicio:
                bloques.append(texto[inicio:corte].strip())
                inicio = corte
            else:
                espacio = texto.rfind(" ", inicio, fin)
                if espacio > inicio:
                    bloques.append(texto[inicio:espacio].strip())
                    inicio = espacio + 1
                else:
                    bloques.append(texto[inicio:fin].strip())
                    inicio = fin
        return [b for b in bloques if b]


class ReproductorMCIWindows:
    """Controlador directo de audio nativo mediante winmm.dll de Windows."""

    def __init__(self, alias: str = "lucia_audio_active") -> None:
        self.alias: str = alias
        self._lock = threading.Lock()

    def detener_inmediato(self) -> None:
        with self._lock:
            try:
                mci = ctypes.windll.winmm.mciSendStringW
                mci(f"stop {self.alias}", None, 0, None)
                mci(f"close {self.alias}", None, 0, None)
            except Exception:
                pass

    def reproducir_mp3(self, ruta_mp3: str, cancel_ev: threading.Event, stop_ev: threading.Event) -> bool:
        if cancel_ev.is_set():
            return False
        with self._lock:
            try:
                mci = ctypes.windll.winmm.mciSendStringW
                mci(f"close {self.alias}", None, 0, None)
                ruta_norm = os.path.abspath(ruta_mp3).replace("\\", "/")
                if mci(f'open "{ruta_norm}" type mpegvideo alias {self.alias}', None, 0, None) != 0:
                    return False
                mci(f"play {self.alias}", None, 0, None)
            except Exception as ex:
                logger.warning(f"Error al abrir audio MCI: {ex}")
                return False

        buf = ctypes.create_unicode_buffer(128)
        time.sleep(0.08)
        while not cancel_ev.is_set() and not stop_ev.is_set():
            time.sleep(0.09)
            with self._lock:
                res = mci(f"status {self.alias} mode", buf, 128, None)
                if res != 0 or buf.value not in ("playing", ""):
                    break

        self.detener_inmediato()
        return True


class VoiceEngine:
    """
    Sintetizador neuronal de voz para LucIA con voz oficial es-ES-ElviraNeural
    y modulacion prosodica adaptativa de estilo juvenil espontaneo (18 anos).
    """

    def __init__(self, voz: str = VOZ_OFICIAL) -> None:
        self.voz: str = voz
        self._cola_voz: queue.Queue[str] = queue.Queue()
        self._detener_evento = threading.Event()
        self._cancelar_actual = threading.Event()
        self._hilo_worker: Optional[threading.Thread] = None
        self.mci = ReproductorMCIWindows()
        self.modulador = ModuladorExpresivoJuvenil()
        self.limpiador = LimpiadorFoneticoLucIA()
        self._edge_tts_disponible: bool = False
        self._nivel_emocion: float = 0.65
        self._habilitado: bool = True
        self._verificar_edge_tts()
        self._iniciar_worker()

    def _verificar_edge_tts(self) -> None:
        try:
            import edge_tts
            self._edge_tts_disponible = True
            logger.info(f"Voz neuronal configurada: {self.voz} (18 anos / ElviraNeural)")
        except ImportError:
            self._edge_tts_disponible = False
            logger.warning("edge_tts no detectado, usando fallback Windows")

    def _iniciar_worker(self) -> None:
        self._hilo_worker = threading.Thread(
            target=self._procesar_cola, daemon=True, name="LucIA-Voice-Worker"
        )
        self._hilo_worker.start()

    def set_emocion(self, nivel: float) -> None:
        """Ajusta la valencia emocional de la prosodia (0.0 a 1.0)."""
        self._nivel_emocion = max(0.0, min(1.0, float(nivel)))

    def limpiar_texto_para_habla(self, texto: str) -> str:
        """Sanitiza markdown, listas y caracteres extraños."""
        return self.limpiador.sanitizar(texto)

    def _sintetizar_y_reproducir_chunk(self, chunk: str) -> bool:
        if self._cancelar_actual.is_set():
            return False

        import edge_tts
        temp_fd, temp_path = tempfile.mkstemp(suffix=".mp3")
        os.close(temp_fd)

        rate_str, pitch_str = self.modulador.calcular_prosodia(chunk, self._nivel_emocion)

        async def _gen() -> None:
            comm = edge_tts.Communicate(chunk, self.voz, rate=rate_str, pitch=pitch_str)
            await comm.save(temp_path)

        try:
            asyncio.run(asyncio.wait_for(_gen(), timeout=18.0))
            if os.path.exists(temp_path) and os.path.getsize(temp_path) > 0:
                if not self._cancelar_actual.is_set():
                    self.mci.reproducir_mp3(temp_path, self._cancelar_actual, self._detener_evento)
                    return True
        except Exception as err:
            logger.warning(f"Fallo en sintesis edge_tts: {err}")
        finally:
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception:
                    pass
        return False

    def _sintetizar_fallback_windows(self, texto_limpio: str) -> None:
        if self._cancelar_actual.is_set():
            return
        t_esc = texto_limpio.replace("'", "''").replace('"', '`"')
        cmd = (
            f"Add-Type -AssemblyName System.Speech; "
            f"$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
            f"$s.Rate = 1; "
            f"$s.Speak('{t_esc}')"
        )
        try:
            subprocess.run(
                ["powershell", "-NoProfile", "-NonInteractive", "-Command", cmd],
                capture_output=True,
                text=True,
                timeout=25,
            )
        except Exception as err:
            logger.warning(f"Error fallback System.Speech: {err}")

    def _procesar_mensaje(self, texto_completo: str) -> None:
        if not self._habilitado:
            return
        limpio = self.limpiador.sanitizar(texto_completo)
        if not limpio:
            return
        humanizado = self.modulador.humanizar_texto(limpio, aplicar_muletillas=True)
        chunks = self.limpiador.segmentar_frases(humanizado, max_chars=420)

        for fragmento in chunks:
            if self._cancelar_actual.is_set() or self._detener_evento.is_set():
                break
            exito = False
            if self._edge_tts_disponible:
                exito = self._sintetizar_y_reproducir_chunk(fragmento)
            if not exito and not self._cancelar_actual.is_set():
                self._sintetizar_fallback_windows(fragmento)

    def _procesar_cola(self) -> None:
        while not self._detener_evento.is_set():
            try:
                mensaje = self._cola_voz.get(timeout=0.3)
            except queue.Empty:
                continue

            if mensaje:
                self._cancelar_actual.clear()
                self._procesar_mensaje(mensaje)
            self._cola_voz.task_done()

    def cancel_tts(self) -> None:
        """Detiene inmediatamente el audio en curso y purga la cola."""
        self._cancelar_actual.set()
        self.mci.detener_inmediato()
        while not self._cola_voz.empty():
            try:
                self._cola_voz.get_nowait()
            except Exception:
                pass

    def speak(self, texto: str, esperar: bool = False, emocion: Optional[float] = None) -> None:
        """Emite un enunciado en la voz natural juvenil de LucIA."""
        if not texto or not texto.strip():
            return
        if emocion is not None:
            self.set_emocion(emocion)
        if esperar:
            self._cancelar_actual.clear()
            self._procesar_mensaje(texto)
        else:
            self._cola_voz.put(texto)

    def activar(self, estado: bool = True) -> None:
        """Habilita o desactiva la sintesis de voz global."""
        self._habilitado = bool(estado)
        if not self._habilitado:
            self.cancel_tts()

    def esta_activa(self) -> bool:
        return self._habilitado

    def detener(self) -> None:
        """Cierre definitivo del motor de audio y sus hilos asociados."""
        self._detener_evento.set()
        self.cancel_tts()


_voice_instance: Optional[VoiceEngine] = None
_global_lock_voice = threading.Lock()


def get_voice_engine() -> VoiceEngine:
    """Singleton thread-safe del VoiceEngine juvenil de LucIA."""
    global _voice_instance
    with _global_lock_voice:
        if _voice_instance is None:
            _voice_instance = VoiceEngine(voz=VOZ_OFICIAL)
    return _voice_instance


def speak(texto: str, esperar: bool = False, emocion: Optional[float] = None) -> None:
    """Conveniencia directa: emite voz con es-ES-ElviraNeural y tono fresco."""
    motor = get_voice_engine()
    motor.speak(texto, esperar=esperar, emocion=emocion)


def cancel_speech() -> None:
    """Interrumpe cualquier reproduccion sonora activa al instante."""
    motor = get_voice_engine()
    motor.cancel_tts()


def configurar_prosodia_juvenil(tono: str = "positivo") -> None:
    """Configura la expresion de la voz segun el tono neuronal detectado."""
    motor = get_voice_engine()
    mapeo_tonos = {"positivo": 0.85, "neutro": 0.60, "tecnico": 0.45, "negativo": 0.30}
    val = mapeo_tonos.get(tono.lower(), 0.60)
    motor.set_emocion(val)


if __name__ == "__main__":
    print("\033[38;5;51m" + "=" * 72 + "\033[0m")
    print("  Motor de Voz LucIA (18 Anos / es-ES-ElviraNeural) - Autotest")
    print("\033[38;5;51m" + "=" * 72 + "\033[0m")
    v = get_voice_engine()
    demo_txt = "Hola! Por lo tanto estoy procesando todo genial y mi voz suena super natural."
    print(f"  Texto original : {demo_txt}")
    h = v.modulador.humanizar_texto(demo_txt)
    r, p = v.modulador.calcular_prosodia(h, 0.75)
    print(f"  Humanizado     : {h}")
    print(f"  Prosodia       : Rate={r} | Pitch={p}")
    print("\033[38;5;48m  [OK] Subsistema de voz listo para integracion en mainLCSTM\033[0m")
    print("\033[38;5;51m" + "=" * 72 + "\033[0m")
