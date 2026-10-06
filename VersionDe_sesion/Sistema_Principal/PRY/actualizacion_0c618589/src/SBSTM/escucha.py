"""Entrada de voz STT para LucIA (SBSTM).

Estrategia por capas, todas opcionales:
  1. `speech_recognition` + Google (requiere internet, sin API key).
  2. `vosk` offline (requiere modelo descargado en STM_IA/modelosSTT/).
  3. Sin backend: devuelve "" y el llamante usa texto.
Sin dependencias obligatorias: si falta todo, degradacion elegante.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Optional

MODELOS_STT_DIR: Path = (
    Path(__file__).resolve().parents[1] / "STM_IA" / "modelosSTT"
)


def backend_disponible() -> Dict[str, bool]:
    """Informa que backends STT estan instalados (sin usar microfono)."""
    estado = {"speech_recognition": False, "vosk": False, "modelo_vosk": False}
    try:
        import speech_recognition  # noqa: F401

        estado["speech_recognition"] = True
    except ImportError:
        pass
    try:
        import vosk  # noqa: F401

        estado["vosk"] = True
    except ImportError:
        pass
    if estado["vosk"]:
        estado["modelo_vosk"] = any(MODELOS_STT_DIR.glob("*/am/final.mdl"))
    return estado


def hay_microfono() -> bool:
    """True si hay backend STT instalado. No garantiza microfono fisico."""
    b = backend_disponible()
    return bool(b["speech_recognition"] or b["vosk"])


def escuchar(
    idioma: str = "es-ES",
    timeout: float = 8.0,
    duracion_max: float = 15.0,
) -> Dict[str, Any]:
    """Escucha el microfono y devuelve {'texto', 'backend', 'ok', 'error'}.

    Nunca lanza excepcion: ante cualquier fallo devuelve ok=False.
    """
    inicio = time.perf_counter()
    b = backend_disponible()
    if b["speech_recognition"]:
        return _escuchar_sr(idioma, timeout, duracion_max, inicio)
    if b["vosk"] and b["modelo_vosk"]:
        return _escuchar_vosk(duracion_max, inicio)
    return {
        "texto": "",
        "backend": "ninguno",
        "ok": False,
        "error": "sin backend STT (instala speech_recognition o vosk)",
        "duracion_ms": round((time.perf_counter() - inicio) * 1000.0, 1),
    }


def _escuchar_sr(
    idioma: str, timeout: float, duracion_max: float, inicio: float
) -> Dict[str, Any]:
    import speech_recognition as sr

    rec = sr.Recognizer()
    try:
        with sr.Microphone() as fuente:
            rec.adjust_for_ambient_noise(fuente, duration=0.5)
            audio = rec.listen(fuente, timeout=timeout, phrase_time_limit=duracion_max)
        texto = rec.recognize_google(audio, language=idioma)
    except Exception as exc:
        return {
            "texto": "",
            "backend": "speech_recognition",
            "ok": False,
            "error": str(exc)[:120],
            "duracion_ms": round((time.perf_counter() - inicio) * 1000.0, 1),
        }
    return {
        "texto": (texto or "").strip(),
        "backend": "speech_recognition",
        "ok": True,
        "error": "",
        "duracion_ms": round((time.perf_counter() - inicio) * 1000.0, 1),
    }


def _escuchar_vosk(duracion_max: float, inicio: float) -> Dict[str, Any]:
    import json as _json

    import vosk

    modelo = next(MODELOS_STT_DIR.glob("*/am/final.mdl")).parents[1]
    try:
        import sounddevice as sd

        samplerate = 16000
        frames = int(samplerate * min(duracion_max, 15.0))
        audio = sd.rec(frames, samplerate=samplerate, channels=1, dtype="int16")
        sd.wait()
    except Exception as exc:
        return {
            "texto": "",
            "backend": "vosk",
            "ok": False,
            "error": f"microfono: {exc}"[:120],
            "duracion_ms": round((time.perf_counter() - inicio) * 1000.0, 1),
        }
    try:
        rec = vosk.KaldiRecognizer(vosk.Model(str(modelo)), 16000)
        rec.AcceptWaveform(audio.tobytes())
        texto = _json.loads(rec.FinalResult()).get("text", "")
    except Exception as exc:
        return {
            "texto": "",
            "backend": "vosk",
            "ok": False,
            "error": str(exc)[:120],
            "duracion_ms": round((time.perf_counter() - inicio) * 1000.0, 1),
        }
    return {
        "texto": (texto or "").strip(),
        "backend": "vosk",
        "ok": True,
        "error": "",
        "duracion_ms": round((time.perf_counter() - inicio) * 1000.0, 1),
    }


def escuchar_o_pedir(
    prompt_texto: str = "> Tu (voz o texto): ",
    idioma: str = "es-ES",
) -> str:
    """Intenta voz primero; si no hay backend, pide texto por teclado."""
    res = escuchar(idioma=idioma)
    if res["ok"] and res["texto"]:
        print(f"  [voz] {res['texto']}")
        return res["texto"]
    if res["error"]:
        print(f"  [voz no disponible: {res['error']}]")
    try:
        return input(prompt_texto).strip()
    except (EOFError, KeyboardInterrupt):
        return ""
