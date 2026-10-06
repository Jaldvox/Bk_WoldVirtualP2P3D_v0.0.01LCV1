"""Fase 3: entrada de voz STT (con mocks, sin microfono real)."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

SRC_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC_DIR))

from SBSTM import escucha


def test_backend_disponible_estructura():
    b = escucha.backend_disponible()
    assert set(b) == {"speech_recognition", "vosk", "modelo_vosk"}
    assert all(isinstance(v, bool) for v in b.values())


def test_escuchar_sin_backend_no_lanza():
    with patch.object(escucha, "backend_disponible", return_value={
        "speech_recognition": False, "vosk": False, "modelo_vosk": False,
    }):
        r = escucha.escuchar()
    assert r["ok"] is False
    assert r["texto"] == ""
    assert r["backend"] == "ninguno"


def test_escuchar_sr_ok_con_mock():
    import types

    fake_sr = types.ModuleType("speech_recognition")

    class FakeRec:
        def adjust_for_ambient_noise(self, fuente, duration=0.5):
            pass

        def listen(self, fuente, timeout=None, phrase_time_limit=None):
            return b"audio"

        def recognize_google(self, audio, language=None):
            return "hola lucia"

    class FakeMic:
        def __enter__(self):
            return object()

        def __exit__(self, *a):
            return False

    fake_sr.Recognizer = FakeRec
    fake_sr.Microphone = FakeMic
    with patch.dict(sys.modules, {"speech_recognition": fake_sr}):
        r = escucha.escuchar()
    assert r["ok"] is True
    assert r["texto"] == "hola lucia"
    assert r["backend"] == "speech_recognition"


def test_escuchar_sr_error_no_lanza():
    import types

    fake_sr = types.ModuleType("speech_recognition")

    class FakeRec:
        def adjust_for_ambient_noise(self, fuente, duration=0.5):
            pass

        def listen(self, fuente, timeout=None, phrase_time_limit=None):
            raise OSError("sin microfono")

    class FakeMic:
        def __enter__(self):
            return object()

        def __exit__(self, *a):
            return False

    fake_sr.Recognizer = FakeRec
    fake_sr.Microphone = FakeMic
    with patch.dict(sys.modules, {"speech_recognition": fake_sr}):
        r = escucha.escuchar()
    assert r["ok"] is False
    assert "error" in r


def test_escuchar_o_pedir_fallback_a_teclado():
    with patch.object(escucha, "escuchar", return_value={
        "ok": False, "texto": "", "error": "sin backend",
    }):
        with patch("builtins.input", return_value="texto manual"):
            assert escucha.escuchar_o_pedir() == "texto manual"


def test_escucha_no_supera_450_lineas():
    n = len((SRC_DIR / "SBSTM" / "escucha.py").read_text(encoding="utf-8").splitlines())
    assert n <= 450, f"escucha.py tiene {n} lineas"