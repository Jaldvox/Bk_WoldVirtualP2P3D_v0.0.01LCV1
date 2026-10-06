"""Fase 5: comandos extensibles + rubrica de calidad."""
from __future__ import annotations

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC_DIR))

from SBSTM import comandos
from SBSTM.rubrica import evaluar_calidad


def test_registrar_y_despachar():
    comandos.registrar_comando("eco-test", lambda args, ctx: f"eco:{args}",
                               ayuda="eco de prueba", alias=["ecotest"])
    assert comandos.es_comando("eco-test")
    assert comandos.es_comando("ecotest")
    assert comandos.despachar("eco-test hola") == "eco:hola"
    assert comandos.despachar("no-existe x") is None


def test_despachar_no_lanza():
    comandos.registrar_comando("roto-test", lambda a, c: 1 / 0, ayuda="siempre falla")
    r = comandos.despachar("roto-test")
    assert r is not None and "error" in r


def test_ayuda_texto_lista():
    comandos.registrar_comando("ayuda-test", lambda a, c: "", ayuda="ayuda de prueba")
    t = comandos.ayuda_texto()
    assert "ayuda-test" in t


def test_rubrica_respuesta_buena():
    r = evaluar_calidad(
        "Imagina un mundo nuevo donde cada bloque cuenta una historia.\n"
        "Primero porque los datos hablan, segundo porque el analisis\n"
        "profundo revela sentido. Prueba ahora este paso concreto?",
        tono="creativo",
    )
    assert 0.0 <= r["puntuacion"] <= 1.0
    assert r["veredicto"] in ("excelente", "buena", "aceptable", "pobre")
    assert set(r["metricas"]) == {
        "diversidad_lexica", "longitud_util", "coherencia_tono",
        "estructura", "cierre",
    }


def test_rubrica_vacia_y_corta():
    assert evaluar_calidad("")["veredicto"] == "vacia"
    r = evaluar_calidad("si")
    assert r["puntuacion"] < 0.5


def test_rubrica_nunca_lanza():
    r = evaluar_calidad(None, tono=None)
    assert "puntuacion" in r


def test_modulos_no_superan_450_lineas():
    for f in ["SBSTM/comandos.py", "SBSTM/rubrica.py"]:
        n = len((SRC_DIR / f).read_text(encoding="utf-8").splitlines())
        assert n <= 450, f"{f} tiene {n} lineas"