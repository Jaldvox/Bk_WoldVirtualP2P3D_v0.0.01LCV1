"""Fase 2: inteligencia emocional de entrada."""
from __future__ import annotations

import sys
import time
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC_DIR))

from SBSTM.emocion import analizar_emocion, modular_prosodia


def test_emocion_positiva():
    r = analizar_emocion("Estoy muy feliz y contento, gracias!")
    assert r["etiqueta"] == "positiva"
    assert r["valencia"] > 0.15


def test_emocion_negativa():
    r = analizar_emocion("Estoy triste y preocupado por este problema")
    assert r["etiqueta"] == "negativa"
    assert r["valencia"] < -0.15


def test_emocion_neutral_y_vacia():
    assert analizar_emocion("el bloque tiene un hash")["etiqueta"] == "neutral"
    assert analizar_emocion("")["valencia"] == 0.0


def test_emocion_negacion_invierte():
    r = analizar_emocion("no estoy contento con esto")
    assert r["valencia"] < 0.15


def test_emocion_rango_acotado():
    for txt in ["muy muy muy feliz genial excelente", "horrible terrible peor imposible"]:
        r = analizar_emocion(txt)
        assert -1.0 <= r["valencia"] <= 1.0
        assert 0.0 <= r["confianza"] <= 1.0


def test_emocion_latencia():
    t0 = time.perf_counter()
    for _ in range(100):
        analizar_emocion("estoy contento con el resultado del minado")
    dt = (time.perf_counter() - t0) * 1000.0 / 100
    assert dt < 5.0, f"latencia media {dt:.2f} ms > 5 ms"


def test_modular_prosodia_consejos():
    _, c1 = modular_prosodia(-0.8)
    _, c2 = modular_prosodia(0.8)
    _, c3 = modular_prosodia(0.0)
    assert (c1, c2, c3) == ("empatico", "celebratorio", "neutral")