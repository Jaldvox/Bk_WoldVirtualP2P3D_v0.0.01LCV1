"""Fase 4: base de conocimiento especifico."""
from __future__ import annotations

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC_DIR))

from STM_IA.conocimiento import buscar, contexto_para_prompt, listar_fuentes


def test_buscar_red_neuronal():
    res = buscar("cuantas neuronas tiene la red neuronal")
    assert res, "deberia encontrar red_neuronal.md"
    assert res[0]["fuente"] == "red_neuronal.md"
    assert "50" in res[0]["fragmento"]


def test_buscar_blockchain():
    res = buscar("que es el consenso PoNL y el minado")
    assert res, "deberia encontrar blockchain.md"
    assert any("BKSVCB" in r["fragmento"] or "PoNL" in r["fragmento"] for r in res)


def test_buscar_sin_resultados():
    assert buscar("xyzzy física cuántica ornitorrinco") == []
    assert buscar("") == []


def test_buscar_top_k_y_puntuacion():
    res = buscar("modelo comando sesion", top_k=1)
    assert len(res) <= 1
    assert 0.0 < res[0]["puntuacion"] <= 1.0


def test_contexto_para_prompt_con_fuentes():
    ctx = contexto_para_prompt("blockchain minado bloques")
    assert "Conocimiento relevante" in ctx
    assert ".md" in ctx


def test_contexto_para_prompt_vacio():
    assert contexto_para_prompt("xyzzy ornitorrinco cuántico") == ""


def test_listar_fuentes():
    fuentes = listar_fuentes()
    assert "red_neuronal.md" in fuentes
    assert "blockchain.md" in fuentes
    assert "comandos.md" in fuentes


def test_conocimiento_no_supera_450_lineas():
    n = len((SRC_DIR / "STM_IA" / "conocimiento.py").read_text(encoding="utf-8").splitlines())
    assert n <= 450, f"conocimiento.py tiene {n} lineas"