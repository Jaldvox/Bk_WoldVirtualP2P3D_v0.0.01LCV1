"""Fase 1: memoria contextual de sesion."""
from __future__ import annotations

import json
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC_DIR))


def _memoria_fresca(sesion_id: str):
    from SBSTM.memoria_sesion import MemoriaSesion

    m = MemoriaSesion(sesion_id=sesion_id)
    m.limpiar()
    return MemoriaSesion(sesion_id=sesion_id)


def test_memoria_registra_y_resume():
    m = _memoria_fresca("test_mem_01")
    assert m.resumen_contexto() == ""
    m.registrar_turno("hola", "hola, soy Lucia", tono="creativo", valencia=0.5)
    ctx = m.resumen_contexto()
    assert "hola" in ctx
    assert "Lucia" in ctx
    m.limpiar()


def test_memoria_ventana_deslizante():
    m = _memoria_fresca("test_mem_02")
    m.ventana = 3
    for i in range(5):
        m.registrar_turno(f"pregunta {i}", f"respuesta {i}")
    assert m.total() == 3
    assert "pregunta 0" not in m.resumen_contexto()
    assert "pregunta 4" in m.resumen_contexto()
    m.limpiar()


def test_memoria_persiste_en_psnrl():
    m = _memoria_fresca("test_mem_03")
    m.registrar_turno("persistir?", "si", tono="analitico", valencia=0.1)
    ruta = m._ruta
    assert ruta.exists()
    data = json.loads(ruta.read_text(encoding="utf-8"))
    assert data["sesion_id"] == "test_mem_03"
    assert len(data["turnos"]) == 1
    m2 = __import__("SBSTM.memoria_sesion", fromlist=["MemoriaSesion"]).MemoriaSesion(
        sesion_id="test_mem_03"
    )
    assert m2.total() == 1
    m.limpiar()


def test_memoria_singleton_por_sesion():
    from SBSTM.memoria_sesion import get_memoria

    a = get_memoria("test_mem_04")
    b = get_memoria("test_mem_04")
    assert a is b
    a.limpiar()
    c = get_memoria("test_mem_05")
    assert c is not a
    c.limpiar()


def test_memoria_no_supera_450_lineas():
    n = len((SRC_DIR / "SBSTM" / "memoria_sesion.py").read_text(encoding="utf-8").splitlines())
    assert n <= 450, f"memoria_sesion.py tiene {n} lineas"