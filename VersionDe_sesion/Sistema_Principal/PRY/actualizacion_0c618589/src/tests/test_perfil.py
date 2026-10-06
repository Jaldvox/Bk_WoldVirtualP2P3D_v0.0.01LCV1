"""Fase 6: perfil de usuario persistente."""
from __future__ import annotations

import json
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC_DIR))

from SBSTM.perfil import PerfilUsuario, get_perfil


def _perfil_limpio() -> PerfilUsuario:
    p = PerfilUsuario()
    p.reiniciar()
    return PerfilUsuario()


def test_perfil_aprende_turnos():
    p = _perfil_limpio()
    p.aprender_turno(tono="creativo", valencia=0.8, modelo="cogito:3b",
                     palabras_clave=["blockchain", "minado"])
    p.aprender_turno(tono="creativo", valencia=0.6, modelo="cogito:3b",
                     palabras_clave=["blockchain"])
    assert p.turnos_totales == 2
    assert p.tono_preferido() == "creativo"
    assert p.modelo_preferido() == "cogito:3b"
    assert p.temas_top(1) == ["blockchain"]
    assert 0.5 < p.valencia_media() < 0.9
    p.reiniciar()


def test_perfil_persiste_en_psnrl():
    p = _perfil_limpio()
    p.aprender_turno(tono="analitico", valencia=-0.2, modelo="qwen2.5:7b")
    assert p._ruta.exists()
    data = json.loads(p._ruta.read_text(encoding="utf-8"))
    assert data["turnos_totales"] == 1
    assert data["tonos"] == {"analitico": 1}
    p2 = PerfilUsuario()
    assert p2.turnos_totales == 1
    assert p2.tono_preferido() == "analitico"
    p.reiniciar()


def test_perfil_singleton():
    a = get_perfil()
    assert get_perfil() is a


def test_perfil_valencia_acotada():
    p = _perfil_limpio()
    p.aprender_turno(valencia=99.0)
    p.aprender_turno(valencia=-99.0)
    assert p.valencias == [1.0, -1.0]
    p.reiniciar()


def test_perfil_resumen_estructura():
    p = _perfil_limpio()
    r = p.resumen()
    assert set(r) == {
        "turnos_totales", "tono_preferido", "valencia_media",
        "modelo_preferido", "temas_top", "actualizado_ts",
    }
    p.reiniciar()


def test_perfil_no_supera_450_lineas():
    n = len((SRC_DIR / "SBSTM" / "perfil.py").read_text(encoding="utf-8").splitlines())
    assert n <= 450, f"perfil.py tiene {n} lineas"