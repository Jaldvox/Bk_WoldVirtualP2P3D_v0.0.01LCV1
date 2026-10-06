"""Registro del manifiesto STM_HRTS/pyproject.toml.

Expone metadatos del proyecto (versión, dependencias, scripts,
config ruff/pytest) sin depender de setuptools en runtime.
"""
from __future__ import annotations
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

STM_HRTS_DIR: Path = Path(__file__).resolve().parent
PYPROJECT_PATH: Path = STM_HRTS_DIR / "pyproject.toml"


def _cargar_toml() -> Dict[str, Any]:
    try:
        import tomllib
        return tomllib.loads(PYPROJECT_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


class RegistroHRTS:
    """Lectura instanciable del manifiesto del sistema."""

    def __init__(self) -> None:
        self._datos = _cargar_toml()
        self._proyecto = self._datos.get("project", {})

    def existe(self) -> bool:
        return PYPROJECT_PATH.exists()

    def nombre(self) -> str:
        return str(self._proyecto.get("name", "desconocido"))

    def version(self) -> str:
        return str(self._proyecto.get("version", "0.0.0"))

    def dependencias(self) -> List[str]:
        return list(self._proyecto.get("dependencies", []))

    def opcionales(self) -> Dict[str, List[str]]:
        return dict(self._proyecto.get("optional-dependencies", {}))

    def scripts(self) -> Dict[str, str]:
        return dict(self._proyecto.get("scripts", {}))

    def linea_estado(self) -> str:
        if not self.existe():
            return "STM_HRTS sin pyproject.toml"
        return f"{self.nombre()} {self.version()} | deps: {len(self.dependencias())}"

    def verificar_entorno(self) -> Dict[str, Any]:
        faltan: List[str] = []
        for dep in self.dependencias():
            mod = dep.split(">=")[0].split("==")[0].split("<")[0].strip().replace("-", "_")
            try:
                __import__(mod)
            except Exception:
                faltan.append(dep)
        return {"python": sys.version.split()[0], "manifiesto": str(PYPROJECT_PATH),
                "dependencias_ok": not faltan, "faltantes": faltan}


_instancia: Optional[RegistroHRTS] = None


def get_registro_hrts() -> RegistroHRTS:
    global _instancia
    if _instancia is None:
        _instancia = RegistroHRTS()
    return _instancia
