"""Paquete IALOCAL runtime (gestor por pregunta)."""
from .gestor import (
    ErrorIALocal, GestorIALocal, IADS_BR, borrar_modelo, elegir_modelo,
    generar, get_gestor_ialocal, modelos_instalados, ollama_disponible,
)

__all__ = ["ErrorIALocal", "GestorIALocal", "IADS_BR", "borrar_modelo", "elegir_modelo",
           "generar", "get_gestor_ialocal", "modelos_instalados", "ollama_disponible"]
