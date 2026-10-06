"""Vault OpenRouter Fernet — STM_SGR. Key cifrada embebida, solo free-models."""
from __future__ import annotations
import base64
import hashlib
import json
import os
import urllib.request
from pathlib import Path
from typing import Dict, List

SRC_DIR = Path(__file__).resolve().parents[1]
VAULT_FILE = Path(__file__).resolve().parent / "vault.enc"
FREE_SUFFIX = ":free"
API_MODELS_URL = "https://openrouter.ai/api/v1/models"

# Derivación simple de clave Fernet sin dependencias externas pesadas.
# Usa hashlib + XOR stream para no exigir `cryptography` en clones mínimos.
_MASTER_SEED = "LucIA-STM-SGR-free-vault-2026-v1"
_VAULT_BLOB_B64 = ""  # Se rellena con set_vault_blob(); vacío = sin key provisionada.


def _derive_key(master: str, salt: bytes) -> bytes:
    return hashlib.pbkdf2_hmac("sha256", master.encode(), salt, 100_000, dklen=32)


def cifrar_key(api_key: str, master: str = None, salt: bytes = None) -> str:
    """Cifra la API key con PBKDF2-HMAC-SHA256 + XOR sobre keystream derivado.

    `salt` es aleatorio por defecto. Pasarlo explicitamente permite cifrado
    determinista (mismo texto -> mismo blob), necesario para deduplicar o
    para regenerar un blob desde una fuente conocida.
    """
    m = master or os.getenv("OPENROUTER_VAULT_PW", _MASTER_SEED)
    sal = salt if salt is not None else os.urandom(16)
    key = _derive_key(m, sal)
    raw = api_key.encode()
    enc = bytes(a ^ b for a, b in zip(raw, key * (len(raw) // 32 + 1)))
    return base64.urlsafe_b64encode(b"V" + sal + enc).decode()


def descifrar_blob(blob_b64: str, master: str = None) -> str:
    """Revierte cifrar_key(). Detecta el formato legacy XOR (sin prefijo de version)."""
    m = master or os.getenv("OPENROUTER_VAULT_PW", _MASTER_SEED)
    data = base64.urlsafe_b64decode(blob_b64.encode())
    if data[:1] == b"V":
        _, sal, enc = data[0:1], data[1:17], data[17:]
        key = _derive_key(m, sal)
        return bytes(a ^ b for a, b in zip(enc, key * (len(enc) // 32 + 1))).decode()
    return _descifrar_legacy_xor(data, m)


def _descifrar_legacy_xor(data: bytes, m: str) -> str:
    """Descifra blobs antiguos generados con XOR + SHA-256 keystream sin salt."""
    def _keystream(n: int) -> bytes:
        out = b""
        ctr = 0
        while len(out) < n:
            out += hashlib.sha256(f"{m}:{ctr}".encode()).digest()
            ctr += 1
        return out[:n]
    return bytes(a ^ b for a, b in zip(data, _keystream(len(data)))).decode()


def _leer_blob() -> str:
    if _VAULT_BLOB_B64:
        return _VAULT_BLOB_B64
    if VAULT_FILE.exists():
        try:
            return VAULT_FILE.read_text(encoding="utf-8").strip()
        except Exception:
            return ""
    # Migración una sola vez desde .env local (luego borrar .env real).
    envf = Path(__file__).resolve().parent / ".env"
    if envf.exists():
        for line in envf.read_text(encoding="utf-8-sig", errors="replace").splitlines():
            k, _, v = line.strip().partition("=")
            if k.strip() == "OPENROUTER_API_KEY" and v.strip():
                return cifrar_key(v.strip().strip("'\""))
    return ""


def obtener_key() -> str:
    blob = _leer_blob()
    if not blob:
        return ""
    try:
        return descifrar_blob(blob).strip()
    except Exception:
        return ""


def origen_key() -> str:
    """usuario (vault.enc) | compartida (blob) | env (.env) | ninguna."""
    if _VAULT_BLOB_B64:
        return "compartida"
    if VAULT_FILE.exists() and VAULT_FILE.stat().st_size > 10:
        return "usuario"
    envf = Path(__file__).resolve().parent / ".env"
    if envf.exists():
        return "env"
    if os.getenv("OPENROUTER_API_KEY", "").strip():
        return "env"
    return "ninguna"


def hay_key_usuario() -> bool:
    return origen_key() == "usuario"


def registrar_key_usuario() -> bool:
    """El usuario registra SU key: se cifra en vault.enc (ignorado por git)."""
    try:
        import getpass
        raw = getpass.getpass("Pega tu API key de OpenRouter (vacío cancela): ").strip()
    except Exception:
        try:
            raw = input("Pega tu API key de OpenRouter (vacío cancela): ").strip()
        except Exception:
            return False
    if not raw or len(raw) < 10:
        print("Registro cancelado.")
        return False
    try:
        VAULT_FILE.write_text(cifrar_key(raw), encoding="utf-8")
        print(f"Key registrada y cifrada en vault.enc. Muestra: {enmascarar(raw)}")
        return True
    except Exception as e:
        print(f"No se pudo guardar: {e}")
        return False


def enmascarar(key: str) -> str:
    if not key:
        return "****"
    return f"{key[:4]}...****{key[-4:]}" if len(key) > 8 else "****"


def es_modelo_free(model_id: str) -> bool:
    return model_id.strip().endswith(FREE_SUFFIX)


def confirmar_conexion() -> bool:
    """Pregunta s/n sin desvelar la key. True si conecta a free-models."""
    try:
        r = input("¿Desea confirmar conexión con OpenRouter a modelos IA free? s/n: ").strip().lower()
    except (KeyboardInterrupt, EOFError):
        print()
        return False
    if r not in ("s", "si", "sí", "y", "yes"):
        print("Conexión OpenRouter omitida. Modo local/reflejo.")
        return False
    key = obtener_key()
    if not key:
        print("Vault sin key provisionada. Modo local/reflejo.")
        return False
    try:
        req = urllib.request.Request(API_MODELS_URL, headers={"Authorization": f"Bearer {key}"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8", "replace"))
        ids: List[str] = [m.get("id", "") for m in data.get("data", [])]
        frees = [i for i in ids if i.endswith(FREE_SUFFIX)]
        print(f"Conectado: {len(frees)} modelos free. Key: {enmascarar(key)}")
        return True
    except Exception as e:
        print(f"Conexión fallida ({e}). Key: {enmascarar(key)}. Modo local/reflejo.")
        return False


def aprovisionar_desde_env() -> bool:
    """Cifra .env -> vault.enc. Uso una sola vez por mantenedor."""
    envf = Path(__file__).resolve().parent / ".env"
    key = ""
    if envf.exists():
        for line in envf.read_text(encoding="utf-8-sig", errors="replace").splitlines():
            k, _, v = line.strip().partition("=")
            if k.strip() == "OPENROUTER_API_KEY" and v.strip():
                key = v.strip().strip("'\"")
    if not key:
        print("No hay OPENROUTER_API_KEY en .env.")
        return False
    VAULT_FILE.write_text(cifrar_key(key), encoding="utf-8")
    print(f"vault.enc generado. Key: {enmascarar(key)}. Borra el .env real y rota si se expuso.")
    return True


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "aprovisionar":
        raise SystemExit(0 if aprovisionar_desde_env() else 1)
    raise SystemExit(0 if confirmar_conexion() else 2)
