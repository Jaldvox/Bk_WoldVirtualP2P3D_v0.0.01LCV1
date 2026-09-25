"""Wrapper Python del .bat IPFS para gestión desde mainLCSTM."""
from __future__ import annotations
import socket
import subprocess
from pathlib import Path
from typing import Dict

BAT_DIR = Path(__file__).resolve().parent.parent / "STM_BAT_IPFS"
BAT_FILE = BAT_DIR / "arrancar_daemon.bat"
API_PORT = 5001
GATEWAY_PORT = 8080


def _puerto_abierto(puerto: int) -> bool:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(1.0)
    try:
        return s.connect_ex(("127.0.0.1", puerto)) == 0
    finally:
        s.close()


def estado_daemon() -> Dict[str, object]:
    return {"bat": str(BAT_FILE), "existe_bat": BAT_FILE.exists(),
            "api_5001": _puerto_abierto(API_PORT), "gateway_8080": _puerto_abierto(GATEWAY_PORT)}


def arrancar_daemon(esperar: bool = False) -> Dict[str, object]:
    """Lanza arrancar_daemon.bat en ventana propia; no bloquea por defecto."""
    if not BAT_FILE.exists():
        return {"ok": False, "error": f"No existe {BAT_FILE}"}
    try:
        if esperar:
            subprocess.run(["cmd", "/c", str(BAT_FILE)], check=False)
        else:
            subprocess.Popen(["cmd", "/c", "start", "IPFS-Daemon", str(BAT_FILE)],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return {"ok": True, "bat": str(BAT_FILE), **estado_daemon()}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def detener_daemon() -> Dict[str, object]:
    try:
        subprocess.run(["taskkill", "/F", "/IM", "ipfs.exe"], capture_output=True, check=False)
        return {"ok": True, **estado_daemon()}
    except Exception as e:
        return {"ok": False, "error": str(e)}
