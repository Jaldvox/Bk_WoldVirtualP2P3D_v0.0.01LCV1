@echo off
:: ============================================================
:: ARRANQUE DAEMON IPFS - WoldVirtualP2P3D Celebro 2026
:: Kubo v0.43.1 | nodo local -> red IPFS global
:: ============================================================
set IPFS_PATH=%~dp0.ipfs
echo.
echo  ================================================
echo   IPFS DAEMON - WoldVirtualP2P3D Celebro 2026
echo   Kubo v0.43.1  ^|  PeerID: 12D3KooWQQMPQ...
echo   IPFS_PATH: %IPFS_PATH%
echo  ================================================
echo.
echo  Arrancando daemon IPFS... (Ctrl+C para detener)
echo  API HTTP disponible en: http://127.0.0.1:5001
echo  Gateway local disponible en: http://127.0.0.1:8080
echo.
"%~dp0ipfs.exe" daemon --enable-pubsub-experiment
pause
