# LucIA · README del sistema — plan de desarrollo y hoja de ruta

Fecha: 2026-09-25 · Árbol vivo: `VersionDe_sesion/CONTRRF/TLLDCD/src` (+ `STM_RFC/RFPRMN.py`)
Arranque: `python src/STMGNRL/mainLCSTM.py` → confirma `s` → 4/4 verde.

---

## 1. Estado actual (verificado en ejecución)

| # | Subsistema | Estado |
|---|-----------|--------|
| 1 | Vault OpenRouter (`STM_SGR/vault_openrouter.py`) — key cifrada, terminal solo pregunta `s/n`, key enmascarada `sk-o...****`, solo modelos `:free` | Operativo (20 modelos free) |
| 2 | Blockchain BKSVCB (`STM_BKCH/`) — PoNL, minado cada 3 turnos, ledger en `STM_CH/` | Operativa |
| 3 | Conversor PSNRCV — **shim local** (`compat_local.py`, 50 neuronas sintéticas) | Funcional, pendiente real (deuda D1) |
| 4 | IPFS — `ipfs_launcher.py` + `bat_runner.py` + `IPFSManager` **shim no-op** (CIDs `QmLocal…`) | Sin daemon real (deuda D2) |
| 5 | Sesión P2P (`SBSTM/`) + STYLOS + RPLC + voz | Operativa |
| 6 | `STM_CH/` — todo el runtime (`pycache`, `PSNRL/`, ledger, cachés) + `rutas.limpiar()` al cierre | Operativo |
| 7 | `STM_JSON/registro_json.py` — 5 JSON registrados (canónicos + volátiles) | Operativo |
| 8 | `STM_HRTS/registro_hrts.py` — manifiesto `pyproject.toml` | Operativo |
| 9 | `RFPRMN.py` (`STM_RFC/`) — comandos `enviar`, `refactorizar`, `actualizar`, `ia local`, `ds ialocal` | Operativos |
| 10 | Flujo de entrega `enviar → RFC → actualizar → PRY/actualizacion_<id>` verificado 614/614 | Operativo |

Reglas: ningún `.py` supera 450 líneas (`mainLCSTM.py:449`); `src/` es el único módulo en raíz del proyecto (orquestador en `STMGNRL/`).

---

## 2. Hoja de ruta hacia la autonomía (desde el diagnóstico de LucIA)

El diagnóstico identifica 10 requisitos. Mapeo a fases:

| Fase | Requisito del diagnóstico | Trabajo concreto |
|------|---------------------------|------------------|
| F1 | Objetivos explícitos y verificables | Definir `objetivos.json` en `STM_JSON` + evaluador que mida éxito/fracaso por turno |
| F2 | Percepción del entorno | Exponer a LucIA el estado real: `verificar_integridad_sbstm()`, telemetría y `envios.json` como contexto |
| F3 | Modelo interno del mundo | Persistir `estado_mundo.json` (bloques, deriva, modelos, daemon IPFS) entre sesiones |
| F4 | Planificación y decisiones | Bucle plan→actúa→verifica reutilizando `FlujoRefactorLucIA` como máquina de estados |
| F5 | Memoria persistente | Sutituir `memoria.md` simulada por log real de turnos + pesos (hoy se borran al cerrar) |
| F6 | Herramientas ejecutables | Habilitar a LucIA los comandos RFPRMN (`enviar`, `actualizar`) vía herramienta, no solo operador humano |
| F7 | Gestión de recursos | Presupuestos: timeout, nº turnos, rotating de modelos con coste/latencia (ya hay métricas `$0.00`) |
| F8 | Autoevaluación y aprendizaje | Comparar respuesta vs objetivo (F1) y ajustar `temperatura`/modelo automáticamente |
| F9 | Seguridad | Vault ya existe; falta: permisos por herramienta, trazabilidad en ledger (ya hay `ACTUALIZACION_SINAPTICA`), límites de escritura |
| F10 | Supervisión humana | Mantener confirmación `s/n` como patrón para toda acción destructiva (ya es norma en RFPRMN) |

Orden sugerido: F1 → F5 → F2 → F9 → F6 → F3 → F8 → F4 → F7 → F10.

---

## 3. Deuda técnica observada

| ID | Deuda | Impacto | Remedio |
|----|-------|---------|---------|
| D1 | `ConversorRespuestaPesos` es un shim sintético; el `PSNRCV` real nunca se portó a `src/` | Pesos/deriva no son aprendizaje real | Portar transductor sobre `LC_STM/red_neuronal` (50 neuronas reales) |
| D2 | `IPFSManager` no-op + daemon Kubo sin arrancar en el flujo | CIDs `QmLocal…` falsos, sin persistencia P2P | Arrancar daemon vía `bat_runner` en `inicializar_subsistemas` |
| D3 | Key OpenRouter compartida embebida + `.env` con key real en disco | Extraíble por ingeniería inversa; rotar si se expuso | Rotar key; evaluar proxy propio |
| D4 | `RFPRMN.py` ~1800 líneas (tope 450) | Inmanejable, duplica `def ejecutar` muerto (líneas 1393/1395) | Dividir en `comandos_*.py` + `flujo_*.py` |
| D5 | Rutas legacy `LC.celebro…`, `Construccion/…` por todo el código | Confusión + ramas muertas | Eliminar tras migrar `encontrar_raiz` solo al layout actual |
| D6 | `.ipfs/` (miles de blobs) dentro de `src/` | Peso del repo, se copia en cada `enviar` | Mover fuera de `src/` o ignorar en copias |
| D7 | Sin tests ejecutables (`pytest` apunta a `LC/celebro/test` inexistente) | Sin red de seguridad | `STM_HRTS/test/` mínimo: vault, registro, enviar/verificar |
| D8 | `envios.json` cuenta ficheros pero no hashes | Una copia corrupta con igual conteo pasaría | Añadir hash SHA-256 por fichero (muestreo) |
| D9 | `memoria.md` con respuestas simuladas (2026-09-22) | Falsa memoria | F5 de la hoja de ruta |
| D10 | Limpieza `STM_CH` depende del cierre limpio; sondas `python -c` dejan `.pyc` | Residuos ocasionales | `PYTHONDONTWRITEBYTECODE=1` en sondas o gancho `atexit` en herramientas |

---

## 4. Comandos operativos

```powershell
# Arrancar LucIA
python src/STMGNRL/mainLCSTM.py     # responder s · salir con: salir
# Herramienta de proyecto
python TLLDCD/STM_RFC/RFPRMN.py     # enviar · refactorizar · actualizar · ia local · ds ialocal
# Flujo de entrega
enviar       # src -> CONTRRF/RFC/src (verificado, registrado)
refactorizar # PRY/ultima/src -> RFC/src (verificado)
actualizar   # RFC -> Sistema_Principal/PRY/actualizacion_<id> + purga + vacía RFC
```
