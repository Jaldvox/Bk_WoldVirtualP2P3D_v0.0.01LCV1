# Registro de desarrollo — comando `ds ialocal`

Fecha: 2026-09-24 · Archivo principal: `VersionDe_sesion/CONTRRF/RFPRMN.py` (1528 líneas)

---

## 1. Qué se pidió

Añadir al gestor el comando `ds ialocal` para que, dado un archivo:

1. Pida la ruta del archivo a refactorizar.
2. LucIA muestre diagnóstico y plan con sus propias palabras, y pregunte `s/n`.
3. Si `no`: ofrezca `1` (añadir algo) o `2` (diagnóstico más completo).
4. Si `sí`: descargue una IA de código, refactorice respetando el tope de **400/450 líneas**,
   registre en `.md` lo que hizo junto al nombre del modelo, y borre la IA para no repetirla.

---

## 2. Lo implementado hoy

| # | Componente | Ubicación | Estado |
|---|-----------|-----------|--------|
| 1 | Comando `ds ialocal` en el menú | `RFPRMN.py:1193` | Funciona |
| 2 | Orquestador `FlujoRefactorLucIA` (máquina de estados) | `RFPRMN.py:1247` | Funciona |
| 3 | Fases `[1/5]…[5/5]` con `try/finally` | `RFPRMN.py:1393` | Funciona |
| 4 | Diagnóstico + plan + `s/n` + opciones `1/2` | `RFPRMN.py:1273` | Funciona |
| 5 | Análisis estático local (AST, métricas) | `RFPRMN.py:363` | Funciona |
| 6 | Validación de sintaxis antes de sobrescribir | `RFPRMN.py:1121` | Funciona (evita destroys) |
| 7 | Rotación de hasta 3 IAs con `ollama rm` | `RFPRMN.py:1308` | Funciona |
| 8 | Registro `.md` con nota de LucIA + `### Errores` + `### Rotación` | `RFPRMN.py:1383` | Funciona |
| 9 | Descarga GGUF de HuggingFace con progreso visible | `RFPRMN.py:735` | Funciona (probado: 1.117.320.768 bytes) |
| 10 | Borrado automático del GGUF al terminar | `RFPRMN.py:852` | Funciona (verificado) |
| 11 | Filtro de candidatos solo de código + sin alias duplicados | `RFPRMN.py:815` | Funciona |
| 12 | Liberación de CPU (`ollama stop`) antes de generar | `RFPRMN.py:977` | Funciona |
| 13 | Vigilante de 60 s por fragmento | `RFPRMN.py:1003` | Funciona |
| 14 | Fallback a OpenRouter free (rotación de modelos) | `RFPRMN.py:1211` | Parcial: límites 429 |
| 15 | Corrección de ruta `.env` para conectar LucIA con OpenRouter | `IAFREE.py:81`, `mainLCSTM.py:20` | Funciona (`auth: True`) |
| 16 | Salida segura en consolas CP1252 | `RFPRMN.py:126` | Funciona |
| 17 | Detección de refactor *no-op* (IA devuelve el código igual) | `RFPRMN.py:1419` | Funciona |
| 18 | **Refactorización real del código** | `RFPRMN.py:1316`, `1343` | **FALLA (ver sección 4)** |

---

## 3. Verificaciones ejecutadas

- `python -m py_compile RFPRMN.py` → **OK**
- Descarga real del GGUF `Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF` → **OK**, 1.117.320.768 bytes,
  con progreso visible y borrado posterior verificado.
- Test unitario de `FlujoRefactorLucIA` con generador falso → **OK**:
  - fragmento roto → 2 reintentos → `irreparable=True` → se conserva el original;
  - fragmento que se arregla en el 2.º intento → `OK`;
  - ensamblado con mixto → `ast.parse` **OK**;
  - `registro_refactor.md` contiene `### Errores` e `IRREPARABLE`.
- Autenticación OpenRouter leída desde `RFC/LC/LC/.env` → **OK**.

---

## 4. Causa raíz de que "no funciona" (verificada, no supuesta)

El problema **no** es la descarga, ni el borrado, ni la validación: es el **troceado del archivo**.

`generar_fragmentos()` (`RFPRMN.py:1316`) corta el archivo cada ~60 líneas en una línea en blanco o
un `def/class/if/for…`. Eso produce fragmentos que **empiezan a mitad de una función**.
`_validar_salida_frag()` (`RFPRMN.py:1335`) valida cada fragmento con `ast.parse` **por separado**:

```
mainLCSTM.py (449 líneas) → 8 fragmentos
  frag 1 NO parsea: expected an indented block after 'for' (línea 65)
  frag 2..8 NO parsean: unexpected indent (línea 1)
  TOTAL 8 · parsean 0 · no parsean 8
```

**Los 8 fragmentos son Python inválido por sí solos.** Por tanto cualquier salida del modelo se
rechaza, el fragmento se marca `irreparable`, se conserva el original y el flujo termina "refactorizando"
un archivo idéntico. Además, como el modelo ve un fragmento que no compila, lo más probable es que lo
devuelva copiado (de ahí el *no-op* con `qwen2-math:1.5b`).

Problemas secundarios detectados:

- `_refactorizar_con_modelo()` (`RFPRMN.py:1078`) quedó **obsoleta** tras la clase: es código muerto.
- El solape de 10 líneas (`SOLAPE_LINEAS`) se concatena **detrás** del fragmento, sin marcar inicio/fin,
  por lo que el modelo no distingue el contexto de lo que debe refactorizar.
- OpenRouter free satura con `HTTP 429` de forma constante; la rotación ayuda pero no hay garantía.

---

## 5. Pendiente

### P0 — Arreglar el troceado (desbloquea todo lo demás)

- [ ] Trocear por **nodos AST de nivel superior** (`ast.get_source_segment`) en vez de por número de líneas.
      Así cada fragmento es una unidad sintáctica completa y `ast.parse` por fragmento es válido.
- [ ] Validar el **ensamblado completo**, no el fragmento aislado (o validar el fragmento solo cuando
      sea una unidad AST completa).
- [ ] Marcar el solape con delimitadores (`# --- CONTEXTO (no repetir) ---`) y colocarlo **antes** del código.

### P1 — Limpieza de código

- [ ] Eliminar `_refactorizar_con_modelo()` (`RFPRMN.py:1078`), código muerto.
- [ ] Eliminar `_leer_candidatos_codigo` reliance en aliases y unificar con `IAlocal.json`.
- [ ] Reducir `RFPRMN.py`: 1528 líneas exceden con creces el propio límite de 450 líneas del proyecto.
      Extraer `FlujoRefactorLucIA` y los clientes de IA a módulos propios.

### P2 — Robustez

- [ ] Prueba automática end-to-end del flujo con IA simulada (rápida, sin descargar modelos).
- [ ] Reintento con espera exponencial ante `429` de OpenRouter.
- [ ] Métrica de calidad del refactor: % de líneas realmente modificadas; si es ~0, tratarlo como fallo.
- [ ] Copia de seguridad (`.bak`) del archivo antes de sobrescribir, por si la validación pasa pero el
      comportamiento se rompe.

### P3 — Opcional

- [ ] Backend `llama.cpp` con SYCL para aprovechar la Intel Arc A750 (Ollama en Windows solo acelera con
      NVIDIA/AMD; hoy el 100 % del cómputo es CPU).
- [ ] Sustituir la IA de diagnóstico por defecto (`cogito:3b`, que agota el tiempo de espera) por un
      modelo más rápido o por el respaldo determinista.

---

## 6. Estado del último archivo refactorizado

`RFC/LC/LC/mainLCSTM.py` — 449 líneas, sintaxis válida, **sin refactor real aplicado** (solo el cambio
manual de `ENV_FILE` en la línea 20 para localizar `RFC/LC/LC/.env`).
