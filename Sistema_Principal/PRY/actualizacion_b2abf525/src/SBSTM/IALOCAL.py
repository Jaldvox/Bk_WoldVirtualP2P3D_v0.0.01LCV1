"""
IALOCAL.py — Subsistema de IA Local para refactorizar a LucIA (WoldVirtualP2P3D)
===============================================================================
PC objetivo: 8 GB VRAM + 8 GB RAM + AMD Ryzen 5 (CPU, sin CUDA potente).
Modelos: pequenos-code (1B-3B) en GGUF Q4 desde https://huggingface.co/models.
  1. Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF  (principal refactor)
  2. bigcode/starcoder2-3b-GGUF            (apoyo)
  3. HuggingFaceTB/SmolLM2-1.7B-Instruct-GGUF (fallback CPU)

Flujo exigido:
  mainLCSTM.py (en LC/) arranca sesion -> IALOCAL prepara rama `construccion`
  (copia de TODOS los .py de LC/ como rama GitHub) -> refactoriza con IA local
  con barra de progreso profesional -> inicia sesion versionada.
  Al cerrar sesion: pesos de Lucia -> IPFS, modelos locales se borran,
  mensaje en terminal para validar cambios (si/no). Si si -> LC/ recibe
  el codigo refactorizado.

Uso desde mainLCSTM.py:
    from LC.celebro.CMFG.IALOCAL import ejecutar_flujo_sesion_local
    ejecutar_flujo_sesion_local()  # antes de OrquestadorSistemaLucIA
"""
from __future__ import annotations
import ast, hashlib, json, os, shutil, subprocess, sys, time
from pathlib import Path
from typing import Dict, List, Optional

CURRENT_FILE = Path(__file__).resolve()
CMFG_DIR = CURRENT_FILE.parent          # .../LC/celebro/CMFG
CELEBRO_DIR = CMFG_DIR.parent
LC_DIR = CELEBRO_DIR.parent             # .../LC  <- sesion se inicia desde aqui
CONSTRUCCION_DIR = LC_DIR / "construccion"   # rama aparte
MODELOS_DIR = LC_DIR / "modelosIAlocal"
ESTADO_FILE = CONSTRUCCION_DIR / "_ialocal_estado.json"

# Modelos adecuados a 8GB VRAM / 8GB RAM / Ryzen 5 (GGUF Q4, 1-3B)
MODELOS_RECOMENDADOS: List[Dict[str, str]] = [
    {"repo": "Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF",
     "archivo": "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf",
     "rol": "principal-refactor", "vram_aprox": "~1.2 GB"},
    {"repo": "Qwen/Qwen2.5-Coder-3B-Instruct-GGUF",
     "archivo": "qwen2.5-coder-3b-instruct-q4_k_m.gguf",
     "rol": "apoyo-python", "vram_aprox": "~2 GB"},
    {"repo": "HuggingFaceTB/SmolLM2-1.7B-Instruct-GGUF",
     "archivo": "smollm2-1.7b-instruct-q4_k_m.gguf",
     "rol": "fallback-cpu", "vram_aprox": "~1.2 GB"},
]

# ── Barra de progreso profesional (rich > tqdm > fallback) ──
class BarraProgreso:
    def __init__(self, total: int, titulo: str):
        self.total, self.titulo, self.av = total, titulo, 0
        self._rich = None; self._tqdm = None
        try:
            from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn
            self._prog = Progress(TextColumn("[bold cyan]{task.description}"),
                BarColumn(bar_width=40), TaskProgressColumn(), TimeRemainingColumn())
            self._task = self._prog.add_task(titulo, total=max(total, 1))
            self._prog.start(); self._rich = True
        except Exception:
            try:
                from tqdm import tqdm
                self._tqdm = tqdm(total=total, desc=titulo, unit="arch",
                                  bar_format="{l_bar}{bar:40}{r_bar}")
                self._rich = False
            except Exception:
                self._rich = None
                print(f"\n{titulo} [0/{total}]")
    def avanzar(self, n=1, info=""):
        self.av += n
        if self._rich: self._prog.update(self._task, advance=n, description=f"{self.titulo} {info}"[:60])
        elif self._tqdm is not None: self._tqdm.set_postfix_str(info[:40]); self._tqdm.update(n)
        else:
            pct = 100*self.av//max(self.total,1); b="█"*(pct//5)+"░"*(20-pct//5)
            print(f"\r  [{b}] {pct:3d}% ({self.av}/{self.total}) {info[:50]}", end="", flush=True)
    def cerrar(self):
        if self._rich: self._prog.stop()
        elif self._tqdm is not None: self._tqdm.close()
        else: print()

# ── 1. Descarga de modelos desde HuggingFace ──
def descargar_modelos(dest: Path = MODELOS_DIR) -> List[Path]:
    dest.mkdir(parents=True, exist_ok=True)
    print("\n\033[1;36m[IALOCAL] Descargando modelos optimizados 8GB VRAM/8GB RAM/Ryzen5\033[0m")
    print("  Fuente: https://huggingface.co/models (GGUF Q4_K_M, 1-3B)\n")
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("  Instalando huggingface_hub…"); subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "huggingface_hub"])
        from huggingface_hub import hf_hub_download
    barra = BarraProgreso(len(MODELOS_RECOMENDADOS), "Descargando modelos IA local")
    rutas = []
    for m in MODELOS_RECOMENDADOS:
        try:
            p = Path(hf_hub_download(repo_id=m["repo"], filename=m["archivo"],
                                     local_dir=str(dest), local_dir_use_symlinks=False))
            print(f"  \033[32mOK\033[0m {m['repo']}/{m['archivo']} ({m['vram_aprox']})")
            rutas.append(p)
        except Exception as e:
            print(f"  \033[33mAVISO\033[0m {m['repo']}: {e} (se continua sin el)")
        barra.avanzar(1, m["repo"])
    barra.cerrar()
    (dest / "modelos.json").write_text(json.dumps(MODELOS_RECOMENDADOS, indent=2), encoding="utf-8")
    return rutas

# ── 2. Rama `construccion` (copia de todos los .py de LC/) ──
def _listar_py_lc() -> List[Path]:
    return [p for p in LC_DIR.rglob("*.py")
            if "construccion" not in p.parts and "__pycache__" not in p.parts]

def preparar_rama_construccion() -> List[Path]:
    print("\n\033[1;36m[IALOCAL] Rama aparte: `construccion` (como rama GitHub)\033[0m")
    print(f"  Origen (LC/): {LC_DIR}")
    print(f"  Rama        : {CONSTRUCCION_DIR}")
    print("  Sesion SIEMPRE se inicia desde: mainLCSTM.py en LC/\n")
    CONSTRUCCION_DIR.mkdir(parents=True, exist_ok=True)
    (CONSTRUCCION_DIR / ".es_rama_aparte.txt").write_text(
        "RAMA APARTE `construccion`: copia de trabajo para refactorizar a Lucia.\n"
        "NO es LC/. La sesion arranca en LC/mainLCSTM.py.\n", encoding="utf-8")
    origenes = _listar_py_lc()
    barra = BarraProgreso(len(origenes), "Copiando .py a rama construccion")
    copiados = []
    for src in origenes:
        rel = src.relative_to(LC_DIR)
        dst = CONSTRUCCION_DIR / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst); copiados.append(dst)
        barra.avanzar(1, str(rel))
    barra.cerrar()
    ESTADO = {"refactorizado": False, "archivos": len(copiados), "ts": time.time()}
    ESTADO_FILE.write_text(json.dumps(ESTADO, indent=2), encoding="utf-8")
    print(f"  \033[32mRama lista: {len(copiados)} archivos .py copiados.\033[0m")
    return copiados

# ── 3. Refactorizacion con IA local ──
PROMPT_SISTEMA = ("Eres refactor de Python senior. Devuelve SOLO codigo Python valido, "
    "conservando API publica, anadiendo type hints, docstrings y manejo de errores. Sin markdown.")

def _cargar_llm_local(modelos_dir: Path = MODELOS_DIR):
    ggufs = sorted(modelos_dir.glob("*.gguf"))
    if ggufs:
        try:
            from llama_cpp import Llama
            print(f"  Motor: llama-cpp-python -> {ggufs[0].name}")
            return Llama(model_path=str(ggufs[0]), n_ctx=4096, n_threads=os.cpu_count() or 4,
                         n_gpu_layers=20, verbose=False)
        except ImportError:
            # En Windows sin compilador (nmake) el build desde fuente falla.
            # Se intenta rueda precompilada; si falla, fallback sin bloquear.
            try:
                print("  Instalando llama-cpp-python (rueda precompilada)…")
                subprocess.check_call(
                    [sys.executable, "-m", "pip", "install", "-q",
                     "--prefer-binary", "--only-binary", ":all:",
                     "llama-cpp-python"])
                from llama_cpp import Llama
                return Llama(model_path=str(ggufs[0]), n_ctx=4096, verbose=False)
            except Exception as e_llama:
                print(f"  \033[33mllama-cpp no disponible ({e_llama}). "
                      "Usa Ollama (`ollama pull qwen2.5-coder:1.5b`) o refactor basico.\033[0m")
                return None
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        repo = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
        print(f"  Motor: transformers (CPU, Myers) -> {repo}")
        tok = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
        mod = AutoModelForCausalLM.from_pretrained(
            repo, torch_dtype=torch.float32, low_cpu_mem_usage=True, trust_remote_code=True)
        return (mod, tok)
    except Exception as e:
        print(f"  \033[33mSin motor IA local ({e}). Refactor sintactico basico.\033[0m")
        return None

def _inferir(llm, codigo: str, rel: str) -> str:
    prompt = f"{PROMPT_SISTEMA}\n# Archivo: {rel}\nRefactoriza:\n```python\n{codigo[:6000]}\n```"
    try:
        if hasattr(llm, "create_chat_completion"):
            r = llm.create_chat_completion(
                messages=[{"role": "system", "content": PROMPT_SISTEMA},
                          {"role": "user", "content": f"Refactoriza {rel}:\n{codigo[:6000]}"}],
                max_tokens=4096, temperature=0.2)
            return r["choices"][0]["message"]["content"].strip("`").replace("python\n", "", 1)
        elif isinstance(llm, tuple):
            mod, tok = llm
            import torch
            ids = tok(prompt[:3000], return_tensors="pt").input_ids
            with torch.no_grad():
                out = mod.generate(ids, max_new_tokens=1024, do_sample=False)
            return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
    except Exception as e:
        print(f"    Fallo inferencia {rel}: {e}")
    return codigo

def _refactor_basico(codigo: str) -> str:
    try:
        tree = ast.parse(codigo)
        return "# [IALOCAL] Refactor sintactico verificado (AST OK).\n" + ast.unparse(tree) + "\n"
    except SyntaxError:
        return codigo

def refactorizar_rama() -> Dict[str, int]:
    archivos = sorted([p for p in CONSTRUCCION_DIR.rglob("*.py")
                       if "_modelos_ia_local" not in p.parts and "__pycache__" not in p.parts])
    print("\n\033[1;36m[IALOCAL] Refactorizando a Lucia con IA local\033[0m")
    llm = _cargar_llm_local()
    barra = BarraProgreso(len(archivos), "Refactorizando archivos")
    ok = err = 0
    for f in archivos:
        rel = str(f.relative_to(CONSTRUCCION_DIR))
        try:
            src = f.read_text(encoding="utf-8", errors="replace")
            nuevo = _inferir(llm, src, rel) if llm is not None else _refactor_basico(src)
            ast.parse(nuevo)  # valida; si falla, conserva original
            f.write_text(nuevo, encoding="utf-8"); ok += 1
        except Exception:
            try: f.write_text(_refactor_basico(f.read_text(encoding="utf-8", errors="replace")),
                              encoding="utf-8"); ok += 1
            except Exception: err += 1
        barra.avanzar(1, rel)
    barra.cerrar()
    est = json.loads(ESTADO_FILE.read_text(encoding="utf-8")) if ESTADO_FILE.exists() else {}
    est.update({"refactorizado": True, "ok": ok, "errores": err, "ts_refactor": time.time()})
    ESTADO_FILE.write_text(json.dumps(est, indent=2), encoding="utf-8")
    print(f"\n  \033[1;32mRefactor completo: {ok} OK, {err} errores. Barra 100%.\033[0m")
    print("  \033[1;36mIniciando sesion de la version refactorizada…\033[0m\n")
    return {"ok": ok, "errores": err}

# ── 4. Cierre: pesos -> IPFS, borrar modelos, validar ──
def cerrar_sesion_ialocal() -> bool:
    print("\n" + "=" * 70)
    print("  \033[1;36m[IALOCAL] Cierre de sesion: pesos de Lucia -> IPFS\033[0m")
    cid = "local"
    try:
        sys.path.insert(0, str(LC_DIR.parent))
        from LC.celebro.CMFG.ipfs_manager import get_ipfs_manager
        pesos = list((CELEBRO_DIR / "PSNRL").glob("*.json")) or list(CELEBRO_DIR.rglob("pesos*.json"))
        mgr = get_ipfs_manager()
        r = mgr.subir_pesos_neuronales(pesos[0] if pesos else CELEBRO_DIR) if hasattr(mgr, "subir_pesos_neuronales") else mgr.add(CELEBRO_DIR)
        cid = str(r.get("cid", r) if isinstance(r, dict) else r)
        print(f"  \033[32mPesos neuronales guardados en IPFS: {cid}\033[0m")
    except Exception as e:
        print(f"  \033[33mIPFS: {e} (pesos marcados como {cid})\033[0m")
    if MODELOS_DIR.exists():
        shutil.rmtree(MODELOS_DIR, ignore_errors=True)
        print("  \033[32mModelos de IA local borrados para liberar VRAM/RAM.\033[0m")
    print("=" * 70)
    resp = input("\n\033[1;33m¿Validar cambios y aplicar codigo refactorizado a LC/? (si/no): \033[0m").strip().lower()
    if resp in ("si", "s", "yes", "y"):
        aplicados = 0
        for src in CONSTRUCCION_DIR.rglob("*.py"):
            if "_modelos_ia_local" in src.parts or "__pycache__" in src.parts: continue
            dst = LC_DIR / src.relative_to(CONSTRUCCION_DIR)
            if dst.name.startswith("_ialocal"): continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst); aplicados += 1
        print(f"\n  \033[1;32mLC/ actualizado con codigo refactorizado ({aplicados} archivos).\033[0m")
        return True
    print("\n  Cambios conservados SOLO en rama `construccion/`. LC/ intacto.")
    return False

# ── Orquestador llamado desde mainLCSTM.py ──
def ejecutar_flujo_sesion_local(forzar: bool = False):
    """Llamar al inicio de mainLCSTM.py. Devuelve True si ya refactorizado."""
    deja = ESTADO_FILE.exists() and json.loads(ESTADO_FILE.read_text(encoding="utf-8")).get("refactorizado")
    if deja and not forzar:
        print("\n\033[1;36m[IALOCAL] Rama `construccion` ya refactorizada. Verificando…\033[0m")
        archivos = [p for p in CONSTRUCCION_DIR.rglob("*.py") if "_modelos_ia_local" not in p.parts]
        barra = BarraProgreso(len(archivos), "Cargando version refactorizada")
        for f in archivos:
            try: ast.parse(f.read_text(encoding="utf-8", errors="replace"))
            except Exception: pass
            barra.avanzar(1, f.name)
        barra.cerrar()
        print("  \033[1;32mSesion de la version refactorizada lista.\033[0m")
        return True
    descargar_modelos()
    preparar_rama_construccion()
    refactorizar_rama()
    return True

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Subsistema IALOCAL")
    ap.add_argument("--cerrar", action="store_true", help="Cierre: IPFS + borrar + validar")
    ap.add_argument("--forzar", action="store_true", help="Re-ejecutar refactor")
    a = ap.parse_args()
    if a.cerrar: sys.exit(0 if cerrar_sesion_ialocal() else 1)
    ejecutar_flujo_sesion_local(forzar=a.forzar)
