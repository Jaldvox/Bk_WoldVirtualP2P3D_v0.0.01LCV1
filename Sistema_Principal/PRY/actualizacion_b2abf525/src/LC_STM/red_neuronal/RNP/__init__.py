"""
RNP - Red Neuronal de Pesos (Neural Weight Optimization Package)
================================================================
Paquete de optimizacion neuronal de pesos de alta precision matematica (2026).

Este paquete integra 10 neuronas especializadas (RN1..RN10), cada una implementando
algoritmos de optimizacion de pesos de ultima generacion:

  RN1  - AdamW          : Adam con desacoplamiento de decaimiento de pesos
  RN2  - RAdam          : Adam rectificado con varianza adaptativa
  RN3  - Lookahead      : Busqueda adelantada para estabilidad de convergencia
  RN4  - NAdam          : Adam con momentum de Nesterov
  RN5  - LAMB           : Layer-wise Adaptive Moments para grandes lotes
  RN6  - AdaBelief      : Adaptacion de creencias de gradiente
  RN7  - Lion           : Optimizador evolucionado de signo de gradiente
  RN8  - SAM / ASAM     : Minimizacion consciente de agudeza (Sharpness-Aware)
  RN9  - SWATS + Muon   : Conmutacion automatica Adam->SGD con descenso ortogonal
  RN10 - Integrado 2026 : Sistema unificado con K-KAN, SAM, Muon y SWATS

Uso basico::

    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from RNP import RNPRegistry, RNPFactory, RNPBenchmark
    optimizer = RNPFactory.create("swats")
    result    = RNPFactory.quick_run("adamw", model)
    report    = RNPBenchmark.run_all(model)

Versionado semantico::

    __version__   = "0.9.1"
    __api_level__ = 2026
"""
from __future__ import annotations

import importlib
import logging
import time
import traceback
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

# ---------------------------------------------------------------------------
# Metadatos del paquete
# ---------------------------------------------------------------------------
__version__   = "0.9.1"
__api_level__ = 2026
__author__    = "WoldVirtualP2P3D Team"
__all__: List[str] = []
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Mapa de modulos: alias -> (modulo, clase principal, factory_fn, quick_fn)
# ---------------------------------------------------------------------------
_MODULE_MAP: Dict[str, Tuple[str, str, str, str]] = {
    "adamw":      ("RN1",  "AdamWOptimizer",
                   "create_adamw_optimizer",                    "quick_adamw"),
    "radam":      ("RN2",  "RAdamOptimizer",
                   "create_radam_optimizer",                    "quick_radam"),
    "lookahead":  ("RN3",  "LookaheadOptimizer",
                   "create_lookahead_optimizer",                "quick_lookahead"),
    "nadam":      ("RN4",  "NadamOptimizer",
                   "create_nadam_optimizer",                    "quick_nadam"),
    "lamb":       ("RN5",  "LAMBOptimizer",
                   "create_lamb_optimizer",                     "quick_lamb"),
    "adabelief":  ("RN6",  "AdaBeliefOptimizer",
                   "create_adabelief_optimizer",                "quick_adabelief"),
    "lion":       ("RN7",  "LionOptimizer",
                   "create_lion_optimizer",                     "quick_lion"),
    "sam":        ("RN8",  "SAMOptimizer",
                   "create_sam_optimizer",                      "quick_sam"),
    "swats":      ("RN9",  "SWATSOptimizer",
                   "create_swats_optimizer",                    "quick_swats"),
    "integrated": ("RN10", "IntegratedNeuralWeightOptimizer",
                   "create_integrated_neural_weight_optimizer", "quick_integrated"),
}

# Alias cortos adicionales (rn1..rn10)
_ALIASES: Dict[str, str] = {
    "rn1": "adamw",      "rn2": "radam",     "rn3": "lookahead",
    "rn4": "nadam",      "rn5": "lamb",       "rn6": "adabelief",
    "rn7": "lion",       "rn8": "sam",        "rn9": "swats",
    "rn10": "integrated",
}

# ===========================================================================
# Cargador perezoso (lazy loader)
# ===========================================================================
class _LazyModule:
    """Envoltorio que retrasa la importacion de un modulo RN hasta el primer acceso."""

    def __init__(self, module_name: str) -> None:
        self._module_name = module_name
        self._module: Optional[Any] = None

    def _load(self) -> Any:
        if self._module is None:
            try:
                self._module = importlib.import_module(
                    f".{self._module_name}", package=__name__
                )
                logger.debug("RNP: modulo %s cargado.", self._module_name)
            except Exception as exc:  # pragma: no cover
                logger.error("RNP: no se pudo cargar %s: %s", self._module_name, exc)
                raise ImportError(
                    f"RNP: fallo al importar {self._module_name}: {exc}"
                ) from exc
        return self._module

    def __getattr__(self, name: str) -> Any:
        return getattr(self._load(), name)

    def __repr__(self) -> str:
        state = "cargado" if self._module is not None else "pendiente"
        return f"<_LazyModule {self._module_name!r} [{state}]>"

# ===========================================================================
# Registro de neuronas (RNPRegistry)
# ===========================================================================
class RNPRegistry:
    """Registro central de todas las neuronas disponibles en el paquete RNP.

    Proporciona acceso perezoso a modulos, clases y funciones de cada RN.
    """

    _cache: Dict[str, _LazyModule] = {}

    @classmethod
    def _resolve_key(cls, key: str) -> str:
        key = key.lower().strip()
        return _ALIASES.get(key, key)

    @classmethod
    def _get_lazy(cls, key: str) -> _LazyModule:
        key = cls._resolve_key(key)
        if key not in _MODULE_MAP:
            available = cls.list_keys()
            raise KeyError(
                f"RNPRegistry: neurona {key!r} no encontrada. Disponibles: {available}"
            )
        if key not in cls._cache:
            module_name = _MODULE_MAP[key][0]
            cls._cache[key] = _LazyModule(module_name)
        return cls._cache[key]

    @classmethod
    def get_module(cls, key: str) -> Any:
        """Devuelve el modulo importado para el algoritmo indicado."""
        return cls._get_lazy(key)._load()

    @classmethod
    def get_class(cls, key: str) -> Type:
        """Devuelve la clase optimizadora principal del algoritmo indicado."""
        key = cls._resolve_key(key)
        class_name = _MODULE_MAP[key][1]
        return getattr(cls.get_module(key), class_name)

    @classmethod
    def get_factory_fn(cls, key: str) -> Callable:
        """Devuelve la funcion factory del algoritmo indicado."""
        key = cls._resolve_key(key)
        fn_name = _MODULE_MAP[key][2]
        return getattr(cls.get_module(key), fn_name)

    @classmethod
    def get_quick_fn(cls, key: str) -> Callable:
        """Devuelve la funcion quick_* del algoritmo indicado."""
        key = cls._resolve_key(key)
        fn_name = _MODULE_MAP[key][3]
        return getattr(cls.get_module(key), fn_name)

    @classmethod
    def list_keys(cls) -> List[str]:
        """Lista todos los alias canonicos disponibles."""
        return sorted(_MODULE_MAP.keys())

    @classmethod
    def list_all(cls) -> Dict[str, Dict[str, str]]:
        """Devuelve un diccionario con todos los metadatos registrados."""
        result: Dict[str, Dict[str, str]] = {}
        for key, (mod, cls_name, factory_fn, quick_fn) in _MODULE_MAP.items():
            result[key] = {
                "module":     mod,
                "class":      cls_name,
                "factory_fn": factory_fn,
                "quick_fn":   quick_fn,
            }
        return result

    @classmethod
    def probe(cls, key: str) -> Dict[str, Any]:
        """Comprueba si un modulo puede importarse y devuelve su estado."""
        k = cls._resolve_key(key)
        info: Dict[str, Any] = {"key": k, "available": False, "error": None}
        try:
            mod = cls.get_module(k)
            info["available"] = True
            info["module"]    = repr(mod)
        except Exception as exc:
            info["error"] = str(exc)
        return info

    @classmethod
    def probe_all(cls) -> Dict[str, Dict[str, Any]]:
        """Comprueba todos los modulos y devuelve un reporte de disponibilidad."""
        return {k: cls.probe(k) for k in cls.list_keys()}

# ===========================================================================
# Fabrica de optimizadores (RNPFactory)
# ===========================================================================
class RNPFactory:
    """Fabrica centralizada para instanciar optimizadores RNP sin importar modulos directamente."""

    @staticmethod
    def create(key: str, config: Optional[Any] = None) -> Any:
        """Crea y devuelve un optimizador. key: alias (p.ej. "swats", "rn9"); config: Config o None."""
        return RNPRegistry.get_factory_fn(key)(config=config)

    @staticmethod
    def quick_run(key: str, model: Any, config: Optional[Any] = None) -> Dict[str, Any]:
        """Ciclo rapido de optimizacion. key: alias del algoritmo; model: capas a optimizar."""
        return RNPRegistry.get_quick_fn(key)(model=model, config=config)

    @staticmethod
    def create_config(key: str, **kwargs: Any) -> Any:
        """Crea NeuralWeightOptimizationConfig del modulo indicado con los kwargs dados."""
        cfg_cls = getattr(RNPRegistry.get_module(key), "NeuralWeightOptimizationConfig")
        return cfg_cls(**kwargs)

    @staticmethod
    def available_algorithms() -> List[str]:
        """Devuelve la lista de algoritmos disponibles."""
        return RNPRegistry.list_keys()

# ===========================================================================
# Benchmark unificado (RNPBenchmark)
# ===========================================================================
class RNPBenchmark:
    """Sistema de benchmarking para comparar todos los algoritmos del paquete RNP.

    Permite evaluar y comparar el rendimiento de todas las neuronas disponibles
    con un modelo de prueba generico o con un modelo proporcionado por el usuario.
    """

    @staticmethod
    def _default_model(layers: int = 3, size: int = 4) -> List[Any]:
        """Genera un modelo sintetico de prueba (lista de matrices numpy)."""
        try:
            import numpy as np  # noqa: PLC0415
            return [np.random.randn(size, size).astype(np.float64) for _ in range(layers)]
        except ImportError:
            return [[[0.0] * size for _ in range(size)] for _ in range(layers)]

    @staticmethod
    def run_single(
        key: str,
        model: Optional[Any] = None,
        config: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Ejecuta el benchmark para un algoritmo individual.

        Returns:
            Diccionario con: key, success, elapsed_s, error, result.
        """
        if model is None:
            model = RNPBenchmark._default_model()
        entry: Dict[str, Any] = {
            "key": key, "success": False,
            "elapsed_s": 0.0, "error": None, "result": None,
        }
        t0 = time.perf_counter()
        try:
            result = RNPFactory.quick_run(key, model=model, config=config)
            entry["success"] = (
                result.get("success", False)
                if isinstance(result, dict) else bool(result)
            )
            entry["result"] = result
        except Exception as exc:
            entry["error"] = traceback.format_exc(limit=4)
            logger.warning("RNPBenchmark[%s] error: %s", key, exc)
        finally:
            entry["elapsed_s"] = round(time.perf_counter() - t0, 6)
        return entry

    @staticmethod
    def run_all(
        model: Optional[Any] = None,
        keys: Optional[List[str]] = None,
        config: Optional[Any] = None,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """Ejecuta el benchmark para todos los algoritmos (o la lista indicada).

        Args:
            model:   Modelo de prueba; se genera uno sintetico si es None.
            keys:    Lista de algoritmos a evaluar; todos si es None.
            config:  Configuracion compartida opcional.
            verbose: Si es True, imprime un resumen por consola.

        Returns:
            Diccionario con claves summary, results, rankings.
        """
        if model is None:
            model = RNPBenchmark._default_model()
        targets = keys if keys is not None else RNPRegistry.list_keys()
        results: Dict[str, Dict[str, Any]] = {}
        for k in targets:
            results[k] = RNPBenchmark.run_single(k, model=model, config=config)
        successful = [k for k, v in results.items() if v["success"]]
        failed     = [k for k, v in results.items() if not v["success"]]
        rankings   = sorted(successful, key=lambda kk: results[kk]["elapsed_s"])
        summary: Dict[str, Any] = {
            "total":      len(targets),
            "successful": len(successful),
            "failed":     len(failed),
            "fastest":    rankings[0] if rankings else None,
            "rankings":   rankings,
        }
        if verbose:
            sep = "=" * 60
            print(f"\n{sep}")
            print(f"  RNP Benchmark v{__version__} (API {__api_level__})")
            print(sep)
            for k in targets:
                r = results[k]
                tick = "OK " if r["success"] else "ERR"
                err_hint = ("  // " + str(r["error"])[:50]) if r["error"] else ""
                print(f"  [{tick}] [{k:12s}] {r['elapsed_s']:.4f}s{err_hint}")
            print(
                f"\n  OK: {len(successful)}/{len(targets)} "
                f"| Mas rapido: {summary['fastest']}"
            )
            print(f"{sep}\n")
        return {"summary": summary, "results": results, "rankings": rankings}

    @staticmethod
    def compare(
        keys: List[str],
        model: Optional[Any] = None,
        config: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Compara un subconjunto de algoritmos y devuelve el ranking."""
        return RNPBenchmark.run_all(model=model, keys=keys, config=config)

# ===========================================================================
# Utilidades del paquete
# ===========================================================================
def get_version() -> str:
    """Devuelve la version del paquete RNP."""
    return __version__

def get_api_level() -> int:
    """Devuelve el nivel de API matematica del paquete (anno de referencia)."""
    return __api_level__

def list_optimizers() -> List[str]:
    """Devuelve la lista de optimizadores disponibles en el paquete."""
    return RNPRegistry.list_keys()

def probe_all() -> Dict[str, Dict[str, Any]]:
    """Verifica la disponibilidad de importacion de todos los modulos RN.

    Returns:
        Diccionario {alias: {available: bool, error: str|None}}.
    """
    return RNPRegistry.probe_all()

def describe() -> str:
    """Devuelve una cadena de descripcion completa del paquete."""
    lines = [
        f"RNP v{__version__} (API {__api_level__}) - {__author__}",
        f"Algoritmos disponibles ({len(_MODULE_MAP)}):",
    ]
    for key, (mod, cls_name, _, _) in sorted(_MODULE_MAP.items()):
        lines.append(f"  * {key:12s} -> {mod}.{cls_name}")
    return "\n".join(lines)

def health_check(verbose: bool = False) -> Dict[str, Any]:
    """Realiza una comprobacion de salud del paquete.

    Args:
        verbose: Si es True, imprime el informe por consola.

    Returns:
        Diccionario con status ("ok" o "degraded"), available,
        unavailable y details.
    """
    details  = probe_all()
    ok_keys  = [k for k, v in details.items() if v["available"]]
    bad_keys = [k for k, v in details.items() if not v["available"]]
    status   = "ok" if not bad_keys else "degraded"
    report: Dict[str, Any] = {
        "status":      status,
        "version":     __version__,
        "api_level":   __api_level__,
        "available":   ok_keys,
        "unavailable": bad_keys,
        "details":     details,
    }
    if verbose:
        print(f"\nRNP health_check -> {status.upper()}")
        for k in ok_keys:
            print(f"  [OK ] {k}")
        for k in bad_keys:
            print(f"  [ERR] {k}: {details[k]['error']}")
    return report

# ===========================================================================
# Atajos de importacion directa (lazy) — acceso sin cargar modulos pesados
# ===========================================================================
# Uso:
#   from RNP import RN1, RN9
#   optimizer = RN9.create_swats_optimizer()
RN1  = _LazyModule("RN1")
RN2  = _LazyModule("RN2")
RN3  = _LazyModule("RN3")
RN4  = _LazyModule("RN4")
RN5  = _LazyModule("RN5")
RN6  = _LazyModule("RN6")
RN7  = _LazyModule("RN7")
RN8  = _LazyModule("RN8")
RN9  = _LazyModule("RN9")
RN10 = _LazyModule("RN10")

# API publica del paquete
__all__ = [
    "__version__", "__api_level__", "__author__",
    "RNPRegistry", "RNPFactory", "RNPBenchmark",
    "get_version", "get_api_level", "list_optimizers",
    "probe_all", "describe", "health_check",
    "RN1", "RN2", "RN3", "RN4", "RN5",
    "RN6", "RN7", "RN8", "RN9", "RN10",
]

logger.info(
    "RNP v%s cargado - %d neuronas disponibles (API %d)",
    __version__, len(_MODULE_MAP), __api_level__,
)