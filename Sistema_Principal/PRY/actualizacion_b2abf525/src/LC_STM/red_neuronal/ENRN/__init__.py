"""
ENRN - Entrada de Neuronas para Red Neuronal LucIA
==================================================

Paquete de neuronas de entrada especializadas para LucIA.
Inicializacion He/Xavier/LeCun, normalizacion y utilidades eficientes.

Autor: LucIA Development Team
Version: 1.1.0
"""

import logging
import time
import json
import os
from typing import Tuple, Optional, List, Dict, Any, Callable
import numpy as np

class LucIANeuronBase:
    """Base local LucIA (sustituye a lucIA.CORE.base sin import externo)."""

    def __init__(self, input_size: int = 0, output_size: int = 0,
                 nombre: str = "LucIANeurona") -> None:
        self.input_size = int(input_size)
        self.output_size = int(output_size)
        self.nombre = str(nombre)
        self.pesos: Optional[np.ndarray] = None
        self.sesgo: Optional[np.ndarray] = None
        self.historial_activaciones: List[np.ndarray] = []
        self.historial_gradientes: List[Dict[str, Any]] = []
        self.llm_connector = None

    def inicializar_pesos(self) -> None:
        raise NotImplementedError

    def forward(self, entrada: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def backward(self, g: np.ndarray, e: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

class LucIAInitializers:
    """Inicializadores locales (sustituyen a lucIA.CORE.initializers)."""

    @staticmethod
    def he(shape: Tuple[int, ...], fan_in: int, rng=None) -> np.ndarray:
        r = rng or np.random.default_rng(LUCIA_CONFIG.get('random_seed', 42))
        return (r.normal(0.0, float(np.sqrt(2.0 / max(1, fan_in))), shape)).astype(_dtype())
    @staticmethod
    def xavier(shape: Tuple[int, ...], fan_in: int, fan_out: int, rng=None) -> np.ndarray:
        r = rng or np.random.default_rng(LUCIA_CONFIG.get('random_seed', 42))
        lim = float(np.sqrt(6.0 / max(1, fan_in + fan_out)))
        return (r.uniform(-lim, lim, shape)).astype(_dtype())
    @staticmethod
    def lecun(shape: Tuple[int, ...], fan_in: int, rng=None) -> np.ndarray:
        r = rng or np.random.default_rng(LUCIA_CONFIG.get('random_seed', 42))
        return (r.normal(0.0, float(np.sqrt(1.0 / max(1, fan_in))), shape)).astype(_dtype())

def core_he(*a, **k):
    return LucIAInitializers.he(a[0] if a else (1, 1), k.get('fan_in', 1))

def core_xavier(*a, **k):
    return LucIAInitializers.xavier(a[0] if a else (1, 1), k.get('fan_in', 1), k.get('fan_out', 1))

def core_lecun(*a, **k):
    return LucIAInitializers.lecun(a[0] if a else (1, 1), k.get('fan_in', 1))

LUCIA_CONFIG = {
    'precision': 'float32',
    'random_seed': 42,
    'default_learning_rate': 0.001,
    'weight_decay': 1e-4,
    'dropout_rate': 0.1,
    'batch_norm_momentum': 0.9,
    'batch_norm_epsilon': 1e-5
}

np.random.seed(LUCIA_CONFIG['random_seed'])
_RNG = np.random.default_rng(LUCIA_CONFIG['random_seed'])
_DTYPE_MAP = {'float32': np.float32, 'float64': np.float64, 'float16': np.float16}
_REGISTRO: Dict[str, str] = {}

def _dtype() -> np.dtype:
    return np.dtype(_DTYPE_MAP.get(LUCIA_CONFIG.get('precision', 'float32'), np.float32))

def configurar_precision(nombre: str) -> None:
    if nombre not in _DTYPE_MAP:
        raise ValueError(f"precision {nombre} no valida")
    LUCIA_CONFIG['precision'] = nombre

def fijar_semilla(semilla: int) -> None:
    LUCIA_CONFIG['random_seed'] = int(semilla)
    np.random.seed(int(semilla))
    global _RNG
    _RNG = np.random.default_rng(int(semilla))

class NeuronaEntradaBase(LucIANeuronBase):
    """Clase base para todas las neuronas de entrada."""

    def __init__(self, input_size: int, output_size: int, nombre: str = "NeuronaEntrada"):
        # CORRECCION 1: no llamar super().__init__ con args si el fallback es object
        try:
            super().__init__(input_size, output_size, nombre)  # type: ignore
        except TypeError:
            pass
        self.input_size = int(input_size)
        self.output_size = int(output_size)
        self.nombre = str(nombre)
        self.pesos: Optional[np.ndarray] = None
        self.sesgo: Optional[np.ndarray] = None
        self.historial_activaciones: List[np.ndarray] = []
        self.historial_gradientes: List[Dict[str, Any]] = []
        self.llm_connector = None
        self.ultima_respuesta_llm: Optional[str] = None
        self._system_ref = None

    def inicializar_pesos(self) -> None:
        raise NotImplementedError

    def forward(self, entrada: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def backward(self, gradiente_salida: np.ndarray, entrada: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def obtener_estadisticas_basicas(self) -> Dict[str, Any]:
        return {'nombre': self.nombre, 'clase': type(self).__name__}

    def obtener_estadisticas(self) -> Dict[str, Any]:
        # CORRECCION 2: usar metodo basico existente en vez de llamada rota
        stats = self.obtener_estadisticas_basicas()
        stats.update({
            'input_size': self.input_size,
            'output_size': self.output_size,
            'total_activaciones': len(self.historial_activaciones),
            'total_gradientes': len(self.historial_gradientes)
        })
        return stats

    def obtener_estadisticas_completas(self) -> Dict[str, Any]:
        return self.obtener_estadisticas()

    def verificar_estabilidad(self) -> Dict[str, bool]:
        return {
            'pesos_inicializados': self.pesos is not None,
            'sesgo_inicializado': self.sesgo is not None
        }

    def resetear_historial(self, limite: Optional[int] = None) -> None:
        if limite is None:
            self.historial_activaciones = []
            self.historial_gradientes = []
        else:
            self.historial_activaciones = self.historial_activaciones[-limite:]
            self.historial_gradientes = self.historial_gradientes[-limite:]

    def recibir_respuesta_llm(self, respuesta: str) -> None:
        print(f"[Neurona {self.nombre}] Respuesta LLM recibida. Procesando...")
        self.ultima_respuesta_llm = respuesta
        prompt_propio = (
            f"Soy la neurona de entrada {self.nombre}. He recibido este analisis global: {respuesta}. "
            f"Basandote en tu fortaleza especifica (logica, codigo o eficiencia), "
            f"genera una directiva tecnica ultra-especifica para el resto de la red neuronal."
        )
        if hasattr(self, 'llm_connector') and self.llm_connector:
            print(f"[Neurona {self.nombre}] Enviando prompt maestro propio a pool de modelos...")
            model_to_use = "nemotron_nano_9b"
            if "ReLU" in self.nombre or "Tanh" in self.nombre:
                model_to_use = "lyria_pro"
            elif "Dropout" in self.nombre:
                model_to_use = "gemma4"
            respuesta_propia = self.llm_connector.generate_response(prompt_propio, model_key=model_to_use, use_internet=True)
            self.distribuir_a_toda_la_red(respuesta_propia)
        else:
            print(f"[Neurona {self.nombre}] No tengo conector LLM para enviar mi prompt.")

    def distribuir_a_toda_la_red(self, directiva: str) -> None:
        print(f"[Neurona {self.nombre}] Distribuyendo directiva: {directiva[:50]}...")
        if hasattr(self, '_system_ref') and self._system_ref is not None:
            self._system_ref.distribuir_directiva_a_todas_las_neuronas(directiva)
        else:
            print("No se puede distribuir la directiva: falta referencia al sistema principal.")

    def __str__(self) -> str:
        return f"{self.nombre}(entrada={self.input_size}, salida={self.output_size})"

    def __repr__(self) -> str:
        return self.__str__()

def he_initialization(fan_in: int, fan_out: int) -> float:
    """Legacy wrapper for CORE He initialization."""
    try:
        return float(core_he((1, 1), fan_in=fan_in).item())
    except Exception:
        return float(np.sqrt(2.0 / max(1, fan_in)))

def xavier_initialization(fan_in: int, fan_out: int) -> float:
    """Legacy wrapper for CORE Xavier initialization."""
    try:
        return float(core_xavier((1, 1), fan_in=fan_in, fan_out=fan_out).item())
    except Exception:
        return float(np.sqrt(1.0 / max(1, (fan_in + fan_out) / 2.0)))

def lecun_initialization(fan_in: int, fan_out: int) -> float:
    """Legacy wrapper for CORE LeCun initialization."""
    try:
        return float(core_lecun((1, 1), fan_in=fan_in).item())
    except Exception:
        return float(np.sqrt(1.0 / max(1, fan_in)))

def normalizar_datos(datos: np.ndarray, metodo: str = 'z_score') -> np.ndarray:
    """Normaliza datos con z_score/min_max/robusta de forma vectorizada."""
    x = np.asarray(datos, dtype=_dtype())
    if metodo == 'z_score':
        return ((x - np.mean(x)) / (np.std(x) + 1e-8)).astype(_dtype())
    elif metodo == 'min_max':
        return ((x - np.min(x)) / (np.max(x) - np.min(x) + 1e-8)).astype(_dtype())
    elif metodo == 'robusta':
        med = np.median(x)
        iqr = float(np.percentile(x, 75) - np.percentile(x, 25)) + 1e-8
        return ((x - med) / iqr).astype(_dtype())
    return x

def crear_neurona(tipo: str, input_size: int, output_size: int, **kw) -> "NeuronaEntradaBase":
    mapa = {'basica': 'NeuronaEntradaBasica', 'xavier': 'NeuronaEntradaXavier',
            'lecun': 'NeuronaEntradaLeCun', 'normalizada': 'NeuronaEntradaNormalizada',
            'regularizada': 'NeuronaEntradaRegularizada', 'dropout': 'NeuronaEntradaDropout',
            'batchnorm': 'NeuronaEntradaBatchNorm', 'relu': 'NeuronaEntradaReLU',
            'tanh': 'NeuronaEntradaTanh', 'sigmoid': 'NeuronaEntradaSigmoid'}
    clases = {'NeuronaEntradaBasica': 'EN1_RN', 'NeuronaEntradaXavier': 'EN2_RN',
              'NeuronaEntradaLeCun': 'EN3_RN', 'NeuronaEntradaNormalizada': 'EN4_RN',
              'NeuronaEntradaRegularizada': 'EN5_RN', 'NeuronaEntradaDropout': 'EN6_RN',
              'NeuronaEntradaBatchNorm': 'EN7_RN', 'NeuronaEntradaReLU': 'EN8_RN',
              'NeuronaEntradaTanh': 'EN9_RN', 'NeuronaEntradaSigmoid': 'EN10_RN'}
    nombre = mapa.get(str(tipo).lower(), 'NeuronaEntradaBasica')
    mod = __import__(f"{__name__}.{clases[nombre]}", fromlist=[nombre])
    return getattr(mod, nombre)(input_size, output_size, **kw)

def benchmark_neurona(neurona: "NeuronaEntradaBase", dim_in: int = 32, dim_out: int = 16,
                      iters: int = 50) -> Dict[str, float]:
    x = np.random.default_rng(0).normal(0, 1, (16, dim_in)).astype(_dtype())
    t0 = time.perf_counter()
    for _ in range(iters):
        neurona.forward(x)
    tf = (time.perf_counter() - t0) / max(1, iters) * 1000.0
    return {'t_medio_forward_ms': tf, 'iters': float(iters)}

def validar_lote(x: np.ndarray, dim: int) -> np.ndarray:
    a = np.asarray(x, dtype=_dtype())
    if a.ndim == 1:
        a = a.reshape(1, -1)
    if a.shape[1] != dim:
        raise ValueError(f"lote dim {a.shape[1]} != {dim}")
    return np.ascontiguousarray(a, dtype=_dtype())

def forward_en_cascada(neuronas: List["NeuronaEntradaBase"], x: np.ndarray) -> np.ndarray:
    a = np.asarray(x, dtype=_dtype())
    for n in neuronas:
        a = n.forward(a)
    return a

def backward_en_cascada(neuronas: List["NeuronaEntradaBase"], g: np.ndarray) -> np.ndarray:
    a = np.asarray(g, dtype=_dtype())
    for n in reversed(neuronas):
        if not hasattr(n, 'grad_entrada'):
            break
        a = n.grad_entrada(a)
    return a

def guardar_red(neuronas: List["NeuronaEntradaBase"], ruta: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(ruta)), exist_ok=True)
    with open(ruta, 'w', encoding='utf-8') as f:
        json.dump([n.a_dict() if hasattr(n, 'a_dict') else {} for n in neuronas], f)

def resumen_red(neuronas: List["NeuronaEntradaBase"]) -> Dict[str, Any]:
    total_params = 0
    for n in neuronas:
        if getattr(n, 'pesos', None) is not None:
            total_params += int(np.size(n.pesos)) + int(np.size(n.sesgo))
    return {'n_neuronas': len(neuronas), 'total_params': total_params,
            'tipos': [type(n).__name__ for n in neuronas]}

def registrar_tipo(clave: str, clase: Callable[..., "NeuronaEntradaBase"]) -> None:
    _REGISTRO[str(clave).lower()] = clase.__name__

def listar_tipos() -> List[str]:
    return sorted(_REGISTRO.keys()) or ['basica', 'xavier', 'lecun', 'normalizada',
                                        'regularizada', 'dropout', 'batchnorm',
                                        'relu', 'tanh', 'sigmoid']

def norma_global(neuronas: List["NeuronaEntradaBase"]) -> float:
    acc = 0.0
    for n in neuronas:
        if getattr(n, 'pesos', None) is not None:
            acc += float(np.sum(n.pesos.astype(np.float64) ** 2))
    return float(np.sqrt(acc))

def clip_global(neuronas: List["NeuronaEntradaBase"], max_norma: float = 5.0) -> float:
    ng = norma_global(neuronas)
    if ng > max_norma and ng > 0:
        f = max_norma / ng
        for n in neuronas:
            if getattr(n, 'pesos', None) is not None:
                n.pesos = (n.pesos.astype(np.float64) * f).astype(_dtype())
    return norma_global(neuronas)

def decaimiento_lr_lineal(paso: int, lr_base: float = 0.001,
                          cada: int = 100, factor: float = 0.9,
                          lr_min: float = 1e-6) -> float:
    return max(lr_min, float(lr_base) * (float(factor) ** (paso // max(1, cada))))

def aplicar_lr(neuronas: List["NeuronaEntradaBase"], lr: float) -> None:
    for n in neuronas:
        if hasattr(n, 'lr_actual'):
            n.lr_actual = float(lr)
        if hasattr(n, 'lr_base'):
            n.lr_base = float(lr)

def semilla_global() -> int:
    return int(LUCIA_CONFIG.get('random_seed', 42))

def info_paquete() -> Dict[str, Any]:
    return {'nombre': 'ENRN', 'version': '1.1.0', 'precision': LUCIA_CONFIG.get('precision'),
            'tipos': listar_tipos(), 'n_tipos': len(listar_tipos())}


__version__ = '1.1.0'
__autor__ = 'LucIA Development Team'

def salud() -> Dict[str, Any]:
    try:
        n = crear_neurona('basica', 4, 2)
        a = n.forward(np.zeros((1, 4), dtype=_dtype()))
        ok = bool(np.all(np.isfinite(a)))
    except Exception as e:
        return {'ok': False, 'error': f"{type(e).__name__}: {e}"}
    return {'ok': ok, 'version': __version__, 'tipos': len(listar_tipos())}

def _cargar_submodulos() -> None:
    """Carga EN1..EN10 tanto como paquete como script directo."""
    pares = [('EN1_RN', 'NeuronaEntradaBasica'), ('EN2_RN', 'NeuronaEntradaXavier'), ('EN3_RN', 'NeuronaEntradaLeCun'), ('EN4_RN', 'NeuronaEntradaNormalizada'),
             ('EN5_RN', 'NeuronaEntradaRegularizada'), ('EN6_RN', 'NeuronaEntradaDropout'), ('EN7_RN', 'NeuronaEntradaBatchNorm'),
             ('EN8_RN', 'NeuronaEntradaReLU'), ('EN9_RN', 'NeuronaEntradaTanh'), ('EN10_RN', 'NeuronaEntradaSigmoid')]
    try:
        for _mod, _cls in pares:
            globals()[_cls] = getattr(__import__(f"{__name__}.{_mod}", fromlist=[_cls]), _cls)
        return
    except ImportError:
        pass
    import importlib.util as _ilu
    base = os.path.dirname(os.path.abspath(__file__))
    for _mod, _cls in pares:
        _spec = _ilu.spec_from_file_location(_mod, os.path.join(base, _mod + '.py'))
        _m = _ilu.module_from_spec(_spec)
        _spec.loader.exec_module(_m)
        globals()[_cls] = getattr(_m, _cls)


_cargar_submodulos()

__all__ = [
    'NeuronaEntradaBase',
    'NeuronaEntradaBasica',
    'NeuronaEntradaXavier',
    'NeuronaEntradaLeCun',
    'NeuronaEntradaNormalizada',
    'NeuronaEntradaRegularizada',
    'NeuronaEntradaDropout',
    'NeuronaEntradaBatchNorm',
    'NeuronaEntradaReLU',
    'NeuronaEntradaTanh',
    'NeuronaEntradaSigmoid',
    'LUCIA_CONFIG', 'crear_neurona', 'benchmark_neurona',
    'normalizar_datos', 'configurar_precision', 'fijar_semilla',
    'validar_lote', 'forward_en_cascada', 'backward_en_cascada',
    'guardar_red', 'resumen_red',
    'registrar_tipo', 'listar_tipos', 'norma_global', 'clip_global',
    'decaimiento_lr_lineal', 'aplicar_lr', 'semilla_global', 'info_paquete',
    'salud', '__version__',
]

logger = logging.getLogger('ENRN')
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(_h)
logger.setLevel(logging.INFO)
logger.info("Paquete ENRN inicializado correctamente para LucIA")
# Fin del modulo ENRN v1.1.0 — base corregida y utilidades eficientes listas.
# Historial: v1.1.0 corrige init fallback y estadisticas basicas.