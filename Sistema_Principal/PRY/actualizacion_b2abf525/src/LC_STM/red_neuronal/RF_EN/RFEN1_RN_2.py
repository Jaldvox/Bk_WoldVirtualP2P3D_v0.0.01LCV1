"""RFEN1_RN_2.py - Neurona de Refuerzo con Policy Gradient (refactor eficiente)."""
import numpy as np
import math
import time
import logging
from typing import Tuple, Optional, Dict, Any, List

LUCIA_RL_CONFIG = {'precision': 'float32', 'random_seed': 42,
                   'default_learning_rate': 0.001}


class NeuronaRefuerzoBase:
    """Base minimalista local (evita importar .base inexistente)."""

    def __init__(self, input_size: int, output_size: int, nombre: str = "NeuronaRefuerzo"):
        self.input_size = int(input_size)
        self.output_size = int(output_size)
        self.nombre = str(nombre)
        self.pesos = None
        self.sesgo = None
        self.historial_activaciones: List[np.ndarray] = []
        self.historial_gradientes: List[Any] = []

    def inicializar_pesos(self) -> None:
        raise NotImplementedError

    def forward(self, e: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def resetear_historial(self) -> None:
        self.historial_activaciones = []
        self.historial_gradientes = []


class InicializadoresRL:
    """Inicializadores minimalistas locales."""

    @staticmethod
    def _rng(seed: int = 42) -> np.random.Generator:
        return np.random.default_rng(seed)

    @classmethod
    def he(cls, shape, fan_in=1, seed=42):
        return cls._rng(seed).normal(0.0, math.sqrt(2.0 / max(1, fan_in)), shape).astype(LUCIA_RL_CONFIG['precision'])

    @classmethod
    def xavier(cls, shape, fan_in=1, fan_out=1, seed=42):
        lim = math.sqrt(6.0 / max(1, fan_in + fan_out))
        return cls._rng(seed).uniform(-lim, lim, shape).astype(LUCIA_RL_CONFIG['precision'])

    @classmethod
    def ortogonal(cls, shape, seed=42, ganancia: float = 1.0):
        a = cls._rng(seed).normal(0, 1, shape)
        q, _ = np.linalg.qr(a) if shape[0] >= shape[1] else np.linalg.qr(a.T)
        q = q if shape[0] >= shape[1] else q.T
        return (q[:, :shape[1]] * ganancia).astype(LUCIA_RL_CONFIG['precision'])

    @classmethod
    def espectral(cls, shape, seed=42):
        w = cls._rng(seed).normal(0, 1, shape)
        s = np.linalg.svd(w, compute_uv=False)
        return (w / max(1e-12, s[0])).astype(LUCIA_RL_CONFIG['precision'])


def inicializar_pesos_he(shape, fan_in=1, seed=42):
    return InicializadoresRL.he(shape, fan_in, seed)


def inicializar_pesos_xavier(shape, fan_in=1, fan_out=1, seed=42):
    return InicializadoresRL.xavier(shape, fan_in, fan_out, seed)


def inicializar_pesos_ortogonal(shape, seed=42):
    return InicializadoresRL.ortogonal(shape, seed)


def inicializar_pesos_espectral(shape, seed=42):
    return InicializadoresRL.espectral(shape, seed)


logger = logging.getLogger('RFENRN1.RFEN1_RN_2')
_DTYPE = {'float32': np.float32, 'float64': np.float64}


def _dt() -> np.dtype:
    return np.dtype(_DTYPE.get(LUCIA_RL_CONFIG.get('precision', 'float32'), np.float32))


class _Ring:
    def __init__(self, cap: int = 512):
        self.cap = cap
        self._d: List[Any] = []

    def append(self, x: Any) -> None:
        self._d.append(x)
        if len(self._d) > self.cap:
            del self._d[0:len(self._d) - self.cap]

    def __len__(self) -> int:
        return len(self._d)

    def tail(self, n: int) -> List[Any]:
        return self._d[-n:]

    def clear(self) -> None:
        self._d.clear()


class NeuronaRefuerzoPolicyGradient(NeuronaRefuerzoBase):
    """Policy Gradient eficiente: REINFORCE vectorizado, baseline, entropia."""

    def __init__(self, input_size: int, output_size: int,
                 nombre: str = "NeuronaRefuerzoPolicyGradient",
                 learning_rate: float = 0.001,
                 gamma: float = 0.99,
                 usar_ortogonal: bool = True,
                 entropy_coef: float = 0.01,
                 grad_clip: float = 5.0,
                 usar_baseline: bool = True,
                 semilla: int = 42,
                 capacidad_historial: int = 512):
        super().__init__(input_size, output_size, nombre)
        self.learning_rate = float(learning_rate)
        self.gamma = float(gamma)
        self.usar_ortogonal = bool(usar_ortogonal)
        self.entropy_coef = float(entropy_coef)
        self.grad_clip = float(grad_clip)
        self.usar_baseline = bool(usar_baseline)
        self.rng = np.random.default_rng(semilla)
        self.pasos = 0
        self.tiempo_fwd = 0.0
        self.tiempo_upd = 0.0
        self.estadisticas_pg = {
            'gradiente_politica_norma': 0.0, 'entropia_politica': 0.0,
            'probabilidad_accion_media': 0.0, 'probabilidad_accion_std': 0.0,
            'convergencia_politica': 0.0, 'estabilidad_politica': 0.0,
            'actualizaciones_politica': 0, 'exploracion_efectiva': 0.0}
        self.historial_politicas = _Ring(capacidad_historial)
        self.historial_gradientes = _Ring(capacidad_historial)
        self.historial_entropias = _Ring(capacidad_historial)
        self.historial_recompensas_descontadas = _Ring(capacidad_historial)
        logger.info(f"NeuronaRefuerzoPolicyGradient creada: {self}")

    def inicializar_pesos(self, modo: Optional[str] = None) -> None:
        m = (modo or ('ortogonal' if self.usar_ortogonal else 'xavier')).lower()
        shp = (self.input_size, self.output_size)
        if m == 'ortogonal':
            self.pesos = inicializar_pesos_ortogonal(shp)
        elif m == 'xavier':
            self.pesos = inicializar_pesos_xavier(shp, self.input_size, self.output_size)
        elif m == 'he':
            self.pesos = inicializar_pesos_he(shp, self.input_size)
        else:
            raise ValueError(f"modo {m} desconocido")
        self.pesos = np.ascontiguousarray(self.pesos, dtype=_dt())
        self.sesgo = np.zeros((1, self.output_size), dtype=_dt())
        logger.info(f"Pesos de politica inicializados con {m}")

    def _asegurar(self) -> None:
        if self.pesos is None:
            self.inicializar_pesos()

    def _prep(self, estado: np.ndarray) -> np.ndarray:
        x = np.asarray(estado, dtype=_dt())
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if x.shape[1] != self.input_size:
            raise ValueError(f"estado dim {x.shape[1]} != input {self.input_size}")
        return np.ascontiguousarray(x, dtype=_dt())

    @staticmethod
    def _softmax(z: np.ndarray) -> np.ndarray:
        z = z - np.max(z, axis=1, keepdims=True)
        e = np.exp(np.clip(z, -30, 30))
        return e / np.maximum(np.sum(e, axis=1, keepdims=True), 1e-12)

    def forward(self, estado: np.ndarray) -> np.ndarray:
        t0 = time.perf_counter()
        self._asegurar()
        assert self.pesos is not None and self.sesgo is not None
        x = self._prep(estado)
        probs = self._softmax(np.dot(x, self.pesos) + self.sesgo).astype(_dt(), copy=False)
        self.historial_politicas.append(probs.copy())
        ent = float(np.mean(-np.sum(probs.astype(np.float64) * np.log(probs.astype(np.float64) + 1e-8), axis=1)))
        self.historial_entropias.append(ent)
        self.tiempo_fwd += time.perf_counter() - t0
        return np.ascontiguousarray(probs, dtype=_dt())

    def seleccionar_accion(self, estado: np.ndarray) -> Tuple[int, float]:
        probs = self.forward(estado)
        p = probs[0].astype(np.float64)
        p = p / max(1e-12, p.sum())
        acc = int(self.rng.choice(self.output_size, p=p))
        return acc, float(probs[0, acc])

    def recompensas_descontadas(self, recompensas: List[float]) -> np.ndarray:
        g = np.zeros(len(recompensas))
        acc = 0.0
        for t in range(len(recompensas) - 1, -1, -1):
            acc = recompensas[t] + self.gamma * acc
            g[t] = acc
        return g

    def calcular_gradiente_politica(self, estados: List[np.ndarray], acciones: List[int],
                                    recompensas_descontadas: List[float]) -> Tuple[np.ndarray, np.ndarray]:
        """REINFORCE vectorizado con baseline y entropia (corrige bug linea 181)."""
        self._asegurar()
        assert self.pesos is not None and self.sesgo is not None
        X = np.vstack([self._prep(s) for s in estados]).astype(np.float64)
        A = np.asarray(acciones, dtype=np.int64)
        G = np.asarray(recompensas_descontadas, dtype=np.float64)
        n = len(X)
        logits = X @ self.pesos.astype(np.float64) + self.sesgo.astype(np.float64)
        probs = self._softmax(logits).astype(np.float64)
        if self.usar_baseline and n > 1:
            G = G - G.mean()
            sd = G.std() + 1e-8
            G = G / sd
        one_hot = np.zeros_like(probs)
        one_hot[np.arange(n), A % self.output_size] = 1.0
        dlogits = (one_hot - probs) * G[:, None] / n
        grad_w = X.T @ dlogits
        grad_b = dlogits.sum(axis=0, keepdims=True)
        if self.entropy_coef > 0:
            logp = np.log(probs + 1e-8)
            ent_coef = self.entropy_coef / n
            grad_w += (X.T @ (-(logp + 1.0) * probs)) * ent_coef
            grad_b += (-(logp + 1.0) * probs).sum(axis=0, keepdims=True) * ent_coef
        ng = float(np.linalg.norm(grad_w))
        if self.grad_clip > 0 and ng > self.grad_clip:
            f = self.grad_clip / (ng + 1e-12)
            grad_w *= f
            grad_b *= f
            ng = float(np.linalg.norm(grad_w))
        self.historial_gradientes.append(ng)
        self.estadisticas_pg['gradiente_politica_norma'] = ng
        self.historial_recompensas_descontadas.append(float(np.mean(np.asarray(recompensas_descontadas, dtype=np.float64))))
        return grad_w.astype(_dt()), grad_b.astype(_dt())

    def actualizar_politica(self, gradiente_politica: np.ndarray, gradiente_sesgo: np.ndarray) -> Dict[str, float]:
        t0 = time.perf_counter()
        if self.pesos is None or self.sesgo is None:
            raise ValueError("Pesos no inicializados")
        self.pesos = np.ascontiguousarray((self.pesos.astype(np.float64) + self.learning_rate * np.asarray(gradiente_politica, dtype=np.float64)).astype(_dt()))
        self.sesgo = np.ascontiguousarray((self.sesgo.astype(np.float64) + self.learning_rate * np.asarray(gradiente_sesgo, dtype=np.float64)).astype(_dt()))
        self.pasos += 1
        self.estadisticas_pg['actualizaciones_politica'] += 1
        self._calcular_convergencia_politica()
        self.tiempo_upd += time.perf_counter() - t0
        return {'paso': float(self.pasos), 'norma_grad': self.estadisticas_pg['gradiente_politica_norma']}

    def entrenar_episodio(self, estados: List[np.ndarray], acciones: List[int],
                          recompensas: List[float]) -> Dict[str, float]:
        g = self.recompensas_descontadas(recompensas)
        gw, gb = self.calcular_gradiente_politica(estados, acciones, g.tolist())
        info = self.actualizar_politica(gw, gb)
        return {'retorno': float(np.sum(recompensas)), **info}

    def _calcular_convergencia_politica(self) -> None:
        if len(self.historial_gradientes) < 10:
            return
        var = float(np.var(np.asarray(self.historial_gradientes.tail(10), dtype=np.float64)))
        self.estadisticas_pg['convergencia_politica'] = 1.0 / (1.0 + var)

    def calcular_estabilidad_politica(self) -> float:
        if len(self.historial_politicas) < 20:
            return 0.0
        arr = np.stack(self.historial_politicas.tail(20), axis=0).astype(np.float64)
        est = 1.0 / (1.0 + float(np.mean(np.var(arr, axis=0))))
        self.estadisticas_pg['estabilidad_politica'] = est
        return est

    def calcular_exploracion_efectiva(self) -> float:
        if not len(self.historial_entropias):
            return 0.0
        ent = float(np.mean(np.asarray(self.historial_entropias.tail(10), dtype=np.float64)))
        exp = ent / max(1e-12, math.log(self.output_size))
        self.estadisticas_pg['exploracion_efectiva'] = exp
        return exp

    def obtener_estadisticas_pg(self) -> Dict[str, Any]:
        if self.pesos is None:
            return {'estado': 'no_inicializada'}
        if len(self.historial_politicas):
            arr = np.stack(self.historial_politicas.tail(10), axis=0).astype(np.float64)
            self.estadisticas_pg['probabilidad_accion_media'] = float(np.mean(arr))
            self.estadisticas_pg['probabilidad_accion_std'] = float(np.std(arr))
        if len(self.historial_entropias):
            self.estadisticas_pg['entropia_politica'] = float(np.mean(np.asarray(self.historial_entropias.tail(10), dtype=np.float64)))
        return {'learning_rate': self.learning_rate, 'gamma': self.gamma, 'entropy_coef': self.entropy_coef,
                'convergencia_politica': self.estadisticas_pg['convergencia_politica'],
                'estabilidad_politica': self.calcular_estabilidad_politica(),
                'exploracion_efectiva': self.calcular_exploracion_efectiva(),
                'actualizaciones_politica': self.estadisticas_pg['actualizaciones_politica'],
                'gradiente_politica_norma': self.estadisticas_pg['gradiente_politica_norma'],
                'entropia_politica': self.estadisticas_pg['entropia_politica'],
                'probabilidad_accion_media': self.estadisticas_pg['probabilidad_accion_media'],
                'probabilidad_accion_std': self.estadisticas_pg['probabilidad_accion_std'],
                't_medio_fwd_us': self.tiempo_fwd / max(1, self.pasos + 1) * 1e6,
                't_medio_upd_us': self.tiempo_upd / max(1, self.pasos) * 1e6}

    def verificar_estabilidad(self) -> Dict[str, bool]:
        est = {'convergencia_ok': self.estadisticas_pg['convergencia_politica'] > 0.7,
               'politica_estable': self.calcular_estabilidad_politica() > 0.7,
               'exploracion_efectiva': 0.1 <= self.calcular_exploracion_efectiva() <= 0.9,
               'gradientes_no_explosivos': self.estadisticas_pg['gradiente_politica_norma'] < 10.0,
               'gradientes_no_desaparecen': self.estadisticas_pg['gradiente_politica_norma'] > 1e-8}
        if self.pesos is not None:
            pmax = float(np.max(np.abs(self.pesos)))
            pmin = max(float(np.min(np.abs(self.pesos))), 1e-12)
            est['pesos_no_explosivos'] = pmax < 5.0
            est['pesos_no_desaparecen'] = pmin > 1e-6
            est['pesos_balanceados'] = pmax / pmin < 1000.0
            est['sin_nan'] = bool(not np.isnan(self.pesos).any())
        else:
            est.update({'pesos_no_explosivos': True, 'pesos_no_desaparecen': True, 'pesos_balanceados': True, 'sin_nan': True})
        return est

    def reinicializar_con_parametros(self, learning_rate: float = None, gamma: float = None,
                                     entropy_coef: float = None) -> None:
        if learning_rate is not None:
            self.learning_rate = float(learning_rate)
        if gamma is not None:
            self.gamma = float(gamma)
        if entropy_coef is not None:
            self.entropy_coef = float(entropy_coef)
        self.inicializar_pesos()
        self.resetear_historial()
        self.historial_politicas.clear()
        self.historial_gradientes.clear()
        self.historial_entropias.clear()
        self.historial_recompensas_descontadas.clear()
        self.estadisticas_pg['actualizaciones_politica'] = 0
        self.pasos = 0
        logger.info(f"Policy Gradient reinicializada: lr={self.learning_rate}, gamma={self.gamma}")

    def a_dict(self) -> Dict[str, Any]:
        return {'input_size': self.input_size, 'output_size': self.output_size, 'nombre': self.nombre,
                'learning_rate': self.learning_rate, 'gamma': self.gamma, 'entropy_coef': self.entropy_coef,
                'pesos': self.pesos.tolist() if self.pesos is not None else None,
                'sesgo': self.sesgo.tolist() if self.sesgo is not None else None}

    def guardar(self, ruta: str) -> None:
        import json
        with open(ruta, 'w', encoding='utf-8') as f:
            json.dump(self.a_dict(), f)

    @classmethod
    def cargar(cls, ruta: str) -> "NeuronaRefuerzoPolicyGradient":
        import json
        with open(ruta, 'r', encoding='utf-8') as f:
            d = json.load(f)
        o = cls(d['input_size'], d['output_size'], d.get('nombre', 'NeuronaRefuerzoPolicyGradient'))
        if d.get('pesos') is not None:
            o.pesos = np.asarray(d['pesos'], dtype=_dt())
            o.sesgo = np.asarray(d['sesgo'], dtype=_dt())
        return o

    def __str__(self) -> str:
        return (f"NeuronaRefuerzoPolicyGradient(entrada={self.input_size}, salida={self.output_size}, "
                f"lr={self.learning_rate}, gamma={self.gamma}, entropy_coef={self.entropy_coef})")

    def __repr__(self) -> str:
        return self.__str__()


def crear_neurona_policy_gradient(input_size: int, output_size: int,
                                  configuracion: Dict[str, Any] = None) -> NeuronaRefuerzoPolicyGradient:
    configuracion = configuracion or {}
    return NeuronaRefuerzoPolicyGradient(
        input_size=input_size, output_size=output_size,
        nombre=configuracion.get('nombre', 'NeuronaRefuerzoPolicyGradient'),
        learning_rate=configuracion.get('learning_rate', 0.001),
        gamma=configuracion.get('gamma', 0.99),
        usar_ortogonal=configuracion.get('usar_ortogonal', True),
        entropy_coef=configuracion.get('entropy_coef', 0.01))


RFEN2_CONFIG = {'inicializacion_preferida': 'ortogonal', 'learning_rate_default': 0.001,
                'gamma_default': 0.99, 'entropy_coef_default': 0.01, 'umbral_convergencia': 0.7,
                'umbral_estabilidad': 0.7, 'umbral_exploracion_min': 0.1, 'umbral_exploracion_max': 0.9}


def analizar_convergencia_pg(neurona: NeuronaRefuerzoPolicyGradient) -> Dict[str, Any]:
    return {'convergencia': neurona.estadisticas_pg['convergencia_politica'],
            'estabilidad': neurona.calcular_estabilidad_politica(),
            'exploracion': neurona.calcular_exploracion_efectiva(),
            'actualizaciones': neurona.estadisticas_pg['actualizaciones_politica'],
            'norma_grad': neurona.estadisticas_pg['gradiente_politica_norma'],
            'estadisticas': neurona.obtener_estadisticas_pg(),
            'estable': neurona.verificar_estabilidad()}


def comparar_inicializaciones_pg(input_size: int, output_size: int, pasos: int = 200) -> Dict[str, Any]:
    rng = np.random.default_rng(0)
    out = {}
    for k, m in (('ortogonal', 'ortogonal'), ('xavier', 'xavier')):
        n = NeuronaRefuerzoPolicyGradient(input_size, output_size)
        n.inicializar_pesos(m)
        X = rng.normal(0, 1, (pasos, input_size)).astype(np.float32)
        A = rng.integers(0, output_size, pasos).tolist()
        G = rng.random(pasos).tolist()
        gw, gb = n.calcular_gradiente_politica([x for x in X], A, G)
        out[k] = {'norma_grad': float(np.linalg.norm(gw)), 'entropia': float(np.mean(n.historial_entropias.tail(10)))}
    return out
logger.info("RFEN1_RN_2.py cargado correctamente - Neurona de Refuerzo Policy Gradient")
