"""RFEN1_RN_4.py - Neurona de Refuerzo con DQN (refactor eficiente)."""
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
    """Inicializadores minimalistas locales (he/xavier/lecun/ortogonal/espectral)."""

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
    def lecun(cls, shape, fan_in=1, seed=42):
        return cls._rng(seed).normal(0.0, math.sqrt(1.0 / max(1, fan_in)), shape).astype(LUCIA_RL_CONFIG['precision'])

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


def inicializar_pesos_lecun(shape, fan_in=1, seed=42):
    return InicializadoresRL.lecun(shape, fan_in, seed)


def inicializar_pesos_ortogonal(shape, seed=42):
    return InicializadoresRL.ortogonal(shape, seed)


def inicializar_pesos_espectral(shape, seed=42):
    return InicializadoresRL.espectral(shape, seed)


logger = logging.getLogger('RFENRN1.RFEN1_RN_4')
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


class ReplayBuffer:
    """Buffer de experiencia circular con muestreo vectorizado."""

    def __init__(self, capacidad: int = 10000, semilla: int = 42):
        self.capacidad = int(capacidad)
        self.rng = np.random.default_rng(semilla)
        self.s: List[np.ndarray] = []
        self.a: List[int] = []
        self.r: List[float] = []
        self.ns: List[np.ndarray] = []
        self.d: List[bool] = []

    def __len__(self) -> int:
        return len(self.s)

    def push(self, s: np.ndarray, a: int, r: float, ns: np.ndarray, d: bool) -> None:
        self.s.append(np.asarray(s, dtype=np.float32).flatten())
        self.a.append(int(a))
        self.r.append(float(r))
        self.ns.append(np.asarray(ns, dtype=np.float32).flatten())
        self.d.append(bool(d))
        if len(self.s) > self.capacidad:
            for lst in (self.s, self.a, self.r, self.ns, self.d):
                del lst[0]

    def sample(self, n: int) -> Dict[str, np.ndarray]:
        idx = self.rng.integers(0, len(self.s), min(n, len(self.s)))
        return {'s': np.stack([self.s[i] for i in idx]).astype(np.float32),
                'a': np.asarray([self.a[i] for i in idx], dtype=np.int64),
                'r': np.asarray([self.r[i] for i in idx], dtype=np.float64),
                'ns': np.stack([self.ns[i] for i in idx]).astype(np.float32),
                'd': np.asarray([self.d[i] for i in idx], dtype=bool)}

    def clear(self) -> None:
        self.s.clear()
        self.a.clear()
        self.r.clear()
        self.ns.clear()
        self.d.clear()


class NeuronaRefuerzoDQN(NeuronaRefuerzoBase):
    """DQN eficiente: replay buffer, red objetivo, Bellman vectorizado."""

    def __init__(self, input_size: int, output_size: int,
                 nombre: str = "NeuronaRefuerzoDQN",
                 learning_rate: float = 0.001,
                 gamma: float = 0.99,
                 epsilon: float = 0.2,
                 epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.01,
                 usar_doble: bool = True,
                 sync_objetivo_cada: int = 100,
                 buffer_capacidad: int = 10000,
                 batch_size: int = 32,
                 grad_clip: float = 5.0,
                 semilla: int = 42,
                 capacidad_historial: int = 512):
        super().__init__(input_size, output_size, nombre)
        self.learning_rate = float(learning_rate)
        self.gamma = float(gamma)
        self.epsilon = float(epsilon)
        self.epsilon_decay = float(epsilon_decay)
        self.epsilon_min = float(epsilon_min)
        self.usar_doble = bool(usar_doble)
        self.sync_objetivo_cada = int(sync_objetivo_cada)
        self.batch_size = int(batch_size)
        self.grad_clip = float(grad_clip)
        self.rng = np.random.default_rng(semilla)
        self.q_online = None
        self.q_objetivo = None
        self.sesgo = None
        self.pasos = 0
        self.tiempo_fwd = 0.0
        self.tiempo_upd = 0.0
        self.buffer = ReplayBuffer(buffer_capacidad, semilla)
        self.estadisticas_dqn = {
            'loss_media': 0.0, 'convergencia': 0.0,
            'exploracion_rate': 0, 'exploitacion_rate': 0,
            'actualizaciones': 0, 'syncs_objetivo': 0}
        self.historial_loss = _Ring(capacidad_historial)
        self.historial_q = _Ring(capacidad_historial)
        self._loss_ema = 0.0
        logger.info(f"NeuronaRefuerzoDQN creada: {self}")

    def inicializar_pesos(self, modo: str = "he") -> None:
        m = str(modo).lower()
        shp = (self.input_size, self.output_size)
        if m == 'he':
            self.q_online = inicializar_pesos_he(shp, self.input_size)
        elif m == 'xavier':
            self.q_online = inicializar_pesos_xavier(shp, self.input_size, self.output_size)
        elif m == 'lecun':
            self.q_online = inicializar_pesos_lecun(shp, self.input_size)
        elif m == 'ortogonal':
            self.q_online = inicializar_pesos_ortogonal(shp)
        elif m == 'espectral':
            self.q_online = inicializar_pesos_espectral(shp)
        elif m == 'ceros':
            self.q_online = np.zeros(shp, dtype=_dt())
        else:
            raise ValueError(f"modo {m} desconocido")
        self.q_online = np.ascontiguousarray(self.q_online, dtype=_dt())
        self.q_objetivo = self.q_online.copy()
        self.sesgo = np.zeros((1, self.output_size), dtype=_dt())
        self.pesos = self.q_online
        logger.info(f"Red DQN inicializada ({modo}): forma={self.q_online.shape}")

    def _asegurar(self) -> None:
        if self.q_online is None:
            self.inicializar_pesos()

    def _prep(self, estado: np.ndarray) -> np.ndarray:
        x = np.asarray(estado, dtype=_dt())
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if x.shape[1] != self.input_size:
            raise ValueError(f"estado dim {x.shape[1]} != input {self.input_size}")
        return np.ascontiguousarray(x, dtype=_dt())

    def forward(self, estado: np.ndarray) -> np.ndarray:
        t0 = time.perf_counter()
        self._asegurar()
        assert self.q_online is not None and self.sesgo is not None
        qv = np.dot(self._prep(estado), self.q_online) + self.sesgo
        qv = np.ascontiguousarray(qv[0]).astype(np.float64)
        if self.rng.random() < self.epsilon:
            acc = int(self.rng.integers(0, self.output_size))
            self.estadisticas_dqn['exploracion_rate'] += 1
        else:
            acc = int(np.argmax(qv))
            self.estadisticas_dqn['exploitacion_rate'] += 1
        self.historial_q.append(qv.copy())
        self.tiempo_fwd += time.perf_counter() - t0
        return np.array([acc])

    def recordar(self, s: np.ndarray, a: int, r: float, ns: np.ndarray, d: bool = False) -> None:
        self.buffer.push(s, a, r, ns, d)

    def valores_q(self, estado: np.ndarray, red: str = "online") -> np.ndarray:
        self._asegurar()
        assert self.q_online is not None and self.sesgo is not None
        x = self._prep(estado)
        w = self.q_online if red == "online" else self.q_objetivo
        return np.ascontiguousarray((np.dot(x, w) + self.sesgo).astype(_dt()))

    def calcular_gradientes_dqn(self, s: np.ndarray, a: np.ndarray,
                                     r: np.ndarray, ns: np.ndarray,
                                     d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        self._asegurar()
        assert self.q_online is not None and self.sesgo is not None
        S = s.astype(np.float64)
        NS = ns.astype(np.float64)
        A = a.astype(np.int64) % self.output_size
        R = r.astype(np.float64)
        D = d.astype(bool)
        wo = self.q_online.astype(np.float64)
        wt = self.q_objetivo.astype(np.float64)
        b = self.sesgo.astype(np.float64)
        q_act = (S @ wo + b)[np.arange(len(S)), A]
        if self.usar_doble:
            a_star = np.argmax(NS @ wo + b, axis=1)
            q_next = (NS @ wt + b)[np.arange(len(NS)), a_star]
        else:
            q_next = np.max(NS @ wt + b, axis=1)
        q_obj = np.where(D, R, R + self.gamma * q_next)
        td = q_obj - q_act
        n = max(1, len(S))
        oh = np.zeros((n, self.output_size))
        oh[np.arange(n), A] = -td / n
        grad_w = S.T @ oh
        grad_b = oh.sum(axis=0, keepdims=True)
        if self.grad_clip > 0:
            ng = float(np.linalg.norm(grad_w))
            if ng > self.grad_clip:
                grad_w *= self.grad_clip / (ng + 1e-12)
                grad_b *= self.grad_clip / (ng + 1e-12)
        return grad_w.astype(_dt()), grad_b.astype(_dt())

    def actualizar_q_batch(self, s: np.ndarray, a: np.ndarray,
                             r: np.ndarray, ns: np.ndarray,
                             d: np.ndarray) -> float:
        t0 = time.perf_counter()
        self._asegurar()
        grad_w, grad_b = self.calcular_gradientes_dqn(s, a, r, ns, d)
        self.q_online = (self.q_online.astype(np.float64) - self.learning_rate * grad_w.astype(np.float64)).astype(_dt())
        self.sesgo = (self.sesgo.astype(np.float64) - self.learning_rate * grad_b.astype(np.float64)).astype(_dt())
        self.q_objetivo = self.q_online.copy()
        self.pesos = self.q_online
        self.pasos += 1
        self.estadisticas_dqn['actualizaciones'] += 1
        wo = self.q_online.astype(np.float64)
        wt = self.q_objetivo.astype(np.float64)
        b = self.sesgo.astype(np.float64)
        S = s.astype(np.float64)
        A = a.astype(np.int64) % self.output_size
        R = r.astype(np.float64)
        D = d.astype(bool)
        NS = ns.astype(np.float64)
        q_act = (S @ wo + b)[np.arange(len(S)), A]
        if self.usar_doble:
            a_star = np.argmax(NS @ wo + b, axis=1)
            q_next = (NS @ wt + b)[np.arange(len(NS)), a_star]
        else:
            q_next = np.max(NS @ wt + b, axis=1)
        q_obj = np.where(D, R, R + self.gamma * q_next)
        loss = float(np.mean((q_act - q_obj) ** 2))
        self.historial_loss.append(loss)
        self._loss_ema = 0.05 * loss + 0.95 * self._loss_ema
        self.estadisticas_dqn['loss_media'] = self._loss_ema
        self.estadisticas_dqn['convergencia'] = 1.0 - min(self._loss_ema, 1.0)
        if self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.tiempo_upd += time.perf_counter() - t0
        return loss

    def entrenar_desde_buffer(self, batch: Optional[int] = None) -> Optional[float]:
        if len(self.buffer) < 4:
            return None
        b = self.buffer.sample(batch or self.batch_size)
        return self.actualizar_q_batch(b['s'], b['a'], b['r'], b['ns'], b['d'])

    def entrenar_paso(self, s: np.ndarray, a: int, r: float, ns: np.ndarray,
                          d: bool = False, usar_buffer: bool = True) -> Dict[str, float]:
        self.recordar(s, a, r, ns, d)
        if usar_buffer and len(self.buffer) >= 4:
            loss = self.entrenar_desde_buffer()
        else:
            loss = self.actualizar_q_batch(
                np.asarray(s).reshape(1, -1), np.array([a]),
                np.array([r]), np.asarray(ns).reshape(1, -1), np.array([d]))
        return {'loss': float(loss) if loss is not None else 0.0, 'paso': float(self.pasos)}

    def obtener_mejor_politica(self, estados: np.ndarray) -> np.ndarray:
        return np.argmax(self.valores_q(estados, "online"), axis=1)

    def obtener_valores_q(self, estado: np.ndarray) -> np.ndarray:
        return self.valores_q(estado, "online")[0].copy()

    def obtener_estadisticas_dqn(self) -> Dict[str, Any]:
        if self.q_online is None:
            return {'estado': 'no_inicializada'}
        qf = self.q_online.astype(np.float64).flatten()
        tot = self.estadisticas_dqn['exploracion_rate'] + self.estadisticas_dqn['exploitacion_rate']
        return {'q_media': float(qf.mean()), 'q_std': float(qf.std()),
                'q_min': float(qf.min()), 'q_max': float(qf.max()),
                'loss_media': self.estadisticas_dqn['loss_media'],
                'convergencia': self.estadisticas_dqn['convergencia'],
                'epsilon': self.epsilon, 'buffer_len': len(self.buffer),
                'actualizaciones': self.estadisticas_dqn['actualizaciones'],
                'syncs_objetivo': self.estadisticas_dqn['syncs_objetivo'],
                'tasa_exploracion': self.estadisticas_dqn['exploracion_rate'] / max(1, tot),
                't_medio_fwd_us': self.tiempo_fwd / max(1, self.pasos + 1) * 1e6,
                't_medio_upd_us': self.tiempo_upd / max(1, self.pasos) * 1e6}

    def verificar_estabilidad(self) -> Dict[str, bool]:
        est = {'convergencia_ok': self.estadisticas_dqn['convergencia'] > 0.7,
               'epsilon_ok': self.epsilon_min <= self.epsilon <= 1.0,
               'buffer_ok': len(self.buffer) > 0}
        if self.q_online is not None:
            est['q_no_explosiva'] = float(np.max(np.abs(self.q_online))) < 100.0
            est['sin_nan'] = bool(not np.isnan(self.q_online).any())
            est['sin_inf'] = bool(not np.isinf(self.q_online).any())
        else:
            est.update({'q_no_explosiva': True, 'sin_nan': True, 'sin_inf': True})
        return est

    def reinicializar_con_parametros(self, learning_rate: float = None, gamma: float = None,
                                        epsilon: float = None) -> None:
        if learning_rate is not None:
            self.learning_rate = float(learning_rate)
        if gamma is not None:
            self.gamma = float(gamma)
        if epsilon is not None:
            self.epsilon = float(epsilon)
        self.inicializar_pesos()
        self.resetear_historial()
        self.buffer.clear()
        self.historial_loss.clear()
        self.historial_q.clear()
        self.pasos = 0
        logger.info(f"DQN reinicializada: lr={self.learning_rate}")

    def a_dict(self) -> Dict[str, Any]:
        return {'input_size': self.input_size, 'output_size': self.output_size,
                'nombre': self.nombre, 'learning_rate': self.learning_rate,
                'gamma': self.gamma, 'epsilon': self.epsilon,
                'q_online': self.q_online.tolist() if self.q_online is not None else None}

    def guardar(self, ruta: str) -> None:
        import json
        with open(ruta, 'w', encoding='utf-8') as f:
            json.dump(self.a_dict(), f)

    @classmethod
    def cargar(cls, ruta: str) -> "NeuronaRefuerzoDQN":
        import json
        with open(ruta, 'r', encoding='utf-8') as f:
            d = json.load(f)
        o = cls(d['input_size'], d['output_size'], d.get('nombre', 'NeuronaRefuerzoDQN'))
        if d.get('q_online') is not None:
            o.q_online = np.asarray(d['q_online'], dtype=_dt())
            o.q_objetivo = o.q_online.copy()
            o.sesgo = np.zeros((1, o.output_size), dtype=_dt())
            o.pesos = o.q_online
        return o

    def __str__(self) -> str:
        return (f"NeuronaRefuerzoDQN(entrada={self.input_size}, salida={self.output_size}, "
                f"lr={self.learning_rate}, gamma={self.gamma}, epsilon={self.epsilon:.3f})")

    def __repr__(self) -> str:
        return self.__str__()


def crear_neurona_dqn(input_size: int, output_size: int,
                          configuracion: Dict[str, Any] = None) -> NeuronaRefuerzoDQN:
    configuracion = configuracion or {}
    return NeuronaRefuerzoDQN(
        input_size=input_size, output_size=output_size,
        nombre=configuracion.get('nombre', 'NeuronaRefuerzoDQN'),
        learning_rate=configuracion.get('learning_rate', 0.001),
        gamma=configuracion.get('gamma', 0.99),
        epsilon=configuracion.get('epsilon', 0.2),
        usar_doble=configuracion.get('usar_doble', True))


def analizar_dqn(neurona: NeuronaRefuerzoDQN) -> Dict[str, Any]:
    return {'estadisticas': neurona.obtener_estadisticas_dqn(),
            'estable': neurona.verificar_estabilidad()}


RFEN4_CONFIG = {'inicializacion_preferida': 'he', 'learning_rate_default': 0.001,
                'gamma_default': 0.99, 'epsilon_default': 0.2, 'doble_default': True,
                'sync_default': 100, 'batch_default': 32, 'umbral_convergencia': 0.7}
logger.info("RFEN1_RN_4.py cargado correctamente - Neurona de Refuerzo DQN")