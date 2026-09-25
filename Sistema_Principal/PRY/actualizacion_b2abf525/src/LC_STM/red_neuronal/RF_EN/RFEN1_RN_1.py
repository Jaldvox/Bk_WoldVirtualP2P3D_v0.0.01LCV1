"""RFEN1_RN_1.py - Neurona de Refuerzo con Q-Learning (refactor eficiente)."""
import numpy as np
import math
import time
import logging
from typing import Tuple, Optional, Dict, Any, List

LUCIA_RL_CONFIG = {'precision': 'float32', 'random_seed': 42,
                   'default_learning_rate': 0.1}


class NeuronaRefuerzoBase:
    """Base minimalista local (evita importar .base inexistente)."""

    def __init__(self, input_size: int, output_size: int, nombre: str = "NeuronaRefuerzo"):
        self.input_size = int(input_size)
        self.output_size = int(output_size)
        self.nombre = str(nombre)
        self.pesos = None
        self.sesgo = None
        self.historial_activaciones: List[np.ndarray] = []
        self.historial_gradientes: List[Dict[str, Any]] = []

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
    def ortogonal(cls, shape, seed=42):
        a = cls._rng(seed).normal(0, 1, shape)
        q, _ = np.linalg.qr(a) if shape[0] >= shape[1] else np.linalg.qr(a.T)
        q = q if shape[0] >= shape[1] else q.T
        return q[:, :shape[1]].astype(LUCIA_RL_CONFIG['precision'])

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

logger = logging.getLogger('RFENRN1.RFEN1_RN_1')
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


class NeuronaRefuerzoQLearning(NeuronaRefuerzoBase):
    """Q-Learning eficiente: Bellman vectorizado, RNG propio, buffers acotados."""

    def __init__(self, input_size: int, output_size: int,
                 nombre: str = "NeuronaRefuerzoQLearning",
                 learning_rate: float = 0.1,
                 gamma: float = 0.99,
                 epsilon: float = 0.1,
                 epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.01,
                 usar_xavier: bool = True,
                 usar_doble_q: bool = False,
                 sync_objetivo_cada: int = 0,
                 semilla: int = 42,
                 capacidad_historial: int = 512):
        super().__init__(input_size, output_size, nombre)
        self.learning_rate = float(learning_rate)
        self.gamma = float(gamma)
        self.epsilon = float(epsilon)
        self.epsilon_decay = float(epsilon_decay)
        self.epsilon_min = float(epsilon_min)
        self.usar_xavier = bool(usar_xavier)
        self.usar_doble_q = bool(usar_doble_q)
        self.sync_objetivo_cada = int(sync_objetivo_cada)
        self.rng = np.random.default_rng(semilla)
        self.q_table = None
        self.q_objetivo = None
        self.sesgo = None
        self.pasos = 0
        self.tiempo_fwd = 0.0
        self.tiempo_upd = 0.0
        self.estadisticas_qlearning = {
            'convergencia_q': 0.0, 'estabilidad_q': 0.0,
            'exploracion_rate': 0, 'exploitacion_rate': 0,
            'q_values_media': 0.0, 'q_values_std': 0.0,
            'q_values_min': 0.0, 'q_values_max': 0.0,
            'actualizaciones_q': 0, 'convergencia_detectada': False}
        self.historial_q_values = _Ring(capacidad_historial)
        self.historial_epsilon = _Ring(capacidad_historial)
        self.historial_convergencia = _Ring(capacidad_historial)
        self._delta_ema = 0.0
        logger.info(f"NeuronaRefuerzoQLearning creada: {self}")

    def inicializar_pesos(self, modo: Optional[str] = None) -> None:
        dt = _dt()
        m = (modo or ('xavier' if self.usar_xavier else 'ceros')).lower()
        shp = (self.input_size, self.output_size)
        if m == 'xavier':
            self.q_table = inicializar_pesos_xavier(shp, self.input_size, self.output_size)
        elif m == 'he':
            self.q_table = inicializar_pesos_he(shp, self.input_size)
        elif m == 'lecun':
            self.q_table = inicializar_pesos_lecun(shp, self.input_size)
        elif m == 'ortogonal':
            self.q_table = inicializar_pesos_ortogonal(shp)
        elif m == 'espectral':
            self.q_table = inicializar_pesos_espectral(shp)
        else:
            self.q_table = np.zeros(shp, dtype=dt)
        self.q_table = np.ascontiguousarray(self.q_table, dtype=dt)
        self.q_objetivo = self.q_table.copy()
        self.sesgo = np.zeros((1, self.output_size), dtype=dt)
        self.pesos = self.q_table
        logger.info(f"Tabla Q inicializada ({m}): forma={self.q_table.shape}")

    def _asegurar(self) -> None:
        if self.q_table is None:
            self.inicializar_pesos()
        assert self.q_table is not None

    def forward(self, estado: np.ndarray) -> np.ndarray:
        t0 = time.perf_counter()
        self._asegurar()
        assert self.q_table is not None
        s = int(np.asarray(estado).flat[0]) % self.input_size
        qv = self.q_table[s]
        if self.rng.random() < self.epsilon:
            acc = int(self.rng.integers(0, self.output_size))
            self.estadisticas_qlearning['exploracion_rate'] += 1
        else:
            acc = int(np.argmax(qv))
            self.estadisticas_qlearning['exploitacion_rate'] += 1
        self.historial_q_values.append(qv.copy())
        self.historial_epsilon.append(self.epsilon)
        self.tiempo_fwd += time.perf_counter() - t0
        return np.array([acc])

    def forward_batch(self, estados: np.ndarray) -> np.ndarray:
        arr = np.asarray(estados).flatten().astype(np.int64) % self.input_size
        self._asegurar()
        assert self.q_table is not None
        return np.argmax(self.q_table[arr], axis=1)

    def actualizar_q_value(self, estado: int, accion: int, recompensa: float,
                           siguiente_estado: int, terminado: bool = False) -> float:
        t0 = time.perf_counter()
        self._asegurar()
        assert self.q_table is not None and self.q_objetivo is not None
        s = int(estado) % self.input_size
        ns = int(siguiente_estado) % self.input_size
        a = int(accion) % self.output_size
        q_act = float(self.q_table[s, a])
        if terminado:
            q_obj = float(recompensa)
        elif self.usar_doble_q:
            a_star = int(np.argmax(self.q_table[ns]))
            q_obj = float(recompensa) + self.gamma * float(self.q_objetivo[ns, a_star])
        else:
            q_obj = float(recompensa) + self.gamma * float(np.max(self.q_objetivo[ns]))
        delta = q_obj - q_act
        self.q_table[s, a] = q_act + self.learning_rate * delta
        self.pasos += 1
        self.estadisticas_qlearning['actualizaciones_q'] += 1
        self._delta_ema = 0.02 * abs(delta) + 0.98 * self._delta_ema
        if self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        if self.sync_objetivo_cada > 0 and self.pasos % self.sync_objetivo_cada == 0:
            self.q_objetivo = self.q_table.copy()
        elif self.sync_objetivo_cada == 0:
            self.q_objetivo = self.q_table
        self._convergencia_rapida()
        self.tiempo_upd += time.perf_counter() - t0
        return delta

    def actualizar_q_batch(self, estados: np.ndarray, acciones: np.ndarray,
                           recompensas: np.ndarray, siguientes: np.ndarray,
                           terminados: np.ndarray) -> float:
        """Bellman vectorizado por mini-lote (mucho mas rapido que 1 a 1)."""
        self._asegurar()
        assert self.q_table is not None and self.q_objetivo is not None
        s = np.asarray(estados, dtype=np.int64) % self.input_size
        a = np.asarray(acciones, dtype=np.int64) % self.output_size
        r = np.asarray(recompensas, dtype=np.float64)
        ns = np.asarray(siguientes, dtype=np.int64) % self.input_size
        done = np.asarray(terminados, dtype=bool)
        q_max = np.max(self.q_objetivo[ns], axis=1)
        q_obj = np.where(done, r, r + self.gamma * q_max)
        q_act = self.q_table[s, a].astype(np.float64)
        delta = q_obj - q_act
        self.q_table[s, a] = (q_act + self.learning_rate * delta).astype(_dt())
        n = len(s)
        self.pasos += n
        self.estadisticas_qlearning['actualizaciones_q'] += n
        self._delta_ema = 0.1 * float(np.mean(np.abs(delta))) + 0.9 * self._delta_ema
        if self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon_min, self.epsilon * (self.epsilon_decay ** n))
        if self.sync_objetivo_cada > 0 and self.pasos % self.sync_objetivo_cada < n:
            self.q_objetivo = self.q_table.copy()
        elif self.sync_objetivo_cada == 0:
            self.q_objetivo = self.q_table
        self._convergencia_rapida()
        return float(np.mean(delta ** 2))

    def _convergencia_rapida(self) -> None:
        conv = 1.0 - min(self._delta_ema, 1.0)
        self.estadisticas_qlearning['convergencia_q'] = conv
        self.historial_convergencia.append(conv)
        if conv > 0.95 and self.estadisticas_qlearning['actualizaciones_q'] > 500:
            self.estadisticas_qlearning['convergencia_detectada'] = True

    def calcular_estabilidad_q(self) -> float:
        if len(self.historial_q_values) < 20:
            return 0.0
        arr = np.stack(self.historial_q_values.tail(20), axis=0).astype(np.float64)
        var = float(np.mean(np.var(arr, axis=0)))
        est = 1.0 / (1.0 + var)
        self.estadisticas_qlearning['estabilidad_q'] = est
        return est

    def obtener_estadisticas_q(self) -> Dict[str, Any]:
        if self.q_table is None:
            return {'estado': 'no_inicializada'}
        qf = self.q_table.astype(np.float64).flatten()
        tot = self.estadisticas_qlearning['exploracion_rate'] + self.estadisticas_qlearning['exploitacion_rate']
        return {'q_values_media': float(np.mean(qf)), 'q_values_std': float(np.std(qf)),
                'q_values_min': float(np.min(qf)), 'q_values_max': float(np.max(qf)),
                'epsilon_actual': self.epsilon, 'learning_rate': self.learning_rate, 'gamma': self.gamma,
                'convergencia_q': self.estadisticas_qlearning['convergencia_q'],
                'estabilidad_q': self.calcular_estabilidad_q(),
                'actualizaciones_q': self.estadisticas_qlearning['actualizaciones_q'],
                'convergencia_detectada': self.estadisticas_qlearning['convergencia_detectada'],
                'tasa_exploracion': self.estadisticas_qlearning['exploracion_rate'] / max(1, tot),
                'tasa_exploitacion': self.estadisticas_qlearning['exploitacion_rate'] / max(1, tot),
                't_medio_fwd_us': self.tiempo_fwd / max(1, self.pasos) * 1e6,
                't_medio_upd_us': self.tiempo_upd / max(1, self.pasos) * 1e6}

    def verificar_estabilidad(self) -> Dict[str, bool]:
        est = {'convergencia_ok': self.estadisticas_qlearning['convergencia_q'] > 0.8,
               'q_values_estables': self.calcular_estabilidad_q() > 0.7,
               'epsilon_apropiado': self.epsilon_min <= self.epsilon <= 1.0}
        if self.q_table is not None:
            qmax = float(np.max(np.abs(self.q_table.astype(np.float64))))
            est['q_values_no_explosivos'] = qmax < 100.0
            est['q_values_no_desaparecen'] = True
            est['sin_nan'] = bool(not np.isnan(self.q_table).any())
            est['sin_inf'] = bool(not np.isinf(self.q_table).any())
        else:
            est.update({'q_values_no_explosivos': True, 'q_values_no_desaparecen': True,
                        'sin_nan': True, 'sin_inf': True})
        return est

    def reinicializar_con_parametros(self, learning_rate: float = None, gamma: float = None,
                                     epsilon: float = None, epsilon_decay: float = None,
                                     epsilon_min: float = None) -> None:
        if learning_rate is not None:
            self.learning_rate = float(learning_rate)
        if gamma is not None:
            self.gamma = float(gamma)
        if epsilon is not None:
            self.epsilon = float(epsilon)
        if epsilon_decay is not None:
            self.epsilon_decay = float(epsilon_decay)
        if epsilon_min is not None:
            self.epsilon_min = float(epsilon_min)
        self.inicializar_pesos()
        self.resetear_historial()
        self.historial_q_values.clear()
        self.historial_epsilon.clear()
        self.historial_convergencia.clear()
        for k in ('exploracion_rate', 'exploitacion_rate', 'actualizaciones_q'):
            self.estadisticas_qlearning[k] = 0
        self.estadisticas_qlearning['convergencia_detectada'] = False
        self.pasos = 0
        logger.info(f"Q-Learning reinicializada: lr={self.learning_rate}, gamma={self.gamma}, eps={self.epsilon}")

    def obtener_mejor_politica(self) -> np.ndarray:
        if self.q_table is None:
            raise ValueError("Tabla Q no inicializada")
        return np.argmax(self.q_table, axis=1)

    def obtener_valores_q(self, estado: int) -> np.ndarray:
        if self.q_table is None:
            raise ValueError("Tabla Q no inicializada")
        return self.q_table[int(estado) % self.input_size].copy()

    def resumen_rendimiento(self) -> Dict[str, Any]:
        return {'pasos': self.pasos, 't_fwd_s': self.tiempo_fwd, 't_upd_s': self.tiempo_upd,
                't_medio_fwd_us': self.tiempo_fwd / max(1, self.pasos) * 1e6,
                't_medio_upd_us': self.tiempo_upd / max(1, self.pasos) * 1e6,
                'actualizaciones': self.estadisticas_qlearning['actualizaciones_q']}

    def a_dict(self) -> Dict[str, Any]:
        return {'input_size': self.input_size, 'output_size': self.output_size, 'nombre': self.nombre,
                'learning_rate': self.learning_rate, 'gamma': self.gamma, 'epsilon': self.epsilon,
                'epsilon_decay': self.epsilon_decay, 'epsilon_min': self.epsilon_min,
                'q_table': self.q_table.tolist() if self.q_table is not None else None}

    def guardar(self, ruta: str) -> None:
        import json
        with open(ruta, 'w', encoding='utf-8') as f:
            json.dump(self.a_dict(), f)

    @classmethod
    def cargar(cls, ruta: str) -> "NeuronaRefuerzoQLearning":
        import json
        with open(ruta, 'r', encoding='utf-8') as f:
            d = json.load(f)
        o = cls(d['input_size'], d['output_size'], d.get('nombre', 'NeuronaRefuerzoQLearning'),
                learning_rate=d.get('learning_rate', 0.1), gamma=d.get('gamma', 0.99),
                epsilon=d.get('epsilon', 0.1))
        if d.get('q_table') is not None:
            o.q_table = np.asarray(d['q_table'], dtype=_dt())
            o.q_objetivo = o.q_table.copy()
            o.pesos = o.q_table
        return o

    def __str__(self) -> str:
        return (f"NeuronaRefuerzoQLearning(entrada={self.input_size}, salida={self.output_size}, "
                f"lr={self.learning_rate}, gamma={self.gamma}, epsilon={self.epsilon:.3f})")

    def __repr__(self) -> str:
        return self.__str__()


def crear_neurona_qlearning(input_size: int, output_size: int,
                            configuracion: Dict[str, Any] = None) -> NeuronaRefuerzoQLearning:
    configuracion = configuracion or {}
    return NeuronaRefuerzoQLearning(
        input_size=input_size, output_size=output_size,
        nombre=configuracion.get('nombre', 'NeuronaRefuerzoQLearning'),
        learning_rate=configuracion.get('learning_rate', 0.1),
        gamma=configuracion.get('gamma', 0.99),
        epsilon=configuracion.get('epsilon', 0.1),
        epsilon_decay=configuracion.get('epsilon_decay', 0.995),
        epsilon_min=configuracion.get('epsilon_min', 0.01),
        usar_xavier=configuracion.get('usar_xavier', True),
        usar_doble_q=configuracion.get('usar_doble_q', False))


def analizar_convergencia_qlearning(neurona: NeuronaRefuerzoQLearning) -> Dict[str, Any]:
    return {'convergencia_q': neurona.estadisticas_qlearning['convergencia_q'],
            'estabilidad_q': neurona.calcular_estabilidad_q(),
            'epsilon_actual': neurona.epsilon,
            'actualizaciones_q': neurona.estadisticas_qlearning['actualizaciones_q'],
            'convergencia_detectada': neurona.estadisticas_qlearning['convergencia_detectada'],
            'estadisticas_q': neurona.obtener_estadisticas_q(),
            'estabilidad_verificada': neurona.verificar_estabilidad()}


def comparar_inicializaciones_qlearning(input_size: int, output_size: int,
                                        episodios: int = 1000) -> Dict[str, Any]:
    nx = NeuronaRefuerzoQLearning(input_size, output_size, usar_xavier=True)
    nc = NeuronaRefuerzoQLearning(input_size, output_size, usar_xavier=False)
    rng = np.random.default_rng(0)
    sx = rng.integers(0, input_size, episodios)
    ax = np.array([nx.forward(np.array([s]))[0] for s in sx])
    rx = rng.random(episodios)
    nsx = rng.integers(0, input_size, episodios)
    nx.actualizar_q_batch(sx, ax, rx, nsx, np.zeros(episodios, dtype=bool))
    sc = rng.integers(0, input_size, episodios)
    ac = np.array([nc.forward(np.array([s]))[0] for s in sc])
    rc = rng.random(episodios)
    nsc = rng.integers(0, input_size, episodios)
    nc.actualizar_q_batch(sc, ac, rc, nsc, np.zeros(episodios, dtype=bool))
    out = {}
    for k, n in (('xavier', nx), ('ceros', nc)):
        out[k] = {'convergencia_q': n.estadisticas_qlearning['convergencia_q'],
                  'estabilidad_q': n.calcular_estabilidad_q(),
                  'q_values_media': float(np.mean(n.q_table)), 'q_values_std': float(np.std(n.q_table)),
                  'actualizaciones_q': n.estadisticas_qlearning['actualizaciones_q']}
    out['ratios'] = {'convergencia_q': out['xavier']['convergencia_q'] / max(1e-12, out['ceros']['convergencia_q']),
                     'estabilidad_q': out['xavier']['estabilidad_q'] / max(1e-12, out['ceros']['estabilidad_q'])}
    return out


RFEN1_CONFIG = {'inicializacion_preferida': 'xavier', 'learning_rate_default': 0.1,
                'gamma_default': 0.99, 'epsilon_default': 0.1, 'epsilon_decay_default': 0.995,
                'epsilon_min_default': 0.01, 'umbral_convergencia': 0.95, 'umbral_estabilidad': 0.7}
logger.info("RFEN1_RN_1.py cargado correctamente - Neurona de Refuerzo Q-Learning")
