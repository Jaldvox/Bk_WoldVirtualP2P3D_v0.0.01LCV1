"""RFEN1_RN_3.py - Neurona Actor-Critic (refactor eficiente)."""
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

    @classmethod
    def xavier(cls, shape, fan_in=1, fan_out=1, seed=42):
        lim = math.sqrt(6.0 / max(1, fan_in + fan_out))
        return cls._rng(seed).uniform(-lim, lim, shape).astype(LUCIA_RL_CONFIG['precision'])


def inicializar_pesos_ortogonal(shape, seed=42):
    return InicializadoresRL.ortogonal(shape, seed)


def inicializar_pesos_espectral(shape, seed=42):
    return InicializadoresRL.espectral(shape, seed)


def inicializar_pesos_xavier(shape, fan_in=1, fan_out=1, seed=42):
    return InicializadoresRL.xavier(shape, fan_in, fan_out, seed)


logger = logging.getLogger('RFENRN1.RFEN1_RN_3')
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


class NeuronaRefuerzoActorCritic(NeuronaRefuerzoBase):
    """Actor-Critic eficiente: GAE vectorizado, espectral por potencia, RNG propio."""

    def __init__(self, input_size: int, output_size: int,
                 nombre: str = "NeuronaRefuerzoActorCritic",
                 learning_rate_actor: float = 0.001,
                 learning_rate_critic: float = 0.002,
                 gamma: float = 0.99,
                 lambda_gae: float = 0.95,
                 usar_espectral: bool = True,
                 meta_learning: bool = True,
                 grad_clip: float = 5.0,
                 semilla: int = 42,
                 capacidad_historial: int = 512):
        super().__init__(input_size, output_size, nombre)
        self.learning_rate_actor = float(learning_rate_actor)
        self.learning_rate_critic = float(learning_rate_critic)
        self.gamma = float(gamma)
        self.lambda_gae = float(lambda_gae)
        self.usar_espectral = bool(usar_espectral)
        self.meta_learning = bool(meta_learning)
        self.grad_clip = float(grad_clip)
        self.rng = np.random.default_rng(semilla)
        self.pesos_actor = None
        self.sesgo_actor = None
        self.pesos_critic = None
        self.sesgo_critic = None
        self.sigma_actor = 1.0
        self.sigma_critic = 1.0
        self.pasos = 0
        self.tiempo_fwd = 0.0
        self.tiempo_upd = 0.0
        self.estadisticas_ac = {
            'convergencia_actor': 0.0, 'convergencia_critic': 0.0,
            'estabilidad_actor': 0.0, 'estabilidad_critic': 0.0,
            'advantage_media': 0.0, 'advantage_std': 0.0,
            'valor_error_medio': 0.0, 'policy_loss_medio': 0.0,
            'meta_adaptaciones': 0, 'espectral_updates': 0}
        self.historial_actor_loss = _Ring(capacidad_historial)
        self.historial_critic_loss = _Ring(capacidad_historial)
        self.historial_advantages = _Ring(capacidad_historial * 4)
        self.historial_valores = _Ring(capacidad_historial)
        self.historial_politicas = _Ring(capacidad_historial)
        logger.info(f"NeuronaRefuerzoActorCritic creada: {self}")

    def inicializar_pesos(self, modo: Optional[str] = None) -> None:
        m = (modo or ('espectral' if self.usar_espectral else 'ortogonal')).lower()
        if m == 'espectral':
            self.pesos_actor = inicializar_pesos_espectral((self.input_size, self.output_size))
            self.pesos_critic = inicializar_pesos_espectral((self.input_size, 1))
        elif m == 'ortogonal':
            self.pesos_actor = inicializar_pesos_ortogonal((self.input_size, self.output_size))
            self.pesos_critic = inicializar_pesos_ortogonal((self.input_size, 1))
        else:
            self.pesos_actor = inicializar_pesos_xavier((self.input_size, self.output_size), self.input_size, self.output_size)
            self.pesos_critic = inicializar_pesos_xavier((self.input_size, 1), self.input_size, 1)
        self.pesos_actor = np.ascontiguousarray(self.pesos_actor, dtype=_dt())
        self.pesos_critic = np.ascontiguousarray(self.pesos_critic, dtype=_dt())
        self.sesgo_actor = np.zeros((1, self.output_size), dtype=_dt())
        self.sesgo_critic = np.zeros((1, 1), dtype=_dt())
        self.sigma_actor = self._sigma(self.pesos_actor)
        self.sigma_critic = self._sigma(self.pesos_critic)
        logger.info(f"Pesos Actor-Critic inicializados con {m}")

    @staticmethod
    def _sigma(w: np.ndarray, iters: int = 3) -> float:
        """Mayor valor singular por iteracion de potencia (barato, sin SVD)."""
        wf = w.astype(np.float64)
        v = np.random.default_rng(0).normal(0, 1, (wf.shape[1], 1))
        v /= max(1e-12, np.linalg.norm(v))
        for _ in range(iters):
            u = wf @ v
            u /= max(1e-12, np.linalg.norm(u))
            v = wf.T @ u
            v /= max(1e-12, np.linalg.norm(v))
        return float((u.T @ wf @ v)[0, 0])

    def _norma_espectral(self, w: np.ndarray, sigma: float) -> np.ndarray:
        if not self.usar_espectral:
            return w
        return (w.astype(np.float64) / max(1e-12, sigma)).astype(_dt())

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

    def forward_actor(self, estado: np.ndarray) -> np.ndarray:
        if self.pesos_actor is None:
            self.inicializar_pesos()
        assert self.pesos_actor is not None and self.sesgo_actor is not None
        x = self._prep(estado)
        w = self._norma_espectral(self.pesos_actor, self.sigma_actor)
        probs = self._softmax(np.dot(x, w) + self.sesgo_actor).astype(_dt(), copy=False)
        self.historial_politicas.append(probs.copy())
        return np.ascontiguousarray(probs, dtype=_dt())

    def forward_critic(self, estado: np.ndarray) -> np.ndarray:
        if self.pesos_critic is None:
            self.inicializar_pesos()
        assert self.pesos_critic is not None and self.sesgo_critic is not None
        x = self._prep(estado)
        w = self._norma_espectral(self.pesos_critic, self.sigma_critic)
        v = (np.dot(x, w) + self.sesgo_critic).astype(_dt(), copy=False)
        self.historial_valores.append(v.copy())
        return np.ascontiguousarray(v, dtype=_dt())

    def forward(self, estado: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        t0 = time.perf_counter()
        out = (self.forward_actor(estado), self.forward_critic(estado))
        self.tiempo_fwd += time.perf_counter() - t0
        return out

    def seleccionar_accion(self, estado: np.ndarray) -> Tuple[int, float, float]:
        probs, valor = self.forward(estado)
        p = probs[0].astype(np.float64)
        p = p / max(1e-12, p.sum())
        acc = int(self.rng.choice(self.output_size, p=p))
        return acc, float(probs[0, acc]), float(valor[0, 0])

    def calcular_gae_advantages(self, recompensas: List[float], valores: List[float],
                                siguiente_valor: float = 0.0) -> np.ndarray:
        r = np.asarray(recompensas, dtype=np.float64)
        v = np.asarray(valores, dtype=np.float64)
        v_next = np.append(v[1:], siguiente_valor)
        deltas = r + self.gamma * v_next - v
        adv = np.zeros_like(deltas)
        acc = 0.0
        for t in range(len(deltas) - 1, -1, -1):
            acc = deltas[t] + self.gamma * self.lambda_gae * acc
            adv[t] = acc
        self.historial_advantages.append(adv.copy())
        self.estadisticas_ac['advantage_media'] = float(adv.mean())
        self.estadisticas_ac['advantage_std'] = float(adv.std())
        return adv

    def calcular_gradientes_ac(self, estados: List[np.ndarray], acciones: List[int],
                               advantages: List[float], valores_objetivo: List[float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if self.pesos_actor is None or self.pesos_critic is None:
            raise ValueError("Pesos no inicializados")
        X = np.vstack([self._prep(s) for s in estados]).astype(np.float64)
        A = np.asarray(acciones, dtype=np.int64) % self.output_size
        Adv = np.asarray(advantages, dtype=np.float64)
        Vt = np.asarray(valores_objetivo, dtype=np.float64).reshape(-1, 1)
        n = len(X)
        wa = self._norma_espectral(self.pesos_actor, self.sigma_actor).astype(np.float64)
        wc = self._norma_espectral(self.pesos_critic, self.sigma_critic).astype(np.float64)
        logits = X @ wa + self.sesgo_actor.astype(np.float64)
        probs = self._softmax(logits).astype(np.float64)
        vals = X @ wc + self.sesgo_critic.astype(np.float64)
        oh = np.zeros_like(probs)
        oh[np.arange(n), A] = 1.0
        dlog = (oh - probs) * (Adv / max(1e-12, abs(Adv).std() + 1.0))[:, None] / n
        ga_w = X.T @ dlog
        ga_b = dlog.sum(axis=0, keepdims=True)
        err = (Vt - vals) / n
        gc_w = X.T @ err
        gc_b = err.sum(axis=0, keepdims=True)
        for g in (ga_w, ga_b, gc_w, gc_b):
            ng = float(np.linalg.norm(g))
            if self.grad_clip > 0 and ng > self.grad_clip:
                g *= self.grad_clip / (ng + 1e-12)
        pl = float(np.mean(-np.log(probs[np.arange(n), A] + 1e-8) * Adv))
        cl = float(0.5 * np.mean((Vt - vals) ** 2))
        self.historial_actor_loss.append(pl)
        self.historial_critic_loss.append(cl)
        self.estadisticas_ac['policy_loss_medio'] = pl
        self.estadisticas_ac['valor_error_medio'] = cl
        return ga_w.astype(_dt()), ga_b.astype(_dt()), gc_w.astype(_dt()), gc_b.astype(_dt())

    def actualizar_pesos(self, grad_actor_pesos: np.ndarray, grad_actor_sesgo: np.ndarray,
                         grad_critic_pesos: np.ndarray, grad_critic_sesgo: np.ndarray) -> Dict[str, float]:
        t0 = time.perf_counter()
        if self.pesos_actor is None or self.pesos_critic is None or self.sesgo_actor is None or self.sesgo_critic is None:
            raise ValueError("Pesos no inicializados")
        self.pesos_actor = (self.pesos_actor.astype(np.float64) + self.learning_rate_actor * np.asarray(grad_actor_pesos, dtype=np.float64)).astype(_dt())
        self.sesgo_actor = (self.sesgo_actor.astype(np.float64) + self.learning_rate_actor * np.asarray(grad_actor_sesgo, dtype=np.float64)).astype(_dt())
        self.pesos_critic = (self.pesos_critic.astype(np.float64) + self.learning_rate_critic * np.asarray(grad_critic_pesos, dtype=np.float64)).astype(_dt())
        self.sesgo_critic = (self.sesgo_critic.astype(np.float64) + self.learning_rate_critic * np.asarray(grad_critic_sesgo, dtype=np.float64)).astype(_dt())
        self.pasos += 1
        if self.meta_learning:
            self._adaptar_meta_learning()
        if self.usar_espectral:
            self.sigma_actor = self._sigma(self.pesos_actor)
            self.sigma_critic = self._sigma(self.pesos_critic)
            self.estadisticas_ac['espectral_updates'] += 1
        self._calcular_convergencia_ac()
        self.tiempo_upd += time.perf_counter() - t0
        return {'paso': float(self.pasos)}

    def entrenar_paso(self, estados: List[np.ndarray], acciones: List[int],
                      recompensas: List[float], valores: List[float]) -> Dict[str, float]:
        adv = self.calcular_gae_advantages(recompensas, valores)
        vt = (adv + np.asarray(valores, dtype=np.float64)).tolist()
        ga = self.calcular_gradientes_ac(estados, acciones, adv.tolist(), vt)
        return {'ventaja_media': float(np.mean(adv)), **self.actualizar_pesos(*ga)}

    def _adaptar_meta_learning(self) -> None:
        if len(self.historial_actor_loss) < 10 or len(self.historial_critic_loss) < 10:
            return
        ta = float(np.mean(np.diff(np.asarray(self.historial_actor_loss.tail(10), dtype=np.float64))))
        tc = float(np.mean(np.diff(np.asarray(self.historial_critic_loss.tail(10), dtype=np.float64))))
        self.learning_rate_actor = float(np.clip(self.learning_rate_actor * (0.95 if ta > 0 else 1.05), 1e-6, 1e-2))
        self.learning_rate_critic = float(np.clip(self.learning_rate_critic * (0.95 if tc > 0 else 1.05), 1e-6, 1e-2))
        self.estadisticas_ac['meta_adaptaciones'] += 1

    def _calcular_convergencia_ac(self) -> None:
        if len(self.historial_actor_loss) < 10 or len(self.historial_critic_loss) < 10:
            return
        va = float(np.var(np.asarray(self.historial_actor_loss.tail(10), dtype=np.float64)))
        vc = float(np.var(np.asarray(self.historial_critic_loss.tail(10), dtype=np.float64)))
        self.estadisticas_ac['convergencia_actor'] = 1.0 / (1.0 + va)
        self.estadisticas_ac['convergencia_critic'] = 1.0 / (1.0 + vc)

    def calcular_estabilidad_ac(self) -> Tuple[float, float]:
        ea = ec = 0.0
        if len(self.historial_politicas) >= 20:
            ea = 1.0 / (1.0 + float(np.mean(np.var(np.stack(self.historial_politicas.tail(20), axis=0).astype(np.float64), axis=0))))
        if len(self.historial_valores) >= 20:
            ec = 1.0 / (1.0 + float(np.mean(np.var(np.stack(self.historial_valores.tail(20), axis=0).astype(np.float64), axis=0))))
        self.estadisticas_ac['estabilidad_actor'] = ea
        self.estadisticas_ac['estabilidad_critic'] = ec
        return ea, ec

    def obtener_estadisticas_ac(self) -> Dict[str, Any]:
        if self.pesos_actor is None or self.pesos_critic is None:
            return {'estado': 'no_inicializada'}
        ea, ec = self.calcular_estabilidad_ac()
        stats: Dict[str, Any] = {
            'learning_rate_actor': self.learning_rate_actor, 'learning_rate_critic': self.learning_rate_critic,
            'gamma': self.gamma, 'lambda_gae': self.lambda_gae,
            'convergencia_actor': self.estadisticas_ac['convergencia_actor'],
            'convergencia_critic': self.estadisticas_ac['convergencia_critic'],
            'estabilidad_actor': ea, 'estabilidad_critic': ec,
            'meta_adaptaciones': self.estadisticas_ac['meta_adaptaciones'],
            'espectral_updates': self.estadisticas_ac['espectral_updates'],
            'usar_espectral': self.usar_espectral, 'meta_learning': self.meta_learning,
            't_medio_fwd_us': self.tiempo_fwd / max(1, self.pasos + 1) * 1e6,
            't_medio_upd_us': self.tiempo_upd / max(1, self.pasos) * 1e6}
        if len(self.historial_advantages):
            arr = np.concatenate([np.asarray(a, dtype=np.float64).flatten() for a in self.historial_advantages.tail(10)])
            stats.update({'advantage_media': float(arr.mean()), 'advantage_std': float(arr.std()),
                          'advantage_min': float(arr.min()), 'advantage_max': float(arr.max())})
        if len(self.historial_actor_loss):
            stats['actor_loss_medio'] = float(np.mean(np.asarray(self.historial_actor_loss.tail(10), dtype=np.float64)))
        if len(self.historial_critic_loss):
            stats['critic_loss_medio'] = float(np.mean(np.asarray(self.historial_critic_loss.tail(10), dtype=np.float64)))
        return stats

    def verificar_estabilidad(self) -> Dict[str, bool]:
        ea, ec = self.calcular_estabilidad_ac()
        est = {'actor_convergencia_ok': self.estadisticas_ac['convergencia_actor'] > 0.7,
               'critic_convergencia_ok': self.estadisticas_ac['convergencia_critic'] > 0.7,
               'actor_estable': ea > 0.7, 'critic_estable': ec > 0.7,
               'lr_actor_ok': 1e-6 <= self.learning_rate_actor <= 1e-2,
               'lr_critic_ok': 1e-6 <= self.learning_rate_critic <= 1e-2}
        if self.pesos_actor is not None and self.pesos_critic is not None:
            est['pesos_actor_no_explosivos'] = float(np.max(np.abs(self.pesos_actor))) < 5.0
            est['pesos_critic_no_explosivos'] = float(np.max(np.abs(self.pesos_critic))) < 5.0
            est['sin_nan'] = bool(not np.isnan(self.pesos_actor).any() and not np.isnan(self.pesos_critic).any())
        else:
            est.update({'pesos_actor_no_explosivos': True, 'pesos_critic_no_explosivos': True, 'sin_nan': True})
        return est

    def reinicializar_con_parametros(self, learning_rate_actor: float = None, learning_rate_critic: float = None,
                                     gamma: float = None, lambda_gae: float = None) -> None:
        if learning_rate_actor is not None:
            self.learning_rate_actor = float(learning_rate_actor)
        if learning_rate_critic is not None:
            self.learning_rate_critic = float(learning_rate_critic)
        if gamma is not None:
            self.gamma = float(gamma)
        if lambda_gae is not None:
            self.lambda_gae = float(lambda_gae)
        self.inicializar_pesos()
        self.resetear_historial()
        self.historial_actor_loss.clear()
        self.historial_critic_loss.clear()
        self.historial_advantages.clear()
        self.historial_valores.clear()
        self.historial_politicas.clear()
        self.estadisticas_ac['meta_adaptaciones'] = 0
        self.estadisticas_ac['espectral_updates'] = 0
        self.pasos = 0
        logger.info(f"Actor-Critic reinicializada: lr_a={self.learning_rate_actor}, lr_c={self.learning_rate_critic}")

    def a_dict(self) -> Dict[str, Any]:
        return {'input_size': self.input_size, 'output_size': self.output_size, 'nombre': self.nombre,
                'lr_actor': self.learning_rate_actor, 'lr_critic': self.learning_rate_critic,
                'pesos_actor': self.pesos_actor.tolist() if self.pesos_actor is not None else None,
                'pesos_critic': self.pesos_critic.tolist() if self.pesos_critic is not None else None}

    def guardar(self, ruta: str) -> None:
        import json
        with open(ruta, 'w', encoding='utf-8') as f:
            json.dump(self.a_dict(), f)

    @classmethod
    def cargar(cls, ruta: str) -> "NeuronaRefuerzoActorCritic":
        import json
        with open(ruta, 'r', encoding='utf-8') as f:
            d = json.load(f)
        o = cls(d['input_size'], d['output_size'], d.get('nombre', 'NeuronaRefuerzoActorCritic'))
        if d.get('pesos_actor') is not None:
            o.pesos_actor = np.asarray(d['pesos_actor'], dtype=_dt())
            o.pesos_critic = np.asarray(d['pesos_critic'], dtype=_dt())
            o.sesgo_actor = np.zeros((1, o.output_size), dtype=_dt())
            o.sesgo_critic = np.zeros((1, 1), dtype=_dt())
        return o

    def __str__(self) -> str:
        return (f"NeuronaRefuerzoActorCritic(entrada={self.input_size}, salida={self.output_size}, "
                f"lr_actor={self.learning_rate_actor}, lr_critic={self.learning_rate_critic}, espectral={self.usar_espectral})")

    def __repr__(self) -> str:
        return self.__str__()


def crear_neurona_actor_critic(input_size: int, output_size: int,
                               configuracion: Dict[str, Any] = None) -> NeuronaRefuerzoActorCritic:
    configuracion = configuracion or {}
    return NeuronaRefuerzoActorCritic(
        input_size=input_size, output_size=output_size,
        nombre=configuracion.get('nombre', 'NeuronaRefuerzoActorCritic'),
        learning_rate_actor=configuracion.get('learning_rate_actor', 0.001),
        learning_rate_critic=configuracion.get('learning_rate_critic', 0.002),
        gamma=configuracion.get('gamma', 0.99),
        lambda_gae=configuracion.get('lambda_gae', 0.95),
        usar_espectral=configuracion.get('usar_espectral', True),
        meta_learning=configuracion.get('meta_learning', True))


def analizar_ac(neurona: NeuronaRefuerzoActorCritic) -> Dict[str, Any]:
    return {'estadisticas': neurona.obtener_estadisticas_ac(), 'estable': neurona.verificar_estabilidad()}


RFEN3_CONFIG = {'inicializacion_preferida': 'espectral', 'learning_rate_actor_default': 0.001,
                'learning_rate_critic_default': 0.002, 'gamma_default': 0.99, 'lambda_gae_default': 0.95,
                'meta_learning_default': True, 'espectral_default': True,
                'umbral_convergencia': 0.7, 'umbral_estabilidad': 0.7}
logger.info("RFEN1_RN_3.py cargado correctamente - Neurona de Refuerzo Actor-Critic Avanzado")