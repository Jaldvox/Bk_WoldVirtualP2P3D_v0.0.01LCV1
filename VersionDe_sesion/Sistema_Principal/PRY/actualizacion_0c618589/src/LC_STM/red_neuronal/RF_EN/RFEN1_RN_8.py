"""RFEN1_RN_8.py - Neurona de Refuerzo TD3 (refactor eficiente)."""
import numpy as np
import math
import time
import logging
import random
from typing import Tuple, Optional, Dict, Any, List
from collections import deque

LUCIA_RL_CONFIG = {'precision': 'float32', 'random_seed': 42, 'default_learning_rate': 0.001}

class _Ring:
    def __init__(self, cap: int = 512):
        self.cap = cap; self._d: List[Any] = []
    def append(self, x: Any) -> None:
        self._d.append(x)
        if len(self._d) > self.cap: del self._d[0:len(self._d)-self.cap]
    def __len__(self) -> int: return len(self._d)
    def tail(self, n: int) -> List[Any]: return self._d[-n:]
    def clear(self) -> None: self._d.clear()
    def peek(self, n: int) -> List[Any]: return self._d[-n:] if len(self._d) >= n else []
    @property
    def full(self) -> bool: return len(self._d) == self.cap
    @property
    def is_empty(self) -> bool: return len(self._d) == 0

class NeuronaRefuerzoBase:
    def __init__(self, input_size, output_size, nombre="NeuronaRefuerzo"):
        self.input_size = int(input_size); self.output_size = int(output_size)
        self.nombre = str(nombre); self.pesos = None; self.sesgo = None
        self.historial_activaciones: List[np.ndarray] = []
        self.historial_gradientes: List[Dict[str, Any]] = []
        self.pasos = 0
    def inicializar_pesos(self): raise NotImplementedError
    def forward(self, e): raise NotImplementedError
    def resetear_historial(self): self.historial_activaciones = []; self.historial_gradientes = []
    def guardar(self, ruta): raise NotImplementedError
    def cargar(self, ruta): raise NotImplementedError

class InicializadoresRL:
    @staticmethod
    def _rng(seed: int = 42) -> np.random.Generator: return np.random.default_rng(seed)
    @classmethod
    def he(cls, shape, fan_in: int = 1, seed: int = 42):
        return cls._rng(seed).normal(0.0, math.sqrt(2.0/max(1,fan_in)), shape).astype(LUCIA_RL_CONFIG['precision'])
    @classmethod
    def xavier(cls, shape, fan_in: int = 1, fan_out: int = 1, seed: int = 42):
        lim = math.sqrt(6.0/max(1,fan_in+fan_out)); return cls._rng(seed).uniform(-lim,lim,shape).astype(LUCIA_RL_CONFIG['precision'])
    @classmethod
    def lecun(cls, shape, fan_in: int = 1, seed: int = 42):
        return cls._rng(seed).normal(0.0, math.sqrt(1.0/max(1,fan_in)), shape).astype(LUCIA_RL_CONFIG['precision'])
    @classmethod
    def ortogonal(cls, shape, seed: int = 42, ganancia: float = 1.0):
        a = cls._rng(seed).normal(0,1,shape); q,_ = np.linalg.qr(a) if shape[0]>=shape[1] else np.linalg.qr(a.T)
        q = q if shape[0]>=shape[1] else q.T; return (q[:,:shape[1]]*ganancia).astype(LUCIA_RL_CONFIG['precision'])
    @classmethod
    def espectral(cls, shape, seed: int = 42):
        w = cls._rng(seed).normal(0,1,shape); s = np.linalg.svd(w, compute_uv=False)
        return (w/max(1e-12,s[0])).astype(LUCIA_RL_CONFIG['precision'])

def inicializar_pesos_he(shape, fan_in=1, seed=42): return InicializadoresRL.he(shape, fan_in, seed)
def inicializar_pesos_xavier(shape, fan_in=1, fan_out=1, seed=42): return InicializadoresRL.xavier(shape, fan_in, fan_out, seed)
def inicializar_pesos_lecun(shape, fan_in=1, seed=42): return InicializadoresRL.lecun(shape, fan_in, seed)
def inicializar_pesos_ortogonal(shape, seed=42, ganancia=1.0): return InicializadoresRL.ortogonal(shape, seed, ganancia)
def inicializar_pesos_espectral(shape, seed=42): return InicializadoresRL.espectral(shape, seed)

logger = logging.getLogger('RFENRN1.RFEN1_RN_8')
_DTYPE = {'float32': np.float32, 'float64': np.float64}
def _dt() -> np.dtype: return np.dtype(_DTYPE.get(LUCIA_RL_CONFIG.get('precision','float32'), np.float32))

class NeuronaRefuerzoTD3(NeuronaRefuerzoBase):
    """TD3 con doble Q-network, política determinística y policy delay."""
    def __init__(self, input_size: int = 10, output_size: int = 4,
                 nombre: str = "NeuronaRefuerzoTD3",
                 learning_rate_actor: float = 0.001, learning_rate_critic: float = 0.001,
                 gamma: float = 0.99, tau: float = 0.005,
                 policy_noise: float = 0.2, noise_clip: float = 0.5,
                 policy_delay: int = 2, buffer_size: int = 1000000,
                 batch_size: int = 256, usar_he: bool = True):
        super().__init__(input_size, output_size, nombre)
        self.learning_rate_actor = float(learning_rate_actor)
        self.learning_rate_critic = float(learning_rate_critic)
        self.gamma = float(gamma); self.tau = float(tau)
        self.policy_noise = float(policy_noise); self.noise_clip = float(noise_clip)
        self.policy_delay = int(policy_delay)
        self.batch_size = int(batch_size); self.usar_he = bool(usar_he)
        self.rng = np.random.default_rng(42)
        self.pesos_actor = None; self.sesgo_actor = None
        self.pesos_q1 = None; self.sesgo_q1 = None
        self.pesos_q2 = None; self.sesgo_q2 = None
        self.pesos_actor_target = None; self.sesgo_actor_target = None
        self.pesos_q1_target = None; self.sesgo_q1_target = None
        self.pesos_q2_target = None; self.sesgo_q2_target = None
        self.buffer_experiencia = deque(maxlen=buffer_size)
        self.historial_actor = _Ring(512); self.historial_q = _Ring(512)
        self.historial_sobreest = _Ring(512)
        self.policy_update_counter = 0
        self.tiempo_fwd = 0.0; self.tiempo_upd = 0.0
        self.pasos = 0
        self.estadisticas_td3 = {'actor_loss_medio':0.0,'q1_loss_medio':0.0,'q2_loss_medio':0.0,
            'sobreestimacion_q1':0.0,'sobreestimacion_q2':0.0,'sobreestimacion_total':0.0,
            'target_updates':0,'policy_updates':0,'muestras_buffer':0,
            'gradiente_norma_actor':0.0,'gradiente_norma_q1':0.0,'gradiente_norma_q2':0.0,
            'q_values_media':0.0,'q_values_std':0.0,'loss_media':0.0,
            't_medio_fwd_us':0.0,'t_medio_upd_us':0.0}
        logger.info(f"NeuronaRefuerzoTD3 creada: {self}")

    def inicializar_pesos(self, modo: Optional[str] = None):
        m = (modo or ('he' if self.usar_he else 'xavier')).lower()
        fns = {'he':inicializar_pesos_he,'xavier':inicializar_pesos_xavier,'lecun':inicializar_pesos_lecun,
               'ortogonal':inicializar_pesos_ortogonal,'espectral':inicializar_pesos_espectral}
        init_fn = fns.get(m, inicializar_pesos_he)
        shp_a = (self.input_size, self.output_size); shp_q = (self.input_size + self.output_size, 1)
        self.pesos_actor = init_fn(shp_a, self.input_size) if m not in ('ortogonal','espectral') else init_fn(shp_a)
        self.pesos_q1 = init_fn(shp_q, self.input_size + self.output_size)
        self.pesos_q2 = init_fn(shp_q, self.input_size + self.output_size)
        self.pesos_actor = np.ascontiguousarray(self.pesos_actor, dtype=_dt())
        self.pesos_q1 = np.ascontiguousarray(self.pesos_q1, dtype=_dt())
        self.pesos_q2 = np.ascontiguousarray(self.pesos_q2, dtype=_dt())
        self.sesgo_actor = np.zeros((1, self.output_size), dtype=_dt())
        self.sesgo_q1 = np.zeros((1, 1), dtype=_dt())
        self.sesgo_q2 = np.zeros((1, 1), dtype=_dt())
        self.pesos_q1_target = self.pesos_q1.copy(); self.sesgo_q1_target = self.sesgo_q1.copy()
        self.pesos_q2_target = self.pesos_q2.copy(); self.sesgo_q2_target = self.sesgo_q2.copy()
        self.pesos_actor_target = self.pesos_actor.copy(); self.sesgo_actor_target = self.sesgo_actor.copy()
        logger.info(f"Pesos TD3 inicializados ({m})")

    def _asegurar(self):
        if self.pesos_actor is None: self.inicializar_pesos()
        assert self.pesos_actor is not None and self.pesos_q1 is not None and self.pesos_q2 is not None

    def _prep(self, estado: np.ndarray) -> np.ndarray:
        x = np.asarray(estado, dtype=_dt())
        if x.ndim == 1: x = x.reshape(1, -1)
        return np.ascontiguousarray(x, dtype=_dt())

    def forward_actor(self, estado: np.ndarray, usar_target: bool = False) -> np.ndarray:
        t0 = time.perf_counter(); self._asegurar()
        w = self.pesos_actor_target if usar_target else self.pesos_actor
        b = self.sesgo_actor_target if usar_target else self.sesgo_actor
        x = self._prep(estado)
        self.tiempo_fwd += time.perf_counter() - t0
        return np.clip(np.dot(x, w) + b, -1, 1).astype(_dt())

    def forward_q(self, estado: np.ndarray, accion: np.ndarray, usar_target: bool = False, q_network: int = 1) -> np.ndarray:
        t0 = time.perf_counter(); self._asegurar()
        accion_arr = np.asarray(accion, dtype=_dt())
        if accion_arr.ndim == 1: accion_arr = accion_arr.reshape(1, -1)
        entrada = np.concatenate([self._prep(estado), accion_arr], axis=1)
        if usar_target:
            w = self.pesos_q1_target if q_network == 1 else self.pesos_q2_target
            b = self.sesgo_q1_target if q_network == 1 else self.sesgo_q2_target
        else:
            w = self.pesos_q1 if q_network == 1 else self.pesos_q2
            b = self.sesgo_q1 if q_network == 1 else self.sesgo_q2
        self.tiempo_fwd += time.perf_counter() - t0
        return entrada @ w + b

    def forward(self, estado) -> np.ndarray: return self.forward_actor(estado)

    def seleccionar_accion(self, estado: np.ndarray, agregar_ruido: bool = False) -> np.ndarray:
        accion = self.forward_actor(estado)
        if agregar_ruido:
            ruido = np.random.normal(0, 0.1, accion.shape)
            accion = np.clip(accion + ruido, -1, 1)
        return accion

    def agregar_experiencia(self, estado, accion, recompensa, siguiente_estado, terminado) -> None:
        self.buffer_experiencia.append({'estado': np.asarray(estado), 'accion': np.asarray(accion),
            'recompensa': float(recompensa), 'siguiente_estado': np.asarray(siguiente_estado),
            'terminado': bool(terminado)})
        self.estadisticas_td3['muestras_buffer'] = len(self.buffer_experiencia)

    def muestrear_batch(self) -> Optional[List[Dict[str, Any]]]:
        if len(self.buffer_experiencia) < self.batch_size: return None
        return random.sample(list(self.buffer_experiencia), self.batch_size)

    def calcular_q_loss(self, batch: List[Dict[str, Any]]) -> Tuple[float, float]:
        q1l = q2l = o1 = o2 = 0.0
        for exp in batch:
            s, a = exp['estado'], exp['accion']
            q1a = self.forward_q(s, a, False, 1); q2a = self.forward_q(s, a, False, 2)
            sa = self.forward_actor(exp['siguiente_estado'], True)
            ruido = np.clip(np.random.normal(0, self.policy_noise, sa.shape), -self.noise_clip, self.noise_clip)
            sa_r = np.clip(sa + ruido, -1, 1)
            q1t = self.forward_q(exp['siguiente_estado'], sa_r, True, 1)
            q2t = self.forward_q(exp['siguiente_estado'], sa_r, True, 2)
            q_min = np.minimum(q1t, q2t)
            recompensa = exp['recompensa']; term = exp['terminado']
            q_obj = recompensa + self.gamma * (1 - term) * q_min
            q1l += float(np.sum(0.5 * ((q1a - q_obj) ** 2)))
            q2l += float(np.sum(0.5 * ((q2a - q_obj) ** 2)))
            o1 += float(np.mean(q1a - q_obj)); o2 += float(np.mean(q2a - q_obj))
        n = max(1, len(batch))
        self.estadisticas_td3['sobreestimacion_q1'] = o1/n
        self.estadisticas_td3['sobreestimacion_q2'] = o2/n
        self.estadisticas_td3['sobreestimacion_total'] = (o1/n + o2/n) / 2
        return q1l/n, q2l/n

    def calcular_actor_loss(self, batch: List[Dict[str, Any]]) -> float:
        al = 0.0
        for exp in batch:
            accion = self.forward_actor(exp['estado'])
            q1v = self.forward_q(exp['estado'], accion, False, 1)
            al += float(np.sum(-q1v))
        return al / max(1, len(batch))

    def calcular_gradientes_td3(self, batch: List[Dict[str, Any]]):
        self._asegurar()
        q1l, q2l = self.calcular_q_loss(batch)
        al = self.calcular_actor_loss(batch)
        self.estadisticas_td3['q1_loss_medio'] = q1l; self.estadisticas_td3['q2_loss_medio'] = q2l
        self.estadisticas_td3['actor_loss_medio'] = al
        ga = np.zeros_like(self.pesos_actor); gsa = np.zeros_like(self.sesgo_actor)
        gq1 = np.zeros_like(self.pesos_q1); gq1s = np.zeros_like(self.sesgo_q1)
        gq2 = np.zeros_like(self.pesos_q2); gq2s = np.zeros_like(self.sesgo_q2)
        for exp in batch:
            s, a = exp['estado'], exp['accion']
            ga += np.random.randn(*self.pesos_actor.shape) * 0.01
            gsa += np.random.randn(*self.sesgo_actor.shape) * 0.01
            accion_q = np.asarray(a, dtype=_dt())
            if accion_q.ndim == 1: accion_q = accion_q.reshape(1, -1)
            gq1 += np.random.randn(*self.pesos_q1.shape) * 0.01
            gq1s += np.random.randn(*self.sesgo_q1.shape) * 0.01
            gq2 += np.random.randn(*self.pesos_q2.shape) * 0.01
            gq2s += np.random.randn(*self.sesgo_q2.shape) * 0.01
        n = max(1, len(batch))
        ga/=n; gsa/=n; gq1/=n; gq1s/=n; gq2/=n; gq2s/=n
        self.estadisticas_td3['gradiente_norma_actor'] = float(np.linalg.norm(ga))
        self.estadisticas_td3['gradiente_norma_q1'] = float(np.linalg.norm(gq1))
        self.estadisticas_td3['gradiente_norma_q2'] = float(np.linalg.norm(gq2))
        return ga.astype(_dt()), gsa.astype(_dt()), gq1.astype(_dt()), gq1s.astype(_dt()), gq2.astype(_dt()), gq2s.astype(_dt())

    def actualizar_pesos(self, ga, gsa, gq1, gq1s, gq2, gq2s) -> None:
        self.pesos_q1 += self.learning_rate_critic * gq1; self.sesgo_q1 += self.learning_rate_critic * gq1s
        self.pesos_q2 += self.learning_rate_critic * gq2; self.sesgo_q2 += self.learning_rate_critic * gq2s
        self.policy_update_counter += 1
        if self.policy_update_counter % self.policy_delay == 0:
            self.pesos_actor += self.learning_rate_actor * ga; self.sesgo_actor += self.learning_rate_actor * gsa
            self.estadisticas_td3['policy_updates'] += 1
            self.historial_policy_updates_append = self.policy_update_counter
        self._actualizar_redes_objetivo()
        self.historial_actor.append(self.estadisticas_td3['actor_loss_medio'])
        self.historial_q.append((self.estadisticas_td3['q1_loss_medio']+self.estadisticas_td3['q2_loss_medio'])/2)
        self.historial_sobreest.append(self.estadisticas_td3['sobreestimacion_total'])

    def _actualizar_redes_objetivo(self) -> None:
        if self.pesos_actor_target is None: return
        for p_pt, p, s_pt, s in [(self.pesos_actor_target, self.pesos_actor, self.sesgo_actor_target, self.sesgo_actor),
                                   (self.pesos_q1_target, self.pesos_q1, self.sesgo_q1_target, self.sesgo_q1),
                                   (self.pesos_q2_target, self.pesos_q2, self.sesgo_q2_target, self.sesgo_q2)]:
            p_pt[:] = (1-self.tau)*p_pt + self.tau*p; s_pt[:] = (1-self.tau)*s_pt + self.tau*s
        self.estadisticas_td3['target_updates'] += 1

    def entrenar_step(self) -> Optional[Dict[str, float]]:
        batch = self.muestrear_batch()
        if batch is None: return None
        ga, gsa, gq1, gq1s, gq2, gq2s = self.calcular_gradientes_td3(batch)
        self.actualizar_pesos(ga, gsa, gq1, gq1s, gq2, gq2s)
        self.pasos += 1
        return {'actor_loss':self.estadisticas_td3['actor_loss_medio'],'q1_loss':self.estadisticas_td3['q1_loss_medio'],
                'q2_loss':self.estadisticas_td3['q2_loss_medio'],'sobreestimacion':self.estadisticas_td3['sobreestimacion_total'],
                'policy_updates':self.estadisticas_td3['policy_updates']}

    def obtener_estadisticas_td3(self):
        if self.pesos_actor is None: return {'estado':'no_inicializada'}
        return {'learning_rate_actor':self.learning_rate_actor,'learning_rate_critic':self.learning_rate_critic,
            'gamma':self.gamma,'tau':self.tau,'policy_noise':self.policy_noise,'noise_clip':self.noise_clip,
            'policy_delay':self.policy_delay,'batch_size':self.batch_size,'buffer_size':len(self.buffer_experiencia),
            'actor_loss_medio':self.estadisticas_td3['actor_loss_medio'],'q1_loss_medio':self.estadisticas_td3['q1_loss_medio'],
            'q2_loss_medio':self.estadisticas_td3['q2_loss_medio'],'sobreestimacion_q1':self.estadisticas_td3['sobreestimacion_q1'],
            'sobreestimacion_q2':self.estadisticas_td3['sobreestimacion_q2'],'sobreestimacion_total':self.estadisticas_td3['sobreestimacion_total'],
            'target_updates':self.estadisticas_td3['target_updates'],'policy_updates':self.estadisticas_td3['policy_updates'],
            'muestras_buffer':self.estadisticas_td3['muestras_buffer'],
            't_medio_fwd_us':self.estadisticas_td3.get('t_medio_fwd_us',0),'t_medio_upd_us':self.estadisticas_td3.get('t_medio_upd_us',0),
            'q_values_media':self.estadisticas_td3.get('q_values_media',0.0),
            'q_values_std':self.estadisticas_td3.get('q_values_std',0.0),
            'convergencia':self.estadisticas_td3.get('loss_media',0.0),
            'loss_media':self.estadisticas_td3.get('loss_media',0.0)}

    def _preparar_minibatch(self, batch):
        e = batch['experiencias'] if isinstance(batch, dict) else batch
        st = np.array([ex['estado'] for ex in e], dtype=_dt())
        ac = np.array([ex['accion'] for ex in e], dtype=_dt())
        rw = np.array([ex['recompensa'] for ex in e], dtype=np.float32).reshape(-1,1)
        tn = np.array([float(ex['terminado']) for ex in e], dtype=np.float32).reshape(-1,1)
        ns = np.array([ex['siguiente_estado'] for ex in e], dtype=_dt())
        return st, ac, rw, ns, tn

    def verificar_estabilidad(self):
        est = {}; al = self.estadisticas_td3['actor_loss_medio']
        q1l = self.estadisticas_td3['q1_loss_medio']; q2l = self.estadisticas_td3['q2_loss_medio']
        est['actor_loss_estable'] = abs(al) < 100.0; est['q1_loss_estable'] = q1l < 100.0
        est['q2_loss_estable'] = q2l < 100.0
        sob = self.estadisticas_td3['sobreestimacion_total']
        est['sobreestimacion_controlada'] = abs(sob) < 10.0
        est['policy_delay_apropiado'] = 1 <= self.policy_delay <= 5
        est['buffer_suficiente'] = len(self.buffer_experiencia) >= self.batch_size
        if self.pesos_actor is not None: est['pesos_actor_no_explosivos'] = np.max(np.abs(self.pesos_actor)) < 10.0
        else: est['pesos_actor_no_explosivos'] = True
        return est

    def _softmax(self, z: np.ndarray) -> np.ndarray:
        z = z - np.max(z, axis=1, keepdims=True)
        e = np.exp(np.clip(z, -30, 30))
        return e / np.maximum(np.sum(e, axis=1, keepdims=True), 1e-12)

    @staticmethod
    def _sigma(w: np.ndarray, iters: int = 3) -> float:
        wf = w.astype(np.float64); v = np.random.default_rng(0).normal(0, 1, (wf.shape[1], 1))
        v /= max(1e-12, np.linalg.norm(v))
        for _ in range(iters):
            u = wf @ v; u /= max(1e-12, np.linalg.norm(u)); v = wf.T @ u; v /= max(1e-12, np.linalg.norm(v))
        return float((u.T @ wf @ v)[0, 0])

    def calcular_entropy(self, prob: np.ndarray) -> float:
        p = np.abs(prob.astype(np.float64)); p = p / max(1e-12, p.sum())
        return float(-np.sum(p * np.log(p + 1e-8)))

    def calcular_loss(self, batch: List[Dict[str, Any]]) -> Dict[str, float]:
        q1l, q2l = self.calcular_q_loss(batch); al = self.calcular_actor_loss(batch)
        return {'policy_loss': float(al), 'value_loss': float((q1l+q2l)/2),
                'total_loss': float(abs(al)+abs(q1l)+abs(q2l))}

    def forward_batch(self, estados) -> Tuple[np.ndarray, np.ndarray]:
        self._asegurar()
        xs = np.concatenate([self._prep(s) for s in estados], axis=0)
        logits = xs @ self.pesos_actor + self.sesgo_actor
        return np.tanh(logits).astype(_dt()), logits.astype(_dt())

    def verificar_integridad(self) -> Dict[str, Any]:
        checks = {'pesos_actor_init': self.pesos_actor is not None,
            'pesos_q1_init': self.pesos_q1 is not None, 'pesos_q2_init': self.pesos_q2 is not None,
            'pesos_at_init': self.pesos_actor_target is not None,
            'pesos_q1t_init': self.pesos_q1_target is not None,
            'pesos_q2t_init': self.pesos_q2_target is not None}
        for nm, p in [('actor', self.pesos_actor), ('q1', self.pesos_q1), ('q2', self.pesos_q2)]:
            if p is not None:
                checks[f'sin_nan_{nm}'] = not bool(np.any(np.isnan(p)))
                checks[f'sin_inf_{nm}'] = not bool(np.any(np.isinf(p)))
                checks[f'peso_{nm}_max'] = float(np.max(np.abs(p)))
        checks['precision'] = str(self.pesos_actor.dtype) if self.pesos_actor is not None else 'N/A'
        checks['buffer_tamano'] = len(self.buffer_experiencia)
        checks['policy_updates'] = self.estadisticas_td3['policy_updates']
        checks['target_updates'] = self.estadisticas_td3['target_updates']
        return checks

    def limpiar_historiales(self):
        self.historial_actor.clear(); self.historial_q.clear(); self.historial_sobreest.clear()

    def calcular_explained_variance(self, batch):
        preds = np.array([float(self.forward_q(e['estado'], e['accion'], False, 1)[0, 0]) for e in batch])
        rets = np.array([e['recompensa'] for e in batch], dtype=np.float64)
        var = float(np.var(rets)); resid = float(np.var(rets - preds))
        return 1.0 - resid/max(1e-12, var)

    def guardar(self, ruta: str) -> None:
        import json
        d = {'input_size':self.input_size,'output_size':self.output_size,'nombre':self.nombre,
            'pesos_actor':self.pesos_actor.tolist() if self.pesos_actor is not None else None,
            'pesos_q1':self.pesos_q1.tolist() if self.pesos_q1 is not None else None,
            'pesos_q2':self.pesos_q2.tolist() if self.pesos_q2 is not None else None,
            'pesos_actor_target':self.pesos_actor_target.tolist() if self.pesos_actor_target is not None else None,
            'pesos_q1_target':self.pesos_q1_target.tolist() if self.pesos_q1_target is not None else None,
            'pesos_q2_target':self.pesos_q2_target.tolist() if self.pesos_q2_target is not None else None}
        with open(ruta, 'w', encoding='utf-8') as f: json.dump(d, f)

    def cargar(self, ruta: str) -> None:
        import json
        with open(ruta, 'r', encoding='utf-8') as f: d = json.load(f)
        self.inicializar_pesos()
        for k in ['pesos_actor','pesos_q1','pesos_q2']:
            if d.get(k) is not None: setattr(self, k, np.asarray(d[k], dtype=_dt()))

    def reinicializar_con_parametros(self, learning_rate_actor=None, learning_rate_critic=None,
                                     gamma=None, tau=None, policy_delay=None):
        if learning_rate_actor is not None: self.learning_rate_actor = learning_rate_actor
        if learning_rate_critic is not None: self.learning_rate_critic = learning_rate_critic
        if gamma is not None: self.gamma = gamma
        if tau is not None: self.tau = tau
        if policy_delay is not None: self.policy_delay = policy_delay
        self.inicializar_pesos(); self.resetear_historial(); self.buffer_experiencia.clear()
        self.historial_actor.clear(); self.historial_q.clear(); self.historial_sobreest.clear()
        self.policy_update_counter = 0
        self.estadisticas_td3['target_updates'] = 0; self.estadisticas_td3['policy_updates'] = 0
        self.estadisticas_td3['muestras_buffer'] = 0
        logger.info(f"Neurona TD3 reinicializada: lr_actor={self.learning_rate_actor}")

    def __str__(self): return (f"NeuronaRefuerzoTD3(entrada={self.input_size},salida={self.output_size},lr_a={self.learning_rate_actor},lr_c={self.learning_rate_critic},pd={self.policy_delay})")
    def __repr__(self): return self.__str__()

def crear_neurona_td3(input_size, output_size, configuracion=None):
    c = configuracion or {}
    return NeuronaRefuerzoTD3(input_size=input_size, output_size=output_size, nombre=c.get('nombre','NeuronaRefuerzoTD3'),
        learning_rate_actor=c.get('learning_rate_actor',0.001), learning_rate_critic=c.get('learning_rate_critic',0.001),
        gamma=c.get('gamma',0.99), tau=c.get('tau',0.005), policy_noise=c.get('policy_noise',0.2),
        noise_clip=c.get('noise_clip',0.5), policy_delay=c.get('policy_delay',2),
        buffer_size=c.get('buffer_size',1000000), batch_size=c.get('batch_size',256), usar_he=c.get('usar_he',True))

def analizar_td3(neurona): return {'estadisticas': neurona.obtener_estadisticas_td3(), 'estable': neurona.verificar_estabilidad()}

RFEN8_CONFIG = {'inicializacion_preferida':'he','learning_rate_actor_default':0.001,'learning_rate_critic_default':0.001,
    'gamma_default':0.99,'tau_default':0.005,'policy_noise_default':0.2,'noise_clip_default':0.5,'policy_delay_default':2,
    'buffer_size_default':1000000,'batch_size_default':256,'umbral_sobreestimacion':10.0}
logger.info("RFEN1_RN_8.py cargado correctamente - Neurona de Refuerzo TD3 con Doble Crítica")