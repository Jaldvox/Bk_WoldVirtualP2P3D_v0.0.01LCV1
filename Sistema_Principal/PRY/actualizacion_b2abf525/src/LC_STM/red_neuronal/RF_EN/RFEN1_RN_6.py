"""RFEN1_RN_6.py - Neurona de Refuerzo PPO (refactor eficiente)."""
import numpy as np
import math
import time
import logging
import threading
import random
from typing import Tuple, Optional, Dict, Any, List
from collections import deque

LUCIA_RL_CONFIG = {'precision': 'float32', 'random_seed': 42, 'default_learning_rate': 0.0003}

class _Ring:
    def __init__(self, cap: int = 512):
        self.cap = cap; self._d: List[Any] = []
    def append(self, x: Any) -> None:
        self._d.append(x)
        if len(self._d) > self.cap: del self._d[0:len(self._d)-self.cap]
    def __len__(self) -> int: return len(self._d)
    def tail(self, n: int) -> List[Any]: return self._d[-n:]
    def clear(self) -> None: self._d.clear()

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

def inicializar_pesos_he(shape, fan_in: int = 1, seed: int = 42): return InicializadoresRL.he(shape, fan_in, seed)
def inicializar_pesos_xavier(shape, fan_in: int = 1, fan_out: int = 1, seed: int = 42): return InicializadoresRL.xavier(shape, fan_in, fan_out, seed)
def inicializar_pesos_lecun(shape, fan_in: int = 1, seed: int = 42): return InicializadoresRL.lecun(shape, fan_in, seed)
def inicializar_pesos_ortogonal(shape, seed: int = 42, ganancia: float = 1.0): return InicializadoresRL.ortogonal(shape, seed, ganancia)
def inicializar_pesos_espectral(shape, seed: int = 42): return InicializadoresRL.espectral(shape, seed)

logger = logging.getLogger('RFENRN1.RFEN1_RN_6')
_DTYPE = {'float32': np.float32, 'float64': np.float64}
def _dt() -> np.dtype: return np.dtype(_DTYPE.get(LUCIA_RL_CONFIG.get('precision','float32'), np.float32))

class NeuronaRefuerzoPPO(NeuronaRefuerzoBase):
    """PPO con clipping adaptativo, normalización de ventajas y buffers."""
    def __init__(self, input_size: int = 10, output_size: int = 4,
                 nombre: str = "NeuronaRefuerzoPPO",
                 learning_rate: float = 0.0003, gamma: float = 0.99,
                 lambda_gae: float = 0.95, clip_ratio: float = 0.2,
                 clip_ratio_adaptativo: bool = True, entropy_coef: float = 0.01,
                 value_coef: float = 0.5, max_grad_norm: float = 0.5,
                 usar_ortogonal: bool = True, epochs_ppo: int = 4,
                 semilla: int = 42, capacidad_historial: int = 512):
        super().__init__(input_size, output_size, nombre)
        self.learning_rate = float(learning_rate); self.gamma = float(gamma)
        self.lambda_gae = float(lambda_gae); self.clip_ratio = float(clip_ratio)
        self.clip_ratio_adaptativo = bool(clip_ratio_adaptativo)
        self.entropy_coef = float(entropy_coef); self.value_coef = float(value_coef)
        self.max_grad_norm = float(max_grad_norm); self.usar_ortogonal = bool(usar_ortogonal)
        self.epochs_ppo = int(epochs_ppo)
        self.rng = np.random.default_rng(semilla)
        self._loss_ema = 0.0; self._delta_ema = 0.0
        self.pesos_actor = None; self.sesgo_actor = None
        self.pesos_critic = None; self.sesgo_critic = None
        self.pesos_lock = threading.Lock()
        self.buffer_estados = deque(); self.buffer_acciones = deque()
        self.buffer_recompensas = deque(); self.buffer_valores = deque()
        self.buffer_log_probs = deque(); self.buffer_advantages = deque()
        self.buffer_returns = deque()
        self.historial_policy = _Ring(capacidad_historial)
        self.historial_value = _Ring(capacidad_historial)
        self.historial_entropy = _Ring(capacidad_historial)
        self.historial_clip = _Ring(capacidad_historial)
        self.historial_kl = _Ring(capacidad_historial)
        self.historial_ratio = _Ring(capacidad_historial)
        self.tiempo_fwd = 0.0; self.tiempo_upd = 0.0; self.pasos = 0
        self.estadisticas_ppo = {'policy_loss_medio':0.0,'value_loss_medio':0.0,
            'entropy_loss_medio':0.0,'total_loss_medio':0.0,'clip_ratio_actual':clip_ratio,
            'kl_divergencia':0.0,'explained_variance':0.0,'policy_ratio_medio':0.0,
            'advantage_norma':0.0,'gradiente_norma':0.0,'epochs_completados':0,
            'loss_media':0.0,'q_max':0.0,'q_min':0.0,'q_media':0.0,'q_std':0.0,
            't_medio_fwd_us':0.0,'t_medio_upd_us':0.0,'buffer_len':0}
        logger.info(f"NeuronaRefuerzoPPO creada: {self}")

    def inicializar_pesos(self, modo: Optional[str] = None):
        m = (modo or ('ortogonal' if self.usar_ortogonal else 'xavier')).lower()
        fns = {'he':inicializar_pesos_he,'xavier':inicializar_pesos_xavier,'lecun':inicializar_pesos_lecun,
               'ortogonal':inicializar_pesos_ortogonal,'espectral':inicializar_pesos_espectral}
        if m in fns:
            self.pesos_actor = fns[m]((self.input_size, self.output_size))
            self.pesos_critic = fns[m]((self.input_size, 1))
        else:
            self.pesos_actor = np.zeros((self.input_size, self.output_size), dtype=_dt())
            self.pesos_critic = np.zeros((self.input_size, 1), dtype=_dt())
        self.pesos_actor = np.ascontiguousarray(self.pesos_actor, dtype=_dt())
        self.pesos_critic = np.ascontiguousarray(self.pesos_critic, dtype=_dt())
        self.sesgo_actor = np.zeros((1, self.output_size), dtype=_dt())
        self.sesgo_critic = np.zeros((1, 1), dtype=_dt())
        logger.info(f"Pesos PPO inicializados ({m})")

    def _asegurar(self):
        if self.pesos_actor is None: self.inicializar_pesos()
        assert self.pesos_actor is not None and self.pesos_critic is not None

    def _softmax(self, z: np.ndarray) -> np.ndarray:
        z = z - np.max(z, axis=1, keepdims=True)
        e = np.exp(np.clip(z, -30, 30))
        return e / np.maximum(np.sum(e, axis=1, keepdims=True), 1e-12)

    def _prep(self, estado: np.ndarray) -> np.ndarray:
        x = np.asarray(estado, dtype=_dt())
        if x.ndim == 1: x = x.reshape(1, -1)
        return np.ascontiguousarray(x, dtype=_dt())

    def _sigma(self, w: np.ndarray, iters: int = 3) -> float:
        wf = w.astype(np.float64); v = np.random.default_rng(0).normal(0, 1, (wf.shape[1], 1))
        v /= max(1e-12, np.linalg.norm(v))
        for _ in range(iters):
            u = wf @ v; u /= max(1e-12, np.linalg.norm(u)); v = wf.T @ u; v /= max(1e-12, np.linalg.norm(v))
        return float((u.T @ wf @ v)[0, 0])

    def forward_actor(self, estado: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        t0 = time.perf_counter(); self._asegurar()
        w = self.pesos_actor; b = self.sesgo_actor
        wa = (w / max(1e-12, self._sigma(w))) if self.usar_ortogonal else w
        x = self._prep(estado); logits = x @ wa + b; probs = self._softmax(logits).astype(_dt(), copy=False)
        log_p = np.log(probs + 1e-8)
        self.tiempo_fwd += time.perf_counter() - t0
        return np.ascontiguousarray(probs, dtype=_dt()), np.ascontiguousarray(log_p, dtype=_dt())

    def forward_critic(self, estado: np.ndarray) -> np.ndarray:
        t0 = time.perf_counter(); self._asegurar()
        w = self.pesos_critic; b = self.sesgo_critic
        wc = (w / max(1e-12, self._sigma(w))) if self.usar_ortogonal else w
        x = self._prep(estado); v = (x @ wc + b).astype(_dt(), copy=False)
        self.tiempo_fwd += time.perf_counter() - t0
        return np.ascontiguousarray(v, dtype=_dt())

    def forward(self, estado: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        prob, lp = self.forward_actor(estado); val = self.forward_critic(estado)
        return prob, lp, val

    def seleccionar_accion(self, estado: np.ndarray) -> Tuple[int, float, float, float]:
        prob, lp, val = self.forward(estado)
        accion = int(np.random.choice(self.output_size, p=prob[0]))
        return accion, float(prob[0, accion]), float(lp[0, accion]), float(val[0, 0])

    def agregar_experiencia(self, estado, accion, recompensa, valor, log_prob) -> None:
        self.buffer_estados.append(np.asarray(estado)); self.buffer_acciones.append(accion)
        self.buffer_recompensas.append(float(recompensa)); self.buffer_valores.append(float(valor))
        self.buffer_log_probs.append(float(log_prob))

    def calcular_gae_advantages(self, recompensas, valores):
        advantages = []; advantage = 0.0
        for t in reversed(range(len(recompensas))):
            nv = 0.0 if t == len(recompensas)-1 else valores[t+1]
            delta = recompensas[t] + self.gamma * nv - valores[t]
            advantage = delta + self.gamma * self.lambda_gae * advantage; advantages.insert(0, advantage)
        adv = np.array(advantages, dtype=np.float32)
        if len(adv) > 1: adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        self.estadisticas_ppo['advantage_norma'] = float(np.linalg.norm(adv))
        self.estadisticas_ppo['advantage_media'] = float(np.mean(advantages))
        return adv.tolist()

    def calcular_returns(self, recompensas, valores):
        returns = []; acum = 0.0
        for t in reversed(range(len(recompensas))):
            nv = 0.0 if t == len(recompensas)-1 else valores[t+1]
            acum = recompensas[t] + self.gamma * acum; returns.insert(0, acum)
        return returns

    def calcular_policy_loss(self, estados, acciones, log_probs_old, advantages):
        plt = klt = 0.0; ratios = []
        for estado, accion, lp_old, adv in zip(estados, acciones, log_probs_old, advantages):
            _, lp = self.forward_actor(estado); lp_act = lp[0, accion]
            ratio = float(np.exp(lp_act - lp_old)); ratios.append(ratio)
            cr = np.clip(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
            plt += float(-np.minimum(ratio * adv, cr * adv))
            klt += float(lp_old - lp_act)
        n = max(1, len(estados)); pm = plt/n; km = klt/n
        self.estadisticas_ppo['policy_ratio_medio'] = float(np.mean(ratios))
        self.estadisticas_ppo['kl_divergencia'] = km
        return pm, km

    def calcular_value_loss(self, estados, returns):
        vlt = 0.0
        for estado, ret in zip(estados, returns):
            v = float(self.forward_critic(estado)[0, 0]); vlt += 0.5*(v - ret)**2
        return vlt / max(1, len(estados))

    def calcular_entropy_loss(self, estados) -> float:
        elt = 0.0
        for estado in estados:
            prob, _ = self.forward_actor(estado)
            elt += float(np.mean(-np.sum(prob * np.log(prob + 1e-8), axis=1)))
        return elt / max(1, len(estados))

    def forward_batch(self, estados) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        self._asegurar()
        xs = np.concatenate([self._prep(s) for s in estados], axis=0)
        wa = (self.pesos_actor / max(1e-12, self._sigma(self.pesos_actor))) if self.usar_ortogonal else self.pesos_actor
        wc = (self.pesos_critic / max(1e-12, self._sigma(self.pesos_critic))) if self.usar_ortogonal else self.pesos_critic
        logits = xs @ wa + self.sesgo_actor; probs = self._softmax(logits).astype(_dt(), copy=False)
        vals = (xs @ wc + self.sesgo_critic).astype(_dt(), copy=False)
        lp = np.log(probs + 1e-8)
        return np.ascontiguousarray(probs, dtype=_dt()), np.ascontiguousarray(lp, dtype=_dt()), vals

    def calcular_loss(self, estados, acciones, advantages, returns) -> Dict[str, float]:
        pl, kl = self.calcular_policy_loss(estados, acciones, [0.0]*len(estados), advantages)
        vl = self.calcular_value_loss(estados, returns)
        el = self.calcular_entropy_loss(estados)
        return {'policy_loss': float(pl), 'value_loss': float(vl), 'entropy_loss': float(el), 'total_loss': float(pl + self.value_coef*vl - self.entropy_coef*el)}

    def limpiar_buffers(self) -> None:
        for buf in [self.buffer_estados, self.buffer_acciones, self.buffer_recompensas, self.buffer_valores, self.buffer_log_probs]: buf.clear()

    def calcular_explained_variance(self, estados, returns) -> float:
        preds = np.array([float(self.forward_critic(s)[0, 0]) for s in estados])
        rets = np.array(returns, dtype=np.float64)
        var = float(np.var(rets)); resid = float(np.var(rets - preds))
        return 1.0 - resid/max(1e-12, var)

    def obtener_ratio_politica(self, estados, acciones) -> List[float]:
        ratios = []
        for estado, accion in zip(estados, acciones):
            _, lp = self.forward_actor(estado); ratios.append(float(np.exp(lp[0, accion])))
        return ratios

    def _calcular_convergencia_ppo(self):
        hp = self.historial_policy; hv = self.historial_value
        if len(hp) < 10 or len(hv) < 10: return
        vp = float(np.var(np.asarray(hp.tail(10), dtype=np.float64)))
        vc = float(np.var(np.asarray(hv.tail(10), dtype=np.float64)))
        self.estadisticas_ppo['convergencia_policy'] = 1.0/(1.0+vp)
        self.estadisticas_ppo['convergencia_value'] = 1.0/(1.0+vc)

    def calcular_gradientes_ppo(self, estados, acciones, log_probs_old, advantages, returns):
        self._asegurar()
        policy_loss, kl = self.calcular_policy_loss(estados, acciones, log_probs_old, advantages)
        value_loss = self.calcular_value_loss(estados, returns)
        entropy_loss = self.calcular_entropy_loss(estados)
        total_loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy_loss
        self.estadisticas_ppo['policy_loss_medio'] = policy_loss
        self.estadisticas_ppo['value_loss_medio'] = value_loss
        self.estadisticas_ppo['entropy_loss_medio'] = entropy_loss
        self.estadisticas_ppo['total_loss_medio'] = total_loss
        gap = np.zeros_like(self.pesos_actor); gas = np.zeros_like(self.sesgo_actor)
        gcp = np.zeros_like(self.pesos_critic); gcs = np.zeros_like(self.sesgo_critic)
        n = max(1, len(estados))
        for estado, accion, lp_old, adv in zip(estados, acciones, log_probs_old, advantages):
            _, lp = self.forward_actor(estado); lp_act = lp[0, accion]
            glp = np.zeros(self.output_size); glp[accion] = 1.0/(lp[0, accion]+1e-8)
            ratio = np.exp(lp_act - lp_old); cr = np.clip(ratio, 1-self.clip_ratio, 1+self.clip_ratio)
            grad = -np.minimum(ratio*adv, cr*adv)
            gap += np.outer(estado[0], glp)*grad; gas += glp*grad
        for estado, ret in zip(estados, returns):
            val = float(self.forward_critic(estado)[0, 0]); err = ret - val
            gcp += np.outer(estado[0], np.array([err])); gcs += np.array([[err]])
        gap/=n; gas/=n; gcp/=n; gcs/=n
        if self.max_grad_norm > 0:
            gnorm = float(np.linalg.norm(np.concatenate([gap.flatten(), gcp.flatten()])))
            if gnorm > self.max_grad_norm:
                cf = self.max_grad_norm/(gnorm+1e-12); gap *= cf; gas *= cf; gcp *= cf; gcs *= cf
        self.estadisticas_ppo['gradiente_norma'] = float(np.linalg.norm(np.concatenate([gap.flatten(), gcp.flatten()])))
        self.estadisticas_ppo['epochs_completados'] += 1
        return gap.astype(_dt()), gas.astype(_dt()), gcp.astype(_dt()), gcs.astype(_dt())

    def actualizar_pesos(self, gap, gas, gcp, gcs) -> None:
        self.pesos_actor += self.learning_rate * gap; self.sesgo_actor += self.learning_rate * gas
        self.pesos_critic += self.learning_rate * gcp; self.sesgo_critic += self.learning_rate * gcs
        if self.clip_ratio_adaptativo: self._adaptar_clip_ratio()
        self.historial_policy.append(self.estadisticas_ppo['policy_loss_medio'])
        self.historial_value.append(self.estadisticas_ppo['value_loss_medio'])
        self.historial_entropy.append(self.estadisticas_ppo['entropy_loss_medio'])
        self.historial_clip.append(self.clip_ratio); self.historial_kl.append(self.estadisticas_ppo['kl_divergencia'])
        self.historial_ratio.append(self.estadisticas_ppo['policy_ratio_medio'])

    def _adaptar_clip_ratio(self) -> None:
        kl = self.estadisticas_ppo['kl_divergencia']
        if kl > 0.02: self.clip_ratio *= 0.9
        elif kl < 0.01: self.clip_ratio *= 1.1
        self.clip_ratio = float(np.clip(self.clip_ratio, 0.1, 0.3))
        self.estadisticas_ppo['clip_ratio_actual'] = self.clip_ratio

    def entrenar_ppo(self) -> Dict[str, float]:
        if len(self.buffer_estados) < self.epochs_ppo: return {'error': 'buffer_insuficiente'}
        estados = list(self.buffer_estados); acciones = list(self.buffer_acciones)
        recompensas = list(self.buffer_recompensas); valores = list(self.buffer_valores)
        log_probs_old = list(self.buffer_log_probs)
        advantages = self.calcular_gae_advantages(recompensas, valores)
        returns = self.calcular_returns(recompensas, valores)
        for epoch in range(self.epochs_ppo):
            gap, gas, gcp, gcs = self.calcular_gradientes_ppo(estados, acciones, log_probs_old, advantages, returns)
            self.actualizar_pesos(gap, gas, gcp, gcs)
        for buf in [self.buffer_estados, self.buffer_acciones, self.buffer_recompensas, self.buffer_valores, self.buffer_log_probs]: buf.clear()
        return {'policy_loss':self.estadisticas_ppo['policy_loss_medio'],'value_loss':self.estadisticas_ppo['value_loss_medio'],
                'entropy_loss':self.estadisticas_ppo['entropy_loss_medio'],'total_loss':self.estadisticas_ppo['total_loss_medio'],
                'kl_divergencia':self.estadisticas_ppo['kl_divergencia'],'clip_ratio':self.clip_ratio,'epochs':self.epochs_ppo}

    def obtener_estadisticas_ppo(self):
        if self.pesos_actor is None: return {'estado':'no_inicializada'}
        return {'learning_rate':self.learning_rate,'gamma':self.gamma,'lambda_gae':self.lambda_gae,
            'clip_ratio':self.clip_ratio,'clip_ratio_adaptativo':self.clip_ratio_adaptativo,
            'entropy_coef':self.entropy_coef,'value_coef':self.value_coef,'max_grad_norm':self.max_grad_norm,
            'epochs_ppo':self.epochs_ppo,'policy_loss_medio':self.estadisticas_ppo['policy_loss_medio'],
            'value_loss_medio':self.estadisticas_ppo['value_loss_medio'],'entropy_loss_medio':self.estadisticas_ppo['entropy_loss_medio'],
            'total_loss_medio':self.estadisticas_ppo['total_loss_medio'],'kl_divergencia':self.estadisticas_ppo['kl_divergencia'],
            'policy_ratio_medio':self.estadisticas_ppo['policy_ratio_medio'],'advantage_norma':self.estadisticas_ppo['advantage_norma'],
            'gradiente_norma':self.estadisticas_ppo['gradiente_norma'],'epochs_completados':self.estadisticas_ppo['epochs_completados'],
            'convergencia_policy':self.estadisticas_ppo.get('convergencia_policy',0.0),
            'convergencia_value':self.estadisticas_ppo.get('convergencia_value',0.0),
            'explained_variance':self.estadisticas_ppo.get('explained_variance',0.0),
            't_medio_fwd_us':self.estadisticas_ppo.get('t_medio_fwd_us',0.0),
            't_medio_upd_us':self.estadisticas_ppo.get('t_medio_upd_us',0.0)}

    def verificar_estabilidad(self):
        est = {}; kl = self.estadisticas_ppo['kl_divergencia']
        est['kl_divergenia_ok'] = 0.01 <= kl <= 0.02; est['clip_ratio_ok'] = 0.1 <= self.clip_ratio <= 0.3
        pl = self.estadisticas_ppo['policy_loss_medio']; vl = self.estadisticas_ppo['value_loss_medio']
        est['policy_loss_estable'] = abs(pl) < 10.0; est['value_loss_estable'] = vl < 10.0
        gn = self.estadisticas_ppo['gradiente_norma']
        est['gradientes_no_explosivos'] = gn < 10.0; est['gradientes_no_desaparecen'] = gn > 1e-8
        if self.pesos_actor is not None and self.pesos_critic is not None:
            est['pesos_actor_no_explosivos'] = np.max(np.abs(self.pesos_actor)) < 5.0
            est['pesos_critic_no_explosivos'] = np.max(np.abs(self.pesos_critic)) < 5.0
        else: est['pesos_actor_no_explosivos'] = est['pesos_critic_no_explosivos'] = True
        return est

    def verificar_integridad(self) -> Dict[str, Any]:
        checks = {'pesos_inicializados': self.pesos_actor is not None,
            'pesos_critic_inicializados': self.pesos_critic is not None,
            'locks_funcionales': hasattr(self, 'pesos_lock')}
        if self.pesos_actor is not None and self.pesos_critic is not None:
            checks['sin_nan_actor'] = not bool(np.any(np.isnan(self.pesos_actor)))
            checks['sin_nan_critic'] = not bool(np.any(np.isnan(self.pesos_critic)))
            checks['sin_inf_actor'] = not bool(np.any(np.isinf(self.pesos_actor)))
            checks['sin_inf_critic'] = not bool(np.any(np.isinf(self.pesos_critic)))
            checks['precision'] = str(self.pesos_actor.dtype)
            checks['peso_actor_max'] = float(np.max(np.abs(self.pesos_actor)))
            checks['peso_critic_max'] = float(np.max(np.abs(self.pesos_critic)))
        checks['buffer_tamano'] = len(self.buffer_estados)
        checks['epochs_pendientes'] = self.epochs_ppo
        return checks

    def guardar(self, ruta: str) -> None:
        import json
        with open(ruta, 'w', encoding='utf-8') as f:
            json.dump({'input_size':self.input_size,'output_size':self.output_size,'nombre':self.nombre,
                'pesos_actor':self.pesos_actor.tolist() if self.pesos_actor is not None else None,
                'pesos_critic':self.pesos_critic.tolist() if self.pesos_critic is not None else None,
                'sesgo_actor':self.sesgo_actor.tolist() if self.sesgo_actor is not None else None,
                'sesgo_critic':self.sesgo_critic.tolist() if self.sesgo_critic is not None else None}, f)

    def cargar(self, ruta: str) -> None:
        import json
        with open(ruta, 'r', encoding='utf-8') as f: d = json.load(f)
        self.inicializar_pesos()
        if d.get('pesos_actor') is not None: self.pesos_actor = np.asarray(d['pesos_actor'], dtype=_dt())
        if d.get('pesos_critic') is not None: self.pesos_critic = np.asarray(d['pesos_critic'], dtype=_dt())
        if d.get('sesgo_actor') is not None: self.sesgo_actor = np.asarray(d['sesgo_actor'], dtype=_dt())
        if d.get('sesgo_critic') is not None: self.sesgo_critic = np.asarray(d['sesgo_critic'], dtype=_dt())

    def limpiar_historiales(self):
        self.historial_policy.clear(); self.historial_value.clear(); self.historial_entropy.clear()
        self.historial_clip.clear(); self.historial_kl.clear(); self.historial_ratio.clear()

    def reinicializar_con_parametros(self, learning_rate=None, gamma=None, clip_ratio=None, entropy_coef=None):
        if learning_rate is not None: self.learning_rate = learning_rate
        if gamma is not None: self.gamma = gamma
        if clip_ratio is not None: self.clip_ratio = clip_ratio
        if entropy_coef is not None: self.entropy_coef = entropy_coef
        self.inicializar_pesos(); self.resetear_historial()
        for buf in [self.buffer_estados, self.buffer_acciones, self.buffer_recompensas, self.buffer_valores, self.buffer_log_probs]: buf.clear()
        self.historial_policy.clear(); self.historial_value.clear(); self.historial_entropy.clear()
        self.historial_clip.clear(); self.historial_kl.clear(); self.historial_ratio.clear()
        self.estadisticas_ppo['epochs_completados'] = 0
        logger.info(f"Neurona PPO reinicializada: lr={self.learning_rate}, clip={self.clip_ratio}")

    def __str__(self): return (f"NeuronaRefuerzoPPO(entrada={self.input_size},salida={self.output_size},lr={self.learning_rate},gamma={self.gamma},clip={self.clip_ratio:.3f})")
    def __repr__(self): return self.__str__()

def crear_neurona_ppo(input_size, output_size, configuracion=None):
    c = configuracion or {}
    return NeuronaRefuerzoPPO(input_size=input_size, output_size=output_size, nombre=c.get('nombre','NeuronaRefuerzoPPO'),
        learning_rate=c.get('learning_rate',0.0003), gamma=c.get('gamma',0.99), lambda_gae=c.get('lambda_gae',0.95),
        clip_ratio=c.get('clip_ratio',0.2), clip_ratio_adaptativo=c.get('clip_ratio_adaptativo',True),
        entropy_coef=c.get('entropy_coef',0.01), value_coef=c.get('value_coef',0.5),
        max_grad_norm=c.get('max_grad_norm',0.5), usar_ortogonal=c.get('usar_ortogonal',True), epochs_ppo=c.get('epochs_ppo',4))

def analizar_ppo(neurona): return {'estadisticas': neurona.obtener_estadisticas_ppo(), 'estable': neurona.verificar_estabilidad()}

RFEN6_CONFIG = {'inicializacion_preferida':'ortogonal','learning_rate_default':0.0003,'gamma_default':0.99,
    'lambda_gae_default':0.95,'clip_ratio_default':0.2,'clip_ratio_adaptativo_default':True,
    'entropy_coef_default':0.01,'value_coef_default':0.5,'max_grad_norm_default':0.5,'epochs_ppo_default':4,
    'umbral_kl_divergencia_min':0.01,'umbral_kl_divergencia_max':0.02}
logger.info("RFEN1_RN_6.py cargado correctamente - Neurona de Refuerzo PPO Adaptativo")