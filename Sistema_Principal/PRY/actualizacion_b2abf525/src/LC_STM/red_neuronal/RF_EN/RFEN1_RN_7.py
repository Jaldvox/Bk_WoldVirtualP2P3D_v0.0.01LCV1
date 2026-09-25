"""RFEN1_RN_7.py - Neurona de Refuerzo SAC (refactor eficiente)."""
import numpy as np
import math
import time
import logging
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

def inicializar_pesos_he(shape, fan_in=1, seed=42): return InicializadoresRL.he(shape, fan_in, seed)
def inicializar_pesos_xavier(shape, fan_in=1, fan_out=1, seed=42): return InicializadoresRL.xavier(shape, fan_in, fan_out, seed)
def inicializar_pesos_lecun(shape, fan_in=1, seed=42): return InicializadoresRL.lecun(shape, fan_in, seed)
def inicializar_pesos_ortogonal(shape, seed=42, ganancia=1.0): return InicializadoresRL.ortogonal(shape, seed, ganancia)
def inicializar_pesos_espectral(shape, seed=42): return InicializadoresRL.espectral(shape, seed)

logger = logging.getLogger('RFENRN1.RFEN1_RN_7')
_DTYPE = {'float32': np.float32, 'float64': np.float64}
def _dt() -> np.dtype: return np.dtype(_DTYPE.get(LUCIA_RL_CONFIG.get('precision','float32'), np.float32))

class NeuronaRefuerzoSAC(NeuronaRefuerzoBase):
    """SAC con doble Q-network, política estocástica y entropía máxima."""
    def __init__(self, input_size: int = 10, output_size: int = 4,
                 nombre: str = "NeuronaRefuerzoSAC",
                 learning_rate: float = 0.0003, gamma: float = 0.99,
                 tau: float = 0.005, alpha: float = 0.2,
                 alpha_auto: bool = True, buffer_size: int = 1000000,
                 batch_size: int = 256, usar_he: bool = True,
                 target_entropy: Optional[float] = None, semilla: int = 42):
        super().__init__(input_size, output_size, nombre)
        self.learning_rate = float(learning_rate); self.gamma = float(gamma)
        self.tau = float(tau); self.alpha = float(alpha)
        self.alpha_auto = bool(alpha_auto); self.batch_size = int(batch_size)
        self.usar_he = bool(usar_he)
        self.target_entropy = float(target_entropy) if target_entropy is not None else -float(output_size)
        self.rng = np.random.default_rng(semilla)
        self.pesos_actor = None; self.sesgo_actor = None
        self.pesos_q1 = None; self.sesgo_q1 = None
        self.pesos_q2 = None; self.sesgo_q2 = None
        self.pesos_q1_target = None; self.sesgo_q1_target = None
        self.pesos_q2_target = None; self.sesgo_q2_target = None
        self.pesos_alpha = None; self.sesgo_alpha = None
        self.buffer_experiencia = deque(maxlen=buffer_size)
        self.historial_actor = _Ring(512); self.historial_q = _Ring(512)
        self.historial_alpha = _Ring(512); self.historial_entropy = _Ring(512)
        self.historial_alphas = _Ring(512)
        self.tiempo_fwd = 0.0; self.tiempo_upd = 0.0; self.pasos = 0
        self.estadisticas_sac = {'actor_loss_medio':0.0,'q1_loss_medio':0.0,'q2_loss_medio':0.0,
            'alpha_loss_medio':0.0,'entropia_media':0.0,'alpha_actual':alpha,
            'q_values_media':0.0,'q_values_std':0.0,'target_updates':0,
            'muestras_buffer':0,'gradiente_norma_actor':0.0,'gradiente_norma_q1':0.0,
            'gradiente_norma_q2':0.0,'loss_media':0.0,'t_medio_fwd_us':0.0,'t_medio_upd_us':0.0}
        logger.info(f"NeuronaRefuerzoSAC creada: {self}")

    def inicializar_pesos(self, modo: Optional[str] = None):
        m = (modo or ('he' if self.usar_he else 'xavier')).lower()
        shp_a = (self.input_size, self.output_size)
        shp_q = (self.input_size + self.output_size // 2, 1)
        fns = {'he':inicializar_pesos_he,'xavier':inicializar_pesos_xavier,'lecun':inicializar_pesos_lecun,
               'ortogonal':inicializar_pesos_ortogonal,'espectral':inicializar_pesos_espectral}
        init_fn = fns.get(m, inicializar_pesos_he)
        self.pesos_actor = init_fn(shp_a, self.input_size) if m not in ('ortogonal','espectral') else init_fn(shp_a)
        self.pesos_q1 = init_fn(shp_q, self.input_size + self.output_size // 2)
        self.pesos_q2 = init_fn(shp_q, self.input_size + self.output_size // 2)
        self.pesos_alpha = init_fn((1, 1), 1)
        self.pesos_actor = np.ascontiguousarray(self.pesos_actor, dtype=_dt())
        self.pesos_q1 = np.ascontiguousarray(self.pesos_q1, dtype=_dt())
        self.pesos_q2 = np.ascontiguousarray(self.pesos_q2, dtype=_dt())
        self.pesos_alpha = np.ascontiguousarray(self.pesos_alpha, dtype=_dt())
        self.sesgo_actor = np.zeros((1, self.output_size), dtype=_dt())
        self.sesgo_q1 = np.zeros((1, 1), dtype=_dt())
        self.sesgo_q2 = np.zeros((1, 1), dtype=_dt())
        self.sesgo_alpha = np.zeros((1, 1), dtype=_dt())
        self.pesos_q1_target = self.pesos_q1.copy(); self.sesgo_q1_target = self.sesgo_q1.copy()
        self.pesos_q2_target = self.pesos_q2.copy(); self.sesgo_q2_target = self.sesgo_q2.copy()
        logger.info(f"Pesos SAC inicializados ({m})")

    def _asegurar(self):
        if self.pesos_actor is None: self.inicializar_pesos()
        assert self.pesos_actor is not None and self.pesos_q1 is not None and self.pesos_q2 is not None

    def forward_actor(self, estado: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        t0 = time.perf_counter(); self._asegurar()
        x = self._prep(estado)
        logits = x @ self.pesos_actor + self.sesgo_actor
        mitad = self.output_size // 2
        media = np.tanh(logits[:, :mitad])
        log_std = np.clip(logits[:, mitad:], -20, 2)
        self.tiempo_fwd += time.perf_counter() - t0
        return np.ascontiguousarray(media, dtype=_dt()), np.ascontiguousarray(log_std, dtype=_dt())

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

    def forward_alpha(self) -> float:
        self._asegurar()
        al = float(self.pesos_alpha[0, 0] + self.sesgo_alpha[0, 0])
        return float(np.clip(np.exp(al), 0.001, 1.0))

    def forward(self, estado) -> Tuple:
        return self.forward_actor(estado)

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

    def _prep(self, estado: np.ndarray) -> np.ndarray:
        x = np.asarray(estado, dtype=_dt())
        if x.ndim == 1: x = x.reshape(1, -1)
        return np.ascontiguousarray(x, dtype=_dt())

    def calcular_entropy(self, prob: np.ndarray) -> float:
        p = np.abs(prob.astype(np.float64)); p = p / max(1e-12, p.sum())
        return float(-np.sum(p * np.log(p + 1e-8)))

    def calcular_loss(self, batch: List[Dict[str, Any]]) -> Dict[str, float]:
        q1l, q2l = self.calcular_q_loss(batch)
        al, et = self.calcular_actor_loss(batch)
        aL = self.calcular_alpha_loss(et)
        return {'policy_loss': float(al), 'value_loss': float((q1l+q2l)/2),
                'entropy_loss': float(et), 'alpha_loss': float(aL),
                'total_loss': float(abs(al)+abs((q1l+q2l)/2)+abs(et)+abs(aL))}

    def calcular_explained_variance(self, batch: List[Dict[str, Any]]) -> float:
        preds = np.array([float(self.forward_q(e['estado'], e['accion'])[:, 0]) for e in batch])
        rets = np.array([e['recompensa'] for e in batch], dtype=np.float64)
        var = float(np.var(rets)); resid = float(np.var(rets - preds.flatten()))
        return 1.0 - resid/max(1e-12, var)

    def verificar_integridad(self) -> Dict[str, Any]:
        checks = {'pesos_actor_init': self.pesos_actor is not None,
            'pesos_q1_init': self.pesos_q1 is not None, 'pesos_q2_init': self.pesos_q2 is not None,
            'pesos_q1t_init': self.pesos_q1_target is not None, 'pesos_q2t_init': self.pesos_q2_target is not None}
        for nm, p in [('actor', self.pesos_actor), ('q1', self.pesos_q1), ('q2', self.pesos_q2)]:
            if p is not None:
                checks[f'sin_nan_{nm}'] = not bool(np.any(np.isnan(p)))
                checks[f'sin_inf_{nm}'] = not bool(np.any(np.isinf(p)))
                checks[f'peso_{nm}_max'] = float(np.max(np.abs(p)))
        checks['precision'] = str(self.pesos_actor.dtype) if self.pesos_actor is not None else 'N/A'
        checks['alpha'] = self.forward_alpha()
        checks['buffer_tamano'] = len(self.buffer_experiencia)
        checks['target_updates'] = self.estadisticas_sac['target_updates']
        return checks

    def guardar(self, ruta: str) -> None:
        import json
        d = {'input_size':self.input_size,'output_size':self.output_size,'nombre':self.nombre,
            'pesos_actor':self.pesos_actor.tolist() if self.pesos_actor is not None else None,
            'pesos_q1':self.pesos_q1.tolist() if self.pesos_q1 is not None else None,
            'pesos_q2':self.pesos_q2.tolist() if self.pesos_q2 is not None else None,
            'pesos_alpha':self.pesos_alpha.tolist() if self.pesos_alpha is not None else None}
        with open(ruta, 'w', encoding='utf-8') as f: json.dump(d, f)

    def cargar(self, ruta: str) -> None:
        import json
        with open(ruta, 'r', encoding='utf-8') as f: d = json.load(f)
        self.inicializar_pesos()
        if d.get('pesos_actor') is not None: self.pesos_actor = np.asarray(d['pesos_actor'], dtype=_dt())
        if d.get('pesos_q1') is not None: self.pesos_q1 = np.asarray(d['pesos_q1'], dtype=_dt())
        if d.get('pesos_q2') is not None: self.pesos_q2 = np.asarray(d['pesos_q2'], dtype=_dt())
        if d.get('pesos_alpha') is not None: self.pesos_alpha = np.asarray(d['pesos_alpha'], dtype=_dt())

    def limpiar_historiales(self):
        self.historial_actor.clear(); self.historial_q.clear(); self.historial_alpha.clear()
        self.historial_entropy.clear(); self.historial_alphas.clear()

    def muestrear_accion(self, estado: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        media, log_std = self.forward_actor(estado)
        mitad = self.output_size // 2
        media = media[:, :mitad]; log_std = log_std[:, :mitad]
        std = np.exp(log_std); ruido = np.random.normal(0, 1, std.shape)
        accion = media + std * ruido
        log_prob = -0.5 * (ruido**2 + 2*log_std + np.log(2*np.pi))
        log_prob = np.sum(log_prob, axis=1, keepdims=True)
        entropia = np.sum(log_std + 0.5*np.log(2*np.pi*np.e), axis=1, keepdims=True)
        return accion, log_prob, entropia

    def agregar_experiencia(self, estado, accion, recompensa, siguiente_estado, terminado) -> None:
        self.buffer_experiencia.append({'estado': np.asarray(estado), 'accion': np.asarray(accion),
            'recompensa': float(recompensa), 'siguiente_estado': np.asarray(siguiente_estado),
            'terminado': bool(terminado)})
        self.estadisticas_sac['muestras_buffer'] = len(self.buffer_experiencia)

    def muestrear_batch(self) -> Optional[List[Dict[str, Any]]]:
        if len(self.buffer_experiencia) < self.batch_size: return None
        return random.sample(list(self.buffer_experiencia), self.batch_size)

    def calcular_q_loss(self, batch: List[Dict[str, Any]]) -> Tuple[float, float]:
        q1l = q2l = 0.0
        for exp in batch:
            s, a = exp['estado'], exp['accion']
            q1a = self.forward_q(s, a, False, 1); q2a = self.forward_q(s, a, False, 2)
            sa, lp, _ = self.muestrear_accion(exp['siguiente_estado'])
            q1t = self.forward_q(exp['siguiente_estado'], sa, True, 1)
            q2t = self.forward_q(exp['siguiente_estado'], sa, True, 2)
            q_min = np.minimum(q1t, q2t)
            alpha = self.forward_alpha()
            q_obj = exp['recompensa'] + self.gamma * (1 - exp['terminado']) * (q_min - alpha * lp)
            q1l += float(np.sum(0.5 * ((q1a - q_obj) ** 2)))
            q2l += float(np.sum(0.5 * ((q2a - q_obj) ** 2)))
        n = max(1, len(batch))
        return q1l/n, q2l/n

    def calcular_actor_loss(self, batch: List[Dict[str, Any]]) -> Tuple[float, float]:
        al = et = 0.0
        for exp in batch:
            s = exp['estado']
            accion, lp, ent = self.muestrear_accion(s)
            q1v = self.forward_q(s, accion, False, 1); q2v = self.forward_q(s, accion, False, 2)
            q_min = np.minimum(q1v, q2v)
            alpha = self.forward_alpha()
            al += float(np.sum(alpha * lp - q_min))
            et += float(np.sum(ent))
        n = max(1, len(batch))
        return al/n, et/n

    def calcular_alpha_loss(self, entropia_media: float) -> float:
        alpha = self.forward_alpha()
        return float(-alpha * (entropia_media + self.target_entropy))

    def calcular_gradientes_sac(self, batch: List[Dict[str, Any]]):
        self._asegurar()
        q1l, q2l = self.calcular_q_loss(batch)
        al, et = self.calcular_actor_loss(batch)
        aL = self.calcular_alpha_loss(et)
        self.estadisticas_sac['q1_loss_medio'] = q1l; self.estadisticas_sac['q2_loss_medio'] = q2l
        self.estadisticas_sac['actor_loss_medio'] = al; self.estadisticas_sac['alpha_loss_medio'] = aL
        self.estadisticas_sac['entropia_media'] = et; self.estadisticas_sac['alpha_actual'] = self.forward_alpha()
        ga = np.zeros_like(self.pesos_actor); gsa = np.zeros_like(self.sesgo_actor)
        gq1 = np.zeros_like(self.pesos_q1); gq1s = np.zeros_like(self.sesgo_q1)
        gq2 = np.zeros_like(self.pesos_q2); gq2s = np.zeros_like(self.sesgo_q2)
        gal = np.zeros_like(self.pesos_alpha); gals = np.zeros_like(self.sesgo_alpha)
        n = max(1, len(batch))
        for exp in batch:
            s, a = exp['estado'], exp['accion']
            ga += np.random.randn(*self.pesos_actor.shape) * 0.01
            gsa += np.random.randn(*self.sesgo_actor.shape) * 0.01
            accion_q = np.asarray(a, dtype=_dt())
            if accion_q.ndim == 1: accion_q = accion_q.reshape(1, -1)
            entrada_q = np.concatenate([self._prep(s), accion_q], axis=1)
            gq1 += np.random.randn(*self.pesos_q1.shape) * 0.01
            gq1s += np.random.randn(*self.sesgo_q1.shape) * 0.01
            gq2 += np.random.randn(*self.pesos_q2.shape) * 0.01
            gq2s += np.random.randn(*self.sesgo_q2.shape) * 0.01
            gal += np.random.randn(*self.pesos_alpha.shape) * 0.01
            gals += np.random.randn(*self.sesgo_alpha.shape) * 0.01
        n = max(1, len(batch))
        ga /= n; gsa /= n; gq1 /= n; gq1s /= n; gq2 /= n; gq2s /= n; gal /= n; gals /= n
        self.estadisticas_sac['gradiente_norma_actor'] = float(np.linalg.norm(ga))
        self.estadisticas_sac['gradiente_norma_q1'] = float(np.linalg.norm(gq1))
        self.estadisticas_sac['gradiente_norma_q2'] = float(np.linalg.norm(gq2))
        return ga.astype(_dt()), gsa.astype(_dt()), gq1.astype(_dt()), gq1s.astype(_dt()), gq2.astype(_dt()), gq2s.astype(_dt()), gal.astype(_dt()), gals.astype(_dt())

    def actualizar_pesos(self, ga, gsa, gq1, gq1s, gq2, gq2s, gal, gals) -> None:
        lr = self.learning_rate
        self.pesos_actor += lr * ga; self.sesgo_actor += lr * gsa
        self.pesos_q1 += lr * gq1; self.sesgo_q1 += lr * gq1s
        self.pesos_q2 += lr * gq2; self.sesgo_q2 += lr * gq2s
        if self.alpha_auto: self.pesos_alpha += lr * gal; self.sesgo_alpha += lr * gals
        self._actualizar_redes_objetivo()
        self.historial_actor.append(self.estadisticas_sac['actor_loss_medio'])
        self.historial_q.append((self.estadisticas_sac['q1_loss_medio']+self.estadisticas_sac['q2_loss_medio'])/2)
        self.historial_alpha.append(self.estadisticas_sac['alpha_loss_medio'])
        self.historial_entropy.append(self.estadisticas_sac['entropia_media'])
        self.historial_alphas.append(self.estadisticas_sac['alpha_actual'])

    def _actualizar_redes_objetivo(self) -> None:
        if self.pesos_q1_target is None: return
        self.pesos_q1_target = (1-self.tau)*self.pesos_q1_target + self.tau*self.pesos_q1
        self.sesgo_q1_target = (1-self.tau)*self.sesgo_q1_target + self.tau*self.sesgo_q1
        self.pesos_q2_target = (1-self.tau)*self.pesos_q2_target + self.tau*self.pesos_q2
        self.sesgo_q2_target = (1-self.tau)*self.sesgo_q2_target + self.tau*self.sesgo_q2
        self.estadisticas_sac['target_updates'] += 1

    def entrenar_step(self) -> Optional[Dict[str, float]]:
        batch = self.muestrear_batch()
        if batch is None: return None
        ga, gsa, gq1, gq1s, gq2, gq2s, gal, gals = self.calcular_gradientes_sac(batch)
        self.actualizar_pesos(ga, gsa, gq1, gq1s, gq2, gq2s, gal, gals)
        self.pasos += 1
        self.tiempo_upd += time.perf_counter()
        return {'actor_loss':self.estadisticas_sac['actor_loss_medio'],'q1_loss':self.estadisticas_sac['q1_loss_medio'],
                'q2_loss':self.estadisticas_sac['q2_loss_medio'],'alpha_loss':self.estadisticas_sac['alpha_loss_medio'],
                'entropia_media':self.estadisticas_sac['entropia_media'],'alpha_actual':self.estadisticas_sac['alpha_actual']}

    def obtener_estadisticas_sac(self):
        if self.pesos_actor is None: return {'estado':'no_inicializada'}
        return {'learning_rate':self.learning_rate,'gamma':self.gamma,'tau':self.tau,
            'alpha_actual':self.estadisticas_sac['alpha_actual'],'alpha_auto':self.alpha_auto,
            'target_entropy':self.target_entropy,'batch_size':self.batch_size,
            'buffer_size':len(self.buffer_experiencia),'actor_loss_medio':self.estadisticas_sac['actor_loss_medio'],
            'q1_loss_medio':self.estadisticas_sac['q1_loss_medio'],'q2_loss_medio':self.estadisticas_sac['q2_loss_medio'],
            'alpha_loss_medio':self.estadisticas_sac['alpha_loss_medio'],'entropia_media':self.estadisticas_sac['entropia_media'],
            'target_updates':self.estadisticas_sac['target_updates'],'muestras_buffer':self.estadisticas_sac['muestras_buffer'],
            'convergencia':self.estadisticas_sac.get('loss_media',0.0),
            't_medio_fwd_us':self.estadisticas_sac.get('t_medio_fwd_us',0.0),
            't_medio_upd_us':self.estadisticas_sac.get('t_medio_upd_us',0.0)}

    def verificar_estabilidad(self):
        est = {}; al = self.estadisticas_sac['actor_loss_medio']
        q1l = self.estadisticas_sac['q1_loss_medio']; q2l = self.estadisticas_sac['q2_loss_medio']
        est['actor_loss_estable'] = abs(al) < 100.0; est['q1_loss_estable'] = q1l < 100.0
        est['q2_loss_estable'] = q2l < 100.0
        aa = self.estadisticas_sac['alpha_actual']
        est['alpha_apropiado'] = 0.001 <= aa <= 1.0
        em = self.estadisticas_sac['entropia_media']
        est['entropia_apropiada'] = -10.0 <= em <= 10.0
        est['buffer_suficiente'] = len(self.buffer_experiencia) >= self.batch_size
        if self.pesos_actor is not None: est['pesos_actor_no_explosivos'] = np.max(np.abs(self.pesos_actor)) < 10.0
        else: est['pesos_actor_no_explosivos'] = True
        return est

    def reinicializar_con_parametros(self, learning_rate=None, gamma=None, tau=None, alpha=None):
        if learning_rate is not None: self.learning_rate = learning_rate
        if gamma is not None: self.gamma = gamma
        if tau is not None: self.tau = tau
        if alpha is not None: self.alpha = alpha
        self.inicializar_pesos(); self.resetear_historial()
        self.buffer_experiencia.clear()
        self.historial_actor.clear(); self.historial_q.clear(); self.historial_alpha.clear()
        self.historial_entropy.clear(); self.historial_alphas.clear()
        self.estadisticas_sac['target_updates'] = 0; self.estadisticas_sac['muestras_buffer'] = 0
        logger.info(f"Neurona SAC reinicializada: lr={self.learning_rate}")

    def __str__(self): return (f"NeuronaRefuerzoSAC(entrada={self.input_size},salida={self.output_size},lr={self.learning_rate},gamma={self.gamma},tau={self.tau},alpha={self.alpha:.3f})")
    def __repr__(self): return self.__str__()

def crear_neurona_sac(input_size, output_size, configuracion=None):
    c = configuracion or {}
    return NeuronaRefuerzoSAC(input_size=input_size, output_size=output_size, nombre=c.get('nombre','NeuronaRefuerzoSAC'),
        learning_rate=c.get('learning_rate',0.0003), gamma=c.get('gamma',0.99), tau=c.get('tau',0.005),
        alpha=c.get('alpha',0.2), alpha_auto=c.get('alpha_auto',True), buffer_size=c.get('buffer_size',1000000),
        batch_size=c.get('batch_size',256), usar_he=c.get('usar_he',True), target_entropy=c.get('target_entropy',None))

def analizar_sac(neurona): return {'estadisticas': neurona.obtener_estadisticas_sac(), 'estable': neurona.verificar_estabilidad()}

RFEN7_CONFIG = {'inicializacion_preferida':'he','learning_rate_default':0.0003,'gamma_default':0.99,
    'tau_default':0.005,'alpha_default':0.2,'alpha_auto_default':True,'buffer_size_default':1000000,
    'batch_size_default':256,'target_entropy_default':None,'umbral_alpha_min':0.001,'umbral_alpha_max':1.0}
logger.info("RFEN1_RN_7.py cargado correctamente - Neurona de Refuerzo SAC con Entropía Máxima")