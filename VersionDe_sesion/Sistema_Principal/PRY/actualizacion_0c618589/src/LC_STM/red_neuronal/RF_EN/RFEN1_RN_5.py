"""RFEN1_RN_5.py - Neurona de Refuerzo A3C (refactor eficiente)."""
import numpy as np
import math
import time
import logging
import threading
import random
from typing import Tuple, Optional, Dict, Any, List

LUCIA_RL_CONFIG = {'precision': 'float32', 'random_seed': 42, 'default_learning_rate': 0.002}

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

logger = logging.getLogger('RFENRN1.RFEN1_RN_5')
_DTYPE = {'float32': np.float32, 'float64': np.float64}
def _dt() -> np.dtype: return np.dtype(_DTYPE.get(LUCIA_RL_CONFIG.get('precision','float32'), np.float32))

class NeuronaRefuerzoA3C(NeuronaRefuerzoBase):
    """A3C con workers asíncronos, sincronización y buffer de experiencia."""
    def __init__(self, input_size: int = 10, output_size: int = 4,
                 nombre: str = "NeuronaRefuerzoA3C",
                 learning_rate_actor: float = 0.001, learning_rate_critic: float = 0.002,
                 gamma: float = 0.99, lambda_gae: float = 0.95, num_workers: int = 4,
                 update_frequency: int = 20, usar_ortogonal: bool = True,
                 entropy_coef: float = 0.01, grad_clip: float = 5.0,
                 epsilon: float = 1.0, epsilon_min: float = 0.01, epsilon_decay: float = 0.995,
                 semilla: int = 42, capacidad_historial: int = 512):
        super().__init__(input_size, output_size, nombre)
        self.learning_rate_actor = float(learning_rate_actor)
        self.learning_rate_critic = float(learning_rate_critic)
        self.gamma = float(gamma); self.lambda_gae = float(lambda_gae)
        self.num_workers = int(num_workers); self.update_frequency = int(update_frequency)
        self.usar_ortogonal = bool(usar_ortogonal); self.entropy_coef = float(entropy_coef)
        self.grad_clip = float(grad_clip)
        self.epsilon = float(epsilon); self.epsilon_min = float(epsilon_min); self.epsilon_decay = float(epsilon_decay)
        self.rng = np.random.default_rng(semilla)
        self._loss_ema = 0.0; self._delta_ema = 0.0
        self.pesos_actor_global = None; self.sesgo_actor_global = None
        self.pesos_critic_global = None; self.sesgo_critic_global = None
        self.pesos_actor_locales = {}; self.sesgo_actor_locales = {}
        self.pesos_critic_locales = {}; self.sesgo_critic_locales = {}
        self.pesos_lock = threading.Lock(); self.buffer = []
        self.historial_qvalues = _Ring(capacidad_historial)
        self.historial_epsilon = _Ring(capacidad_historial)
        self.historial_advantages = _Ring(capacidad_historial * 4)
        self.historial_policy = _Ring(capacidad_historial)
        self.historial_sincronizaciones = []
        self.tiempo_fwd = 0.0; self.tiempo_upd = 0.0
        self.pasos = 0
        self.estadisticas_a3c = {'convergencia_actor_global':0.0,'convergencia_critic_global':0.0,
            'estabilidad_actor_global':0.0,'estabilidad_critic_global':0.0,'sincronizaciones':0,
            'updates_globales':0,'advantage_media':0.0,'advantage_std':0.0,'entropia_media':0.0,
            'workers_activos':0,'loss_media':0.0,'q_max':0.0,'q_min':0.0,'q_media':0.0,'q_std':0.0,
            'exploracion_rate':0,'exploitacion_rate':0,'t_medio_fwd_us':0.0,'t_medio_upd_us':0.0,'buffer_len':0,
            'valor_error_medio':0.0,'policy_loss_medio':0.0}
        self.historial_actor_loss_global = []; self.historial_critic_loss_global = []
        self.historial_advantages_global = []; self.historial_entropias_global = []
        logger.info(f"NeuronaRefuerzoA3C creada: {self}")

    def inicializar_pesos(self, modo: Optional[str] = None):
        m = (modo or ('ortogonal' if self.usar_ortogonal else 'xavier')).lower()
        shp_a = (self.input_size, self.output_size); shp_c = (self.input_size, 1)
        fns = {'he':inicializar_pesos_he,'xavier':inicializar_pesos_xavier,'lecun':inicializar_pesos_lecun,
               'ortogonal':inicializar_pesos_ortogonal,'espectral':inicializar_pesos_espectral}
        if m in fns:
            self.pesos_actor_global = fns[m](shp_a, self.input_size, self.output_size)
            self.pesos_critic_global = fns[m](shp_c, self.input_size, 1)
        else:
            self.pesos_actor_global = np.zeros(shp_a, dtype=_dt())
            self.pesos_critic_global = np.zeros(shp_c, dtype=_dt())
        self.pesos_actor_global = np.ascontiguousarray(self.pesos_actor_global, dtype=_dt())
        self.pesos_critic_global = np.ascontiguousarray(self.pesos_critic_global, dtype=_dt())
        self.sesgo_actor_global = np.zeros((1, self.output_size), dtype=_dt())
        self.sesgo_critic_global = np.zeros((1, 1), dtype=_dt())
        for wid in range(self.num_workers):
            self.pesos_actor_locales[wid] = self.pesos_actor_global.copy()
            self.sesgo_actor_locales[wid] = self.sesgo_actor_global.copy()
            self.pesos_critic_locales[wid] = self.pesos_critic_global.copy()
            self.sesgo_critic_locales[wid] = self.sesgo_critic_global.copy()
        logger.info(f"Pesos A3C inicializados ({m}) para {self.num_workers} workers")

    def _asegurar(self):
        if self.pesos_actor_global is None: self.inicializar_pesos()
        assert self.pesos_actor_global is not None

    @staticmethod
    def _softmax(z: np.ndarray) -> np.ndarray:
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

    def _norma_espectral(self, w: np.ndarray, sigma: float) -> np.ndarray:
        if not self.usar_ortogonal: return w
        return (w.astype(np.float64) / max(1e-12, sigma)).astype(_dt())

    def _prep(self, estado: np.ndarray) -> np.ndarray:
        x = np.asarray(estado, dtype=_dt())
        if x.ndim == 1: x = x.reshape(1, -1)
        if x.shape[1] != self.input_size: raise ValueError(f"dim {x.shape[1]} != input {self.input_size}")
        return np.ascontiguousarray(x, dtype=_dt())

    def calcular_entropy(self, prob: np.ndarray) -> float:
        p = prob.astype(np.float64); p = p / max(1e-12, p.sum())
        return float(-np.sum(p * np.log(p + 1e-8)))

    def forward_actor(self, estado, worker_id: int = 0):
        t0 = time.perf_counter(); self._asegurar()
        w = self.pesos_actor_locales.get(worker_id, self.pesos_actor_global)
        b = self.sesgo_actor_locales.get(worker_id, self.sesgo_actor_global)
        x = self._prep(estado) if np.asarray(estado).ndim >= 2 else self._prep(estado.reshape(1,-1))
        wa = self._norma_espectral(w, self._sigma(w)) if self.usar_ortogonal else w
        probs = self._softmax(np.dot(x, wa) + b).astype(_dt(), copy=False)
        self.tiempo_fwd += time.perf_counter() - t0
        return np.ascontiguousarray(probs, dtype=_dt())

    def forward_critic(self, estado, worker_id: int = 0):
        t0 = time.perf_counter(); self._asegurar()
        w = self.pesos_critic_locales.get(worker_id, self.pesos_critic_global)
        b = self.sesgo_critic_locales.get(worker_id, self.sesgo_critic_global)
        x = self._prep(estado) if np.asarray(estado).ndim >= 2 else self._prep(estado.reshape(1,-1))
        wc = self._norma_espectral(w, self._sigma(w)) if self.usar_ortogonal else w
        v = (np.dot(x, wc) + b).astype(_dt(), copy=False)
        self.tiempo_fwd += time.perf_counter() - t0
        return np.ascontiguousarray(v, dtype=_dt())

    def forward(self, estado, worker_id: int = 0):
        return self.forward_actor(estado, worker_id), self.forward_critic(estado, worker_id)

    def forward_batch(self, estados, worker_id: int = 0):
        self._asegurar()
        w = self.pesos_actor_locales.get(worker_id, self.pesos_actor_global)
        b = self.sesgo_actor_locales.get(worker_id, self.sesgo_actor_global)
        s = np.asarray(estados, dtype=np.float64)
        logits = s @ w + b; exp = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        return exp / np.sum(exp, axis=1, keepdims=True), s @ self.pesos_critic_locales.get(worker_id, self.pesos_critic_global) + self.sesgo_critic_locales.get(worker_id, self.sesgo_critic_global)

    def seleccionar_accion(self, estado, worker_id: int = 0):
        prob, val = self.forward(estado, worker_id)
        if self.epsilon > self.rng.random():
            accion = int(self.rng.integers(0, self.output_size)); self.estadisticas_a3c['exploracion_rate'] += 1
        else:
            accion = int(np.random.choice(self.output_size, p=prob[0])); self.estadisticas_a3c['exploitacion_rate'] += 1
        return accion, float(prob[0, accion]), float(val[0, 0])

    def calcular_gae_advantages(self, recompensas, valores):
        advantages = []; advantage = 0.0
        for t in reversed(range(len(recompensas))):
            nv = 0.0 if t == len(recompensas)-1 else valores[t+1]
            delta = recompensas[t] + self.gamma * nv - valores[t]
            advantage = delta + self.gamma * self.lambda_gae * advantage; advantages.insert(0, advantage)
        adv = np.array(advantages, dtype=np.float32)
        if len(adv) > 1: adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        self.estadisticas_a3c['advantage_media'] = float(np.mean(advantages)); self.estadisticas_a3c['advantage_std'] = float(np.std(advantages))
        return adv.tolist()

    def calcular_gradientes_worker(self, worker_id, estados, acciones, advantages, valores_objetivo):
        gap = np.zeros_like(self.pesos_actor_global); gas = np.zeros_like(self.sesgo_actor_global)
        gcp = np.zeros_like(self.pesos_critic_global); gcs = np.zeros_like(self.sesgo_critic_global)
        al = cl = 0.0; n = max(1, len(estados))
        for s, a, adv, vo in zip(estados, acciones, advantages, valores_objetivo):
            prob = self.forward_actor(s, worker_id); v = self.forward_critic(s, worker_id)
            glp = np.zeros(self.output_size); glp[a] = 1.0/(prob[0,a]+1e-8)
            gap += np.outer(s[0], glp)*adv; gas += glp*adv; al += -np.log(prob[0,a]+1e-8)*adv
            err = vo - v[0,0]; gcp += np.outer(s[0], np.array([err])); gcs += np.array([[err]]); cl += 0.5*err**2
        gap/=n; gas/=n; gcp/=n; gcs/=n
        if self.grad_clip > 0:
            for g in [gap, gas, gcp, gcs]:
                ng = float(np.linalg.norm(g))
                if ng > self.grad_clip: g *= self.grad_clip/(ng+1e-12)
        entropia = float(-np.mean([np.log(self.forward_actor(s,worker_id)[0,aa]+1e-8) for s,aa in zip(estados,acciones)]))
        pl = float(-np.mean([np.log(self.forward_actor(s,worker_id)[0,aa]+1e-8)*adv for s,aa,adv in zip(estados,acciones,advantages)]))
        ce = float(np.mean([(vo - self.forward_critic(s,worker_id)[0,0])**2 for s,vo in zip(estados,valores_objetivo)]))
        self.estadisticas_a3c['entropia_media'] = entropia * self.entropy_coef
        self.estadisticas_a3c['policy_loss_medio'] = pl
        self.estadisticas_a3c['valor_error_medio'] = ce
        return gap.astype(_dt()), gas.astype(_dt()), gcp.astype(_dt()), gcs.astype(_dt())

    def actualizar_pesos_worker(self, worker_id, gap, gas, gcp, gcs):
        with self.pesos_lock:
            t0 = time.perf_counter()
            lr_a = self.learning_rate_actor; lr_c = self.learning_rate_critic
            self.pesos_actor_locales[worker_id] -= lr_a * gap
            self.sesgo_actor_locales[worker_id] -= lr_a * gas
            self.pesos_critic_locales[worker_id] -= lr_c * gcp
            self.sesgo_critic_locales[worker_id] -= lr_c * gcs
            self.estadisticas_a3c['t_medio_upd_us'] = 0.95*self.estadisticas_a3c.get('t_medio_upd_us',0) + 0.05*(time.perf_counter()-t0)*1e6

    def sincronizar_pesos_globales(self):
        with self.pesos_lock:
            t0 = time.perf_counter()
            pa = np.zeros_like(self.pesos_actor_global); sa = np.zeros_like(self.sesgo_actor_global)
            pc = np.zeros_like(self.pesos_critic_global); sc = np.zeros_like(self.sesgo_critic_global)
            for wid in range(self.num_workers):
                pa += self.pesos_actor_locales[wid]; sa += self.sesgo_actor_locales[wid]
                pc += self.pesos_critic_locales[wid]; sc += self.sesgo_critic_locales[wid]
            pa/=self.num_workers; sa/=self.num_workers; pc/=self.num_workers; sc/=self.num_workers
            self.pesos_actor_global = pa; self.sesgo_actor_global = sa
            self.pesos_critic_global = pc; self.sesgo_critic_global = sc
            for wid in range(self.num_workers):
                self.pesos_actor_locales[wid] = pa.copy(); self.sesgo_actor_locales[wid] = sa.copy()
                self.pesos_critic_locales[wid] = pc.copy(); self.sesgo_critic_locales[wid] = sc.copy()
            self.estadisticas_a3c['sincronizaciones'] += 1; self.estadisticas_a3c['updates_globales'] += 1
            self.historial_sincronizaciones.append(time.time())
            self.estadisticas_a3c['t_medio_fwd_us'] = 0.95*self.estadisticas_a3c.get('t_medio_fwd_us',0) + 0.05*(time.perf_counter()-t0)*1e6

    def entrenar_paso(self, s, a, r, ns, d: bool = False):
        t0 = time.perf_counter()
        gap, gas, gcp, gcs = self.calcular_gradientes_worker(0, [s], [a], [r], [ns])
        self.actualizar_pesos_worker(0, gap, gas, gcp, gcs)
        self.pasos += 1; self.estadisticas_a3c['buffer_len'] = len(self.buffer)
        self.tiempo_upd += time.perf_counter() - t0
        return {'loss': float(np.mean(np.square(gap))), 'steps': self.pasos}

    def entrenar_desde_buffer(self, batch: Optional[int] = None):
        if len(self.buffer) < 4: return None
        b = self.buffer.sample(batch or self.update_frequency) if hasattr(self.buffer, 'sample') else self.buffer[-batch:] if batch else self.buffer
        return self.entrenar_paso(b['s'], b['a'], b['r'], b['ns'], b['d']) if isinstance(b, dict) else self.entrenar_paso(*b)

    def actualizar_epsilon(self):
        if self.epsilon > self.epsilon_min: self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def _convergencia_rapida(self):
        if self.pesos_actor_global is not None:
            pm = float(np.mean(np.abs(self.pesos_actor_global))); self._delta_ema = 0.02*pm + 0.98*self._delta_ema
            self.estadisticas_a3c['convergencia_actor_global'] = 1.0 - min(self._delta_ema, 1.0)

    def calcular_loss(self, estados, acciones, ventajas, valores_objetivo):
        self._asegurar(); n = max(1, len(estados)); pl = cl = 0.0
        for s, a, adv, vo in zip(estados, acciones, ventajas, valores_objetivo):
            prob = self.forward_actor(s, 0); val = self.forward_critic(s, 0)
            pl += -np.log(prob[0,a]+1e-8) * adv; cl += 0.5*(vo - val[0,0])**2
        return {'policy_loss': float(pl/n), 'value_loss': float(cl/n), 'total_loss': float((pl+cl)/n)}

    def limpiar_historiales(self):
        self.historial_qvalues.clear(); self.historial_epsilon.clear()
        self.historial_advantages.clear(); self.historial_policy.clear()
        self.historial_actor_loss_global.clear(); self.historial_critic_loss_global.clear()
        self.historial_advantages_global.clear(); self.historial_entropias_global.clear()
        self.historial_sincronizaciones.clear()

    def actualizar_buffer(self, experiencia):
        self.buffer.append(experiencia)
        if len(self.buffer) > 10240: del self.buffer[0:len(self.buffer)-10240]

    def __len__(self): return len(self.buffer)

    def verificar_integridad(self):
        checks = {}
        checks['pesos_inicializados'] = self.pesos_actor_global is not None
        checks['workers_activos'] = len(self.pesos_actor_locales) == self.num_workers
        checks['locks_funcionales'] = self.pesos_lock is not None
        if self.pesos_actor_global is not None:
            checks['sin_nan'] = not bool(np.any(np.isnan(self.pesos_actor_global)))
            checks['sin_inf'] = not bool(np.any(np.isinf(self.pesos_actor_global)))
            checks['precision'] = str(self.pesos_actor_global.dtype)
        return checks

    def _calcular_convergencia_ac(self):
        hl = self.historial_actor_loss; hlc = self.historial_critic_loss
        if len(hl) < 10 or len(hlc) < 10: return
        va = float(np.var(np.asarray(hl.tail(10), dtype=np.float64)))
        vc = float(np.var(np.asarray(hlc.tail(10), dtype=np.float64)))
        self.estadisticas_a3c['convergencia_actor_global'] = 1.0/(1.0+va)
        self.estadisticas_a3c['convergencia_critic_global'] = 1.0/(1.0+vc)

    def calcular_convergencia_global(self):
        if len(self.historial_sincronizaciones) < 5: return
        if hasattr(self, 'historial_pesos_actor_global'):
            pa = np.array(self.historial_pesos_actor_global[-5:]); vp = np.mean(np.var(pa, axis=0))
            c = 1.0/(1.0+vp); self.estadisticas_a3c['convergencia_actor_global'] = c; self.estadisticas_a3c['convergencia_critic_global'] = c

    def calcular_estabilidad_global(self):
        ea = ec = 0.0
        if len(self.pesos_actor_locales) > 1:
            ea = 1.0/(1.0+np.mean(np.var(np.array(list(self.pesos_actor_locales.values())), axis=0)))
        if len(self.pesos_critic_locales) > 1:
            ec = 1.0/(1.0+np.mean(np.var(np.array(list(self.pesos_critic_locales.values())), axis=0)))
        self.estadisticas_a3c['estabilidad_actor_global'] = ea; self.estadisticas_a3c['estabilidad_critic_global'] = ec
        return ea, ec

    def obtener_estadisticas_a3c(self):
        if self.pesos_actor_global is None: return {'estado':'no_inicializada'}
        ea, ec = self.calcular_estabilidad_global()
        if self.pesos_actor_global is not None and self.pesos_critic_global is not None:
            self.estadisticas_a3c['q_max'] = float(np.max(np.abs(self.pesos_actor_global)))
            self.estadisticas_a3c['q_min'] = float(np.min(np.abs(self.pesos_actor_global)))
            self.estadisticas_a3c['q_media'] = float(np.mean(np.abs(self.pesos_actor_global)))
            self.estadisticas_a3c['q_std'] = float(np.std(np.abs(self.pesos_actor_global)))
        self.estadisticas_a3c['loss_media'] = self._loss_ema; self.estadisticas_a3c['buffer_len'] = len(self.buffer)
        return {'learning_rate_actor':self.learning_rate_actor,'learning_rate_critic':self.learning_rate_critic,
            'gamma':self.gamma,'lambda_gae':self.lambda_gae,'num_workers':self.num_workers,
            'update_frequency':self.update_frequency,'convergencia_actor_global':self.estadisticas_a3c['convergencia_actor_global'],
            'convergencia_critic_global':self.estadisticas_a3c['convergencia_critic_global'],
            'estabilidad_actor_global':ea,'estabilidad_critic_global':ec,
            'sincronizaciones':self.estadisticas_a3c['sincronizaciones'],'updates_globales':self.estadisticas_a3c['updates_globales'],
            'entropy_coef':self.entropy_coef,'epsilon':self.epsilon,'loss_media':self.estadisticas_a3c.get('loss_media',0.0),
            'q_max':self.estadisticas_a3c['q_max'],'q_min':self.estadisticas_a3c['q_min'],
            'q_media':self.estadisticas_a3c['q_media'],'q_std':self.estadisticas_a3c['q_std'],
            't_medio_fwd_us':self.estadisticas_a3c.get('t_medio_fwd_us',0),'t_medio_upd_us':self.estadisticas_a3c.get('t_medio_upd_us',0),
            'policy_loss_medio':self.estadisticas_a3c.get('policy_loss_medio',0.0),'valor_error_medio':self.estadisticas_a3c.get('valor_error_medio',0.0)}

    def verificar_estabilidad(self):
        est = {}; ca = self.estadisticas_a3c['convergencia_actor_global']; cc = self.estadisticas_a3c['convergencia_critic_global']
        est['actor_convergencia_ok'] = ca > 0.7; est['critic_convergencia_ok'] = cc > 0.7
        ea, ec = self.calcular_estabilidad_global()
        est['actor_estable'] = ea > 0.7; est['critic_estable'] = ec > 0.7
        est['sincronizacion_ok'] = self.estadisticas_a3c['sincronizaciones'] > 0
        if self.pesos_actor_global is not None and self.pesos_critic_global is not None:
            est['pesos_actor_no_explosivos'] = np.max(np.abs(self.pesos_actor_global)) < 5.0
            est['pesos_critic_no_explosivos'] = np.max(np.abs(self.pesos_critic_global)) < 5.0
        else: est['pesos_actor_no_explosivos'] = est['pesos_critic_no_explosivos'] = True
        return est

    def reinicializar_con_parametros(self, learning_rate_actor=None, learning_rate_critic=None,
                                     gamma=None, lambda_gae=None, num_workers=None):
        if learning_rate_actor is not None: self.learning_rate_actor = learning_rate_actor
        if learning_rate_critic is not None: self.learning_rate_critic = learning_rate_critic
        if gamma is not None: self.gamma = gamma
        if lambda_gae is not None: self.lambda_gae = lambda_gae
        if num_workers is not None: self.num_workers = num_workers
        self.inicializar_pesos(); self.resetear_historial()
        self.historial_actor_loss_global.clear(); self.historial_critic_loss_global.clear()
        self.historial_advantages_global.clear(); self.historial_entropias_global.clear()
        self.historial_sincronizaciones.clear()
        self.estadisticas_a3c['sincronizaciones'] = 0; self.estadisticas_a3c['updates_globales'] = 0
        logger.info(f"Neurona A3C reinicializada: lr_actor={self.learning_rate_actor}, workers={self.num_workers}")

    def a_dict(self) -> Dict[str, Any]:
        return {'input_size':self.input_size,'output_size':self.output_size,'nombre':self.nombre,
            'pesos_actor':self.pesos_actor_global.tolist() if self.pesos_actor_global is not None else None,
            'pesos_critic':self.pesos_critic_global.tolist() if self.pesos_critic_global is not None else None,
            'sesgo_actor':self.sesgo_actor_global.tolist() if self.sesgo_actor_global is not None else None,
            'sesgo_critic':self.sesgo_critic_global.tolist() if self.sesgo_critic_global is not None else None}

    def guardar(self, ruta: str) -> None:
        import json
        with open(ruta, 'w', encoding='utf-8') as f: json.dump(self.a_dict(), f)

    def cargar(self, ruta: str) -> None:
        import json
        with open(ruta, 'r', encoding='utf-8') as f: d = json.load(f)
        self.inicializar_pesos()
        if d.get('pesos_actor') is not None: self.pesos_actor_global = np.asarray(d['pesos_actor'], dtype=_dt())
        if d.get('pesos_critic') is not None: self.pesos_critic_global = np.asarray(d['pesos_critic'], dtype=_dt())
        if d.get('sesgo_actor') is not None: self.sesgo_actor_global = np.asarray(d['sesgo_actor'], dtype=_dt())
        if d.get('sesgo_critic') is not None: self.sesgo_critic_global = np.asarray(d['sesgo_critic'], dtype=_dt())
        for wid in range(self.num_workers):
            self.pesos_actor_locales[wid] = self.pesos_actor_global.copy()
            self.sesgo_actor_locales[wid] = self.sesgo_actor_global.copy()
            self.pesos_critic_locales[wid] = self.pesos_critic_global.copy()
            self.sesgo_critic_locales[wid] = self.sesgo_critic_global.copy()

    def __str__(self): return (f"NeuronaRefuerzoA3C(entrada={self.input_size},salida={self.output_size},lr_a={self.learning_rate_actor},lr_c={self.learning_rate_critic},workers={self.num_workers})")
    def __repr__(self): return self.__str__()

def crear_neurona_a3c(input_size, output_size, configuracion=None):
    c = configuracion or {}
    return NeuronaRefuerzoA3C(input_size=input_size, output_size=output_size, nombre=c.get('nombre','NeuronaRefuerzoA3C'),
        learning_rate_actor=c.get('learning_rate_actor',0.001), learning_rate_critic=c.get('learning_rate_critic',0.002),
        gamma=c.get('gamma',0.99), lambda_gae=c.get('lambda_gae',0.95), num_workers=c.get('num_workers',4),
        update_frequency=c.get('update_frequency',20), usar_ortogonal=c.get('usar_ortogonal',True), entropy_coef=c.get('entropy_coef',0.01))

def analizar_a3c(neurona): return {'estadisticas': neurona.obtener_estadisticas_a3c(), 'estable': neurona.verificar_estabilidad()}

RFEN5_CONFIG = {'inicializacion_preferida':'ortogonal','learning_rate_actor_default':0.001,'learning_rate_critic_default':0.002,
    'gamma_default':0.99,'lambda_gae_default':0.95,'num_workers_default':4,'update_frequency_default':20,
    'entropy_coef_default':0.01,'umbral_convergencia':0.7,'umbral_estabilidad':0.7}
logger.info("RFEN1_RN_5.py cargado correctamente - Neurona de Refuerzo A3C Distribuida")