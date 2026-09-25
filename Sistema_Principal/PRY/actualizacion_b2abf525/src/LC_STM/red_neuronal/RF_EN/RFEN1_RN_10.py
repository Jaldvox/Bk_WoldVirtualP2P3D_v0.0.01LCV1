import numpy as np, math, logging, threading, time
from typing import Tuple, Optional, Dict, Any, List
from collections import deque
import random

logger = logging.getLogger('RFENRN1.RFEN1_RN_10')
LUCIA_RL_CONFIG = {'precision': 'float32', 'random_seed': 42, 'default_learning_rate': 0.001}

def _dt() -> np.dtype: return np.float32

def inicializar_pesos_he(shape, fan_in: int = 1) -> np.ndarray:
    return np.random.normal(0, math.sqrt(2.0/max(1,fan_in)), shape).astype(LUCIA_RL_CONFIG['precision'])
def inicializar_pesos_xavier(shape, fan_in: int = 1, fan_out: int = 1) -> np.ndarray:
    lim = math.sqrt(6.0/max(1,fan_in+fan_out)); return np.random.uniform(-lim,lim,shape).astype(LUCIA_RL_CONFIG['precision'])
def inicializar_pesos_lecun(shape, fan_in: int = 1) -> np.ndarray:
    return np.random.normal(0, math.sqrt(1.0/max(1,fan_in)), shape).astype(LUCIA_RL_CONFIG['precision'])
def inicializar_pesos_ortogonal(shape) -> np.ndarray:
    wf = np.random.randn(*shape).astype(np.float64); iters = 3
    v = np.random.default_rng(0).normal(0,1,(wf.shape[1],1)); v /= max(1e-12,np.linalg.norm(v))
    for _ in range(iters):
        u = wf @ v; u /= max(1e-12,np.linalg.norm(u)); v = wf.T @ u; v /= max(1e-12,np.linalg.norm(v))
    r = (u.T @ wf @ v)[0,0]; return np.random.randn(*shape).astype(LUCIA_RL_CONFIG['precision']) * abs(r)
def inicializar_pesos_espectral(shape) -> np.ndarray:
    wf = np.random.randn(*shape).astype(np.float64); v = np.random.default_rng(0).normal(0,1,(wf.shape[1],1))
    v /= max(1e-12,np.linalg.norm(v))
    for _ in range(3):
        u = wf @ v; u /= max(1e-12,np.linalg.norm(u)); v = wf.T @ u; v /= max(1e-12,np.linalg.norm(v))
    s = abs((u.T @ wf @ v)[0,0]); return (np.random.randn(*shape) * s / max(1e-12,np.linalg.norm(wf))).astype(LUCIA_RL_CONFIG['precision'])

class _Ring:
    def __init__(self, cap: int): self.cap = cap; self._d: List[Any] = []
    def append(self, x) -> None:
        self._d.append(x)
        if len(self._d) > self.cap: del self._d[0:len(self._d)-self.cap]
    def __len__(self) -> int: return len(self._d)
    def __getitem__(self, i: int) -> Any: return self._d[i]
    def __iter__(self): return iter(self._d)
    def tail(self, n: int) -> List[Any]: return self._d[-n:]
    def peek(self, n: int) -> List[Any]: return self._d[-n:] if len(self._d) >= n else []
    def clear(self) -> None: self._d.clear()
    @property
    def full(self) -> bool: return len(self._d) == self.cap
    @property
    def is_empty(self) -> bool: return len(self._d) == 0

class NeuronaRefuerzoBase:
    def __init__(self, input_size, output_size, nombre="NeuronaRefuerzo"):
        self.input_size = int(input_size); self.output_size = int(output_size); self.nombre = str(nombre)
        self.pesos = None; self.sesgo = None; self.historial_activaciones = []; self.historial_gradientes = []; self.pasos = 0
    def inicializar_pesos(self): raise NotImplementedError
    def forward(self, e): raise NotImplementedError
    def resetear_historial(self): self.historial_activaciones = []; self.historial_gradientes = []

class NeuronaRefuerzoIMPALA(NeuronaRefuerzoBase):
    def __init__(self, input_size: int, output_size: int, nombre: str = "NeuronaRefuerzoIMPALA",
                 learning_rate: float = 0.0003, gamma: float = 0.99, lambda_gae: float = 0.95,
                 num_actors: int = 8, num_learners: int = 2, batch_size: int = 32,
                 sequence_length: int = 20, usar_ortogonal: bool = True, entropy_coef: float = 0.01,
                 value_coef: float = 0.5, max_grad_norm: float = 0.5, update_frequency: int = 100):
        super().__init__(input_size, output_size, nombre)
        self.learning_rate = learning_rate; self.gamma = gamma; self.lambda_gae = lambda_gae
        self.num_actors = num_actors; self.num_learners = num_learners; self.batch_size = batch_size
        self.sequence_length = sequence_length; self.usar_ortogonal = usar_ortogonal
        self.entropy_coef = entropy_coef; self.value_coef = value_coef; self.max_grad_norm = max_grad_norm
        self.update_frequency = update_frequency
        self.pesos_actor_global = self.sesgo_actor_global = None
        self.pesos_critic_global = self.sesgo_critic_global = None
        self.pesos_actor_locales = {}; self.sesgo_actor_locales = {}
        self.pesos_critic_locales = {}; self.sesgo_critic_locales = {}
        self.buffer_experiencias = {}; self.buffer_prioridades = {}
        self.pesos_lock = threading.Lock()
        self.estadisticas_impala = {'actor_loss_medio':0.0,'critic_loss_medio':0.0,'entropy_loss_medio':0.0,
            'total_loss_medio':0.0,'importance_weights_medio':0.0,'importance_weights_std':0.0,
            'convergencia_actor':0.0,'convergencia_critic':0.0,'estabilidad_actor':0.0,'estabilidad_critic':0.0,
            'sincronizaciones':0,'updates_globales':0,'muestras_totales':0,
            'gradiente_norma_actor':0.0,'gradiente_norma_critic':0.0,'actors_activos':0,'learners_activos':0}
        self.historial_actor_loss = []; self.historial_critic_loss = []; self.historial_entropy_loss = []
        self.historial_importance_weights = []; self.historial_sincronizaciones = []

    def inicializar_pesos(self) -> None:
        init_o = inicializar_pesos_ortogonal if self.usar_ortogonal else inicializar_pesos_xavier
        self.pesos_actor_global = init_o((self.input_size, self.output_size))
        self.pesos_critic_global = init_o((self.input_size, 1))
        self.sesgo_actor_global = np.zeros((1,self.output_size),dtype=LUCIA_RL_CONFIG['precision'])
        self.sesgo_critic_global = np.zeros((1,1),dtype=LUCIA_RL_CONFIG['precision'])
        for aid in range(self.num_actors):
            self.pesos_actor_locales[aid] = self.pesos_actor_global.copy(); self.sesgo_actor_locales[aid] = self.sesgo_actor_global.copy()
            self.pesos_critic_locales[aid] = self.pesos_critic_global.copy(); self.sesgo_critic_locales[aid] = self.sesgo_critic_global.copy()
            self.buffer_experiencias[aid] = deque(maxlen=1000); self.buffer_prioridades[aid] = deque(maxlen=1000)

    def _prep(self, estado: np.ndarray) -> np.ndarray:
        x = np.asarray(estado, dtype=_dt())
        if x.ndim == 1: x = x.reshape(1, -1)
        return np.ascontiguousarray(x, dtype=_dt())

    @staticmethod
    def _sigma(w: np.ndarray, iters: int = 3) -> float:
        wf = w.astype(np.float64); v = np.random.default_rng(0).normal(0,1,(wf.shape[1],1))
        v /= max(1e-12,np.linalg.norm(v))
        for _ in range(iters):
            u = wf @ v; u /= max(1e-12,np.linalg.norm(u)); v = wf.T @ u; v /= max(1e-12,np.linalg.norm(v))
        return float((u.T @ wf @ v)[0, 0])

    def forward_actor(self, estado: np.ndarray, actor_id: int = 0) -> np.ndarray:
        if self.pesos_actor_global is None: self.inicializar_pesos()
        wp = self.pesos_actor_locales.get(actor_id, self.pesos_actor_global); sb = self.sesgo_actor_locales.get(actor_id, self.sesgo_actor_global)
        x = self._prep(estado); logits = x @ wp + sb
        e = np.exp(logits - np.max(logits, axis=1, keepdims=True)); return e/np.sum(e, axis=1, keepdims=True)

    def forward_critic(self, estado: np.ndarray, actor_id: int = 0) -> np.ndarray:
        if self.pesos_critic_global is None: self.inicializar_pesos()
        wp = self.pesos_critic_locales.get(actor_id, self.pesos_critic_global); sb = self.sesgo_critic_locales.get(actor_id, self.sesgo_critic_global)
        return self._prep(estado) @ wp + sb

    def forward(self, estado: np.ndarray, actor_id: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        return self.forward_actor(estado, actor_id), self.forward_critic(estado, actor_id)

    def forward_batch(self, estados) -> Tuple[np.ndarray, np.ndarray]:
        if self.pesos_actor_global is None: self.inicializar_pesos()
        xs = np.concatenate([self._prep(s) for s in estados], axis=0)
        return np.exp(xs @ self.pesos_actor_global + self.sesgo_actor_global), xs @ self.pesos_critic_global + self.sesgo_critic_global

    def forward_q(self, estado, accion, usar_objetivo=False, q_network=1) -> np.ndarray:
        wp = self.pesos_critic_global; sb = self.sesgo_critic_global
        return self._prep(estado) @ wp + sb

    def seleccionar_accion(self, estado: np.ndarray, actor_id: int = 0) -> Tuple[int, float, float]:
        probs, val = self.forward(estado, actor_id); accion = int(np.random.choice(self.output_size, p=probs[0]))
        return accion, float(probs[0, accion]), float(val[0, 0])

    def agregar_experiencia(self, actor_id: int, estado, accion, recompensa, siguiente_estado, terminado, probabilidad_accion) -> None:
        exp = {'estado':np.asarray(estado),'accion':accion,'recompensa':recompensa,'siguiente_estado':np.asarray(siguiente_estado),'terminado':terminado,'probabilidad_accion':probabilidad_accion,'timestamp':time.time()}
        self.buffer_experiencias[actor_id].append(exp); self.estadisticas_impala['muestras_totales'] = sum(len(b) for b in self.buffer_experiencias.values())

    def calcular_gae_advantages(self, recompensas: List[float], valores: List[float]) -> np.ndarray:
        rews = np.array(recompensas, dtype=np.float64); vals = np.array(valores, dtype=np.float64); n = len(rews)
        advantages = np.zeros(n, dtype=np.float64); adv = 0.0
        for t in range(n - 1, -1, -1):
            nv = 0.0 if t == n - 1 else vals[t + 1]; delta = rews[t] + self.gamma * nv - vals[t]; adv = delta + self.gamma * self.lambda_gae * adv; advantages[t] = adv
        return advantages

    def calcular_importance_weights(self, experiencias: List[Dict[str, Any]], actor_id: int) -> np.ndarray:
        estados = np.array([e['estado'] for e in experiencias], dtype=_dt())
        acciones = np.array([e['accion'] for e in experiencias]); p_ant = np.array([e['probabilidad_accion'] for e in experiencias], dtype=np.float64)
        p_act = self.forward_actor(estados, actor_id)[np.arange(len(experiencias)), acciones]
        iw = np.clip(p_act / np.maximum(p_ant, 1e-8), 0.1, 10.0)
        if len(iw):
            self.estadisticas_impala['importance_weights_medio'] = float(np.mean(iw)); self.estadisticas_impala['importance_weights_std'] = float(np.std(iw))
            self.historial_importance_weights.extend(iw.tolist())
        return iw

    def calcular_gradientes_impala(self, experiencias: List[Dict[str, Any]], actor_id: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if self.pesos_actor_global is None: raise ValueError("Pesos no inicializados")
        iw = self.calcular_importance_weights(experiencias, actor_id)
        estados = np.array([e['estado'] for e in experiencias], dtype=_dt()).reshape(-1, self.input_size)
        acciones = np.array([e['accion'] for e in experiencias])
        recompensas = np.array([e['recompensa'] for e in experiencias]); terminados = np.array([e['terminado'] for e in experiencias])
        siguientes = np.array([e['siguiente_estado'] for e in experiencias], dtype=_dt()).reshape(-1, self.input_size)
        valores = np.array([self.forward_critic(e['estado'], actor_id)[0, 0] for e in experiencias], dtype=np.float64)
        advantages = self.calcular_gae_advantages(recompensas, valores)
        q_sig = self.forward_critic(siguientes, actor_id)
        q_objetivo = np.where(terminados, recompensas, recompensas + self.gamma * np.max(q_sig, axis=1))
        error_critic = q_objetivo - valores
        probs = self.forward_actor(estados, actor_id)
        grad_acc = np.zeros((len(experiencias), self.output_size), dtype=_dt())
        grad_acc[np.arange(len(experiencias)), acciones] = 1.0 / np.maximum(probs[np.arange(len(experiencias)), acciones], 1e-8)
        iw_exp = iw.reshape(-1, 1)
        gw_a = estados.T @ (grad_acc * advantages.reshape(-1, 1) * iw_exp) / len(experiencias)
        gs_a = np.mean(grad_acc * advantages * iw, axis=0)
        gw_c = estados.T @ (error_critic.reshape(-1, 1) * iw_exp) / len(experiencias)
        gs_c = np.mean(error_critic * iw, axis=0)
        actor_loss = float(-np.mean(np.log(probs[np.arange(len(experiencias)), acciones] + 1e-8) * advantages * iw))
        critic_loss = float(0.5 * np.mean(error_critic**2 * iw))
        entropy = -np.mean(np.sum(probs * np.log(probs + 1e-8), axis=1) * iw)
        self.estadisticas_impala['actor_loss_medio'] = actor_loss; self.estadisticas_impala['critic_loss_medio'] = critic_loss; self.estadisticas_impala['entropy_loss_medio'] = entropy
        self.estadisticas_impala['total_loss_medio'] = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy
        self.estadisticas_impala['gradiente_norma_actor'] = float(np.linalg.norm(gw_a)); self.estadisticas_impala['gradiente_norma_critic'] = float(np.linalg.norm(gw_c))
        return gw_a, gs_a, gw_c, gs_c

    def actualizar_pesos_learner(self, gwa, gsa, gwc, gsc) -> None:
        if self.pesos_actor_global is None or self.pesos_critic_global is None: self.inicializar_pesos()
        with self.pesos_lock:
            gn_a = float(np.linalg.norm(gwa)); gn_c = float(np.linalg.norm(gwc))
            if gn_a > self.max_grad_norm: gwa *= self.max_grad_norm/(gn_a+1e-8); gsa *= self.max_grad_norm/(gn_a+1e-8)
            if gn_c > self.max_grad_norm: gwc *= self.max_grad_norm/(gn_c+1e-8); gsc *= self.max_grad_norm/(gn_c+1e-8)
            assert self.pesos_actor_global is not None and self.pesos_critic_global is not None
            self.pesos_actor_global += self.learning_rate * gwa; self.sesgo_actor_global += self.learning_rate * gsa
            self.pesos_critic_global += self.learning_rate * gwc; self.sesgo_critic_global += self.learning_rate * gsc
            self.estadisticas_impala['gradiente_norma_actor'] = gn_a; self.estadisticas_impala['gradiente_norma_critic'] = gn_c; self.estadisticas_impala['updates_globales'] += 1

    def sincronizar_actores(self) -> None:
        if self.pesos_actor_global is None or self.pesos_critic_global is None: self.inicializar_pesos()
        assert self.pesos_actor_global is not None and self.pesos_critic_global is not None
        with self.pesos_lock:
            for aid in range(self.num_actors):
                self.pesos_actor_locales[aid] = self.pesos_actor_global.copy(); self.sesgo_actor_locales[aid] = self.sesgo_actor_global.copy()
                self.pesos_critic_locales[aid] = self.pesos_critic_global.copy(); self.sesgo_critic_locales[aid] = self.sesgo_critic_global.copy()
            self.estadisticas_impala['sincronizaciones'] += 1; self.historial_sincronizaciones.append(time.time())

    def entrenar_step(self, actor_id: int) -> Optional[Dict[str, float]]:
        if len(self.buffer_experiencias[actor_id]) < self.batch_size: return None
        exp = random.sample(list(self.buffer_experiencias[actor_id]), self.batch_size)
        gwa, gsa, gwc, gsc = self.calcular_gradientes_impala(exp, actor_id); self.actualizar_pesos_learner(gwa, gsa, gwc, gsc)
        if self.estadisticas_impala['updates_globales'] % self.update_frequency == 0: self.sincronizar_actores()
        self.historial_actor_loss.append(self.estadisticas_impala['actor_loss_medio']); self.historial_critic_loss.append(self.estadisticas_impala['critic_loss_medio']); self.historial_entropy_loss.append(self.estadisticas_impala['entropy_loss_medio'])
        return {'actor_loss':self.estadisticas_impala['actor_loss_medio'],'critic_loss':self.estadisticas_impala['critic_loss_medio'],'entropy_loss':self.estadisticas_impala['entropy_loss_medio'],'total_loss':self.estadisticas_impala['total_loss_medio'],'importance_weights_medio':self.estadisticas_impala['importance_weights_medio'],'sincronizaciones':self.estadisticas_impala['sincronizaciones']}

    def calcular_loss(self, experiencias, actor_id) -> Dict[str, float]:
        self.calcular_gradientes_impala(experiencias, actor_id); return {'actor_loss':self.estadisticas_impala['actor_loss_medio'],'critic_loss':self.estadisticas_impala['critic_loss_medio'],'entropy_loss':self.estadisticas_impala['entropy_loss_medio'],'total_loss':self.estadisticas_impala['total_loss_medio']}

    def calcular_entropy(self, estado, actor_id=0) -> float:
        probs = self.forward_actor(estado, actor_id); return float(-np.sum(probs * np.log(probs + 1e-8)))

    def calcular_critico_valor(self, estado, actor_id=0) -> float: return float(self.forward_critic(estado, actor_id)[0, 0])

    def calcular_value_function(self, estado, actor_id=0) -> float: return float(self.forward_critic(estado, actor_id)[0, 0])

    def calcular_value_loss(self, experiencias, actor_id) -> float:
        preds = np.array([self.forward_critic(e['estado'], actor_id)[0,0] for e in experiencias])
        targets = np.array([e['recompensa'] for e in experiencias], dtype=np.float64)
        return float(np.mean((targets - preds)**2))

    def calcular_policy_loss(self, experiencias, actor_id) -> float:
        probs = self.forward_actor(np.array([e['estado'] for e in experiencias],dtype=_dt()), actor_id)
        acciones = np.array([e['accion'] for e in experiencias])
        return float(-np.mean(np.log(probs[np.arange(len(experiencias)), acciones] + 1e-8)))

    def calcular_distribucion_loss(self, experiencias, actor_id) -> float:
        probs = self.forward_actor(np.array([e['estado'] for e in experiencias],dtype=_dt()), actor_id)
        return float(-np.mean(np.sum(probs * np.log(probs + 1e-8), axis=1)))

    def calcular_q_target(self, estado, accion, recompensa, siguiente_estado, terminado) -> float:
        if terminado: return float(recompensa)
        return float(recompensa + self.gamma * np.max(self.forward_critic(siguiente_estado)[0]))

    def calcular_prioridad(self, experiencia) -> float:
        probs = self.forward_actor(experiencia['estado'])
        q_act = float(probs[0, experiencia['accion']])
        q_obj = self.calcular_q_target(experiencia['estado'], experiencia['accion'], experiencia['recompensa'], experiencia['siguiente_estado'], experiencia['terminado'])
        return float((abs(q_obj - q_act) + 1e-6) ** 0.6)

    def calcular_prioridad_batch(self, batch) -> np.ndarray:
        estados = np.array([e['estado'] for e in batch], dtype=_dt())
        acciones = np.array([e['accion'] for e in batch])
        preds = self.forward_actor(estados)[np.arange(len(batch)), acciones]
        return (np.abs(preds) + 1e-6) ** 0.6

    def _preparar_minibatch(self, batch):
        st = np.array([e['estado'] for e in batch], dtype=_dt())
        ac = np.array([e['accion'] for e in batch])
        rw = np.array([e['recompensa'] for e in batch], dtype=np.float32).reshape(-1,1)
        tn = np.array([float(e['terminado']) for e in batch], dtype=np.float32).reshape(-1,1)
        ns = np.array([e['siguiente_estado'] for e in batch], dtype=_dt())
        return st, ac, rw, tn, ns

    def calcular_explained_variance(self, experiencias, actor_id) -> float:
        preds = np.array([self.forward_critic(e['estado'], actor_id)[0, 0] for e in experiencias]); rets = np.array([e['recompensa'] for e in experiencias], dtype=np.float64)
        var = float(np.var(rets)); resid = float(np.var(rets - preds)); return 1.0 - resid/max(1e-12, var)

    def forward_dueling(self, estado, actor_id=0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.pesos_actor_global is None: self.inicializar_pesos()
        vs = self._prep(estado) @ (self.pesos_actor_global * 0.5)
        as_ = self._prep(estado) @ self.pesos_actor_global
        return self.forward_actor(estado, actor_id), vs, as_

    def calcular_gradientes_vectoriales(self, estados, acciones, ventajas) -> Tuple[np.ndarray, np.ndarray]:
        if self.pesos_actor_global is None: raise ValueError("Pesos no inicializados")
        estados = np.array(estados, dtype=_dt()).reshape(-1, self.input_size)
        acciones = np.array(acciones); ventajas = np.array(ventajas)
        probs = self.forward_actor(estados, 0)
        grad_acc = np.zeros((len(acciones), self.output_size), dtype=_dt())
        grad_acc[np.arange(len(acciones)), acciones] = 1.0 / np.maximum(probs[np.arange(len(acciones)), acciones], 1e-8)
        return estados.T @ (grad_acc * ventajas.reshape(-1,1)), np.mean(grad_acc * ventajas, axis=0)

    def calcular_estabilidad_pesos(self) -> Dict[str, float]:
        r = {}
        for nm, p in [('actor',self.pesos_actor_global),('critic',self.pesos_critic_global)]:
            if p is not None: r[f'{nm}_max'] = float(np.max(np.abs(p))); r[f'{nm}_mean'] = float(np.mean(np.abs(p))); r[f'{nm}_min'] = float(np.min(np.abs(p)))
        return r

    def obtener_ratio_exploracion(self) -> float:
        s = self.estadisticas_impala; ta = s.get('exploracion_rate',0) + s.get('exploitacion_rate',0)
        return float(s.get('exploracion_rate',0)/ta) if ta > 0 else 0.0

    def _aplicar_beta(self) -> None:
        pass

    def forward_critic_batch(self, estados) -> np.ndarray:
        if self.pesos_critic_global is None: self.inicializar_pesos()
        xs = np.concatenate([self._prep(s) for s in estados], axis=0)
        return xs @ self.pesos_critic_global + self.sesgo_critic_global

    def calcular_actor_critic_loss(self, estados, acciones, ventajas) -> Tuple[float, float]:
        probs = self.forward_actor(np.array(estados,dtype=_dt()))
        accs = np.array(acciones)
        al = float(-np.mean(np.log(probs[np.arange(len(accs)),accs]+1e-8)*np.array(ventajas,dtype=np.float64)))
        vals = self.forward_critic_batch(estados).flatten()
        vl = float(0.5*np.mean((np.array(ventajas,dtype=np.float64)-vals)**2))
        return al, vl

    def obtener_peso_exploracion(self) -> float: return float(self.learning_rate)

    def obtener_configuracion_actual(self) -> Dict[str, Any]:
        return {'learning_rate':self.learning_rate,'gamma':self.gamma,'lambda_gae':self.lambda_gae,'num_actors':self.num_actors,'num_learners':self.num_learners,'batch_size':self.batch_size,'entropy_coef':self.entropy_coef,'value_coef':self.value_coef,'max_grad_norm':self.max_grad_norm,'update_frequency':self.update_frequency,'usar_ortogonal':self.usar_ortogonal}

    def verificar_integridad(self) -> Dict[str, Any]:
        checks = {'pesos_actor_init':self.pesos_actor_global is not None,'pesos_critic_init':self.pesos_critic_global is not None,'actores':self.num_actors,'learners':self.num_learners}
        for nm,p in [('actor',self.pesos_actor_global),('critic',self.pesos_critic_global)]:
            if p is not None: checks[f'sin_nan_{nm}'] = not bool(np.any(np.isnan(p))); checks[f'sin_inf_{nm}'] = not bool(np.any(np.isinf(p))); checks[f'peso_{nm}_max'] = float(np.max(np.abs(p)))
        checks['muestras_totales'] = self.estadisticas_impala['muestras_totales']; checks['sincronizaciones'] = self.estadisticas_impala['sincronizaciones']
        return checks

    def resetear_prioridades(self) -> None:
        for aid in range(self.num_actors):
            if aid in self.buffer_prioridades: self.buffer_prioridades[aid].clear()

    def limpiar_historiales(self) -> None:
        self.historial_actor_loss.clear(); self.historial_critic_loss.clear(); self.historial_entropy_loss.clear(); self.historial_importance_weights.clear(); self.historial_sincronizaciones.clear()

    def calcular_convergencia_global(self) -> None:
        if len(self.historial_sincronizaciones) < 5: return
        if len(self.historial_actor_loss) >= 5:
            vl = np.array(self.historial_actor_loss[-5:]); c = float(1.0/(1.0+np.var(vl)))
            self.estadisticas_impala['convergencia_actor'] = c; self.estadisticas_impala['convergencia_critic'] = c

    def calcular_estabilidad_global(self) -> Tuple[float, float]:
        ea = 0.0; ec = 0.0
        if len(self.pesos_actor_locales) > 1: ea = float(1.0/(1.0+np.mean(np.var(np.array(list(self.pesos_actor_locales.values())),axis=0))))
        if len(self.pesos_critic_locales) > 1: ec = float(1.0/(1.0+np.mean(np.var(np.array(list(self.pesos_critic_locales.values())),axis=0))))
        self.estadisticas_impala['estabilidad_actor'] = ea; self.estadisticas_impala['estabilidad_critic'] = ec
        return ea, ec

    def calcular_estabilidad_q(self) -> float:
        if len(self.historial_actor_loss) < 5: return 0.0
        vals = np.array(self.historial_actor_loss[-5:], dtype=np.float64); return float(1.0/(1.0+np.var(vals)))

    def verificar_estabilidad(self) -> Dict[str, bool]:
        e = {}; ca = self.estadisticas_impala['convergencia_actor']; cc = self.estadisticas_impala['convergencia_critic']
        e['actor_convergencia_ok'] = ca > 0.7; e['critic_convergencia_ok'] = cc > 0.7
        ea, ec = self.calcular_estabilidad_global(); e['actor_estable'] = ea > 0.7; e['critic_estable'] = ec > 0.7
        e['sincronizacion_ok'] = self.estadisticas_impala['sincronizaciones'] > 0
        iw = self.estadisticas_impala['importance_weights_medio']; e['importance_weights_estables'] = 0.5 <= iw <= 2.0
        if self.pesos_actor_global is not None and self.pesos_critic_global is not None:
            e['pesos_actor_no_explosivos'] = np.max(np.abs(self.pesos_actor_global)) < 5.0; e['pesos_critic_no_explosivos'] = np.max(np.abs(self.pesos_critic_global)) < 5.0
        else: e.update({'pesos_actor_no_explosivos':True,'pesos_critic_no_explosivos':True})
        return e

    def obtener_estadisticas_impala(self) -> Dict[str, Any]:
        if self.pesos_actor_global is None: return {'estado':'no_inicializada'}
        ea, ec = self.calcular_estabilidad_global(); s = self.estadisticas_impala
        return {'learning_rate':self.learning_rate,'gamma':self.gamma,'lambda_gae':self.lambda_gae,'num_actors':self.num_actors,'num_learners':self.num_learners,'batch_size':self.batch_size,'sequence_length':self.sequence_length,'entropy_coef':self.entropy_coef,'value_coef':self.value_coef,'max_grad_norm':self.max_grad_norm,'update_frequency':self.update_frequency,'actor_loss_medio':s['actor_loss_medio'],'critic_loss_medio':s['critic_loss_medio'],'entropy_loss_medio':s['entropy_loss_medio'],'total_loss_medio':s['total_loss_medio'],'importance_weights_medio':s['importance_weights_medio'],'importance_weights_std':s['importance_weights_std'],'convergencia_actor':s['convergencia_actor'],'convergencia_critic':s['convergencia_critic'],'estabilidad_actor':ea,'estabilidad_critic':ec,'sincronizaciones':s['sincronizaciones'],'updates_globales':s['updates_globales'],'muestras_totales':s['muestras_totales'],'gradiente_norma_actor':s['gradiente_norma_actor'],'gradiente_norma_critic':s['gradiente_norma_critic']}

    def obtener_estadisticas_impala_extended(self) -> Dict[str, Any]:
        s = self.obtener_estadisticas_impala(); s.update({'sigma_actor':float(self._sigma(self.pesos_actor_global)) if self.pesos_actor_global is not None else 0.0,'exploracion_ratio':self.obtener_ratio_exploracion(),'configuracion':self.obtener_configuracion_actual(),'verificar_integridad':self.verificar_integridad(),'estabilidad_pesos':self.calcular_estabilidad_pesos()})
        return s

    def guardar(self, ruta: str) -> None:
        import json
        d = {'input_size':self.input_size,'output_size':self.output_size,'nombre':self.nombre,
            'pesos_actor_global':self.pesos_actor_global.tolist() if self.pesos_actor_global is not None else None,
            'pesos_critic_global':self.pesos_critic_global.tolist() if self.pesos_critic_global is not None else None,
            'learning_rate':self.learning_rate,'gamma':self.gamma,'lambda_gae':self.lambda_gae,
            'num_actors':self.num_actors,'num_learners':self.num_learners,'batch_size':self.batch_size}
        with open(ruta, 'w', encoding='utf-8') as f: json.dump(d, f)

    def cargar(self, ruta: str) -> None:
        import json
        with open(ruta, 'r', encoding='utf-8') as f: d = json.load(f)
        self.inicializar_pesos()
        for k in ['pesos_actor_global','pesos_critic_global']:
            if d.get(k) is not None: setattr(self, k, np.asarray(d[k], dtype=_dt()))

    def reinicializar_con_parametros(self, learning_rate=None, gamma=None, lambda_gae=None, num_actors=None) -> None:
        if learning_rate is not None: self.learning_rate = learning_rate
        if gamma is not None: self.gamma = gamma
        if lambda_gae is not None: self.lambda_gae = lambda_gae
        if num_actors is not None: self.num_actors = num_actors
        self.inicializar_pesos(); self.resetear_historial(); self.limpiar_historiales()
        for aid in range(self.num_actors): self.buffer_experiencias[aid].clear(); self.buffer_prioridades[aid].clear()
        self.estadisticas_impala['sincronizaciones'] = 0; self.estadisticas_impala['updates_globales'] = 0; self.estadisticas_impala['muestras_totales'] = 0
        logger.info(f"Neurona IMPALA reinicializada: lr={self.learning_rate}, gamma={self.gamma}, actors={self.num_actors}")

    def _aplicar_epsilon_decay(self) -> None: pass

    def __str__(self): return f"NeuronaRefuerzoIMPALA(entrada={self.input_size},salida={self.output_size},lr={self.learning_rate},gamma={self.gamma},actors={self.num_actors},learners={self.num_learners})"
    def __repr__(self): return self.__str__()

def crear_neurona_impala(input_size: int, output_size: int, configuracion: Optional[Dict[str, Any]] = None) -> NeuronaRefuerzoIMPALA:
    c = configuracion or {}
    return NeuronaRefuerzoIMPALA(input_size=input_size,output_size=output_size,nombre=c.get('nombre','NeuronaRefuerzoIMPALA'),learning_rate=c.get('learning_rate',0.0003),gamma=c.get('gamma',0.99),lambda_gae=c.get('lambda_gae',0.95),num_actors=c.get('num_actors',8),num_learners=c.get('num_learners',2),batch_size=c.get('batch_size',32),sequence_length=c.get('sequence_length',20),usar_ortogonal=c.get('usar_ortogonal',True),entropy_coef=c.get('entropy_coef',0.01),value_coef=c.get('value_coef',0.5),max_grad_norm=c.get('max_grad_norm',0.5),update_frequency=c.get('update_frequency',100))

def _impala_advantage_batch(estados, acciones, recompensas, gamma: float, lambda_gae: float) -> np.ndarray:
    rews = np.array(recompensas, dtype=np.float64); n = len(rews); adv = np.zeros(n)
    a = 0.0
    for t in range(n - 1, -1, -1): nv = 0.0 if t == n - 1 else 0.0; delta = rews[t] + gamma * nv - 0.0; a = delta + gamma * lambda_gae * a; adv[t] = a
    return adv

RFEN10_CONFIG = {'inicializacion_preferida':'ortogonal','learning_rate_default':0.0003,'gamma_default':0.99,'lambda_gae_default':0.95,'num_actors_default':8,'num_learners_default':2,'batch_size_default':32,'sequence_length_default':20,'entropy_coef_default':0.01,'value_coef_default':0.5,'max_grad_norm_default':0.5,'update_frequency_default':100,'umbral_convergencia':0.7,'umbral_estabilidad':0.7}
logger.info("RFEN1_RN_10.py cargado - Neurona IMPALA Distribuida")