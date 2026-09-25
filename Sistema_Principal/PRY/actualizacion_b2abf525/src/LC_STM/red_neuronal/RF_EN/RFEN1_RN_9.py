import numpy as np, math, logging
from typing import Tuple, Optional, Dict, Any, List
from collections import deque
import random

logger = logging.getLogger('RFENRN1.RFEN1_RN_9')
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
    def __bool__(self) -> bool: return len(self._d) > 0
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

class NeuronaRefuerzoRainbowDQN(NeuronaRefuerzoBase):
    def __init__(self, input_size: int, output_size: int, nombre: str = "NeuronaRefuerzoRainbowDQN",
                 learning_rate: float = 0.00025, gamma: float = 0.99, epsilon: float = 0.1,
                 epsilon_decay: float = 0.995, epsilon_min: float = 0.01, tau: float = 0.005,
                 buffer_size: int = 1000000, batch_size: int = 32, usar_he: bool = True,
                 usar_dueling: bool = True, usar_prioritized: bool = True, usar_multi_step: bool = True,
                 usar_distributional: bool = True, usar_noisy: bool = True, n_atoms: int = 51,
                 v_min: float = -10.0, v_max: float = 10.0, n_steps: int = 3,
                 alpha_prioritized: float = 0.6, beta_prioritized: float = 0.4):
        super().__init__(input_size, output_size, nombre)
        self.learning_rate = learning_rate; self.gamma = gamma; self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay; self.epsilon_min = epsilon_min; self.tau = tau
        self.batch_size = batch_size; self.usar_he = usar_he; self.usar_dueling = usar_dueling
        self.usar_prioritized = usar_prioritized; self.usar_multi_step = usar_multi_step
        self.usar_distributional = usar_distributional; self.usar_noisy = usar_noisy
        self.n_atoms = n_atoms; self.v_min = v_min; self.v_max = v_max; self.n_steps = n_steps
        self.alpha_prioritized = alpha_prioritized; self.beta_prioritized = beta_prioritized
        self.pesos_q_principal = None; self.sesgo_q_principal = None; self.pesos_q_objetivo = None; self.sesgo_q_objetivo = None
        if self.usar_dueling:
            self.pesos_value_stream = self.sesgo_value_stream = None
            self.pesos_advantage_stream = self.sesgo_advantage_stream = None
            self.pesos_value_stream_target = self.sesgo_value_stream_target = None
            self.pesos_advantage_stream_target = self.sesgo_advantage_stream_target = None
        if self.usar_distributional:
            self.pesos_distributional = self.sesgo_distributional = None
            self.pesos_distributional_target = self.sesgo_distributional_target = None
        if self.usar_noisy:
            self.pesos_noisy = self.sesgo_noisy = None; self.pesos_noisy_target = self.sesgo_noisy_target = None
        self.buffer_experiencia = _Ring(buffer_size)
        self.prioridades = _Ring(buffer_size) if self.usar_prioritized else None
        self.beta_schedule = 0.4
        self.estadisticas_rainbow = {'td_error_medio':0.0,'td_error_std':0.0,'convergencia_q':0.0,'estabilidad_q':0.0,
            'actualizaciones_target':0,'muestras_buffer':0,'exploracion_rate':0,'exploitacion_rate':0,'loss_medio':0.0,
            'gradiente_norma':0.0,'prioridad_media':0.0,'kl_divergencia_distribucional':0.0,'entropia_distribucional':0.0,'noise_factor':0.0}
        self.historial_td_errors = []; self.historial_losses = []; self.historial_q_values = []
        self.historial_epsilon = []; self.historial_prioridades = []; self.historial_kl_divergences = []

    def inicializar_pesos(self) -> None:
        init_h = inicializar_pesos_he if self.usar_he else inicializar_pesos_xavier
        self.pesos_q_principal = init_h((self.input_size, self.output_size), self.input_size) if self.usar_he else inicializar_pesos_xavier((self.input_size,self.output_size),self.input_size,self.output_size)
        self.pesos_q_objetivo = init_h((self.input_size, self.output_size), self.input_size) if self.usar_he else inicializar_pesos_xavier((self.input_size,self.output_size),self.input_size,self.output_size)
        self.sesgo_q_principal = np.zeros((1,self.output_size),dtype=LUCIA_RL_CONFIG['precision'])
        self.sesgo_q_objetivo = np.zeros((1,self.output_size),dtype=LUCIA_RL_CONFIG['precision'])
        if self.usar_dueling:
            pv = inicializar_pesos_he((self.input_size,1),self.input_size) if self.usar_he else inicializar_pesos_xavier((self.input_size,1),self.input_size,1)
            pa = inicializar_pesos_he((self.input_size,self.output_size),self.input_size) if self.usar_he else inicializar_pesos_xavier((self.input_size,self.output_size),self.input_size,self.output_size)
            self.pesos_value_stream = pv; self.pesos_advantage_stream = pa; self.pesos_value_stream_target = pv.copy(); self.pesos_advantage_stream_target = pa.copy()
            self.sesgo_value_stream = np.zeros((1,1),dtype=LUCIA_RL_CONFIG['precision']); self.sesgo_advantage_stream = np.zeros((1,self.output_size),dtype=LUCIA_RL_CONFIG['precision'])
            self.sesgo_value_stream_target = self.sesgo_value_stream.copy(); self.sesgo_advantage_stream_target = self.sesgo_advantage_stream.copy()
        if self.usar_distributional:
            pd = inicializar_pesos_he((self.input_size,self.output_size*self.n_atoms),self.input_size) if self.usar_he else inicializar_pesos_xavier((self.input_size,self.output_size*self.n_atoms),self.input_size,self.output_size*self.n_atoms)
            self.pesos_distributional = pd; self.sesgo_distributional = np.zeros((1,self.output_size*self.n_atoms),dtype=LUCIA_RL_CONFIG['precision'])
            self.pesos_distributional_target = pd.copy(); self.sesgo_distributional_target = self.sesgo_distributional.copy()
        if self.usar_noisy:
            pn = inicializar_pesos_he((self.input_size,self.output_size),self.input_size) if self.usar_he else inicializar_pesos_xavier((self.input_size,self.output_size),self.input_size,self.output_size)
            self.pesos_noisy = pn; self.sesgo_noisy = np.zeros((1,self.output_size),dtype=LUCIA_RL_CONFIG['precision'])
            self.pesos_noisy_target = pn.copy(); self.sesgo_noisy_target = self.sesgo_noisy.copy()

    def _prep(self, estado: np.ndarray) -> np.ndarray:
        x = np.asarray(estado, dtype=_dt())
        if x.ndim == 1: x = x.reshape(1, -1)
        return np.ascontiguousarray(x, dtype=_dt())

    @staticmethod
    def _sigma(w: np.ndarray, iters: int = 3) -> float:
        wf = w.astype(np.float64); v = np.random.default_rng(0).normal(0, 1, (wf.shape[1], 1))
        v /= max(1e-12, np.linalg.norm(v))
        for _ in range(iters):
            u = wf @ v; u /= max(1e-12, np.linalg.norm(u)); v = wf.T @ u; v /= max(1e-12, np.linalg.norm(v))
        return float((u.T @ wf @ v)[0, 0])

    def forward(self, estado: np.ndarray, usar_objetivo: bool = False) -> np.ndarray:
        if self.pesos_q_principal is None: self.inicializar_pesos()
        wp = self.pesos_q_objetivo if usar_objetivo else self.pesos_q_principal
        sb = self.sesgo_q_objetivo if usar_objetivo else self.sesgo_q_principal
        x = self._prep(estado); q = x @ wp + sb
        if self.usar_dueling: q = self._dueling(x, q, usar_objetivo)
        if self.usar_distributional: q = self._distrib(x, q, usar_objetivo)
        if self.usar_noisy: q = self._aplicar_noisy(x, q, usar_objetivo)
        self.historial_q_values.append(q.copy()); return q

    def forward_q(self, estado: np.ndarray, accion: np.ndarray, usar_objetivo: bool = False, q_network: int = 1) -> np.ndarray:
        q = self.forward(estado, usar_objetivo)
        if q_network == 1: w = self.pesos_q_principal; b = self.sesgo_q_principal
        else: w = self.pesos_q_objetivo; b = self.sesgo_q_objetivo
        entrada = np.concatenate([self._prep(estado), accion.reshape(-1,1) if accion.ndim == 1 else accion], axis=1) if False else self._prep(estado)
        return entrada @ w + b

    def forward_batch(self, estados) -> Tuple[np.ndarray, np.ndarray]:
        if self.pesos_q_principal is None: self.inicializar_pesos()
        xs = np.concatenate([self._prep(s) for s in estados], axis=0)
        logits = xs @ self.pesos_q_principal + self.sesgo_q_principal
        return np.tanh(logits).astype(_dt()), logits.astype(_dt())

    def _dueling(self, estado, q, uo):
        pv = self.pesos_value_stream_target if uo else self.pesos_value_stream; sbv = self.sesgo_value_stream_target if uo else self.sesgo_value_stream
        pa = self.pesos_advantage_stream_target if uo else self.pesos_advantage_stream; sba = self.sesgo_advantage_stream_target if uo else self.sesgo_advantage_stream
        vs = estado @ pv + sbv; as_ = estado @ pa + sba
        return vs + as_ - np.mean(as_, axis=1, keepdims=True)

    def _distrib(self, estado, q, uo):
        pd = self.pesos_distributional_target if uo else self.pesos_distributional; sbd = self.sesgo_distributional_target if uo else self.sesgo_distributional
        l = (estado @ pd + sbd).reshape(-1, self.output_size, self.n_atoms)
        e = np.exp(l - np.max(l, axis=2, keepdims=True)); p = e/np.sum(e, axis=2, keepdims=True)
        atoms = np.linspace(self.v_min, self.v_max, self.n_atoms, dtype=_dt())
        return np.sum(p * atoms, axis=2)

    def _calc_distrib_probs(self, estado, uo):
        pd = self.pesos_distributional_target if uo else self.pesos_distributional; sbd = self.sesgo_distributional_target if uo else self.sesgo_distributional
        l = (estado @ pd + sbd).reshape(-1, self.output_size, self.n_atoms)
        e = np.exp(l - np.max(l, axis=2, keepdims=True)); return e/np.sum(e, axis=2, keepdims=True)

    def _aplicar_noisy(self, estado, q, uo):
        pn = self.pesos_noisy_target if uo else self.pesos_noisy
        noise_factor = 0.1; noise = np.random.normal(0, noise_factor, q.shape)
        self.estadisticas_rainbow['noise_factor'] = noise_factor; return q + noise

    def _aplicar_epsilon_decay(self) -> None:
        if not self.usar_noisy and self.epsilon > self.epsilon_min: self.epsilon *= self.epsilon_decay

    def _actualizar_beta(self) -> None:
        if self.usar_prioritized: self.beta_schedule = min(1.0, self.beta_schedule + 0.001)

    def calcular_td_error(self, estado, accion, recompensa, siguiente_estado, terminado) -> float:
        q_actual = float(self.forward(estado)[0, accion])
        if terminado: q_objetivo = float(recompensa)
        else: q_objetivo = float(recompensa + self.gamma * np.max(self.forward(siguiente_estado, True)[0]))
        return q_objetivo - q_actual

    def calcular_prioridad_batch(self, batch) -> np.ndarray:
        estados = np.array([e['estado'] for e in batch], dtype=_dt())
        acciones = np.array([e['accion'] for e in batch])
        recompensas = np.array([e['recompensa'] for e in batch])
        terminados = np.array([e['terminado'] for e in batch])
        siguientes = np.array([e['siguiente_estado'] for e in batch], dtype=_dt())
        q_actual = self.forward(estados)[np.arange(len(batch)), acciones]
        q_sig = self.forward(siguientes, True)
        q_objetivo = np.where(terminados, recompensas, recompensas + self.gamma * np.max(q_sig, axis=1))
        td_errors = q_objetivo - q_actual
        return (np.abs(td_errors) + 1e-6) ** self.alpha_prioritized

    def seleccionar_accion(self, estado: np.ndarray) -> int:
        if self.usar_noisy: return int(np.argmax(self.forward(estado)[0]))
        if np.random.random() < self.epsilon: self.estadisticas_rainbow['exploracion_rate'] += 1; return int(np.random.randint(0, self.output_size))
        self.estadisticas_rainbow['exploitacion_rate'] += 1; return int(np.argmax(self.forward(estado)[0]))

    def obtener_peso_exploracion(self) -> float: return float(self.epsilon)

    def agregar_experiencia(self, estado, accion, recompensa, siguiente_estado, terminado) -> None:
        exp = {'estado':np.asarray(estado),'accion':accion,'recompensa':recompensa,'siguiente_estado':np.asarray(siguiente_estado),'terminado':terminado}
        self.buffer_experiencia.append(exp); self.estadisticas_rainbow['muestras_buffer'] = len(self.buffer_experiencia)
        if self.usar_prioritized:
            pr = self._prioridad(exp); self.prioridades.append(pr); self.estadisticas_rainbow['prioridad_media'] = float(np.mean(list(self.prioridades)))

    def _prioridad(self, exp): return float((abs(self.forward(exp['estado'])[0,exp['accion']] - (exp['recompensa'] + self.gamma*np.max(self.forward(exp['siguiente_estado'],True)[0])) + 1e-6) ** self.alpha_prioritized))

    def _preparar_minibatch(self, batch):
        e = batch['experiencias'] if isinstance(batch, dict) else batch
        st = np.array([ex['estado'] for ex in e], dtype=_dt())
        ac = np.array([ex['accion'] for ex in e], dtype=_dt())
        rw = np.array([ex['recompensa'] for ex in e], dtype=np.float32).reshape(-1,1)
        tn = np.array([float(ex['terminado']) for ex in e], dtype=np.float32).reshape(-1,1)
        ns = np.array([ex['siguiente_estado'] for ex in e], dtype=_dt())
        return st, ac, rw, ns, tn

    def muestrear_batch(self) -> Optional[List[Dict[str, Any]]]:
        if len(self.buffer_experiencia) < self.batch_size: return None
        if self.usar_prioritized:
            pa = np.array(list(self.prioridades)); pb = pa**self.alpha_prioritized; pb /= np.sum(pb)
            idx = np.random.choice(len(self.buffer_experiencia),size=self.batch_size,replace=False,p=pb)
            return [list(self.buffer_experiencia)[i] for i in idx]
        return random.sample(list(self.buffer_experiencia), self.batch_size)

    def calcular_loss(self, batch: List[Dict[str, Any]]) -> Dict[str, float]:
        gw, gs = self.calcular_gradientes_rainbow(batch); q1l = float(np.mean(gw**2)); q2l = float(np.mean(gs**2))
        return {'policy_loss': q2l, 'value_loss': q1l, 'total_loss': float(abs(q1l)+abs(q2l))}

    def calcular_entropy(self, prob: np.ndarray) -> float:
        p = np.abs(prob.astype(np.float64)); p = p / max(1e-12, p.sum())
        return float(-np.sum(p * np.log(p + 1e-8)))

    def calcular_critico_valor(self, batch) -> float:
        q_vals = np.array([float(self.forward(e['estado'])[0, e['accion']]) for e in batch])
        return float(np.mean(q_vals))

    def calcular_distribucion_loss(self, batch) -> float:
        probs = self._calc_distrib_probs(np.array([e['estado'] for e in batch], dtype=_dt()), False)
        target = np.zeros_like(probs); target[np.arange(len(batch)), [e['accion'] for e in batch], :] = 1.0
        return float(-np.mean(np.sum(target * np.log(probs + 1e-8), axis=2)))

    def calcular_explained_variance(self, batch) -> float:
        preds = np.array([float(self.forward(e['estado'])[0, e['accion']]) for e in batch])
        rets = np.array([e['recompensa'] for e in batch], dtype=np.float64)
        var = float(np.var(rets)); resid = float(np.var(rets - preds))
        return 1.0 - resid/max(1e-12, var)

    def calcular_q_target(self, estado, accion, recompensa, siguiente_estado, terminado) -> float:
        if terminado: return float(recompensa)
        return float(recompensa + self.gamma * np.max(self.forward(siguiente_estado, True)[0]))

    def _dueling_extended(self, estado, usar_objetivo=False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        vs = self.forward(estado, usar_objetivo) if self.usar_dueling else np.zeros((1, self.output_size), dtype=_dt())
        v = estado @ (self.pesos_value_stream_target if usar_objetivo else self.pesos_value_stream) + (self.sesgo_value_stream_target if usar_objetivo else self.sesgo_value_stream)
        a = estado @ (self.pesos_advantage_stream_target if usar_objetivo else self.pesos_advantage_stream) + (self.sesgo_advantage_stream_target if usar_objetivo else self.sesgo_advantage_stream)
        q = v + a - np.mean(a, axis=1, keepdims=True) if self.usar_dueling else vs
        return q, v, a

    def calcular_policy_loss(self, batch) -> float:
        actions = np.array([e['accion'] for e in batch])
        q_vals = self.forward(np.array([e['estado'] for e in batch], dtype=_dt()))
        return float(-np.mean(q_vals[np.arange(len(batch)), actions]))

    def calcular_value_loss(self, batch) -> float:
        recompensas = np.array([e['recompensa'] for e in batch])
        q_targets = np.array([self.calcular_q_target(e['estado'],e['accion'],e['recompensa'],e['siguiente_estado'],e['terminado']) for e in batch])
        return float(np.mean((recompensas - q_targets)**2))

    def obtener_configuracion_actual(self) -> Dict[str, Any]:
        return {'learning_rate':self.learning_rate,'gamma':self.gamma,'epsilon':self.epsilon,'tau':self.tau,'batch_size':self.batch_size,'usar_dueling':self.usar_dueling,'usar_prioritized':self.usar_prioritized,'usar_distributional':self.usar_distributional,'usar_noisy':self.usar_noisy,'n_atoms':self.n_atoms,'alpha_prioritized':self.alpha_prioritized,'beta_schedule':self.beta_schedule}

    def verificar_integridad(self) -> Dict[str, Any]:
        checks = {'pesos_q_init': self.pesos_q_principal is not None, 'pesos_qt_init': self.pesos_q_objetivo is not None}
        for nm, p in [('q1', self.pesos_q_principal), ('q2', self.pesos_q_objetivo)]:
            if p is not None:
                checks[f'sin_nan_{nm}'] = not bool(np.any(np.isnan(p))); checks[f'sin_inf_{nm}'] = not bool(np.any(np.isinf(p)))
                checks[f'peso_{nm}_max'] = float(np.max(np.abs(p)))
        checks['buffer_tamano'] = len(self.buffer_experiencia); checks['prioridades_tamano'] = len(self.prioridades) if self.prioridades else 0
        checks['beta_schedule'] = self.beta_schedule; checks['epsilon'] = self.epsilon
        return checks

    def resetear_prioridades(self) -> None:
        if self.usar_prioritized and self.prioridades is not None: self.prioridades.clear()

    def obtener_ratio_exploracion(self) -> float:
        ta = self.estadisticas_rainbow['exploracion_rate'] + self.estadisticas_rainbow['exploitacion_rate']
        return float(self.estadisticas_rainbow['exploracion_rate']/ta) if ta > 0 else 0.0

    def verificar_estabilidad(self) -> Dict[str, bool]:
        e = {}; e['convergencia_ok'] = self.estadisticas_rainbow['convergencia_q'] > 0.8
        e['q_values_estables'] = self.calcular_estabilidad_q() > 0.7
        e['epsilon_apropiado'] = True if self.usar_noisy else (self.epsilon_min <= self.epsilon <= 1.0)
        e['buffer_suficiente'] = len(self.buffer_experiencia) >= self.batch_size
        if self.pesos_q_principal is not None:
            pm = np.max(np.abs(self.pesos_q_principal)); pn = np.min(np.abs(self.pesos_q_principal))
            e['pesos_no_explosivos'] = pm < 10.0; e['pesos_no_desaparecen'] = pn > 1e-6; e['pesos_balanceados'] = pm/max(1e-8,pn) < 1000.0
        else: e.update({'pesos_no_explosivos':True,'pesos_no_desaparecen':True,'pesos_balanceados':True})
        e['gradientes_no_explosivos'] = self.estadisticas_rainbow['gradiente_norma'] < 10.0
        e['gradientes_no_desaparecen'] = self.estadisticas_rainbow['gradiente_norma'] > 1e-8
        return e

    def calcular_estabilidad_q(self) -> float:
        if len(self.historial_q_values) < 20: return 0.0
        vals = np.array([float(np.mean(np.ravel(qv))) for qv in self.historial_q_values[-20:]])
        return float(1.0/(1.0+np.var(vals)))

    def entrenar_step(self) -> Optional[float]:
        batch = self.muestrear_batch()
        if batch is None: return None
        gw, gs = self.calcular_gradientes_rainbow(batch); self.actualizar_pesos(gw, gs)
        return float(np.mean(self.historial_losses[-len(batch):]))

    def calcular_gradientes_rainbow(self, batch: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
        if self.pesos_q_principal is None: raise ValueError("Pesos no inicializados")
        estados = np.array([e['estado'] for e in batch], dtype=_dt()).reshape(-1, self.input_size)
        acciones = np.array([e['accion'] for e in batch])
        recompensas = np.array([e['recompensa'] for e in batch])
        terminados = np.array([e['terminado'] for e in batch])
        siguientes = np.array([e['siguiente_estado'] for e in batch], dtype=_dt()).reshape(-1, self.input_size)
        q_actual = self.forward(estados)[np.arange(len(batch)), acciones]
        q_sig = self.forward(siguientes, True)
        q_objetivo = np.where(terminados, recompensas, recompensas + self.gamma * np.max(q_sig, axis=1))
        td_errors = q_objetivo - q_actual
        grad_acc = np.zeros((len(batch), self.output_size), dtype=_dt())
        grad_acc[np.arange(len(batch)), acciones] = td_errors
        gw = estados.T @ grad_acc / len(batch); gs = np.mean(grad_acc, axis=0)
        loss = float(np.mean(0.5 * td_errors**2))
        self.historial_losses.append(loss); self.estadisticas_rainbow['gradiente_norma'] = float(np.linalg.norm(gw))
        return gw, gs

    def actualizar_pesos(self, gw, gs) -> None:
        if self.pesos_q_principal is None: raise ValueError("Pesos no inicializados")
        self.pesos_q_principal += self.learning_rate * gw; self.sesgo_q_principal += self.learning_rate * gs
        self._actualizar_red_objetivo(); self._aplicar_epsilon_decay(); self._convergencia()

    def _actualizar_red_objetivo(self) -> None:
        if self.pesos_q_objetivo is None: return
        t = self.tau; self.pesos_q_objetivo = (1-t)*self.pesos_q_objetivo + t*self.pesos_q_principal; self.sesgo_q_objetivo = (1-t)*self.sesgo_q_objetivo + t*self.sesgo_q_principal
        if self.usar_dueling:
            for at,ap in [(self.pesos_value_stream_target,self.pesos_value_stream),(self.pesos_advantage_stream_target,self.pesos_advantage_stream)]: at[:] = (1-t)*at + t*ap
            for at,ap in [(self.sesgo_value_stream_target,self.sesgo_value_stream),(self.sesgo_advantage_stream_target,self.sesgo_advantage_stream)]: at[:] = (1-t)*at + t*ap
        if self.usar_distributional: self.pesos_distributional_target[:] = (1-t)*self.pesos_distributional_target + t*self.pesos_distributional; self.sesgo_distributional_target[:] = (1-t)*self.sesgo_distributional_target + t*self.sesgo_distributional
        if self.usar_noisy: self.pesos_noisy_target[:] = (1-t)*self.pesos_noisy_target + t*self.pesos_noisy; self.sesgo_noisy_target[:] = (1-t)*self.sesgo_noisy_target + t*self.sesgo_noisy
        self.estadisticas_rainbow['actualizaciones_target'] += 1

    def _convergencia(self) -> None:
        if len(self.historial_td_errors) >= 10: self.estadisticas_rainbow['convergencia_q'] = float(1.0/(1.0+np.var(self.historial_td_errors[-10:])))

    def _calcular_convergencia_rainbow(self) -> None: self._convergencia()

    def limpiar_historiales(self) -> None:
        self.historial_td_errors.clear(); self.historial_losses.clear(); self.historial_q_values.clear()
        self.historial_epsilon.clear(); self.historial_prioridades.clear(); self.historial_kl_divergences.clear()

    def guardar(self, ruta: str) -> None:
        import json
        d = {'input_size':self.input_size,'output_size':self.output_size,'nombre':self.nombre,
            'pesos_q_principal':self.pesos_q_principal.tolist() if self.pesos_q_principal is not None else None,
            'pesos_q_objetivo':self.pesos_q_objetivo.tolist() if self.pesos_q_objetivo is not None else None,
            'pesos_value_stream':self.pesos_value_stream.tolist() if self.pesos_value_stream is not None else None,
            'pesos_advantage_stream':self.pesos_advantage_stream.tolist() if self.pesos_advantage_stream is not None else None,
            'pesos_distributional':self.pesos_distributional.tolist() if self.pesos_distributional is not None else None,
            'pesos_noisy':self.pesos_noisy.tolist() if self.pesos_noisy is not None else None,
            'learning_rate':self.learning_rate,'gamma':self.gamma,'epsilon':self.epsilon,'tau':self.tau,
            'batch_size':self.batch_size,'buffer_size':len(self.buffer_experiencia)}
        with open(ruta, 'w', encoding='utf-8') as f: json.dump(d, f)

    def cargar(self, ruta: str) -> None:
        import json
        with open(ruta, 'r', encoding='utf-8') as f: d = json.load(f)
        self.inicializar_pesos()
        for k in ['pesos_q_principal','pesos_q_objetivo']:
            if d.get(k) is not None: setattr(self, k, np.asarray(d[k], dtype=_dt()))

    def obtener_estadisticas_rainbow(self) -> Dict[str, Any]:
        if self.pesos_q_principal is None: return {'estado':'no_inicializada'}
        if self.historial_td_errors: a = np.array(self.historial_td_errors[-100:]); self.estadisticas_rainbow['td_error_medio'] = float(np.mean(a)); self.estadisticas_rainbow['td_error_std'] = float(np.std(a))
        if self.historial_losses: self.estadisticas_rainbow['loss_medio'] = float(np.mean(self.historial_losses[-10:]))
        ta = self.estadisticas_rainbow['exploracion_rate'] + self.estadisticas_rainbow['exploitacion_rate']
        self.estadisticas_rainbow['tasa_exploracion'] = float(self.estadisticas_rainbow['exploracion_rate']/ta) if ta > 0 else 0.0
        self.estadisticas_rainbow['tasa_exploitacion'] = float(self.estadisticas_rainbow['exploitacion_rate']/ta) if ta > 0 else 0.0
        self.estadisticas_rainbow['estabilidad_q'] = float(self.calcular_estabilidad_q())
        return {'learning_rate':self.learning_rate,'gamma':self.gamma,'epsilon_actual':self.epsilon,'tau':self.tau,'batch_size':self.batch_size,'buffer_size':len(self.buffer_experiencia),'convergencia_q':self.estadisticas_rainbow['convergencia_q'],'estabilidad_q':self.estadisticas_rainbow['estabilidad_q'],'actualizaciones_target':self.estadisticas_rainbow['actualizaciones_target'],'muestras_buffer':self.estadisticas_rainbow['muestras_buffer'],'td_error_medio':self.estadisticas_rainbow['td_error_medio'],'td_error_std':self.estadisticas_rainbow['td_error_std'],'loss_medio':self.estadisticas_rainbow['loss_medio'],'gradiente_norma':self.estadisticas_rainbow['gradiente_norma'],'tasa_exploracion':self.estadisticas_rainbow['tasa_exploracion'],'tasa_exploitacion':self.estadisticas_rainbow['tasa_exploitacion'],'prioridad_media':self.estadisticas_rainbow['prioridad_media'],'noise_factor':self.estadisticas_rainbow['noise_factor'],'usar_dueling':self.usar_dueling,'usar_prioritized':self.usar_prioritized,'usar_multi_step':self.usar_multi_step,'usar_distributional':self.usar_distributional,'usar_noisy':self.usar_noisy}

    def obtener_estadisticas_rainbow_extended(self) -> Dict[str, Any]:
        s = self.obtener_estadisticas_rainbow(); s.update({'sigma_actor':float(self._sigma(self.pesos_q_principal)) if self.pesos_q_principal is not None else 0.0,'exploracion_ratio':self.obtener_ratio_exploracion(),'beta_schedule':self.beta_schedule,'entropy':self.calcular_entropy(self.forward(np.zeros((1,self.input_size),dtype=_dt()))),'verificar_integridad':self.verificar_integridad()})
        return s

    def reinicializar_con_parametros(self, learning_rate=None, gamma=None, epsilon=None, tau=None) -> None:
        if learning_rate is not None: self.learning_rate = learning_rate
        if gamma is not None: self.gamma = gamma
        if epsilon is not None: self.epsilon = epsilon
        if tau is not None: self.tau = tau
        self.inicializar_pesos(); self.resetear_historial(); self.buffer_experiencia.clear()
        if self.usar_prioritized: self.prioridades.clear()
        self.limpiar_historiales()
        self.estadisticas_rainbow['actualizaciones_target'] = 0; self.estadisticas_rainbow['muestras_buffer'] = 0; self.estadisticas_rainbow['exploracion_rate'] = 0; self.estadisticas_rainbow['exploitacion_rate'] = 0

    def __str__(self): return f"NeuronaRefuerzoRainbowDQN(entrada={self.input_size},salida={self.output_size},lr={self.learning_rate},gamma={self.gamma},epsilon={self.epsilon:.3f},buffer={len(self.buffer_experiencia)})"
    def __repr__(self): return self.__str__()

def crear_neurona_rainbow_dqn(input_size: int, output_size: int, configuracion: Dict[str, Any] = None) -> NeuronaRefuerzoRainbowDQN:
    c = configuracion or {}
    return NeuronaRefuerzoRainbowDQN(input_size=input_size,output_size=output_size,nombre=c.get('nombre','NeuronaRefuerzoRainbowDQN'),learning_rate=c.get('learning_rate',0.00025),gamma=c.get('gamma',0.99),epsilon=c.get('epsilon',0.1),epsilon_decay=c.get('epsilon_decay',0.995),epsilon_min=c.get('epsilon_min',0.01),tau=c.get('tau',0.005),buffer_size=c.get('buffer_size',1000000),batch_size=c.get('batch_size',32),usar_he=c.get('usar_he',True),usar_dueling=c.get('usar_dueling',True),usar_prioritized=c.get('usar_prioritized',True),usar_multi_step=c.get('usar_multi_step',True),usar_distributional=c.get('usar_distributional',True),usar_noisy=c.get('usar_noisy',True),n_atoms=c.get('n_atoms',51),v_min=c.get('v_min',-10.0),v_max=c.get('v_max',10.0),n_steps=c.get('n_steps',3),alpha_prioritized=c.get('alpha_prioritized',0.6),beta_prioritized=c.get('beta_prioritized',0.4))

RFEN9_CONFIG = {'inicializacion_preferida':'he','learning_rate_default':0.00025,'gamma_default':0.99,'epsilon_default':0.1,'epsilon_decay_default':0.995,'epsilon_min_default':0.01,'tau_default':0.005,'buffer_size_default':1000000,'batch_size_default':32,'usar_dueling_default':True,'usar_prioritized_default':True,'usar_multi_step_default':True,'usar_distributional_default':True,'usar_noisy_default':True,'n_atoms_default':51,'v_min_default':-10.0,'v_max_default':10.0,'n_steps_default':3,'alpha_prioritized_default':0.6,'beta_prioritized_default':0.4,'umbral_convergencia':0.8,'umbral_estabilidad':0.7}
logger.info("RFEN1_RN_9.py cargado - Rainbow DQN Multi-Método")
