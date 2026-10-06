import numpy as np, math, logging, time, hashlib
from typing import Dict, Any, List, Optional, Tuple, Callable
from collections import deque
import random
import pickle
import io

from .RFEN1_RN_10 import NeuronaRefuerzoIMPALA
from .RFEN1_RN_9 import NeuronaRefuerzoRainbowDQN
from .RFEN1_RN_8 import NeuronaRefuerzoTD3
from .RFEN1_RN_7 import NeuronaRefuerzoSAC
from .RFEN1_RN_6 import NeuronaRefuerzoPPO
from .RFEN1_RN_5 import NeuronaRefuerzoA3C
from .RFEN1_RN_4 import NeuronaRefuerzoDQN
from .RFEN1_RN_3 import NeuronaRefuerzoActorCritic
from .RFEN1_RN_2 import NeuronaRefuerzoPolicyGradient
from .RFEN1_RN_1 import NeuronaRefuerzoQLearning

LUCIA_RL_CONFIG = {'precision': 'float32', 'random_seed': 42, 'default_learning_rate': 0.001, 'cache_max_size': 1024, 'profiler_enabled': False}

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
def inicializar_pesos_eficiente(shape, method: str = 'he', fan_in: int = 1) -> np.ndarray:
    metodos = {'he': inicializar_pesos_he, 'xavier': inicializar_pesos_xavier, 'lecun': inicializar_pesos_lecun, 'ortogonal': inicializar_pesos_ortogonal, 'espectral': inicializar_pesos_espectral}
    fn = metodos.get(method, inicializar_pesos_he)
    return fn(shape, fan_in) if method != 'ortogonal' and method != 'espectral' else fn(shape)

class NeuronaRefuerzoBase:
    def __init__(self, input_size, output_size, nombre="NeuronaRefuerzo"):
        self.input_size = int(input_size); self.output_size = int(output_size); self.nombre = str(nombre)
        self.pesos = None; self.sesgo = None; self.historial_activaciones = []; self.historial_gradientes = []; self.pasos = 0
    def inicializar_pesos(self): raise NotImplementedError
    def forward(self, e): raise NotImplementedError
    def resetear_historial(self): self.historial_activaciones = []; self.historial_gradientes = []
    def info(self) -> str: return f"{self.nombre}(in={self.input_size},out={self.output_size},pasos={self.pasos})"
    def params_count(self) -> int:
        total = 0
        if self.pesos is not None: total += self.pesos.size
        if self.sesgo is not None: total += self.sesgo.size
        return total
    def add_historial_activacion(self, activacion: np.ndarray) -> None: self.historial_activaciones.append(activacion.copy())
    def add_historial_gradiente(self, gradiente: Dict[str, np.ndarray]) -> None: self.historial_gradientes.append({k: v.copy() for k, v in gradiente.items()})

def calcular_recompensa_descontada(recompensas: List[float], gamma: float) -> np.ndarray:
    n = len(recompensas); discounted = np.zeros(n, dtype=np.float64); acum = 0.0
    for t in range(n - 1, -1, -1): acum = recompensas[t] + gamma * acum; discounted[t] = acum
    return discounted
def normalizar_recompensas(recompensas: np.ndarray) -> np.ndarray:
    media = np.mean(recompensas); std = np.std(recompensas)
    return (recompensas - media) / max(1e-8, std)
def calcular_gae(recompensas: List[float], valores: List[float], gamma: float = 0.99, lambda_gae: float = 0.95) -> np.ndarray:
    rews = np.array(recompensas, dtype=np.float64); vals = np.array(valores, dtype=np.float64); n = len(rews)
    advantages = np.zeros(n, dtype=np.float64); adv = 0.0
    for t in range(n - 1, -1, -1):
        nv = 0.0 if t == n - 1 else vals[t + 1]
        delta = rews[t] + gamma * nv - vals[t]; adv = delta + gamma * lambda_gae * adv; advantages[t] = adv
    return advantages
def aplicar_clip_gradientes(gradiente: np.ndarray, max_norm: float = 0.5) -> np.ndarray:
    norma = np.linalg.norm(gradiente)
    if norma > max_norm: return gradiente * max_norm / (norma + 1e-8)
    return gradiente
def crear_buffer_experiencia(capacidad: int = 10000) -> deque: return deque(maxlen=capacidad)
def muestrear_buffer(buffer: deque, tamano: int) -> List[Any]: return random.sample(list(buffer), min(tamano, len(buffer)))
def calcular_entropy(prob: np.ndarray) -> float:
    p = np.abs(prob); p = p / max(1e-12, p.sum()); return float(-np.sum(p * np.log(p + 1e-8)))
def calcular_loss(y_pred: np.ndarray, y_target: np.ndarray) -> float: return float(np.mean((y_pred - y_target)**2))
def crear_buffer_priorizado(capacidad: int = 10000, alpha: float = 0.6) -> Dict: return {'buffer': deque(maxlen=capacidad), 'prioridades': deque(maxlen=capacidad), 'alpha': alpha}
def muestrear_priorizado(buffer: Dict, tamano: int, beta: float = 0.4) -> List[Any]:
    buf = buffer['buffer']; pri = buffer['prioridades']
    if len(buf) < tamano: return []
    pa = np.array(list(pri)); pb = pa ** buffer['alpha']; pb /= np.sum(pb)
    idx = np.random.choice(len(buf), size=tamano, replace=False, p=pb)
    return [buf[i] for i in idx], idx, pb[idx]
def actualizar_prioridades(buffer: Dict, indices: List[int], errores: np.ndarray, beta: float = 0.4) -> None:
    for idx, err in zip(indices, errores):
        if idx < len(buffer['prioridades']): buffer['prioridades'][idx] = (abs(err) + 1e-6) ** beta
def compute_returns(recompensas: List[float], gamma: float) -> np.ndarray: return calcular_recompensa_descontada(recompensas, gamma)
def discount_cumulative_returns(recompensas: List[float], gamma: float) -> np.ndarray: return calcular_recompensa_descontada(recompensas, gamma)
def huber_loss(error: np.ndarray, delta: float = 1.0) -> float:
    abs_err = np.abs(error); quadratic = np.minimum(abs_err, delta); linear = abs_err - quadratic
    return float(np.mean(0.5 * quadratic**2 + delta * linear))
def mean_std_normalize(data: np.ndarray) -> Tuple[np.ndarray, float, float]:
    media = float(np.mean(data)); std = float(np.std(data)); return (data - media) / max(1e-8, std), media, std
def sample_gumbel_softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    gumbel = -np.log(-np.log(np.random.uniform(1e-8, 1.0, size=logits.shape)))
    y = logits + gumbel; y /= max(1e-12, temperature)
    e = np.exp(y - np.max(y, axis=-1, keepdims=True)); return e / np.sum(e, axis=-1, keepdims=True)
def layer_init(layer: np.ndarray, method: str = 'xavier', fan_in: int = 1) -> np.ndarray:
    fn = {'he': inicializar_pesos_he, 'xavier': inicializar_pesos_xavier, 'lecun': inicializar_pesos_lecun, 'ortogonal': inicializar_pesos_ortogonal}
    return fn.get(method, inicializar_pesos_xavier)(layer.shape if hasattr(layer, 'shape') else (fan_in, layer), fan_in)
def count_parameters(model) -> int:
    total = 0
    for attr in dir(model):
        if attr.startswith('pesos') or attr.startswith('sesgo'):
            val = getattr(model, attr)
            if isinstance(val, np.ndarray) and val.size > 0: total += val.size
    return total
def get_optimizer(name: str, params, lr: float = 0.001):
    optimizers = {'sgd': 'SGD', 'adam': 'Adam', 'rmsprop': 'RMSprop', 'adagrad': 'Adagrad'}
    return optimizers.get(name.lower(), 'Adam')
def lr_scheduler(step: int, initial_lr: float, decay: float = 0.99) -> float: return initial_lr * (decay ** step)
def early_stopping_check(historial: List[float], paciencia: int = 20, umbral: float = 1e-4) -> bool:
    if len(historial) < paciencia + 5: return False
    reciente = historial[-paciencia:]; mejor = min(reciente); mejor_idx = reciente.index(mejor)
    if paciencia - mejor_idx > paciencia * 0.8: return True
    var = np.var(reciente); return var < umbral
def neuron_info(neurona) -> Dict[str, Any]:
    return {'nombre': neurona.nombre, 'input_size': neurona.input_size, 'output_size': neurona.output_size,
            'pasos': neurona.pasos, 'params': count_parameters(neurona), 'historial_len': len(neurona.historial_gradientes)}
def train_epoch(neurona, dataset: List, batch_size: int = 32, max_norm: float = 0.5) -> float:
    random.shuffle(dataset); total_loss = 0.0; n_batches = 0
    for i in range(0, len(dataset), batch_size):
        batch = dataset[i:i+batch_size]; result = train_batch_neuron(neurona, batch, max_norm)
        if result is not None: total_loss += result; n_batches += 1
    return total_loss / max(1, n_batches)
def parallel_train(neuronas: List, dataset: List, batch_size: int = 32) -> List[float]: return [train_epoch(n, dataset, batch_size) for n in neuronas]
def train_batch_neuron(neurona, batch: List[Dict[str, Any]], max_norm: float = 0.5) -> Optional[float]:
    if hasattr(neurona, 'calcular_gradientes_rainbow'):
        gw, gs = neurona.calcular_gradientes_rainbow(batch); gw = aplicar_clip_gradientes(gw, max_norm); gs = aplicar_clip_gradientes(gs, max_norm); neurona.actualizar_pesos(gw, gs)
    elif hasattr(neurona, 'calcular_gradientes_impala'):
        gw = neurona.calcular_gradientes_impala(batch, 0); neurona.actualizar_pesos_learner(*gw)
    else: return None
    return neurona.estadisticas_td3.get('loss_medio', 0.0) if hasattr(neurona, 'estadisticas_td3') else 0.0
def evaluate_neurona(neurona, estados_test, acciones_test, recompensas_test) -> Dict[str, float]:
    if not hasattr(neurona, 'pesos_q_principal') and not hasattr(neurona, 'pesos_actor') and not hasattr(neurona, 'pesos_actor_global'):
        neurona.inicializar_pesos()
    preds = neurona.forward(estados_test)
    if isinstance(preds, tuple): preds = preds[0]
    preds_2d = np.atleast_2d(preds) if preds.ndim == 1 else preds
    if preds_2d.shape[1] > 1:
        acc = float(np.mean(np.argmax(preds_2d, axis=1) == acciones_test))
        mse = float(np.mean((preds_2d[np.arange(len(acciones_test)), acciones_test] - recompensas_test)**2))
    else:
        acc = 0.0; mse = float(np.mean((preds_2d.flatten() - recompensas_test)**2))
    return {'accuracy': acc, 'mse': mse, 'n_muestras': len(estados_test)}
def analizar_convergencia(historial: List[float], ventana: int = 50) -> Dict[str, float]:
    if len(historial) < 2: return {'convergencia': 0.0, 'tendencia': 0.0, 'varianza': 0.0}
    reciente = np.array(historial[-ventana:]); var = float(np.var(reciente))
    if len(reciente) >= 10: pendiente = float(np.polyfit(range(len(reciente)), reciente, 1)[0])
    else: pendiente = 0.0
    return {'convergencia': float(1.0/(1.0+var)), 'tendencia': pendiente, 'varianza': var, 'media': float(np.mean(reciente)), 'min': float(np.min(reciente)), 'max': float(np.max(reciente))}
def comparar_inicializaciones(input_size: int, output_size: int, n_pruebas: int = 100) -> Dict[str, float]:
    resultados = {}
    for nombre, fn in [('he', inicializar_pesos_he), ('xavier', inicializar_pesos_xavier), ('lecun', inicializar_pesos_lecun), ('ortogonal', inicializar_pesos_ortogonal)]:
        pesos = [fn((input_size, output_size)) for _ in range(n_pruebas)]
        medias = [float(np.mean(np.abs(p))) for p in pesos]
        resultado = {'media': float(np.mean(medias)), 'std': float(np.std(medias)), 'max': float(np.max(medias))}
        resultados[nombre] = resultado
    return resultados

class _Ring:
    def __init__(self, cap: int): self.cap = cap; self._d: List[Any] = []
    def append(self, x) -> None:
        self._d.append(x)
        if len(self._d) > self.cap: del self._d[0:len(self._d)-self.cap]
    def __len__(self) -> int: return len(self._d)
    def __getitem__(self, i: int) -> Any: return self._d[i]
    def __iter__(self): return iter(self._d)
    def clear(self) -> None: self._d.clear()
    @property
    def full(self) -> bool: return len(self._d) == self.cap

class _RingBufferNumpy:
    def __init__(self, cap: int, dtype: np.dtype = np.float32):
        self.cap = cap; self._pos = 0; self._count = 0
        self._data = np.zeros((cap,), dtype=object); self._dtype = dtype
    def append(self, x) -> None:
        self._data[self._pos] = x; self._pos = (self._pos + 1) % self.cap
        if self._count < self.cap: self._count += 1
    def __len__(self) -> int: return self._count
    def __getitem__(self, i: int) -> Any:
        if i >= self._count: raise IndexError(i); return self._data[(self._pos - self._count + i) % self.cap]
    def to_array(self) -> np.ndarray: return np.array([self._data[(self._pos - self._count + i) % self.cap] for i in range(self._count)], dtype=self._dtype)
    def sample(self, n: int) -> List[Any]:
        if n >= self._count: return list(self._data[:self._count])
        idx = np.random.choice(self._count, size=n, replace=False); return [self._data[i] for i in idx]
    def clear(self) -> None: self._pos = 0; self._count = 0; self._data = np.zeros((self.cap,), dtype=object)
    @property
    def full(self) -> bool: return self._count == self.cap

class GradientAccumulator:
    def __init__(self, n_accumulate: int = 4):
        self._n = n_accumulate; self._count = 0; self._grads: Dict[str, np.ndarray] = {}
    def add(self, grads: Dict[str, np.ndarray]) -> None:
        for k, v in grads.items():
            if k not in self._grads: self._grads[k] = np.zeros_like(v)
            self._grads[k] += v
        self._count += 1
    def ready(self) -> bool: return self._count >= self._n
    def get(self) -> Dict[str, np.ndarray]:
        return {k: v / self._n for k, v in self._grads.items()}
    def reset(self) -> None: self._count = 0; self._grads.clear()

class LRUSymbolCache:
    def __init__(self, max_size: int = LUCIA_RL_CONFIG['cache_max_size']):
        self._max = max_size; self._cache: Dict[str, Any] = {}; self._order: deque = deque()
    def get(self, key: str) -> Any:
        if key in self._cache: self._order.remove(key); self._order.append(key); return self._cache[key]
        return None
    def put(self, key: str, value: Any) -> None:
        if key in self._cache: self._order.remove(key)
        elif len(self._cache) >= self._max: oldest = self._order.popleft(); del self._cache[oldest]
        self._cache[key] = value; self._order.append(key)
    def invalidate(self, key: str) -> None:
        if key in self._cache: del self._cache[key]; self._order.remove(key)
    def clear(self) -> None: self._cache.clear(); self._order.clear()
    @property
    def size(self) -> int: return len(self._cache)

class PerformanceProfiler:
    def __init__(self, enabled: bool = LUCIA_RL_CONFIG['profiler_enabled']):
        self._enabled = enabled; self._timings: Dict[str, List[float]] = {}; self._start_times: Dict[str, float] = {}
    def start(self, label: str) -> None:
        if not self._enabled: return
        self._start_times[label] = time.perf_counter()
    def stop(self, label: str) -> float:
        if not self._enabled or label not in self._start_times: return 0.0
        elapsed = time.perf_counter() - self._start_times.pop(label)
        if label not in self._timings: self._timings[label] = []
        self._timings[label].append(elapsed); return elapsed
    def get_avg(self, label: str) -> float:
        if label not in self._timings or not self._timings[label]: return 0.0
        return float(np.mean(self._timings[label]))
    def get_all_avg(self) -> Dict[str, float]: return {k: self.get_avg(k) for k in self._timings}
    def reset(self) -> None: self._timings.clear(); self._start_times.clear()

class OptimizerFactory:
    def __init__(self):
        self._optimizers: Dict[str, Callable] = {
            'sgd': self._sgd, 'adam': self._adam, 'rmsprop': self._rmsprop, 'adagrad': self._adagrad
        }
    def create(self, name: str, lr: float = 0.001, **kwargs) -> Callable:
        fn = self._optimizers.get(name.lower(), self._adam)
        return lambda params, grads: fn(params, grads, lr, **kwargs)
    @staticmethod
    def _sgd(params: List[np.ndarray], grads: List[np.ndarray], lr: float, momentum: float = 0.9) -> List[np.ndarray]:
        updated = []; v = [np.zeros_like(p) for p in params]
        for i, (p, g) in enumerate(zip(params, grads)):
            v[i] = momentum * v[i] + g; updated.append(p - lr * v[i])
        return updated
    @staticmethod
    def _adam(params: List[np.ndarray], grads: List[np.ndarray], lr: float, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8) -> List[np.ndarray]:
        m = [np.zeros_like(p) for p in params]; v = [np.zeros_like(p) for p in params]; t = 0
        updated = []
        for i, (p, g) in enumerate(zip(params, grads)):
            t += 1; m[i] = beta1 * m[i] + (1 - beta1) * g; v[i] = beta2 * v[i] + (1 - beta2) * g * g
            mh = m[i] / (1 - beta1 ** t); vh = v[i] / (1 - beta2 ** t); updated.append(p - lr * mh / (np.sqrt(vh) + eps))
        return updated
    @staticmethod
    def _rmsprop(params: List[np.ndarray], grads: List[np.ndarray], lr: float, decay: float = 0.9, eps: float = 1e-8) -> List[np.ndarray]:
        cache = [np.zeros_like(p) for p in params]; updated = []
        for i, (p, g) in enumerate(zip(params, grads)):
            cache[i] = decay * cache[i] + (1 - decay) * g * g; updated.append(p - lr * g / (np.sqrt(cache[i]) + eps))
        return updated
    @staticmethod
    def _adagrad(params: List[np.ndarray], grads: List[np.ndarray], lr: float, eps: float = 1e-8) -> List[np.ndarray]:
        cache = [np.zeros_like(p) for p in params]; updated = []
        for i, (p, g) in enumerate(zip(params, grads)):
            cache[i] += g * g; updated.append(p - lr * g / (np.sqrt(cache[i]) + eps))
        return updated

class WeightStabilityTracker:
    def __init__(self, window: int = 100):
        self._window = window; self._norms: deque = deque(maxlen=window)
    def record(self, weights: np.ndarray) -> None: self._norms.append(float(np.linalg.norm(weights)))
    def get_stability(self) -> float:
        if len(self._norms) < 2: return 1.0
        diffs = np.diff(list(self._norms)); return float(1.0 / (1.0 + np.std(diffs)))
    def get_variance(self) -> float:
        if len(self._norms) == 0:
            return 0.0
        return float(np.var(list(self._norms)))
    def is_diverging(self, threshold: float = 10.0) -> bool:
        if len(self._norms) < 10: return False
        recent = list(self._norms)[-10:]
        return max(recent) / max(1e-8, min(recent)) > threshold

class BatchPreprocessor:
    def __init__(self): self._normalizer = None; self._running_mean = 0.0; self._running_var = 1.0; self._count = 0
    def update(self, batch: np.ndarray) -> None:
        bs = np.mean(batch); bv = np.var(batch); self._count += 1
        alpha = 1.0 / self._count; self._running_mean = (1-alpha)*self._running_mean + alpha*bs; self._running_var = (1-alpha)*self._running_var + alpha*bv
    def normalize(self, batch: np.ndarray) -> np.ndarray: return (batch - self._running_mean) / max(1e-8, np.sqrt(self._running_var))
    def denormalize(self, batch: np.ndarray) -> np.ndarray: return batch * max(1e-8, np.sqrt(self._running_var)) + self._running_mean
    def should_normalize(self) -> bool: return self._count > 5

class TensorAccelerator:
    @staticmethod
    def fast_matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray: return np.dot(a.astype(np.float32), b.astype(np.float32))
    @staticmethod
    def batch_dot(vectors: np.ndarray, matrix: np.ndarray) -> np.ndarray: return vectors @ matrix.T
    @staticmethod
    def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
        e = np.exp(x - np.max(x, axis=axis, keepdims=True)); return e / np.sum(e, axis=axis, keepdims=True)
    @staticmethod
    def relu(x: np.ndarray) -> np.ndarray: return np.maximum(0, x)
    @staticmethod
    def relu_derivative(x: np.ndarray) -> np.ndarray: return (x > 0).astype(np.float32)
    @staticmethod
    def sigmoid(x: np.ndarray) -> np.ndarray: return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
    @staticmethod
    def tanh(x: np.ndarray) -> np.ndarray: return np.tanh(x)
    @staticmethod
    def leaky_relu(x: np.ndarray, alpha: float = 0.01) -> np.ndarray: return np.where(x > 0, x, alpha * x)
    @staticmethod
    def clip(x: np.ndarray, min_val: float, max_val: float) -> np.ndarray: return np.clip(x, min_val, max_val)
    @staticmethod
    def l2_regularize(weights: np.ndarray, lambda_: float = 0.001) -> np.ndarray: return lambda_ * weights
    @staticmethod
    def l1_regularize(weights: np.ndarray, lambda_: float = 0.001) -> np.ndarray: return lambda_ * np.sign(weights)

class WeightScheduler:
    def __init__(self, initial_lr: float, strategy: str = 'cosine', total_steps: int = 1000):
        self._lr = initial_lr; self._strategy = strategy; self._total = total_steps; self._step = 0
    def step(self) -> float:
        self._step += 1; t = self._step / self._total
        if self._strategy == 'cosine': self._lr = 0.5 * self._lr * (1 + math.cos(math.pi * t))
        elif self._strategy == 'linear': self._lr = self._lr * (1 - t)
        elif self._strategy == 'step': self._lr = self._lr * (0.5 ** (self._step // 100))
        return self._lr
    def get_lr(self) -> float: return self._lr
    def reset(self) -> None: self._step = 0; self._lr = None if not hasattr(self, '_lr') else self._lr

class ConvergenceEngine:
    def __init__(self, patience: int = 20, min_delta: float = 1e-4, window: int = 50):
        self._patience = patience; self._delta = min_delta; self._window = window; self._best = float('inf'); self._counter = 0; self._history: List[float] = []
    def update(self, value: float) -> bool:
        self._history.append(value); improved = value < self._best - self._delta
        if improved: self._best = value; self._counter = 0; return False
        self._counter += 1; return self._counter >= self._patience
    def is_converged(self) -> bool: return self._counter >= self._patience
    def get_improvement_rate(self) -> float:
        if len(self._history) < 2: return 0.0
        d = self._history[-1] - self._history[0]
        return float(abs(d) / max(1e-8, len(self._history)))
    def reset(self) -> None: self._best = float('inf'); self._counter = 0; self._history.clear()

class MemoryEfficientBuffer:
    def __init__(self, capacity: int, dtype: np.dtype = np.float32):
        self.cap = capacity; self.pos = 0; self.count = 0
        self.data = np.zeros((capacity,), dtype=object)
    def push(self, item) -> None:
        self.data[self.pos] = item; self.pos = (self.pos + 1) % self.cap
        if self.count < self.cap: self.count += 1
    def sample_batch(self, batch_size: int) -> List:
        n = min(batch_size, self.count); idx = np.random.choice(self.count, size=n, replace=False)
        return [self.data[i] for i in idx]
    def __len__(self) -> int: return self.count
    def is_full(self) -> bool: return self.count == self.cap
    def clear(self) -> None: self.pos = 0; self.count = 0; self.data = np.zeros((self.cap,), dtype=object)

class SerializationHelper:
    @staticmethod
    def serialize_neuron(neurona) -> bytes:
        state = {'input_size': neurona.input_size, 'output_size': neurona.output_size, 'nombre': neurona.nombre, 'pasos': neurona.pasos}
        if neurona.pesos is not None: state['pesos'] = neurona.pesos.tobytes()
        if neurona.sesgo is not None: state['sesgo'] = neurona.sesgo.tobytes()
        return pickle.dumps(state)
    @staticmethod
    def deserialize_neuron(data: bytes, neurona_class):
        state = pickle.loads(data); return neurona_class(state['input_size'], state['output_size'])
    @staticmethod
    def save_to_file(neurona, filepath: str) -> None:
        with open(filepath, 'wb') as f: f.write(SerializationHelper.serialize_neuron(neurona))
    @staticmethod
    def load_from_file(filepath: str, neurona_class):
        with open(filepath, 'rb') as f: return SerializationHelper.deserialize_neuron(f.read(), neurona_class)

logger = logging.getLogger('RFENRN1')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger.info("Paquete RFENRN1 inicializado para LucIA Reinforcement Learning")

__all__ = [
    'NeuronaRefuerzoBase','NeuronaRefuerzoQLearning','NeuronaRefuerzoPolicyGradient','NeuronaRefuerzoActorCritic',
    'NeuronaRefuerzoDQN','NeuronaRefuerzoA3C','NeuronaRefuerzoPPO','NeuronaRefuerzoSAC','NeuronaRefuerzoTD3',
    'NeuronaRefuerzoRainbowDQN','NeuronaRefuerzoIMPALA','LUCIA_RL_CONFIG',
    'inicializar_pesos_he','inicializar_pesos_xavier','inicializar_pesos_lecun','inicializar_pesos_ortogonal','inicializar_pesos_espectral','inicializar_pesos_eficiente',
    'calcular_recompensa_descontada','normalizar_recompensas','calcular_gae','aplicar_clip_gradientes',
    'crear_buffer_experiencia','muestrear_buffer','calcular_entropy','calcular_loss','compute_returns','discount_cumulative_returns',
    'huber_loss','mean_std_normalize','sample_gumbel_softmax','layer_init','count_parameters',
    'get_optimizer','lr_scheduler','early_stopping_check','neuron_info','train_batch_neuron','evaluate_neurona',
    'analizar_convergencia','comparar_inicializaciones','train_epoch','parallel_train',
    'crear_buffer_priorizado','muestrear_priorizado','actualizar_prioridades','_Ring','_RingBufferNumpy',
    'GradientAccumulator','LRUSymbolCache','PerformanceProfiler','OptimizerFactory','WeightStabilityTracker',
    'BatchPreprocessor','TensorAccelerator','WeightScheduler','ConvergenceEngine','MemoryEfficientBuffer',
    'SerializationHelper',
]