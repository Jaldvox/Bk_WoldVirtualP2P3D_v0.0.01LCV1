"""
RF_SL1_4.py - Neural Networks Avanzadas
Optimizado con clases inline y estructura eficiente.
"""

import numpy as np
import logging
import time
import pickle
import math
from typing import Dict, Any, List, Optional, Tuple


class NeuronaMemoriaBase:
    def __init__(self, input_size: int, output_size: int, nombre: str = "Neurona"):
        self.input_size = int(input_size)
        self.output_size = int(output_size)
        self.nombre = str(nombre)
        self.pesos = None
        self.sesgo = None
        self.historial_activaciones: List[np.ndarray] = []
        self.historial_gradientes: List[Dict[str, np.ndarray]] = []
        self.pasos = 0

    def inicializar_pesos(self) -> None:
        raise NotImplementedError

    def forward(self, e: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def obtener_estadisticas(self) -> Dict[str, Any]:
        stats = {
            'nombre': self.nombre, 'input_size': self.input_size,
            'output_size': self.output_size, 'pasos': self.pasos,
            'historial_activaciones_len': len(self.historial_activaciones),
            'historial_gradientes_len': len(self.historial_gradientes),
        }
        if self.pesos is not None:
            stats['pesos_shape'] = self.pesos.shape
            stats['pesos_size'] = self.pesos.size
        if self.sesgo is not None:
            stats['sesgo_shape'] = self.sesgo.shape
        return stats

    def resetear_historial(self) -> None:
        self.historial_activaciones.clear()
        self.historial_gradientes.clear()

    def info(self) -> str:
        return f"{self.nombre}(in={self.input_size},out={self.output_size},pasos={self.pasos})"

    def params_count(self) -> int:
        total = 0
        if self.pesos is not None: total += self.pesos.size
        if self.sesgo is not None: total += self.sesgo.size
        return total

logger = logging.getLogger(__name__)


class NeuralNetworkOptimizer(NeuronaMemoriaBase):
    def __init__(self, input_size: int = 4, output_size: int = 8, nombre: str = "NeuralNetworkOptimizer"):
        super().__init__(input_size, output_size, nombre)
        self.inicializar_pesos()
        self._learning_rate: float = 0.001
        self._momentum: float = 0.9
        self._weight_decay: float = 0.0001
        self._clip_norm: float = 5.0
        self._activation: str = "relu"
        self._dropout_rate: float = 0.0
        self._is_fitted: bool = False
        self._velocity_w: Optional[np.ndarray] = None
        self._velocity_b: Optional[np.ndarray] = None
        self._convergence_history: List[float] = []
        self._training_time: float = 0.0

    def inicializar_pesos(self) -> None:
        std = 0.1
        self.pesos = np.random.randn(self.input_size, self.output_size).astype(np.float32) * std
        self.sesgo = np.zeros((1, self.output_size), dtype=np.float32)
        self._velocity_w = np.zeros_like(self.pesos)
        self._velocity_b = np.zeros_like(self.sesgo)

    def forward(self, entrada: np.ndarray) -> np.ndarray:
        if self.pesos is None: self.inicializar_pesos()
        salida = np.dot(entrada, self.pesos) + self.sesgo
        salida = self._apply_activation(salida)
        self.historial_activaciones.append(salida.copy())
        self.pasos += 1
        return salida

    def _apply_activation(self, x: np.ndarray) -> np.ndarray:
        if self._activation == "relu": return np.maximum(0, x)
        if self._activation == "sigmoid": return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
        if self._activation == "tanh": return np.tanh(x)
        if self._activation == "leaky_relu": return np.where(x > 0, x, 0.01 * x)
        if self._activation == "softmax":
            e = np.exp(x - np.max(x, axis=-1, keepdims=True)); return e / np.sum(e, axis=-1, keepdims=True)
        if self._activation == "linear": return x
        return x

    def compute_loss(self, prediccion: np.ndarray, objetivo: np.ndarray) -> float:
        diff = prediccion - objetivo
        return float(np.mean(diff ** 2))

    def compute_gradient(self, prediccion: np.ndarray, objetivo: np.ndarray) -> np.ndarray:
        return 2.0 * (prediccion - objetivo) / max(1, len(prediccion))

    def apply_gradient_clipping(self, grad: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(grad)
        if norm > self._clip_norm: return grad * self._clip_norm / (norm + 1e-8)
        return grad

    def apply_weight_decay(self) -> None:
        if self._weight_decay > 0 and self.pesos is not None:
            self.pesos *= (1 - self._learning_rate * self._weight_decay)

    def backward(self, gradiente_salida: np.ndarray, entrada: np.ndarray):
        grad_pesos = np.dot(entrada.T, gradiente_salida)
        grad_sesgo = np.sum(gradiente_salida, axis=0, keepdims=True)
        self.historial_gradientes.append({'pesos': grad_pesos.copy(), 'norma': float(np.linalg.norm(grad_pesos))})
        return grad_pesos, grad_sesgo

    def update_weights_momentum(self, grad_pesos: np.ndarray, grad_sesgo: np.ndarray) -> None:
        if self._velocity_w is None or self._velocity_w.shape != grad_pesos.shape:
            self._velocity_w = np.zeros_like(grad_pesos)
            self._velocity_b = np.zeros_like(grad_sesgo)
        self._velocity_w = self._momentum * self._velocity_w - self._learning_rate * grad_pesos
        self._velocity_b = self._momentum * self._velocity_b - self._learning_rate * grad_sesgo
        self.pesos += self._velocity_w
        self.sesgo += self._velocity_b

    def update_weights_adam(self, grad_pesos: np.ndarray, grad_sesgo: np.ndarray, t: int = 1,
                            beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8) -> None:
        if self._velocity_w is None: self._velocity_w = np.zeros_like(grad_pesos)
        if self._velocity_b is None: self._velocity_b = np.zeros_like(grad_sesgo)
        m_w = beta1 * self._velocity_w + (1 - beta1) * grad_pesos
        m_b = beta1 * self._velocity_b + (1 - beta1) * grad_sesgo
        v_w = np.zeros_like(grad_pesos); v_b = np.zeros_like(grad_sesgo)
        self._velocity_w = m_w / (1 - beta1 ** t)
        self._velocity_b = m_b / (1 - beta1 ** t)
        self.pesos -= self._learning_rate * self._velocity_w
        self.sesgo -= self._learning_rate * self._velocity_b

    def train_step(self, entrada: np.ndarray, objetivo: np.ndarray) -> Dict[str, float]:
        pred = self.forward(entrada)
        loss = self.compute_loss(pred, objetivo)
        grad = self.compute_gradient(pred, objetivo)
        grad = self.apply_gradient_clipping(grad)
        gp, gs = self.backward(grad, entrada)
        self.update_weights_momentum(gp, gs)
        self.apply_weight_decay()
        return {'loss': loss, 'grad_norm': float(np.linalg.norm(gp)), 'improvement': loss}

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 50, verbose: bool = False) -> List[float]:
        t0 = time.perf_counter()
        losses = []
        for epoch in range(epochs):
            pred = self.forward(X)
            loss = self.compute_loss(pred, y)
            grad = self.compute_gradient(pred, y)
            grad = self.apply_gradient_clipping(grad)
            gp, gs = self.backward(grad, X)
            self.update_weights_momentum(gp, gs)
            self.apply_weight_decay()
            losses.append(loss)
            self._convergence_history.append(loss)
            if verbose and (epoch + 1) % 10 == 0:
                logger.info(f"NN Epoch {epoch+1}, loss={loss:.6f}")
        self._training_time = time.perf_counter() - t0
        self._is_fitted = True
        return losses

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        pred = self.predict(X)
        mse = self.compute_loss(pred, y)
        mae = float(np.mean(np.abs(pred - y)))
        return {'mse': mse, 'mae': mae, 'n_samples': len(X)}

    def obtener_estadisticas(self) -> Dict[str, Any]:
        stats = super().obtener_estadisticas()
        stats['learning_rate'] = self._learning_rate
        stats['momentum'] = self._momentum
        stats['weight_decay'] = self._weight_decay
        stats['clip_norm'] = self._clip_norm
        stats['activation'] = self._activation
        stats['dropout_rate'] = self._dropout_rate
        stats['is_fitted'] = self._is_fitted
        stats['training_time'] = self._training_time
        if self._convergence_history:
            stats['final_loss'] = float(self._convergence_history[-1])
        return stats

    def verificar_estabilidad(self) -> Dict[str, bool]:
        ok = {'pesos_inicializados': self.pesos is not None}
        if self.pesos is not None:
            ok['pesos_no_explosivos'] = float(np.max(np.abs(self.pesos))) < 10.0
            ok['pesos_no_nan'] = not bool(np.any(np.isnan(self.pesos)))
            ok['pesos_no_inf'] = not bool(np.any(np.isinf(self.pesos)))
            ok['estado_entrenado'] = self._is_fitted
            ok['velocidad_inicializada'] = self._velocity_w is not None
        return ok

    def save(self, filepath: str) -> None:
        state = {
            'pesos': self.pesos, 'sesgo': self.sesgo, 'nombre': self.nombre,
            'input_size': self.input_size, 'output_size': self.output_size,
            'learning_rate': self._learning_rate, 'momentum': self._momentum,
            'weight_decay': self._weight_decay, 'clip_norm': self._clip_norm,
            'activation': self._activation, 'dropout_rate': self._dropout_rate,
            'is_fitted': self._is_fitted, 'pasos': self.pasos,
            'convergence_history': self._convergence_history,
        }
        with open(filepath, 'wb') as f: pickle.dump(state, f)

    def load(self, filepath: str) -> None:
        with open(filepath, 'rb') as f: state = pickle.load(f)
        self.__dict__.update({k: v for k, v in state.items() if k != 'convergence_history'})
        self._convergence_history = state.get('convergence_history', [])

    def get_config(self) -> Dict[str, Any]:
        return {
            'nombre': self.nombre, 'input_size': self.input_size, 'output_size': self.output_size,
            'learning_rate': self._learning_rate, 'momentum': self._momentum,
            'activation': self._activation, 'weight_decay': self._weight_decay,
        }

    def summary(self) -> str:
        cfg = self.get_config()
        stats = self.obtener_estadisticas()
        return "\n".join([
            f"=== {cfg['nombre']} ===",
            f"LR: {cfg['learning_rate']}, Momentum: {cfg['momentum']}",
            f"Activation: {cfg['activation']}, Weight decay: {cfg['weight_decay']}",
            f"Input: {cfg['input_size']}, Output: {cfg['output_size']}",
            f"Fitted: {stats.get('is_fitted', False)}, Params: {self.params_count()}",
            f"Final loss: {stats.get('final_loss', 'N/A')}",
        ])

    def __str__(self) -> str:
        return f"{self.nombre}(ent={self.input_size},sal={self.output_size})"


class ActivationEngine:
    @staticmethod
    def relu(x: np.ndarray) -> np.ndarray: return np.maximum(0, x)

    @staticmethod
    def sigmoid(x: np.ndarray) -> np.ndarray: return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

    @staticmethod
    def tanh(x: np.ndarray) -> np.ndarray: return np.tanh(x)

    @staticmethod
    def leaky_relu(x: np.ndarray, alpha: float = 0.01) -> np.ndarray: return np.where(x > 0, x, alpha * x)

    @staticmethod
    def softmax(x: np.ndarray) -> np.ndarray:
        e = np.exp(x - np.max(x, axis=-1, keepdims=True)); return e / np.sum(e, axis=-1, keepdims=True)

    @staticmethod
    def relu_derivative(x: np.ndarray) -> np.ndarray: return (x > 0).astype(np.float32)

    @staticmethod
    def sigmoid_derivative(x: np.ndarray) -> np.ndarray: s = ActivationEngine.sigmoid(x); return s * (1 - s)

    @staticmethod
    def apply(x: np.ndarray, activation: str) -> np.ndarray:
        if activation == "relu": return ActivationEngine.relu(x)
        if activation == "sigmoid": return ActivationEngine.sigmoid(x)
        if activation == "tanh": return ActivationEngine.tanh(x)
        if activation == "leaky_relu": return ActivationEngine.leaky_relu(x)
        if activation == "softmax": return ActivationEngine.softmax(x)
        return x


class GradientClipper:
    def __init__(self, max_norm: float = 5.0):
        self.max_norm = max_norm

    def clip(self, gradient: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(gradient)
        if norm > self.max_norm: return gradient * self.max_norm / (norm + 1e-8)
        return gradient

    def get_norm(self, gradient: np.ndarray) -> float: return float(np.linalg.norm(gradient))

    def is_clipped(self, gradient: np.ndarray) -> bool: return np.linalg.norm(gradient) > self.max_norm


class WeightInitializer:
    @staticmethod
    def he(shape: tuple, fan_in: int = 1) -> np.ndarray: return np.random.normal(0, math.sqrt(2.0/max(1,fan_in)), shape).astype(np.float32)

    @staticmethod
    def xavier(shape: tuple) -> np.ndarray:
        lim = math.sqrt(6.0/max(1,shape[0]+shape[1])); return np.random.uniform(-lim,lim,shape).astype(np.float32)

    @staticmethod
    def lecun(shape: tuple, fan_in: int = 1) -> np.ndarray: return np.random.normal(0, math.sqrt(1.0/max(1,fan_in)), shape).astype(np.float32)

    @staticmethod
    def orthogonal(shape: tuple) -> np.ndarray:
        wf = np.random.randn(*shape).astype(np.float64); iters = 3
        v = np.random.default_rng(0).normal(0,1,(wf.shape[1],1)); v /= max(1e-12,np.linalg.norm(v))
        for _ in range(iters):
            u = wf @ v; u /= max(1e-12,np.linalg.norm(u)); v = wf.T @ u; v /= max(1e-12,np.linalg.norm(v))
        s = abs((u.T @ wf @ v)[0,0]); return np.random.randn(*shape).astype(np.float32) * abs(s)


class DropoutLayer:
    def __init__(self, rate: float = 0.5):
        self.rate = rate
        self._mask: Optional[np.ndarray] = None

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        if not training or self.rate == 0.0: return x
        self._mask = np.random.random(x.shape) > self.rate
        return x * self._mask / (1.0 - self.rate)

    def backward(self, grad: np.ndarray) -> np.ndarray:
        if self._mask is None: return grad
        return grad * self._mask / (1.0 - self.rate)

    def is_active(self) -> bool: return self.rate > 0 and self._mask is not None


class LossFunctions:
    @staticmethod
    def mse(pred: np.ndarray, target: np.ndarray) -> float: return float(np.mean((pred - target) ** 2))

    @staticmethod
    def mae(pred: np.ndarray, target: np.ndarray) -> float: return float(np.mean(np.abs(pred - target)))

    @staticmethod
    def cross_entropy(pred: np.ndarray, target: np.ndarray) -> float:
        p = np.clip(pred, 1e-8, 1 - 1e-8); return float(-np.mean(target * np.log(p) + (1 - target) * np.log(1 - p)))

    @staticmethod
    def huber(pred: np.ndarray, target: np.ndarray, delta: float = 1.0) -> float:
        abs_err = np.abs(pred - target); q = np.minimum(abs_err, delta); l = abs_err - q
        return float(np.mean(0.5 * q ** 2 + delta * l))


class LearningRateScheduler:
    def __init__(self, initial_lr: float, strategy: str = "cosine", total_steps: int = 100):
        self.lr = initial_lr; self.strategy = strategy; self.total_steps = total_steps; self._step = 0

    def step(self) -> float:
        self._step += 1; t = self._step / self.total_steps
        if self.strategy == "cosine": self.lr = 0.5 * self.lr * (1 + math.cos(math.pi * t))
        elif self.strategy == "linear": self.lr = self.lr * (1 - t)
        elif self.strategy == "step": self.lr = self.lr * (0.5 ** (self._step // 20))
        return self.lr

    def get_lr(self) -> float: return self.lr


class NeuralNetTrainer:
    def __init__(self, network: NeuralNetworkOptimizer, max_norm: float = 5.0):
        self.network = network; self.max_norm = max_norm
        self._train_losses: List[float] = []; self._val_losses: List[float] = []

    def train_epoch(self, X: np.ndarray, y: np.ndarray) -> float:
        pred = self.network.forward(X)
        loss = self.network.compute_loss(pred, y)
        grad = self.network.compute_gradient(pred, y)
        grad = GradientClipper(self.max_norm).clip(grad)
        gp, gs = self.network.backward(grad, X)
        self.network.update_weights_momentum(gp, gs)
        self._train_losses.append(loss)
        return loss

    def validate(self, X: np.ndarray, y: np.ndarray) -> float:
        pred = self.network.predict(X)
        loss = LossFunctions.mse(pred, y)
        self._val_losses.append(loss)
        return loss

    def train_with_validation(self, X: np.ndarray, y: np.ndarray,
                                X_val: np.ndarray, y_val: np.ndarray,
                                epochs: int = 10) -> Dict[str, List[float]]:
        for epoch in range(epochs):
            train_loss = self.train_epoch(X, y)
            val_loss = self.validate(X_val, y_val)
            logger.info(f"Epoch {epoch+1}: train={train_loss:.6f}, val={val_loss:.6f}")
        return {'train': self._train_losses, 'val': self._val_losses}

    def get_train_losses(self) -> List[float]: return self._train_losses.copy()

    def get_val_losses(self) -> List[float]: return self._val_losses.copy()


def create_neural_network_optimizer(input_size: int = 4, output_size: int = 8) -> NeuralNetworkOptimizer:
    return NeuralNetworkOptimizer(input_size=input_size, output_size=output_size)


if __name__ == "__main__":
    logger.info("RF_SL1_4.py cargado exitosamente")