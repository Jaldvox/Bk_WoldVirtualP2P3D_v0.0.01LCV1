"""
RF_SL1_9.py - Linear Regression Avanzada
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


class LinearRegressionOptimizer(NeuronaMemoriaBase):
    def __init__(self, input_size: int = 4, output_size: int = 8, nombre: str = "LinearRegressionOptimizer"):
        super().__init__(input_size, output_size, nombre)
        self.inicializar_pesos()
        self._learning_rate: float = 0.01
        self._epochs: int = 1000
        self._batch_size: int = 32
        self._regularization: str = "l2"
        self._reg_lambda: float = 0.01
        self._is_fitted: bool = False
        self._convergence_history: List[float] = []
        self._training_time: float = 0.0
        self._loss_history: List[float] = []
        self._r2_score: float = 0.0
        self._n_samples: int = 0
        self._use_normal_eq: bool = False

    def inicializar_pesos(self) -> None:
        std = 0.1
        self.pesos = np.random.randn(self.input_size, self.output_size).astype(np.float32) * std
        self.sesgo = np.zeros((1, self.output_size), dtype=np.float32)

    def forward(self, entrada: np.ndarray) -> np.ndarray:
        if self.pesos is None: self.inicializar_pesos()
        salida = np.dot(entrada, self.pesos) + self.sesgo
        self.historial_activaciones.append(salida.copy())
        self.pasos += 1
        return salida

    def compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        mse = np.mean((y_true - y_pred) ** 2)
        if self._regularization == "l2" and self.pesos is not None:
            mse += 0.5 * self._reg_lambda * float(np.sum(self.pesos ** 2))
        elif self._regularization == "l1" and self.pesos is not None:
            mse += self._reg_lambda * float(np.sum(np.abs(self.pesos)))
        return float(mse)

    def backward(self, gradiente_salida: np.ndarray, entrada: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.pesos is None: raise RuntimeError("Modelo no entrenado")
        n = max(1, len(entrada))
        grad_pesos = np.dot(entrada.T, gradiente_salida) / n
        grad_sesgo = np.sum(gradiente_salida, axis=0, keepdims=True) / n
        if self._regularization == "l2" and self.pesos is not None:
            grad_pesos += self._reg_lambda * self.pesos
        elif self._regularization == "l1" and self.pesos is not None:
            grad_pesos += self._reg_lambda * np.sign(self.pesos)
        self.historial_gradientes.append({
            'pesos': grad_pesos.copy(),
            'norma': float(np.linalg.norm(grad_pesos)),
        })
        return grad_pesos, grad_sesgo

    def train(self, X: np.ndarray, y: np.ndarray, epochs: Optional[int] = None,
               learning_rate: Optional[float] = None, verbose: bool = False) -> Dict[str, Any]:
        if epochs is not None: self._epochs = epochs
        if learning_rate is not None: self._learning_rate = learning_rate
        if self._use_normal_eq:
            return self._train_normal_equation(X, y)
        start_time = time.time()
        n_samples = len(X)
        if y.ndim == 1 and self.output_size == 1: y = y.reshape(-1, 1)
        losses = []
        for epoch in range(self._epochs):
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            epoch_loss = 0.0
            for i in range(0, n_samples, self._batch_size):
                batch_X = X_shuffled[i:i+self._batch_size]
                batch_y = y_shuffled[i:i+self._batch_size]
                y_pred = self.forward(batch_X)
                loss = self.compute_loss(batch_y, y_pred)
                epoch_loss += loss
                error = y_pred - batch_y
                grad_pesos, grad_sesgo = self.backward(error, batch_X)
                self.pesos -= self._learning_rate * grad_pesos
                self.sesgo -= self._learning_rate * grad_sesgo
            avg_loss = epoch_loss / max(1, n_samples // self._batch_size)
            losses.append(avg_loss)
            self._loss_history.append(avg_loss)
            self._convergence_history.append(avg_loss)
            if verbose and (epoch + 1) % 100 == 0:
                logger.info(f"Epoch {epoch+1}/{self._epochs}, Loss: {avg_loss:.6f}")
        self._training_time = time.time() - start_time
        self._is_fitted = True
        self._n_samples = n_samples
        return {'losses': losses, 'final_loss': float(losses[-1]), 'training_time': self._training_time}

    def _train_normal_equation(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        start_time = time.time()
        if y.ndim == 1: y = y.reshape(-1, 1)
        X_bias = np.hstack([X, np.ones((len(X), 1))])
        theta = np.linalg.pinv(np.dot(X_bias.T, X_bias))
        theta = np.dot(theta, np.dot(X_bias.T, y))
        self.pesos = theta[:-1].reshape(self.input_size, self.output_size).astype(np.float32)
        self.sesgo = theta[-1:].reshape(1, self.output_size).astype(np.float32)
        self._is_fitted = True
        self._n_samples = len(X)
        self._training_time = time.time() - start_time
        return {'method': 'normal_equation', 'training_time': self._training_time}

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self._is_fitted and self.pesos is None: self.inicializar_pesos()
        return self.forward(X)

    def predict_flat(self, X: np.ndarray) -> np.ndarray:
        preds = self.predict(X)
        return preds.flatten() if preds.size > 1 else preds.flatten()

    def r2_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        ss_res = float(np.sum((y_true - y_pred) ** 2))
        ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
        score = 1.0 - ss_res / max(1e-8, ss_tot)
        self._r2_score = score
        return score

    def mae(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.mean(np.abs(y_true - y_pred)))

    def rmse(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        preds = self.predict(X)
        return {
            'predictions': preds, 'r2': self.r2_score(y, preds),
            'mae': self.mae(y, preds), 'rmse': self.rmse(y, preds),
            'n_samples': len(X),
        }

    def obtener_estadisticas_completas(self) -> Dict[str, Any]:
        stats = self.obtener_estadisticas()
        if self.pesos is not None:
            stats['pesos_mean'] = float(np.mean(self.pesos))
            stats['pesos_std'] = float(np.std(self.pesos))
            stats['pesos_min'] = float(np.min(self.pesos))
            stats['pesos_max'] = float(np.max(self.pesos))
        stats['learning_rate'] = self._learning_rate
        stats['epochs'] = self._epochs
        stats['regularization'] = self._regularization
        stats['is_fitted'] = self._is_fitted
        stats['r2_score'] = self._r2_score
        stats['use_normal_equation'] = self._use_normal_eq
        return stats

    def verificar_estabilidad(self) -> Dict[str, bool]:
        ok = {'pesos_inicializados': self.pesos is not None}
        if self.pesos is not None:
            ok['pesos_no_explosivos'] = float(np.max(np.abs(self.pesos))) < 10.0
            ok['pesos_no_nan'] = not bool(np.any(np.isnan(self.pesos)))
            ok['pesos_no_inf'] = not bool(np.any(np.isinf(self.pesos)))
        ok['sesgo_inicializado'] = self.sesgo is not None
        ok['modelo_entrenado'] = self._is_fitted
        return ok

    def set_regularization(self, reg_type: str, lambda_val: float = 0.01) -> None:
        self._regularization = reg_type if reg_type in ("l1", "l2") else "l2"
        self._reg_lambda = max(0.0, float(lambda_val))

    def enable_normal_equation(self, enable: bool = True) -> None:
        self._use_normal_eq = bool(enable)

    def get_loss_history(self) -> List[float]: return self._loss_history.copy()

    def __str__(self) -> str:
        return f"{self.nombre}(ent={self.input_size},sal={self.output_size},pasos={self.pasos})"


class LossMSE:
    def __init__(self):
        self._name: str = "MSE"

    def compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.mean((y_true - y_pred) ** 2))

    def gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return 2.0 * (y_pred - y_true) / max(1, len(y_true))

    def name(self) -> str: return self._name


class GradientDescentLR:
    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.0, decay: float = 0.0):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.decay = decay
        self._velocity: Optional[np.ndarray] = None
        self._t: int = 0

    def update(self, params: np.ndarray, gradient: np.ndarray) -> np.ndarray:
        if self.decay > 0.0:
            self.learning_rate *= 1.0 / (1.0 + self.decay * self._t)
        if self.momentum > 0.0:
            if self._velocity is None: self._velocity = np.zeros_like(params)
            self._velocity = self.momentum * self._velocity + self.learning_rate * gradient
            self._t += 1
            return params - self._velocity
        self._t += 1
        return params - self.learning_rate * gradient

    def reset(self) -> None:
        self._velocity = None
        self._t = 0


class NormalEquation:
    @staticmethod
    def solve(X: np.ndarray, y: np.ndarray) -> np.ndarray:
        X_bias = np.hstack([X, np.ones((len(X), 1))])
        theta = np.linalg.pinv(np.dot(X_bias.T, X_bias))
        return np.dot(theta, np.dot(X_bias.T, y))

    @staticmethod
    def ridge_solve(X: np.ndarray, y: np.ndarray, alpha: float = 0.01) -> np.ndarray:
        X_bias = np.hstack([X, np.ones((len(X), 1))])
        n_features = X_bias.shape[1]
        reg_matrix = alpha * np.eye(n_features)
        reg_matrix[-1, -1] = 0.0
        theta = np.linalg.solve(
            np.dot(X_bias.T, X_bias) + reg_matrix,
            np.dot(X_bias.T, y)
        )
        return theta


class FeatureStandardizer:
    def __init__(self):
        self._mean: Optional[np.ndarray] = None
        self._std: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> None:
        self._mean = np.mean(X, axis=0)
        self._std = np.std(X, axis=0) + 1e-8

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self._mean is None: raise RuntimeError("No ajustado")
        return (X - self._mean) / self._std

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)

    def get_params(self) -> Dict[str, np.ndarray]:
        return {'mean': self._mean.copy() if self._mean is not None else None,
                'std': self._std.copy() if self._std is not None else None}


class LinearPredictor:
    def __init__(self, model: Optional[LinearRegressionOptimizer] = None):
        self._model = model

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> LinearRegressionOptimizer:
        self._model = LinearRegressionOptimizer(input_size=X.shape[1], output_size=1)
        self._model.train(X, y, **kwargs)
        return self._model

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._model is None: raise RuntimeError("Modelo no entrenado")
        return self._model.predict(X).flatten()

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        preds = self.predict(X)
        ss_res = float(np.sum((y - preds) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        return 1.0 - ss_res / max(1e-8, ss_tot)


class ModelMetrics:
    @staticmethod
    def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        ss_res = float(np.sum((y_true - y_pred) ** 2))
        ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
        return 1.0 - ss_res / max(1e-8, ss_tot)

    @staticmethod
    def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.mean(np.abs(y_true - y_pred)))

    @staticmethod
    def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    @staticmethod
    def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        mask = y_true != 0
        if not np.any(mask): return float('inf')
        return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])))

    @staticmethod
    def evaluate_all(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        return {
            'r2': ModelMetrics.r2(y_true, y_pred),
            'mae': ModelMetrics.mae(y_true, y_pred),
            'rmse': ModelMetrics.rmse(y_true, y_pred),
            'mape': ModelMetrics.mape(y_true, y_pred),
        }


class PolynomialFeatures:
    def __init__(self, degree: int = 2, interaction_only: bool = False):
        self.degree = max(1, int(degree))
        self.interaction_only = bool(interaction_only)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        n_samples, n_features = X.shape
        features = [X]
        for d in range(2, self.degree + 1):
            if self.interaction_only:
                for i in range(n_features):
                    for j in range(i + 1, n_features):
                        features.append((X[:, i] * X[:, j]).reshape(-1, 1))
            else:
                for i in range(n_features):
                    features.append((X[:, i] ** d).reshape(-1, 1))
        return np.hstack(features).astype(np.float32)

    def get_output_dim(self, input_dim: int) -> int:
        if self.interaction_only:
            return input_dim + input_dim * (input_dim - 1) // 2 * self.degree
        return input_dim * self.degree


class LearningRateScheduler:
    def __init__(self, initial_lr: float = 0.01, strategy: str = "step"):
        self.initial_lr = initial_lr
        self.strategy = strategy if strategy in ("step", "exponential", "cosine") else "step"
        self._current_lr: float = initial_lr
        self._step: int = 0

    def get_lr(self) -> float:
        if self.strategy == "step":
            return self.initial_lr * (0.5 ** (self._step // 100))
        elif self.strategy == "exponential":
            return self.initial_lr * math.exp(-0.01 * self._step)
        elif self.strategy == "cosine":
            return self.initial_lr * 0.5 * (1.0 + math.cos(math.pi * self._step / 1000))
        return self.initial_lr

    def step(self) -> None:
        self._step += 1
        self._current_lr = self.get_lr()

    def reset(self) -> None:
        self._step = 0
        self._current_lr = self.initial_lr


def create_linear_regression_optimizer(input_size: int = 4, output_size: int = 8) -> LinearRegressionOptimizer:
    return LinearRegressionOptimizer(input_size=input_size, output_size=output_size)

if __name__ == "__main__":
    logger.info("RF_SL1_9.py cargado exitosamente")