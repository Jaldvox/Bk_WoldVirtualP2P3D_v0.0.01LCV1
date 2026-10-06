"""
RF_SL1_8.py - Logistic Regression Avanzada
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


class LogisticRegressionOptimizer(NeuronaMemoriaBase):
    def __init__(self, input_size: int = 4, output_size: int = 8, nombre: str = "LogisticRegressionOptimizer"):
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
        self._accuracy: float = 0.0
        self._n_samples: int = 0
        self._threshold: float = 0.5

    def inicializar_pesos(self) -> None:
        std = 0.1
        self.pesos = np.random.randn(self.input_size, self.output_size).astype(np.float32) * std
        self.sesgo = np.zeros((1, self.output_size), dtype=np.float32)

    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    def forward(self, entrada: np.ndarray) -> np.ndarray:
        if self.pesos is None: self.inicializar_pesos()
        z = np.dot(entrada, self.pesos) + self.sesgo
        salida = self.sigmoid(z)
        self.historial_activaciones.append(salida.copy())
        self.pasos += 1
        return salida

    def compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        epsilon = 1e-8
        y_pred = np.clip(y_pred, epsilon, 1.0 - epsilon)
        loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        if self._regularization == "l2" and self.pesos is not None:
            loss += 0.5 * self._reg_lambda * float(np.sum(self.pesos ** 2))
        elif self._regularization == "l1" and self.pesos is not None:
            loss += self._reg_lambda * float(np.sum(np.abs(self.pesos)))
        return float(loss)

    def backward(self, gradiente_salida: np.ndarray, entrada: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.pesos is None: raise RuntimeError("Modelo no entrenado")
        grad_pesos = np.dot(entrada.T, gradiente_salida) / max(1, len(entrada))
        grad_sesgo = np.sum(gradiente_salida, axis=0, keepdims=True) / max(1, len(entrada))
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
        if y.ndim == 1 and self.output_size == 1: y = y.reshape(-1, 1)
        start_time = time.time()
        n_samples = len(X)
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

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self._is_fitted: raise RuntimeError("Modelo no entrenado")
        return self.forward(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        return (probs >= self._threshold).astype(int)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        preds = self.predict(X)
        accuracy = float(np.mean(preds == y))
        self._accuracy = accuracy
        return {
            'accuracy': accuracy, 'n_samples': len(X),
            'predictions': preds, 'true_labels': y,
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
        stats['accuracy'] = self._accuracy
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

    def set_threshold(self, threshold: float) -> None:
        self._threshold = max(0.0, min(1.0, float(threshold)))

    def get_loss_history(self) -> List[float]: return self._loss_history.copy()

    def __str__(self) -> str:
        return f"{self.nombre}(ent={self.input_size},sal={self.output_size},pasos={self.pasos})"


class SigmoidEngine:
    def __init__(self, clip_range: float = 500.0):
        self.clip_range = clip_range

    def compute(self, z: np.ndarray) -> np.ndarray:
        z = np.clip(z, -self.clip_range, self.clip_range)
        return 1.0 / (1.0 + np.exp(-z))

    def derivative(self, sigmoid_output: np.ndarray) -> np.ndarray:
        return sigmoid_output * (1.0 - sigmoid_output)

    def log_odds(self, probability: np.ndarray) -> np.ndarray:
        epsilon = 1e-8
        p = np.clip(probability, epsilon, 1.0 - epsilon)
        return np.log(p / (1.0 - p))


class LossCalculator:
    def __init__(self, epsilon: float = 1e-8):
        self.epsilon = epsilon

    def binary_crossentropy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_pred = np.clip(y_pred, self.epsilon, 1.0 - self.epsilon)
        loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return float(loss)

    def mean_squared_error(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.mean((y_true - y_pred) ** 2))

    def log_loss_with_reg(self, y_true: np.ndarray, y_pred: np.ndarray,
                           pesos: np.ndarray, reg_lambda: float = 0.01) -> float:
        base_loss = self.binary_crossentropy(y_true, y_pred)
        reg_term = 0.5 * reg_lambda * float(np.sum(pesos ** 2))
        return base_loss + reg_term


class GradientDescent:
    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.0,
                 decay: float = 0.0, optimizer_type: str = "sgd"):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.decay = decay
        self.optimizer_type = optimizer_type if optimizer_type in ("sgd", "momentum", "adam") else "sgd"
        self._velocity: Optional[np.ndarray] = None
        self._t: int = 0

    def update(self, params: np.ndarray, gradient: np.ndarray) -> np.ndarray:
        if self.decay > 0.0:
            self.learning_rate *= 1.0 / (1.0 + self.decay * self._t)
        if self.optimizer_type == "momentum" or self.momentum > 0.0:
            if self._velocity is None:
                self._velocity = np.zeros_like(params)
            self._velocity = self.momentum * self._velocity + self.learning_rate * gradient
            self._t += 1
            return params - self._velocity
        self._t += 1
        return params - self.learning_rate * gradient

    def reset(self) -> None:
        self._velocity = None
        self._t = 0


class Regularizer:
    def __init__(self, reg_type: str = "l2", lambda_val: float = 0.01):
        self.reg_type = reg_type if reg_type in ("l1", "l2", "elasticnet") else "l2"
        self.lambda_val = max(0.0, float(lambda_val))
        self._l1_ratio: float = 0.5

    def penalty(self, pesos: np.ndarray) -> float:
        if self.reg_type == "l1":
            return self.lambda_val * float(np.sum(np.abs(pesos)))
        if self.reg_type == "l2":
            return 0.5 * self.lambda_val * float(np.sum(pesos ** 2))
        l1 = self.lambda_val * 0.5 * float(np.sum(np.abs(pesos)))
        l2 = 0.5 * self.lambda_val * float(np.sum(pesos ** 2))
        return l1 * self._l1_ratio + l2 * (1.0 - self._l1_ratio)

    def gradient(self, pesos: np.ndarray) -> np.ndarray:
        if self.reg_type == "l1":
            return self.lambda_val * np.sign(pesos)
        if self.reg_type == "l2":
            return self.lambda_val * pesos
        l1_grad = self.lambda_val * np.sign(pesos)
        l2_grad = self.lambda_val * pesos
        return l1_grad * self._l1_ratio + l2_grad * (1.0 - self._l1_ratio)

    def set_elasticnet(self, l1_ratio: float) -> None:
        self._l1_ratio = max(0.0, min(1.0, float(l1_ratio)))
        self.reg_type = "elasticnet"


class DecisionBoundary:
    def __init__(self, threshold: float = 0.5):
        self.threshold = max(0.0, min(1.0, float(threshold)))

    def classify(self, probabilities: np.ndarray) -> np.ndarray:
        return (probabilities >= self.threshold).astype(int)

    def set_threshold(self, threshold: float) -> None:
        self.threshold = max(0.0, min(1.0, float(threshold)))

    def margin_analysis(self, probabilities: np.ndarray) -> Dict[str, Any]:
        margin = np.abs(probabilities - self.threshold)
        return {
            'mean_margin': float(np.mean(margin)),
            'min_margin': float(np.min(margin)),
            'max_margin': float(np.max(margin)),
            'uncertain_predictions': int(np.sum(margin < 0.1)),
        }


class FeatureScaler:
    def __init__(self, method: str = "standard"):
        self.method = method if method in ("standard", "minmax", "robust") else "standard"
        self._mean: Optional[np.ndarray] = None
        self._std: Optional[np.ndarray] = None
        self._min: Optional[np.ndarray] = None
        self._max: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> None:
        if self.method == "standard":
            self._mean = np.mean(X, axis=0)
            self._std = np.std(X, axis=0) + 1e-8
        elif self.method == "minmax":
            self._min = np.min(X, axis=0)
            self._max = np.max(X, axis=0)
        elif self.method == "robust":
            self._mean = np.median(X, axis=0)
            self._std = np.percentile(X, 75, axis=0) - np.percentile(X, 25, axis=0) + 1e-8

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.method == "standard":
            return (X - self._mean) / self._std
        elif self.method == "minmax":
            return (X - self._min) / (self._max - self._min + 1e-8)
        elif self.method == "robust":
            return (X - self._mean) / self._std
        return X

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)


class LRClassifier:
    def __init__(self, input_size: int, output_size: int = 1, threshold: float = 0.5):
        self.input_size = input_size
        self.output_size = output_size
        self.threshold = threshold
        self._model: Optional[LogisticRegressionOptimizer] = None

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> LogisticRegressionOptimizer:
        self._model = LogisticRegressionOptimizer(
            input_size=X.shape[1], output_size=self.output_size)
        training_result = self._model.train(X, y, **kwargs)
        return self._model

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._model is None: raise RuntimeError("Modelo no entrenado")
        return self._model.predict(X)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        preds = self.predict(X)
        return float(np.mean(preds == y))


class ModelEvaluator:
    def __init__(self):
        self._results: Dict[str, Any] = {}

    def accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.mean(y_true == y_pred))

    def precision(self, y_true: np.ndarray, y_pred: np.ndarray, average: str = "binary") -> float:
        y_true = y_true.astype(int)
        y_pred = y_pred.astype(int)
        if average == "binary":
            tp = int(np.sum((y_pred == 1) & (y_true == 1)))
            fp = int(np.sum((y_pred == 1) & (y_true == 0)))
            return tp / max(1, tp + fp)
        return 0.0

    def recall(self, y_true: np.ndarray, y_pred: np.ndarray, average: str = "binary") -> float:
        y_true = y_true.astype(int)
        y_pred = y_pred.astype(int)
        if average == "binary":
            tp = int(np.sum((y_pred == 1) & (y_true == 1)))
            fn = int(np.sum((y_pred == 0) & (y_true == 1)))
            return tp / max(1, tp + fn)
        return 0.0

    def f1_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        p = self.precision(y_true, y_pred)
        r = self.recall(y_true, y_pred)
        return 2.0 * p * r / max(1e-8, p + r)

    def evaluate_all(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        return {
            'accuracy': self.accuracy(y_true, y_pred),
            'precision': self.precision(y_true, y_pred),
            'recall': self.recall(y_true, y_pred),
            'f1': self.f1_score(y_true, y_pred),
        }


def create_logistic_regression_optimizer(input_size: int = 4, output_size: int = 8) -> LogisticRegressionOptimizer:
    return LogisticRegressionOptimizer(input_size=input_size, output_size=output_size)


if __name__ == "__main__":
    logger.info("RF_SL1_8.py cargado exitosamente")