"""
RF_SL1_10.py - Sistema Integrado de Supervised Learning Avanzado
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


class IntegratedSupervisedLearningOptimizer(NeuronaMemoriaBase):
    def __init__(self, input_size: int = 4, output_size: int = 8, nombre: str = "IntegratedSupervisedLearningOptimizer"):
        super().__init__(input_size, output_size, nombre)
        self.inicializar_pesos()
        self._learning_rate: float = 0.01
        self._epochs: int = 500
        self._batch_size: int = 32
        self._regularization: str = "l2"
        self._reg_lambda: float = 0.01
        self._is_fitted: bool = False
        self._convergence_history: List[float] = []
        self._training_time: float = 0.0
        self._loss_history: List[float] = []
        self._val_accuracy: float = 0.0
        self._n_samples: int = 0
        self._models: List[Any] = []
        self._ensemble_method: str = "majority"
        self._validation_split: float = 0.2
        self._shuffle: bool = True
        self._early_stopping: bool = False
        self._patience: int = 20
        self._best_loss: float = float('inf')
        self._no_improve_count: int = 0

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
        if len(y_true.shape) > 1 and y_true.shape[1] > 1:
            diff = y_pred - y_true
            mse = float(np.mean(diff ** 2))
        else:
            diff = (y_pred.flatten() - y_true.flatten())
            mse = float(np.mean(diff ** 2))
        if self._regularization == "l2" and self.pesos is not None:
            mse += 0.5 * self._reg_lambda * float(np.sum(self.pesos ** 2))
        elif self._regularization == "l1" and self.pesos is not None:
            mse += self._reg_lambda * float(np.sum(np.abs(self.pesos)))
        return mse

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
        start_time = time.time()
        n_samples = len(X)
        if y.ndim == 1 and self.output_size == 1: y = y.reshape(-1, 1)
        X_val, y_val = self._split_data(X, y, self._validation_split)
        losses = []
        for epoch in range(self._epochs):
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            epoch_loss = 0.0
            batches = 0
            for i in range(0, n_samples, self._batch_size):
                batch_X = X_shuffled[i:i+self._batch_size]
                batch_y = y_shuffled[i:i+self._batch_size]
                y_pred = self.forward(batch_X)
                loss = self.compute_loss(batch_y, y_pred)
                epoch_loss += loss
                batches += 1
                error = y_pred - batch_y
                grad_pesos, grad_sesgo = self.backward(error, batch_X)
                self.pesos -= self._learning_rate * grad_pesos
                self.sesgo -= self._learning_rate * grad_sesgo
            avg_loss = epoch_loss / max(1, batches)
            losses.append(avg_loss)
            self._loss_history.append(avg_loss)
            self._convergence_history.append(avg_loss)
            if self._early_stopping and X_val is not None:
                val_pred = self.predict(X_val)
                val_loss = self.compute_loss(y_val, val_pred)
                if val_loss < self._best_loss:
                    self._best_loss = val_loss
                    self._no_improve_count = 0
                else:
                    self._no_improve_count += 1
                    if self._no_improve_count >= self._patience:
                        if verbose: logger.info(f"Early stopping en epoch {epoch+1}")
                        break
            if verbose and (epoch + 1) % 100 == 0:
                logger.info(f"Epoch {epoch+1}/{self._epochs}, Loss: {avg_loss:.6f}")
        self._training_time = time.time() - start_time
        self._is_fitted = True
        self._n_samples = n_samples
        if X_val is not None and y_val is not None:
            val_pred = self.predict(X_val)
            self._val_accuracy = float(np.mean(np.argmax(val_pred, axis=1) == np.argmax(y_val, axis=1))) if len(y_val.shape) > 1 and y_val.shape[1] > 1 else float(np.mean(val_pred.flatten() == y_val.flatten()))
        return {'losses': losses, 'final_loss': float(losses[-1]), 'training_time': self._training_time}

    def _split_data(self, X: np.ndarray, y: np.ndarray, split: float) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if split <= 0.0 or split >= 1.0: return None, None
        n = len(X)
        n_val = int(n * split)
        indices = np.random.permutation(n)
        val_idx = indices[:n_val]
        X_val = X[val_idx]
        y_val = y[val_idx]
        return X_val, y_val

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self._is_fitted and self.pesos is None: self.inicializar_pesos()
        return self.forward(X)

    def predict_classes(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict(X)
        if len(probs.shape) > 1 and probs.shape[1] > 1:
            return np.argmax(probs, axis=1)
        return (probs.flatten() >= 0.5).astype(int)

    def r2_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        ss_res = float(np.sum((y_true - y_pred) ** 2))
        ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
        return 1.0 - ss_res / max(1e-8, ss_tot)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        preds = self.predict(X)
        return {
            'predictions': preds, 'r2': self.r2_score(y, preds),
            'n_samples': len(X), 'classes': self.predict_classes(X),
        }

    def add_model(self, model: Any) -> None:
        self._models.append(model)

    def get_ensemble_predictions(self, X: np.ndarray) -> np.ndarray:
        if not self._models: return self.predict(X)
        all_preds = [m.predict(X) if hasattr(m, 'predict') else m for m in self._models]
        all_preds.append(self.predict(X))
        return np.mean(all_preds, axis=0)

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
        stats['val_accuracy'] = self._val_accuracy
        stats['n_models'] = len(self._models)
        stats['ensemble_method'] = self._ensemble_method
        stats['early_stopping'] = self._early_stopping
        stats['training_time'] = self._training_time
        return stats

    def verificar_estabilidad(self) -> Dict[str, bool]:
        ok = {'pesos_inicializados': self.pesos is not None}
        if self.pesos is not None:
            ok['pesos_no_explosivos'] = float(np.max(np.abs(self.pesos))) < 10.0
            ok['pesos_no_nan'] = not bool(np.any(np.isnan(self.pesos)))
            ok['pesos_no_inf'] = not bool(np.any(np.isinf(self.pesos)))
        ok['sesgo_inicializado'] = self.sesgo is not None
        ok['modelo_entrenado'] = self._is_fitted
        ok['estable'] = self._no_improve_count < self._patience
        return ok

    def set_regularization(self, reg_type: str, lambda_val: float = 0.01) -> None:
        self._regularization = reg_type if reg_type in ("l1", "l2") else "l2"
        self._reg_lambda = max(0.0, float(lambda_val))

    def enable_ensemble(self, method: str = "majority") -> None:
        self._ensemble_method = method if method in ("majority", "average", "weighted") else "majority"

    def enable_early_stopping(self, patience: int = 20) -> None:
        self._early_stopping = True
        self._patience = max(1, int(patience))
        self._best_loss = float('inf')
        self._no_improve_count = 0

    def get_loss_history(self) -> List[float]: return self._loss_history.copy()

    def get_models_count(self) -> int: return len(self._models)

    def __str__(self) -> str:
        return f"{self.nombre}(ent={self.input_size},sal={self.output_size},pasos={self.pasos})"


class EnsembleManager:
    def __init__(self, method: str = "average"):
        self.method = method if method in ("majority", "average", "weighted") else "average"
        self._models: List[Any] = []
        self._weights: List[float] = []

    def add(self, model: Any, weight: float = 1.0) -> None:
        self._models.append(model)
        self._weights.append(max(0.01, float(weight)))

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self._models: raise RuntimeError("Sin modelos")
        preds = []
        for m in self._models:
            p = m.predict(X) if hasattr(m, 'predict') else m
            preds.append(p)
        if self.method == "majority":
            stacked = np.array([np.argmax(p, axis=1) if len(p.shape) > 1 else p for p in preds])
            result = []
            for i in range(stacked.shape[1]):
                vals, counts = np.unique(stacked[:, i], return_counts=True)
                result.append(vals[np.argmax(counts)])
            return np.array(result)
        return np.mean(preds, axis=0)

    def get_count(self) -> int: return len(self._models)


class DataSplitter:
    def __init__(self, val_ratio: float = 0.2, test_ratio: float = 0.2, seed: int = 42):
        self.val_ratio = max(0.0, min(0.99, float(val_ratio)))
        self.test_ratio = max(0.0, min(0.99, float(test_ratio)))
        self.seed = int(seed)
        self._train_idx: Optional[np.ndarray] = None
        self._val_idx: Optional[np.ndarray] = None
        self._test_idx: Optional[np.ndarray] = None

    def split(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> Tuple[Tuple[np.ndarray, ...], ...]:
        np.random.seed(self.seed)
        n = len(X)
        n_test = int(n * self.test_ratio)
        n_val = int(n * self.val_ratio)
        indices = np.random.permutation(n)
        test_idx = indices[:n_test]
        val_idx = indices[n_test:n_test + n_val]
        train_idx = indices[n_test + n_val:]
        self._train_idx = train_idx
        self._val_idx = val_idx
        self._test_idx = test_idx
        if y is not None:
            return ((X[train_idx], y[train_idx]), (X[val_idx], y[val_idx]), (X[test_idx], y[test_idx]))
        return ((X[train_idx],), (X[val_idx],), (X[test_idx],))

    def get_sizes(self, n: int) -> Dict[str, int]:
        n_test = int(n * self.test_ratio)
        n_val = int(n * self.val_ratio)
        return {'train': n - n_test - n_val, 'val': n_val, 'test': n_test}


class ModelValidator:
    def __init__(self, n_folds: int = 5):
        self.n_folds = max(2, int(n_folds))
        self._scores: List[float] = []

    def validate(self, model: Any, X: np.ndarray, y: np.ndarray, metric: str = "accuracy") -> float:
        n = len(X)
        fold_size = n // self.n_folds
        scores = []
        for fold in range(self.n_folds):
            start = fold * fold_size
            end = start + fold_size if fold < self.n_folds - 1 else n
            X_test_f = X[start:end]
            y_test_f = y[start:end]
            X_train_f = np.vstack([X[:start], X[end:]]) if start > 0 else X[end:]
            y_train_f = np.vstack([y[:start], y[end:]]) if start > 0 else y[end:]
            if hasattr(model, 'train'):
                model.train(X_train_f, y_train_f, epochs=1)
            pred = model.predict(X_test_f) if hasattr(model, 'predict') else None
            if pred is not None and metric == "accuracy":
                acc = float(np.mean(np.argmax(pred, axis=1) == np.argmax(y_test_f, axis=1))) if len(y_test_f.shape) > 1 else float(np.mean(pred.flatten() == y_test_f.flatten()))
                scores.append(acc)
        avg = float(np.mean(scores)) if scores else 0.0
        self._scores = scores
        return avg

    def get_scores(self) -> List[float]: return self._scores.copy()


class SupervisedTrainer:
    def __init__(self, model: Any, X: np.ndarray, y: np.ndarray):
        self.model = model
        self.X = X
        self.y = y
        self._history: List[Dict[str, Any]] = []

    def run_training(self, epochs: int, **kwargs) -> Dict[str, Any]:
        result = self.model.train(self.X, self.y, epochs=epochs, **kwargs)
        self._history.append({'epochs': epochs, 'result': result})
        return result

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        if hasattr(self.model, 'evaluate'):
            return self.model.evaluate(X, y)
        preds = self.model.predict(X) if hasattr(self.model, 'predict') else self.model
        return {'predictions': preds}

    def get_history(self) -> List[Dict[str, Any]]: return self._history.copy()


class PredictionAggregator:
    def __init__(self, strategy: str = "mean"):
        self.strategy = strategy if strategy in ("mean", "median", "weighted") else "mean"

    def aggregate(self, predictions: List[np.ndarray]) -> np.ndarray:
        if not predictions: raise ValueError("Sin predicciones")
        if self.strategy == "median":
            return np.median(np.array(predictions), axis=0)
        return np.mean(np.array(predictions), axis=0)

    def aggregate_weighted(self, predictions: List[np.ndarray], weights: List[float]) -> np.ndarray:
        w = np.array(weights)
        w = w / w.sum()
        return np.sum(np.array(predictions) * w.reshape(-1, 1), axis=0)

    def get_strategy(self) -> str: return self.strategy

    def count_predictions(self, predictions: List[np.ndarray]) -> int:
        return sum(len(p) for p in predictions)

def create_integrated_supervised_learning_optimizer(input_size: int = 4, output_size: int = 8) -> IntegratedSupervisedLearningOptimizer:
    return IntegratedSupervisedLearningOptimizer(input_size=input_size, output_size=output_size)

if __name__ == "__main__":
    logger.info("RF_SL1_10.py cargado exitosamente")