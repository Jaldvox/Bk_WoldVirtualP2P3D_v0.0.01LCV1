"""
RF_SL1_7.py - K-Nearest Neighbors Avanzado (KNN)
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


class KNearestNeighborsOptimizer(NeuronaMemoriaBase):
    def __init__(self, input_size: int = 4, output_size: int = 8, nombre: str = "KNearestNeighborsOptimizer"):
        super().__init__(input_size, output_size, nombre)
        self.inicializar_pesos()
        self._k: int = 5
        self._distance_metric: str = "euclidean"
        self._data: Optional[np.ndarray] = None
        self._labels: Optional[np.ndarray] = None
        self._is_fitted: bool = False
        self._n_samples: int = 0
        self._convergence_history: List[float] = []
        self._training_time: float = 0.0
        self._accuracy: float = 0.0

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

    def compute_loss(self, prediccion: np.ndarray, objetivo: np.ndarray) -> float:
        if prediccion.ndim > 1 and prediccion.shape[1] > 1:
            diff = prediccion - objetivo
            return float(np.mean(diff ** 2))
        diff = prediccion - objetivo
        return float(np.mean(diff ** 2))

    def _euclidean(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        diff = a[:, np.newaxis, :] - b[np.newaxis, :, :]
        return np.sqrt(np.sum(diff ** 2, axis=2))

    def _manhattan(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        diff = a[:, np.newaxis, :] - b[np.newaxis, :, :]
        return np.sum(np.abs(diff), axis=2)

    def _cosine(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        dot = np.sum(a[:, np.newaxis, :] * b[np.newaxis, :, :], axis=2)
        norm_a = np.sqrt(np.sum(a ** 2, axis=1)).reshape(-1, 1)
        norm_b = np.sqrt(np.sum(b ** 2, axis=1)).reshape(1, -1)
        return 1.0 - dot / np.maximum(1e-8, norm_a * norm_b)

    def _compute_distances(self, X: np.ndarray) -> np.ndarray:
        if self._distance_metric == "manhattan": return self._manhattan(X, self._data)
        if self._distance_metric == "cosine": return self._cosine(X, self._data)
        return self._euclidean(X, self._data)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self._is_fitted: raise RuntimeError("Modelo no entrenado")
        distances = self._compute_distances(X)
        n = len(X)
        n_classes = len(np.unique(self._labels))
        probs = np.zeros((n, n_classes))
        for i in range(n):
            idx = np.argpartition(distances[i], self._k)[:self._k]
            for c in range(n_classes):
                probs[i][c] = np.sum(self._labels[idx] == c) / self._k
        return probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        if len(probs) == 0: return np.array([])
        return np.argmax(probs, axis=1)

    def compute_accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        pred = self.predict(X)
        if len(pred) == 0: return 0.0
        return float(np.mean(pred == y.flatten()))

    def compute_margin(self, X: np.ndarray) -> float:
        if not self._is_fitted: return 0.0
        probs = self.predict_proba(X)
        margins = np.max(probs, axis=1) - np.max(np.sort(probs, axis=1)[:, -2], axis=1)
        return float(np.mean(margins))

    def backward(self, gradiente_salida: np.ndarray, entrada: np.ndarray):
        grad_pesos = np.dot(entrada.T, gradiente_salida)
        grad_sesgo = np.sum(gradiente_salida, axis=0, keepdims=True)
        self.historial_gradientes.append({'pesos': grad_pesos.copy(), 'norma': float(np.linalg.norm(grad_pesos))})
        return grad_pesos, grad_sesgo

    def update_weights(self, grad_pesos: np.ndarray, grad_sesgo: np.ndarray) -> None:
        self.pesos -= 0.01 * grad_pesos
        self.sesgo -= 0.01 * grad_sesgo

    def apply_gradient(self, entrada: np.ndarray, objetivo: np.ndarray) -> float:
        pred = self.forward(entrada)
        grad = pred - objetivo
        gp, gs = self.backward(grad, entrada)
        self.update_weights(gp, gs)
        return self.compute_loss(pred, objetivo)

    def train_step(self, entrada: np.ndarray, objetivo: np.ndarray) -> Dict[str, float]:
        loss_before = self.compute_loss(self.forward(entrada), objetivo)
        loss = self.apply_gradient(entrada, objetivo)
        return {'loss_before': loss_before, 'loss_after': loss, 'improvement': loss_before - loss}

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 1, verbose: bool = False) -> List[float]:
        t0 = time.perf_counter()
        self._data = X.copy()
        self._labels = y.astype(int) if y.dtype != int else y
        self._n_samples = len(X)
        self._is_fitted = True
        losses = []
        for epoch in range(max(1, epochs)):
            pred = self.predict(X)
            loss = self.compute_loss(pred.astype(float), self._labels.astype(float))
            losses.append(loss)
            self._convergence_history.append(loss)
            if verbose and (epoch + 1) % 5 == 0:
                logger.info(f"KNN Epoch {epoch+1}, loss={loss:.6f}")
        self._accuracy = self.compute_accuracy(X, self._labels)
        self._training_time = time.perf_counter() - t0
        return losses

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        acc = self.compute_accuracy(X, y)
        margin = self.compute_margin(X)
        pred = self.predict(X)
        mse = self.compute_loss(pred.astype(float), y.flatten().astype(float))
        return {'accuracy': acc, 'margin': margin, 'mse': mse, 'n_samples': len(X)}

    def obtener_estadisticas(self) -> Dict[str, Any]:
        stats = super().obtener_estadisticas()
        stats['k'] = self._k
        stats['distance_metric'] = self._distance_metric
        stats['is_fitted'] = self._is_fitted
        stats['n_samples'] = self._n_samples
        stats['accuracy'] = self._accuracy
        stats['training_time'] = self._training_time
        if self._convergence_history:
            stats['final_loss'] = float(self._convergence_history[-1])
        return stats

    def verificar_estabilidad(self) -> Dict[str, bool]:
        ok = {'pesos_inicializados': self.pesos is not None}
        if self.pesos is not None:
            ok['pesos_no_explosivos'] = float(np.max(np.abs(self.pesos))) < 10.0
            ok['pesos_no_nan'] = not bool(np.any(np.isnan(self.pesos)))
            ok['estado_entrenado'] = self._is_fitted
            ok['datos_cargados'] = self._data is not None
        return ok

    def save(self, filepath: str) -> None:
        state = {
            'pesos': self.pesos, 'sesgo': self.sesgo, 'nombre': self.nombre,
            'input_size': self.input_size, 'output_size': self.output_size,
            'k': self._k, 'distance_metric': self._distance_metric,
            'is_fitted': self._is_fitted, 'pasos': self.pasos,
            'data': self._data, 'labels': self._labels,
            'accuracy': self._accuracy, 'convergence_history': self._convergence_history,
        }
        with open(filepath, 'wb') as f: pickle.dump(state, f)

    def load(self, filepath: str) -> None:
        with open(filepath, 'rb') as f: state = pickle.load(f)
        self.__dict__.update({k: v for k, v in state.items() if k != 'convergence_history'})
        self._convergence_history = state.get('convergence_history', [])

    def get_config(self) -> Dict[str, Any]:
        return {
            'nombre': self.nombre, 'input_size': self.input_size, 'output_size': self.output_size,
            'k': self._k, 'distance_metric': self._distance_metric,
        }

    def summary(self) -> str:
        cfg = self.get_config()
        stats = self.obtener_estadisticas()
        return "\n".join([
            f"=== {cfg['nombre']} ===",
            f"K: {cfg['k']}, Metric: {cfg['distance_metric']}",
            f"Input: {cfg['input_size']}, Output: {cfg['output_size']}",
            f"Fitted: {stats.get('is_fitted', False)}, Acc: {stats.get('accuracy', 0):.4f}",
            f"Samples: {stats.get('n_samples', 0)}, Params: {self.params_count()}",
        ])

    def __str__(self) -> str:
        return f"{self.nombre}(ent={self.input_size},sal={self.output_size})"


class DistanceMetric:
    @staticmethod
    def euclidean(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.sqrt(np.sum((a - b) ** 2, axis=1))

    @staticmethod
    def manhattan(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.sum(np.abs(a - b), axis=1)

    @staticmethod
    def cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        dot = np.sum(a * b, axis=1)
        norm_a = np.sqrt(np.sum(a ** 2, axis=1))
        norm_b = np.sqrt(np.sum(b ** 2, axis=1))
        return 1.0 - dot / np.maximum(1e-8, norm_a * norm_b)

    @staticmethod
    def minkowski(a: np.ndarray, b: np.ndarray, p: int = 3) -> np.ndarray:
        return np.sum(np.abs(a - b) ** p, axis=1) ** (1.0 / p)

    @staticmethod
    def get_metric(name: str) -> str:
        valid = ["euclidean", "manhattan", "cosine", "minkowski"]
        return name if name in valid else "euclidean"


class KNNSearcher:
    def __init__(self, k: int = 5, metric: str = "euclidean"):
        self.k = k
        self.metric = metric

    def find_neighbors(self, query: np.ndarray, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        distances = DistanceMetric.euclidean(query.reshape(1, -1), data)
        indices = np.argpartition(distances, self.k)[:self.k]
        return indices, distances[indices]

    def find_neighbors_batch(self, queries: np.ndarray, data: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        results = []
        for q in queries:
            idx, dist = self.find_neighbors(q, data)
            results.append((idx, dist))
        return results


class VoteAggregator:
    def __init__(self, weighted: bool = False):
        self.weighted = weighted

    def aggregate(self, labels: np.ndarray, distances: Optional[np.ndarray] = None) -> int:
        if distances is not None and self.weighted:
            weights = 1.0 / (distances + 1e-8)
            unique, inverse = np.unique(labels, return_inverse=True)
            weighted_counts = np.bincount(inverse, weights=weights, minlength=len(unique))
            return int(unique[np.argmax(weighted_counts)])
        unique, counts = np.unique(labels, return_counts=True)
        return int(unique[np.argmax(counts)])

    def aggregate_proba(self, labels: np.ndarray, n_classes: int) -> np.ndarray:
        probs = np.zeros(n_classes)
        for label in labels:
            probs[label] += 1
        return probs / max(1, len(labels))


class FeatureNormalizer:
    def __init__(self):
        self._mean: Optional[np.ndarray] = None
        self._std: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> 'FeatureNormalizer':
        self._mean = np.mean(X, axis=0)
        self._std = np.std(X, axis=0) + 1e-8
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self._mean is None: raise RuntimeError("Normalizer no ajustado")
        return (X - self._mean) / self._std

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


class KNNClassifier:
    def __init__(self, k: int = 5, metric: str = "euclidean", normalize: bool = False):
        self.k = k
        self.metric = metric
        self.normalize = normalize
        self.normalizer = FeatureNormalizer() if normalize else None
        self.searcher = KNNSearcher(k, metric)
        self.voter = VoteAggregator()
        self._data: Optional[np.ndarray] = None
        self._labels: Optional[np.ndarray] = None
        self._is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'KNNClassifier':
        if self.normalize:
            X = self.normalizer.fit_transform(X)
        self._data = X.copy()
        self._labels = y.astype(int) if y.dtype != int else y
        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self._is_fitted: raise RuntimeError("No entrenado")
        if self.normalize: X = self.normalizer.transform(X)
        return np.array([self.voter.aggregate(self._labels[np.argpartition(
            DistanceMetric.euclidean(X[i:i+1], self._data)[0], self.k)[:self.k]]) for i in range(len(X))])


class ConfusionMatrix:
    def __init__(self, n_classes: int):
        self.n_classes = n_classes
        self._matrix = np.zeros((n_classes, n_classes), dtype=int)

    def update(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        for t, p in zip(y_true.astype(int), y_pred.astype(int)):
            if 0 <= t < self.n_classes and 0 <= p < self.n_classes:
                self._matrix[t][p] += 1

    def get_matrix(self) -> np.ndarray: return self._matrix.copy()

    def accuracy(self) -> float:
        total = self._matrix.sum()
        return float(np.trace(self._matrix) / max(1, total))


class RadiusNeighborSearcher:
    def __init__(self, radius: float = 1.0):
        self.radius = radius

    def find(self, query: np.ndarray, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        distances = DistanceMetric.euclidean(query.reshape(1, -1), data)
        mask = distances <= self.radius
        indices = np.where(mask)[0]
        return indices, distances[indices]

    def find_batch(self, queries: np.ndarray, data: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        return [self.find(q, data) for q in queries]


class ModelEvaluator:
    def __init__(self, k_values: List[int] = None):
        self.k_values = k_values or [1, 3, 5, 7, 9]
        self._scores: Dict[int, float] = {}

    def evaluate_k_range(self, X_train: np.ndarray, y_train: np.ndarray,
                             X_test: np.ndarray, y_test: np.ndarray) -> Dict[int, float]:
        for k in self.k_values:
            knn = KNearestNeighborsOptimizer(X_train.shape[1], len(np.unique(y_train)))
            knn._data = X_train.copy()
            knn._labels = y_train.astype(int) if y_train.dtype != int else y_train
            knn._k = k
            knn._is_fitted = True
            acc = knn.compute_accuracy(X_test, y_test)
            self._scores[k] = acc
        return self._scores

    def get_best_k(self) -> int:
        if not self._scores: return 5
        return max(self._scores, key=self._scores.get)

    def get_scores(self) -> Dict[int, float]: return self._scores.copy()


def create_k_nearest_neighbors_optimizer(input_size: int = 4, output_size: int = 8) -> KNearestNeighborsOptimizer:
    return KNearestNeighborsOptimizer(input_size=input_size, output_size=output_size)


if __name__ == "__main__":
    logger.info("RF_SL1_7.py cargado exitosamente")