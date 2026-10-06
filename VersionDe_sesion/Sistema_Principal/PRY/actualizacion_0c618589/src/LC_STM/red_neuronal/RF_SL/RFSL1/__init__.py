"""
RFSL1 __init__.py - Neuronas de memoria RF_SL
Optimizado con clases inline y estructura eficiente.
Paquete de 10 optimizadores de aprendizaje supervisado.
"""

import logging
import time
import pickle
import itertools
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Callable

logger = logging.getLogger(__name__)

from .RF_SL1_1 import GradientBoostingOptimizer
from .RF_SL1_10 import IntegratedSupervisedLearningOptimizer
from .RF_SL1_2 import SupportVectorMachineOptimizer
from .RF_SL1_3 import RandomForestOptimizer
from .RF_SL1_4 import NeuralNetworkOptimizer
from .RF_SL1_5 import DecisionTreeOptimizer
from .RF_SL1_6 import NaiveBayesOptimizer
from .RF_SL1_7 import KNearestNeighborsOptimizer
from .RF_SL1_8 import LogisticRegressionOptimizer
from .RF_SL1_9 import LinearRegressionOptimizer


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


class ModelRegistry:
    def __init__(self):
        self._models: Dict[str, Any] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}

    def register(self, name: str, model: Any, info: Optional[Dict[str, Any]] = None) -> None:
        self._models[name] = model
        self._metadata[name] = info or {}

    def get(self, name: str) -> Any:
        if name not in self._models: raise KeyError(f"Modelo '{name}' no registrado")
        return self._models[name]

    def has(self, name: str) -> bool: return name in self._models

    def list_models(self) -> List[str]: return list(self._models.keys())

    def get_metadata(self, name: str) -> Dict[str, Any]: return self._metadata.get(name, {})

    def count(self) -> int: return len(self._models)

    def clear(self) -> None:
        self._models.clear()
        self._metadata.clear()


class TrainingConfig:
    def __init__(self, epochs: int = 100, learning_rate: float = 0.01,
                 batch_size: int = 32, validation_split: float = 0.2,
                 regularization: str = "l2", reg_lambda: float = 0.01,
                 early_stopping: bool = False, patience: int = 20,
                 shuffle: bool = True, verbose: bool = False):
        self.epochs = max(1, int(epochs))
        self.learning_rate = max(1e-6, float(learning_rate))
        self.batch_size = max(1, int(batch_size))
        self.validation_split = max(0.0, min(0.99, float(validation_split)))
        self.regularization = regularization if regularization in ("l1", "l2") else "l2"
        self.reg_lambda = max(0.0, float(reg_lambda))
        self.early_stopping = bool(early_stopping)
        self.patience = max(1, int(patience))
        self.shuffle = bool(shuffle)
        self.verbose = bool(verbose)

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "TrainingConfig":
        return cls(**{k: v for k, v in config.items() if k in cls().__dict__})


class MetricsCollector:
    def __init__(self):
        self._metrics: Dict[str, List[float]] = {}
        self._timestamps: List[float] = []

    def record(self, name: str, value: float) -> None:
        if name not in self._metrics: self._metrics[name] = []
        self._metrics[name].append(float(value))
        self._timestamps.append(time.time())

    def get_metric(self, name: str) -> Optional[List[float]]: return self._metrics.get(name)

    def get_average(self, name: str) -> Optional[float]:
        vals = self._metrics.get(name)
        return float(np.mean(vals)) if vals else None

    def get_all_averages(self) -> Dict[str, float]:
        return {k: self.get_average(k) for k in self._metrics if self.get_average(k) is not None}

    def reset(self) -> None:
        self._metrics.clear()
        self._timestamps.clear()


class DataPipeline:
    def __init__(self, batch_size: int = 32, shuffle: bool = True):
        self.batch_size = max(1, int(batch_size))
        self.shuffle = bool(shuffle)

    def generate_batches(self, X: np.ndarray, y: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        n = len(X)
        indices = np.arange(n)
        if self.shuffle: np.random.shuffle(indices)
        batches = []
        for i in range(0, n, self.batch_size):
            batch_idx = indices[i:i+self.batch_size]
            batches.append((X[batch_idx], y[batch_idx]))
        return batches

    def get_n_batches(self, n_samples: int) -> int: return max(1, n_samples // self.batch_size)


def build_model(model_type: str, input_size: int, output_size: int) -> Any:
    builders = {
        "gradient_boosting": lambda: GradientBoostingOptimizer(input_size, output_size),
        "svm": lambda: SupportVectorMachineOptimizer(input_size, output_size),
        "random_forest": lambda: RandomForestOptimizer(input_size, output_size),
        "neural_network": lambda: NeuralNetworkOptimizer(input_size, output_size),
        "decision_tree": lambda: DecisionTreeOptimizer(input_size, output_size),
        "naive_bayes": lambda: NaiveBayesOptimizer(input_size, output_size),
        "knn": lambda: KNearestNeighborsOptimizer(input_size, output_size),
        "logistic_regression": lambda: LogisticRegressionOptimizer(input_size, output_size),
        "linear_regression": lambda: LinearRegressionOptimizer(input_size, output_size),
        "integrated": lambda: IntegratedSupervisedLearningOptimizer(input_size, output_size),
    }
    builder = builders.get(model_type.lower())
    if builder is None: raise ValueError(f"Tipo desconocido: {model_type}")
    return builder()


def train_all_models(X: np.ndarray, y: np.ndarray, epochs: int = 100,
                       learning_rate: float = 0.01, **kwargs) -> Dict[str, Any]:
    results = {}
    for model_type in _MODEL_TYPES:
        model = build_model(model_type, X.shape[1], y.shape[1] if len(y.shape) > 1 else 1)
        if hasattr(model, 'train'):
            result = model.train(X, y, epochs=epochs, learning_rate=learning_rate, verbose=False)
            results[model_type] = result
    return results

_MODEL_TYPES = [
    "gradient_boosting", "svm", "random_forest", "neural_network",
    "decision_tree", "naive_bayes", "knn", "logistic_regression",
    "linear_regression", "integrated",
]


def compare_models(models: Dict[str, Any], X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    comparison = {}
    for name, model in models.items():
        try:
            if hasattr(model, 'evaluate'):
                comparison[name] = model.evaluate(X, y).get('r2', 0.0)
            elif hasattr(model, 'predict') and hasattr(model, 'r2_score'):
                comparison[name] = model.r2_score(y, model.predict(X))
        except RuntimeError:
            comparison[name] = validate_model(model, X, y)
    return comparison


def validate_model(model: Any, X: np.ndarray, y: np.ndarray,
                    metric: str = "r2") -> float:
    try:
        if hasattr(model, 'evaluate'): return model.evaluate(X, y).get(metric, 0.0)
        if hasattr(model, 'predict') and hasattr(model, 'r2_score'):
            return model.r2_score(y, model.predict(X))
    except RuntimeError:
        if hasattr(model, 'train'):
            model.train(X, y, epochs=5, verbose=False)
            return validate_model(model, X, y, metric)
    return 0.0


class HyperparameterSearch:
    def __init__(self, param_grid: Dict[str, List[Any]]):
        self.param_grid = param_grid
        self._results: List[Dict[str, Any]] = []

    def enumerate_combinations(self) -> List[Dict[str, Any]]:
        keys = list(self.param_grid.keys())
        values = list(self.param_grid.values())
        combos = []
        for combo in itertools.product(*values):
            combos.append(dict(zip(keys, combo)))
        return combos

    def search(self, X: np.ndarray, y: np.ndarray,
                  model_type: str, metric: str = "r2") -> Dict[str, Any]:
        best_score = float('-inf')
        best_params = {}
        for params in self.enumerate_combinations():
            model = build_model(model_type, X.shape[1],
                                y.shape[1] if len(y.shape) > 1 else 1)
            if hasattr(model, 'train'):
                import inspect as _ins
                sig = _ins.signature(model.train)
                valid = {k: v for k, v in params.items() if k in sig.parameters}
                model.train(X, y, epochs=5, verbose=False, **valid)
            score = validate_model(model, X, y, metric=metric)
            if score > best_score:
                best_score = score
                best_params = params
        return {'best_score': best_score, 'best_params': best_params}


class PerformanceMonitor:
    def __init__(self):
        self._start_time: Optional[float] = None
        self._checkpoints: Dict[str, float] = {}

    def start(self) -> None: self._start_time = time.time()

    def checkpoint(self, name: str) -> None:
        self._checkpoints[name] = time.time()

    def elapsed(self) -> Optional[float]:
        if self._start_time is None: return None
        return time.time() - self._start_time

    def checkpoint_duration(self, name: str) -> Optional[float]:
        if name not in self._checkpoints: return None
        return time.time() - self._checkpoints[name]

    def get_all_durations(self) -> Dict[str, float]:
        return {k: self.checkpoint_duration(k) or 0.0 for k in self._checkpoints}


class FeatureSelector:
    def __init__(self, n_features: int = 10, method: str = "variance"):
        self.n_features = max(1, int(n_features))
        self.method = method if method in ("variance", "correlation", "mutual_info") else "variance"
        self._selected_indices: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> None:
        if self.method == "variance":
            variance = np.var(X, axis=0)
            self._selected_indices = np.argsort(variance)[-self.n_features:]
        else:
            variance = np.var(X, axis=0) + 1e-8
            self._selected_indices = np.argsort(variance)[-self.n_features:]

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self._selected_indices is None: raise RuntimeError("No ajustado")
        return X[:, self._selected_indices]

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)

    def get_selected(self) -> Optional[np.ndarray]:
        return self._selected_indices.copy() if self._selected_indices is not None else None


def serialize_model(model: Any, filepath: str) -> None:
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)


def deserialize_model(filepath: str) -> Any:
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def benchmark_model(model: Any, X: np.ndarray, y: np.ndarray,
                     n_runs: int = 5) -> Dict[str, float]:
    times = []
    for _ in range(n_runs):
        start = time.time()
        if hasattr(model, 'predict'):
            model.predict(X)
        elapsed = time.time() - start
        times.append(elapsed)
    return {
        'avg_time': float(np.mean(times)),
        'min_time': float(np.min(times)),
        'max_time': float(np.max(times)),
        'std_time': float(np.std(times)),
    }


def summary(model: Any) -> Dict[str, Any]:
    info = model.info() if hasattr(model, 'info') else str(model)
    params = model.params_count() if hasattr(model, 'params_count') else 0
    stats = model.obtener_estadisticas() if hasattr(model, 'obtener_estadisticas') else {}
    stability = model.verificar_estabilidad() if hasattr(model, 'verificar_estabilidad') else {}
    return {
        'info': info, 'params': params, 'stats': stats, 'stability': stability,
    }


def create_default_pipeline(input_size: int = 4, output_size: int = 8) -> Dict[str, Any]:
    return {
        'data_pipeline': DataPipeline(batch_size=32, shuffle=True),
        'training_config': TrainingConfig(epochs=100, learning_rate=0.01),
        'metrics_collector': MetricsCollector(),
        'model_registry': ModelRegistry(),
        'input_size': input_size, 'output_size': output_size,
    }


def create_ensemble_predictions(models: List[Any], X: np.ndarray) -> np.ndarray:
    preds = []
    for m in models:
        if not hasattr(m, 'predict'): continue
        try:
            preds.append(m.predict(X))
        except RuntimeError:
            if hasattr(m, 'train'):
                m.train(X, np.zeros(len(X)), epochs=5, verbose=False)
                preds.append(m.predict(X))
    if not preds: raise ValueError("Sin predicciones")
    return np.mean(np.array(preds), axis=0)


def quick_train(model: Any, X: np.ndarray, y: np.ndarray,
                epochs: int = 10, **kwargs) -> Dict[str, Any]:
    if hasattr(model, 'train'):
        return model.train(X, y, epochs=epochs, verbose=False, **kwargs)
    raise RuntimeError("Modelo no soporta train")


def quick_predict(model: Any, X: np.ndarray) -> np.ndarray:
    try:
        if hasattr(model, 'predict'): return model.predict(X)
        if hasattr(model, 'forward'): return model.forward(X)
    except RuntimeError:
        if hasattr(model, 'train'):
            y_dummy = np.zeros((len(X), 1))
            model.train(X, y_dummy, epochs=3, verbose=False)
            if hasattr(model, 'predict'): return model.predict(X)
    raise RuntimeError("Modelo no soporta predict")


def quick_evaluate(model: Any, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    try:
        if hasattr(model, 'evaluate'): return model.evaluate(X, y)
        return {'predictions': quick_predict(model, X)}
    except RuntimeError:
        return {'predictions': None}


def get_available_models() -> List[str]:
    return [
        "gradient_boosting", "svm", "random_forest", "neural_network",
        "decision_tree", "naive_bayes", "knn", "logistic_regression",
        "linear_regression", "integrated",
    ]


__all__ = [
    'GradientBoostingOptimizer',
    'IntegratedSupervisedLearningOptimizer',
    'SupportVectorMachineOptimizer',
    'RandomForestOptimizer',
    'NeuralNetworkOptimizer',
    'DecisionTreeOptimizer',
    'NaiveBayesOptimizer',
    'KNearestNeighborsOptimizer',
    'LogisticRegressionOptimizer',
    'LinearRegressionOptimizer',
    'NeuronaMemoriaBase',
    'ModelRegistry',
    'TrainingConfig',
    'MetricsCollector',
    'DataPipeline',
    'build_model',
    'train_all_models',
    'compare_models',
    'validate_model',
    'serialize_model',
    'deserialize_model',
    'benchmark_model',
    'summary',
    'create_default_pipeline',
    'create_ensemble_predictions',
    'quick_train',
    'quick_predict',
    'quick_evaluate',
    'get_available_models',
    'HyperparameterSearch',
    'PerformanceMonitor',
    'FeatureSelector',
]

logger.info("RFSL1 cargado correctamente")