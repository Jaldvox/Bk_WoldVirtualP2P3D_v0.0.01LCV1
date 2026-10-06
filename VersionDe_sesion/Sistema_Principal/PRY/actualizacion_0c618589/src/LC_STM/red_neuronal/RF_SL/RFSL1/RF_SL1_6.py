"""
RF_SL1_6.py - Naive Bayes Avanzado
Optimizado con clases inline y estructura eficiente.
"""

import numpy as np
import logging
import time
import pickle
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


class NaiveBayesOptimizer(NeuronaMemoriaBase):
    def __init__(self, input_size: int = 4, output_size: int = 8, nombre: str = "NaiveBayesOptimizer"):
        super().__init__(input_size, output_size, nombre)
        self.inicializar_pesos()
        self._var_smoothing: float = 1e-9
        self._distribution: str = "gaussian"
        self._classes: Optional[np.ndarray] = None
        self._class_priors: Optional[np.ndarray] = None
        self._feature_mean: Optional[Dict[int, np.ndarray]] = None
        self._feature_var: Optional[Dict[int, np.ndarray]] = None
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

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self._is_fitted: raise RuntimeError("Modelo no entrenado")
        n = len(X)
        n_classes = len(self._classes)
        log_probs = np.zeros((n, n_classes))
        for c in range(n_classes):
            prior = float(self._class_priors[c])
            log_prior = np.log(max(1e-8, prior))
            class_probs = np.zeros(n)
            for feat in range(self.input_size):
                mean = self._feature_mean[feat][c]
                var = self._feature_var[feat][c]
                diff = X[:, feat] - mean
                log_lik = -0.5 * np.log(2 * np.pi * var + self._var_smoothing)
                log_lik += -(diff ** 2) / (2 * var + self._var_smoothing)
                class_probs += log_lik
            log_probs[:, c] = log_prior + class_probs
        return self._softmax(log_probs)

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        return self._classes[np.argmax(probs, axis=1)]

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        e = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e / np.sum(e, axis=-1, keepdims=True)

    def compute_loss(self, prediccion: np.ndarray, objetivo: np.ndarray) -> float:
        if prediccion.ndim > 1 and prediccion.shape[1] > 1:
            ce = -np.log(np.clip(prediccion[np.arange(len(prediccion)), objetivo.astype(int)], 1e-8, 1))
            return float(np.mean(ce))
        diff = prediccion - objetivo
        return float(np.mean(diff ** 2))

    def compute_accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        pred = self.predict(X)
        return float(np.mean(pred == y.flatten()))

    def backward(self, gradiente_salida: np.ndarray, entrada: np.ndarray):
        grad_pesos = np.dot(entrada.T, gradiente_salida)
        grad_sesgo = np.sum(gradiente_salida, axis=0, keepdims=True)
        self.historial_gradientes.append({'pesos': grad_pesos.copy(), 'norma': float(np.linalg.norm(grad_pesos))})
        return grad_pesos, grad_sesgo

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 1, verbose: bool = False) -> List[float]:
        t0 = time.perf_counter()
        y_int = y.astype(int) if y.dtype != int else y
        self._classes = np.unique(y_int)
        self._n_samples = len(X)
        n_classes = len(self._classes)
        self._class_priors = np.zeros(n_classes)
        self._feature_mean = {}
        self._feature_var = {}
        for c in range(n_classes):
            mask = y_int == c
            self._class_priors[c] = max(1e-8, len(X[mask]) / len(X))
            for feat in range(self.input_size):
                feat_vals = X[mask][:, feat]
                if self._distribution == "gaussian":
                    self._feature_mean.setdefault(feat, np.zeros(n_classes))
                    self._feature_var.setdefault(feat, np.ones(n_classes))
                    self._feature_mean[feat][c] = float(np.mean(feat_vals))
                    self._feature_var[feat][c] = max(1e-8, float(np.var(feat_vals)))
                else:
                    self._feature_mean.setdefault(feat, np.zeros(n_classes))
                    self._feature_var.setdefault(feat, np.ones(n_classes))
                    self._feature_mean[feat][c] = float(np.mean(feat_vals))
                    self._feature_var[feat][c] = max(1e-8, float(np.var(feat_vals)) + self._var_smoothing)
        self._is_fitted = True
        self._accuracy = self.compute_accuracy(X, y_int)
        losses = []
        for epoch in range(max(1, epochs)):
            pred = self.predict_proba(X)
            pred_classes = self._classes[np.argmax(pred, axis=1)]
            loss = self.compute_loss(pred.astype(float), y_int.astype(float))
            losses.append(loss)
            self._convergence_history.append(loss)
            if verbose and (epoch + 1) % 5 == 0:
                logger.info(f"NB Epoch {epoch+1}, loss={loss:.6f}")
        self._training_time = time.perf_counter() - t0
        return losses

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        acc = self.compute_accuracy(X, y)
        pred = self.predict(X)
        mse = self.compute_loss(pred.astype(float), y.flatten().astype(float))
        return {'accuracy': acc, 'mse': mse, 'n_samples': len(X)}

    def obtener_estadisticas(self) -> Dict[str, Any]:
        stats = super().obtener_estadisticas()
        stats['distribution'] = self._distribution
        stats['var_smoothing'] = self._var_smoothing
        stats['is_fitted'] = self._is_fitted
        stats['n_classes'] = len(self._classes) if self._classes is not None else 0
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
            ok['priores_inicializados'] = self._class_priors is not None
        return ok

    def save(self, filepath: str) -> None:
        state = {
            'pesos': self.pesos, 'sesgo': self.sesgo, 'nombre': self.nombre,
            'input_size': self.input_size, 'output_size': self.output_size,
            'var_smoothing': self._var_smoothing, 'distribution': self._distribution,
            'is_fitted': self._is_fitted, 'pasos': self.pasos,
            'class_priors': self._class_priors, 'feature_mean': self._feature_mean,
            'feature_var': self._feature_var, 'accuracy': self._accuracy,
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
            'distribution': self._distribution, 'var_smoothing': self._var_smoothing,
        }

    def summary(self) -> str:
        cfg = self.get_config()
        stats = self.obtener_estadisticas()
        return "\n".join([
            f"=== {cfg['nombre']} ===",
            f"Dist: {cfg['distribution']}, Smoothing: {cfg['var_smoothing']}",
            f"Input: {cfg['input_size']}, Output: {cfg['output_size']}",
            f"Fitted: {stats.get('is_fitted', False)}, Acc: {stats.get('accuracy', 0):.4f}",
            f"Classes: {stats.get('n_classes', 0)}, Params: {self.params_count()}",
        ])

    def __str__(self) -> str:
        return f"{self.nombre}(ent={self.input_size},sal={self.output_size})"


class GaussianDistribution:
    @staticmethod
    def pdf(x: np.ndarray, mean: float, var: float) -> np.ndarray:
        return np.exp(-0.5 * ((x - mean) ** 2) / (var + 1e-8)) / np.sqrt(2 * np.pi * (var + 1e-8))

    @staticmethod
    def log_pdf(x: np.ndarray, mean: float, var: float) -> np.ndarray:
        return -0.5 * np.log(2 * np.pi * (var + 1e-8)) - ((x - mean) ** 2) / (2 * (var + 1e-8))

    @staticmethod
    def fit(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return np.mean(X, axis=0), np.var(X, axis=0) + 1e-8


class LaplaceSmoother:
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def smooth(self, counts: np.ndarray, n_classes: int) -> np.ndarray:
        return (counts + self.alpha) / (counts.sum() + self.alpha * n_classes)

    def get_smoothed_prob(self, count: float, total: float, n_classes: int) -> float:
        return (count + self.alpha) / (total + self.alpha * n_classes)


class FeatureLikelihood:
    def __init__(self, n_features: int, n_classes: int, distribution: str = "gaussian"):
        self.n_features = n_features
        self.n_classes = n_classes
        self.distribution = distribution
        self._means = np.zeros((n_features, n_classes))
        self._vars = np.ones((n_features, n_classes))

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        for c in range(self.n_classes):
            mask = y == c
            for f in range(self.n_features):
                self._means[f][c] = float(np.mean(X[mask][:, f])) if mask.sum() > 0 else 0.0
                self._vars[f][c] = max(1e-8, float(np.var(X[mask][:, f]))) if mask.sum() > 0 else 1.0

    def get_means(self) -> np.ndarray: return self._means.copy()

    def get_vars(self) -> np.ndarray: return self._vars.copy()


class PriorEstimator:
    def __init__(self, smoothing: float = 1e-8):
        self.smoothing = smoothing

    def estimate(self, y: np.ndarray, n_classes: int) -> np.ndarray:
        counts = np.bincount(y.astype(int), minlength=n_classes)
        return (counts + self.smoothing) / (counts.sum() + self.smoothing * n_classes)

    def log_prior(self, y: np.ndarray, n_classes: int) -> np.ndarray:
        priors = self.estimate(y, n_classes)
        return np.log(np.maximum(1e-8, priors))


class BayesianClassifier:
    def __init__(self, distribution: str = "gaussian", var_smoothing: float = 1e-9):
        self.distribution = distribution
        self.var_smoothing = var_smoothing
        self.prior = PriorEstimator()
        self.likelihood = None
        self._is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BayesianClassifier':
        n_classes = len(np.unique(y))
        self.priors = self.prior.estimate(y, n_classes)
        self.likelihood = FeatureLikelihood(X.shape[1], n_classes, self.distribution)
        self.likelihood.fit(X, y)
        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self._is_fitted: raise RuntimeError("No entrenado")
        n = len(X)
        n_classes = len(self.priors)
        log_probs = np.zeros((n, n_classes))
        for c in range(n_classes):
            log_probs[:, c] = np.log(self.priors[c] + 1e-8)
            for f in range(X.shape[1]):
                mean = self.likelihood.get_means()[f][c]
                var = self.likelihood.get_vars()[f][c]
                log_probs[:, c] += GaussianDistribution.log_pdf(X[:, f], mean, var)
        return np.argmax(log_probs, axis=1)


class ModelEvaluator:
    def __init__(self, cv_folds: int = 5):
        self.cv_folds = cv_folds; self._scores: List[float] = []

    def cross_validate(self, model, X: np.ndarray, y: np.ndarray) -> List[float]:
        n = len(X); fold_size = n // self.cv_folds; scores = []
        for i in range(self.cv_folds):
            start = i * fold_size; end = start + fold_size if i < self.cv_folds - 1 else n
            X_test = X[start:end]; y_test = y[start:end]
            X_train = np.concatenate([X[:start], X[end:]]); y_train = np.concatenate([y[:start], y[end:]])
            model_copy = type(model)(X.shape[1], max(2, int(np.max(y)) + 1))
            model_copy.fit(X_train, y_train)
            scores.append(model_copy.compute_accuracy(X_test, y_test))
        self._scores = scores; return scores

    def get_mean_score(self) -> float: return float(np.mean(self._scores)) if self._scores else 0.0

    def get_std_score(self) -> float: return float(np.std(self._scores)) if self._scores else 0.0

    def get_scores(self) -> List[float]: return self._scores.copy()


class BernoulliLikelihood:
    @staticmethod
    def fit(X: np.ndarray, y: np.ndarray, n_classes: int, alpha: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        n_features = X.shape[1]; means = np.zeros((n_features, n_classes)); vars_ = np.ones((n_features, n_classes))
        for c in range(n_classes):
            mask = y == c
            if mask.sum() > 0:
                means[:, c] = np.mean(X[mask], axis=0) * alpha
                vars_[:, c] = np.var(X[mask], axis=0) + alpha
        return means, vars_

    @staticmethod
    def log_likelihood(X: np.ndarray, means: np.ndarray, vars_: np.ndarray) -> np.ndarray:
        n, n_features = X.shape; n_classes = means.shape[1]; ll = np.zeros((n, n_classes))
        for c in range(n_classes):
            diff = X - means[:, c]; ll[:, c] = -0.5 * np.sum(np.log(vars_[:, c] + 1e-8) + diff ** 2 / (vars_[:, c] + 1e-8), axis=1)
        return ll


class MultinomialNB:
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha; self._classes = None; self._log_prior = None; self._log_likelihood = None; self._is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'MultinomialNB':
        n_classes = len(np.unique(y)); self._classes = np.unique(y); n_features = X.shape[1]
        self._log_prior = np.zeros(n_classes); self._log_likelihood = np.zeros((n_features, n_classes))
        for c, cls in enumerate(self._classes):
            mask = y == cls; count = mask.sum(); self._log_prior[c] = np.log(count / len(y))
            class_counts = np.zeros(n_features)
            for i in range(n_features): class_counts[i] = X[mask][:, i].sum()
            total = class_counts.sum() + self.alpha * n_features
            self._log_likelihood[:, c] = np.log((class_counts + self.alpha) / total)
        self._is_fitted = True; return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self._is_fitted: raise RuntimeError("No entrenado")
        scores = X @ self._log_likelihood + self._log_prior.T
        return self._classes[np.argmax(scores, axis=1)]


class ConfusionMatrix:
    def __init__(self, n_classes: int):
        self.n_classes = n_classes; self._matrix = np.zeros((n_classes, n_classes), dtype=int)

    def update(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        for t, p in zip(y_true.astype(int), y_pred.astype(int)):
            if 0 <= t < self.n_classes and 0 <= p < self.n_classes:
                self._matrix[t][p] += 1

    def get_matrix(self) -> np.ndarray: return self._matrix.copy()

    def accuracy(self) -> float:
        total = self._matrix.sum(); return float(np.trace(self._matrix) / max(1, total))


def create_naive_bayes_optimizer(input_size: int = 4, output_size: int = 8) -> NaiveBayesOptimizer:
    return NaiveBayesOptimizer(input_size=input_size, output_size=output_size)


if __name__ == "__main__":
    logger.info("RF_SL1_6.py cargado exitosamente")