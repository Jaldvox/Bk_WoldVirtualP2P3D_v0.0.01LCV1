"""
RF_SL1_2.py - Support Vector Machines Avanzados (SVM)
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


class SupportVectorMachineOptimizer(NeuronaMemoriaBase):
    def __init__(self, input_size: int = 4, output_size: int = 8, nombre: str = "SupportVectorMachineOptimizer"):
        super().__init__(input_size, output_size, nombre)
        self.inicializar_pesos()
        self._kernel_type: str = "rbf"
        self._C: float = 1.0
        self._gamma: float = 0.1
        self._degree: int = 3
        self._tol: float = 1e-3
        self._max_iter: int = 1000
        self._support_vectors: Optional[np.ndarray] = None
        self._support_labels: Optional[np.ndarray] = None
        self._alphas: Optional[np.ndarray] = None
        self._bias: float = 0.0
        self._is_fitted: bool = False
        self._convergence_history: List[float] = []

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

    def predict(self, entrada: np.ndarray) -> np.ndarray:
        return self.forward(entrada)

    def kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        if self._kernel_type == "linear": return X1 @ X2.T
        if self._kernel_type == "poly": return (X1 @ X2.T + 1) ** self._degree
        if self._kernel_type == "sigmoid": return np.tanh(self._gamma * (X1 @ X2.T) + self._degree)
        sq1 = np.sum(X1 ** 2, axis=1).reshape(-1, 1)
        sq2 = np.sum(X2 ** 2, axis=1).reshape(1, -1)
        dist = sq1 + sq2 - 2 * (X1 @ X2.T)
        return np.exp(-self._gamma * np.clip(dist, 0, None))

    def compute_loss(self, prediccion: np.ndarray, objetivo: np.ndarray) -> float:
        hinge = np.maximum(0, 1 - objetivo * prediccion)
        return float(np.mean(self._C * hinge + 0.5 * np.mean(prediccion ** 2)))

    def compute_margin(self) -> float:
        if self.pesos is not None: return float(1.0 / (np.linalg.norm(self.pesos) + 1e-8))
        return 0.0

    def compute_decision_function(self, X: np.ndarray) -> np.ndarray:
        if self._is_fitted and self._alphas is not None:
            K = self.kernel(self._support_vectors, X)
            return (self._alphas * self._support_labels) @ K + self._bias
        return self.forward(X).flatten()

    def backward(self, gradiente_salida: np.ndarray, entrada: np.ndarray):
        grad_pesos = np.dot(entrada.T, gradiente_salida)
        grad_sesgo = np.sum(gradiente_salida, axis=0, keepdims=True)
        self.historial_gradientes.append({'pesos': grad_pesos.copy(), 'norma': float(np.linalg.norm(grad_pesos))})
        return grad_pesos, grad_sesgo

    def update_weights(self, grad_pesos: np.ndarray, grad_sesgo: np.ndarray) -> None:
        self.pesos -= self._C * grad_pesos
        self.sesgo -= self._C * grad_sesgo

    def apply_gradient(self, entrada: np.ndarray, objetivo: np.ndarray) -> float:
        pred = self.forward(entrada)
        grad = self.compute_margin() * (pred - objetivo)
        gp, gs = self.backward(grad, entrada)
        self.update_weights(gp, gs)
        return self.compute_loss(pred, objetivo)

    def train_step(self, entrada: np.ndarray, objetivo: np.ndarray) -> Dict[str, float]:
        loss_before = self.compute_loss(self.forward(entrada), objetivo)
        loss = self.apply_gradient(entrada, objetivo)
        return {'loss_before': loss_before, 'loss_after': loss, 'improvement': loss_before - loss}

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 50, verbose: bool = False) -> List[float]:
        t0 = time.perf_counter()
        losses = []
        for epoch in range(epochs):
            loss = self.apply_gradient(X, y)
            losses.append(loss)
            self._convergence_history.append(loss)
            if verbose and (epoch + 1) % 10 == 0:
                logger.info(f"SVM Epoch {epoch+1}/{epochs}, loss={loss:.6f}")
        self._timer_train = time.perf_counter() - t0
        self._is_fitted = True
        return losses

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        pred = self.predict(X)
        margin = self.compute_margin()
        accuracy = float(np.mean(np.sign(pred.flatten()) == np.sign(y.flatten()))) if len(y) > 0 else 0.0
        return {'accuracy': accuracy, 'margin': margin, 'mse': self.compute_loss(pred, y), 'n_samples': len(X)}

    def obtener_estadisticas(self) -> Dict[str, Any]:
        stats = super().obtener_estadisticas()
        stats['kernel_type'] = self._kernel_type
        stats['C'] = self._C
        stats['gamma'] = self._gamma
        stats['degree'] = self._degree
        stats['max_iter'] = self._max_iter
        stats['is_fitted'] = self._is_fitted
        stats['margin'] = self.compute_margin()
        stats['bias'] = self._bias
        stats['n_support_vectors'] = len(self._support_vectors) if self._support_vectors is not None else 0
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
        return ok

    def save(self, filepath: str) -> None:
        state = {
            'pesos': self.pesos, 'sesgo': self.sesgo, 'nombre': self.nombre,
            'input_size': self.input_size, 'output_size': self.output_size,
            'kernel_type': self._kernel_type, 'C': self._C, 'gamma': self._gamma,
            'degree': self._degree, 'bias': self._bias, 'is_fitted': self._is_fitted,
            'support_vectors': self._support_vectors, 'support_labels': self._support_labels,
            'alphas': self._alphas, 'pasos': self.pasos, 'convergence_history': self._convergence_history,
        }
        with open(filepath, 'wb') as f: pickle.dump(state, f)

    def load(self, filepath: str) -> None:
        with open(filepath, 'rb') as f: state = pickle.load(f)
        self.__dict__.update({k: v for k, v in state.items() if k != 'convergence_history'})
        self._convergence_history = state.get('convergence_history', [])

    def get_config(self) -> Dict[str, Any]:
        return {
            'nombre': self.nombre, 'input_size': self.input_size, 'output_size': self.output_size,
            'kernel_type': self._kernel_type, 'C': self._C, 'gamma': self._gamma,
            'degree': self._degree, 'max_iter': self._max_iter,
        }

    def summary(self) -> str:
        cfg = self.get_config()
        stats = self.obtener_estadisticas()
        return "\n".join([
            f"=== {cfg['nombre']} ===",
            f"Kernel: {cfg['kernel_type']}, C: {cfg['C']}, gamma: {cfg['gamma']}",
            f"Input: {cfg['input_size']}, Output: {cfg['output_size']}",
            f"Fitted: {stats.get('is_fitted', False)}, Margin: {stats.get('margin', 0):.4f}",
            f"Support vectors: {stats.get('n_support_vectors', 0)}, Params: {self.params_count()}",
        ])

    def __str__(self) -> str:
        return f"{self.nombre}(ent={self.input_size},sal={self.output_size})"


class KernelEngine:
    def __init__(self, kernel_type: str = "rbf", gamma: float = 0.1, degree: int = 3):
        self.kernel_type = kernel_type
        self.gamma = gamma
        self.degree = degree

    def compute(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        if self.kernel_type == "linear": return X1 @ X2.T
        if self.kernel_type == "poly": return (X1 @ X2.T + 1) ** self.degree
        if self.kernel_type == "sigmoid": return np.tanh(self.gamma * (X1 @ X2.T) + self.degree)
        sq1 = np.sum(X1 ** 2, axis=1).reshape(-1, 1)
        sq2 = np.sum(X2 ** 2, axis=1).reshape(1, -1)
        dist = sq1 + sq2 - 2 * (X1 @ X2.T)
        return np.exp(-self.gamma * np.clip(dist, 0, None))

    def gram_matrix(self, X: np.ndarray) -> np.ndarray:
        return self.compute(X, X)

    def get_params(self) -> Dict[str, Any]:
        return {'kernel_type': self.kernel_type, 'gamma': self.gamma, 'degree': self.degree}

    def get_gram_matrix(self, X: np.ndarray) -> np.ndarray:
        return self.compute(X, X)

    def set_gamma(self, gamma: float) -> None:
        self.gamma = gamma


class SMOTrainer:
    def __init__(self, C: float = 1.0, tol: float = 1e-3, max_iter: int = 1000):
        self.C = C
        self.tol = tol
        self.max_iter = max_iter
        self._objective_history: List[float] = []

    def optimize(self, K: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, float]:
        n = len(y)
        alphas = np.zeros(n)
        bias = 0.0
        for _ in range(self.max_iter):
            alpha_prev = alphas.copy()
            for i in range(n):
                Ei = np.sum(alphas * y * K[:, i]) + bias - y[i]
                if (y[i] * Ei < -self.tol and alphas[i] < self.C) or (y[i] * Ei > self.tol and alphas[i] > 0):
                    j = np.random.randint(0, n)
                    Ej = np.sum(alphas * y * K[:, j]) + bias - y[j]
                    alpha_i_old, alpha_j_old = alphas[i], alphas[j]
                    if y[i] != y[j]:
                        L = max(0, alphas[j] - alphas[i])
                        H = min(self.C, self.C + alphas[j] - alphas[i])
                    else:
                        L = max(0, alphas[i] + alphas[j] - self.C)
                        H = min(self.C, alphas[i] + alphas[j])
                    if L == H: continue
                    eta = 2 * K[i, j] - K[i, i] - K[j, j]
                    if eta >= 0: continue
                    alphas[j] -= y[j] * (Ei - Ej) / eta
                    alphas[j] = np.clip(alphas[j], L, H)
                    if abs(alphas[j] - alpha_j_old) < 1e-5: continue
                    alphas[i] += y[i] * y[j] * (alpha_j_old - alphas[j])
            if np.max(np.abs(alphas - alpha_prev)) < self.tol: break
        sv = alphas > 1e-5
        self._objective_history.append(float(np.sum(alphas[sv])))
        return alphas, bias

    def get_objective_history(self) -> List[float]: return self._objective_history.copy()

    def is_converged(self, tol: float = 1e-4) -> bool:
        if len(self._objective_history) < 2: return False
        return abs(self._objective_history[-1] - self._objective_history[-2]) < tol

    def reset(self) -> None: self._objective_history.clear()


class SupportVectorIdentifier:
    def __init__(self, threshold: float = 1e-5):
        self.threshold = threshold
        self._sv_indices: Optional[np.ndarray] = None

    def identify(self, alphas: np.ndarray) -> np.ndarray:
        self._sv_indices = np.where(alphas > self.threshold)[0]
        return self._sv_indices

    def count(self) -> int: return len(self._sv_indices) if self._sv_indices is not None else 0

    def get_mask(self, n_samples: int) -> np.ndarray:
        mask = np.zeros(n_samples, dtype=bool)
        if self._sv_indices is not None:
            mask[self._sv_indices] = True
        return mask

    @property
    def indices(self) -> Optional[np.ndarray]: return self._sv_indices


class MarginOptimizer:
    def __init__(self):
        self._best_margin: float = 0.0
        self._margin_history: List[float] = []

    def compute(self, weights: np.ndarray) -> float:
        margin = float(1.0 / (np.linalg.norm(weights) + 1e-8))
        self._margin_history.append(margin)
        self._best_margin = max(self._best_margin, margin)
        return margin

    def get_best_margin(self) -> float: return self._best_margin

    def get_avg_margin(self) -> float:
        if not self._margin_history: return 0.0
        return float(np.mean(self._margin_history))

    def get_margin_variance(self) -> float:
        if len(self._margin_history) < 2: return 0.0
        return float(np.var(self._margin_history))

    def is_stable(self, threshold: float = 0.1) -> bool:
        return self.get_margin_variance() < threshold

    def reset(self) -> None:
        self._best_margin = 0.0
        self._margin_history.clear()


class SVMClassifier:
    def __init__(self, kernel: str = "rbf", C: float = 1.0, gamma: float = 0.1):
        self.kernel_type = kernel
        self.C = C
        self.gamma = gamma
        self.kernel_engine = KernelEngine(kernel, gamma)
        self.trainer = SMOTrainer(C)
        self.identifier = SupportVectorIdentifier()
        self.margin_opt = MarginOptimizer()
        self._is_trained: bool = False
        self._training_time: float = 0.0

    def train(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        t0 = time.perf_counter()
        K = self.kernel_engine.gram_matrix(X)
        alphas, bias = self.trainer.optimize(K, y)
        sv_mask = alphas > self.identifier.threshold
        self._support_vectors = X[sv_mask]
        self._support_labels = y[sv_mask]
        self._alphas = alphas[sv_mask]
        self._bias = bias
        self._is_trained = True
        self._training_time = time.perf_counter() - t0
        self.identifier.identify(alphas)
        margin = self.margin_opt.compute(self.pesos_svm) if hasattr(self, 'pesos_svm') else 0.0
        return {
            'n_support_vectors': self.identifier.count(),
            'margin': margin, 'training_time': self._training_time,
            'kernel': self.kernel_type, 'C': self.C, 'converged': True,
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self._is_trained: raise RuntimeError("SVM no entrenado")
        K = self.kernel_engine.compute(self._support_vectors, X)
        decision = (self._alphas * self._support_labels) @ K + self._bias
        return np.sign(decision)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        if not self._is_trained: raise RuntimeError("SVM no entrenado")
        K = self.kernel_engine.compute(self._support_vectors, X)
        return (self._alphas * self._support_labels) @ K + self._bias

    @property
    def is_trained(self) -> bool: return self._is_trained

    @property
    def n_support_vectors(self) -> int: return self.identifier.count()

    @property
    def training_time(self) -> float: return self._training_time


def create_support_vector_machine_optimizer(input_size: int = 4, output_size: int = 8) -> SupportVectorMachineOptimizer:
    return SupportVectorMachineOptimizer(input_size=input_size, output_size=output_size)


def train_svm(X: np.ndarray, y: np.ndarray, kernel: str = "rbf", C: float = 1.0, gamma: float = 0.1) -> SVMClassifier:
    clf = SVMClassifier(kernel=kernel, C=C, gamma=gamma)
    clf.train(X, y)
    return clf


def evaluate_svm(clf: SVMClassifier, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    pred = clf.predict(X)
    accuracy = float(np.mean(np.sign(pred) == np.sign(y)))
    return {'accuracy': accuracy, 'n_support_vectors': clf.n_support_vectors, 'n_samples': len(X)}


if __name__ == "__main__":
    logger.info("RF_SL1_2.py cargado exitosamente")