"""
RF_SL1_5.py - Decision Trees Avanzados
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


class DecisionTreeOptimizer(NeuronaMemoriaBase):
    def __init__(self, input_size: int = 4, output_size: int = 8, nombre: str = "DecisionTreeOptimizer"):
        super().__init__(input_size, output_size, nombre)
        self.inicializar_pesos()
        self._max_depth: int = 10
        self._min_samples_split: int = 2
        self._criterion: str = "gini"
        self._is_fitted: bool = False
        self._tree: Optional[Dict] = None
        self._feature_importances: Optional[np.ndarray] = None
        self._classes: Optional[np.ndarray] = None
        self._n_samples: int = 0
        self._convergence_history: List[float] = []
        self._training_time: float = 0.0

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

    def compute_loss(self, prediccion: np.ndarray, objetivo: np.ndarray) -> float:
        return float(np.mean((prediccion - objetivo) ** 2))

    def _criterion_score(self, y: np.ndarray) -> float:
        if self._criterion == "gini":
            p = np.bincount(y.astype(int)) / len(y)
            return float(1.0 - np.sum(p ** 2))
        if self._criterion == "entropy":
            p = np.bincount(y.astype(int)) / len(y)
            p = p[p > 0]
            return float(-np.sum(p * np.log2(p)))
        if self._criterion == "mse":
            return float(np.var(y))
        return 0.0

    def compute_gini(self, y: np.ndarray) -> float:
        p = np.bincount(y.astype(int)) / max(1, len(y))
        return float(1.0 - np.sum(p ** 2))

    def compute_entropy(self, y: np.ndarray) -> float:
        p = np.bincount(y.astype(int)) / max(1, len(y))
        p = p[p > 0]
        return float(-np.sum(p * np.log2(p)))

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

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 10, verbose: bool = False) -> List[float]:
        t0 = time.perf_counter()
        y_int = y.astype(int) if y.dtype != int else y
        self._classes = np.unique(y_int)
        self._n_samples = len(X)
        self._feature_importances = np.zeros(self.input_size)
        self._tree = self._build_tree(X, y_int, depth=0)
        self._is_fitted = True
        self._compute_importances(self._tree)
        self._is_fitted = True
        self._training_time = time.perf_counter() - t0
        losses = []
        for epoch in range(epochs):
            pred = self.predict(X)
            loss = self.compute_loss(pred, y)
            losses.append(loss)
            self._convergence_history.append(loss)
            if verbose and (epoch + 1) % 5 == 0:
                logger.info(f"DT Epoch {epoch+1}, loss={loss:.6f}")
        return losses

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int) -> Dict:
        n = len(X)
        if depth >= self._max_depth or n < self._min_samples_split or len(np.unique(y)) == 1:
            return {'leaf': True, 'value': float(np.bincount(y.astype(int)).argmax()) if len(y) > 0 else 0.0}
        feat, thresh, gain = self._best_split(X, y)
        if gain < 1e-8: return {'leaf': True, 'value': float(np.bincount(y.astype(int)).argmax())}
        left_mask = X[:, feat] <= thresh
        self._feature_importances[feat] += gain
        return {
            'leaf': False, 'feature': feat, 'threshold': thresh, 'gain': float(gain),
            'left': self._build_tree(X[left_mask], y[left_mask], depth + 1),
            'right': self._build_tree(X[~left_mask], y[~left_mask], depth + 1),
        }

    def _best_split(self, X: np.ndarray, y: np.ndarray) -> Tuple[int, float, float]:
        best_f, best_t, best_g = 0, 0.0, 0.0
        parent_score = self._criterion_score(y)
        for f in range(X.shape[1]):
            for t in np.unique(X[:, f]):
                lm = X[:, f] <= t
                if lm.sum() == 0 or lm.sum() == len(y): continue
                left_score = self._criterion_score(y[lm])
                right_score = self._criterion_score(y[~lm])
                n_l, n_r = lm.sum(), (~lm).sum()
                gain = parent_score - (n_l * left_score + n_r * right_score) / len(y)
                if gain > best_g: best_g = gain; best_f = f; best_t = float(t)
        return best_f, best_t, best_g

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._tree is None: raise RuntimeError("Tree no ajustado")
        return np.array([self._pred_one(x, self._tree) for x in X])

    def _pred_one(self, x: np.ndarray, node: Dict) -> float:
        if node['leaf']: return node['value']
        if x[node['feature']] <= node['threshold']: return self._pred_one(x, node['left'])
        return self._pred_one(x, node['right'])

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        pred = self.predict(X)
        if y.ndim > 1 and y.shape[1] > 1:
            acc = float(np.mean(np.argmax(pred, axis=1) == np.argmax(y, axis=1)))
        else:
            acc = float(np.mean(pred.flatten() == y.flatten())) if len(y) == len(pred) else 0.0
        mse = self.compute_loss(pred, y.flatten() if y.ndim > 1 else y)
        return {'accuracy': acc, 'mse': mse, 'n_samples': len(X)}

    def prune(self, X: np.ndarray, y: np.ndarray, validation_fraction: float = 0.2) -> None:
        n = len(X)
        val_size = max(1, int(n * validation_fraction))
        idx = np.random.choice(n, size=val_size, replace=False)
        if self._tree is not None: self._tree = self._prune_tree(self._tree, X[idx], y[idx])

    def _prune_tree(self, node: Dict, X: np.ndarray, y: np.ndarray) -> Dict:
        if node.get('leaf'): return node
        if len(X) < self._min_samples_split: return {'leaf': True, 'value': float(np.bincount(y.astype(int)).argmax())}
        left_mask = X[:, node['feature']] <= node['threshold']
        left = self._prune_tree(node['left'], X[left_mask], y[left_mask])
        right = self._prune_tree(node['right'], X[~left_mask], y[~left_mask])
        leaf_acc = self._leaf_accuracy(y)
        node_acc = self._subtree_accuracy(node, X, y)
        if leaf_acc >= node_acc: return {'leaf': True, 'value': float(np.bincount(y.astype(int)).argmax())}
        node['left'] = left; node['right'] = right
        return node

    def _leaf_accuracy(self, y: np.ndarray) -> float:
        pred = np.full(len(y), np.bincount(y.astype(int)).argmax())
        return float(np.mean(pred == y))

    def _subtree_accuracy(self, node: Dict, X: np.ndarray, y: np.ndarray) -> float:
        try:
            pred = self._pred_one_batch(X, node)
            return float(np.mean(pred == y))
        except: return 0.0

    def _pred_one_batch(self, X: np.ndarray, node: Dict) -> np.ndarray:
        return np.array([self._pred_one(x, node) for x in X])

    def _compute_importances(self, node: Dict) -> None:
        if node.get('leaf'): return
        if 'feature' in node:
            self._feature_importances[node['feature']] += node.get('gain', 0)
            self._compute_importances(node.get('left', {}))
            self._compute_importances(node.get('right', {}))

    def obtener_estadisticas(self) -> Dict[str, Any]:
        stats = super().obtener_estadisticas()
        stats['max_depth'] = self._max_depth
        stats['min_samples_split'] = self._min_samples_split
        stats['criterion'] = self._criterion
        stats['is_fitted'] = self._is_fitted
        stats['n_classes'] = len(self._classes) if self._classes is not None else 0
        if self._feature_importances is not None:
            stats['top_feature'] = int(np.argmax(self._feature_importances))
            stats['feature_importances'] = self._feature_importances.tolist()
        if self._convergence_history:
            stats['final_loss'] = float(self._convergence_history[-1])
        stats['training_time'] = self._training_time
        return stats

    def verificar_estabilidad(self) -> Dict[str, bool]:
        ok = {'pesos_inicializados': self.pesos is not None}
        if self.pesos is not None:
            ok['pesos_no_explosivos'] = float(np.max(np.abs(self.pesos))) < 10.0
            ok['pesos_no_nan'] = not bool(np.any(np.isnan(self.pesos)))
            ok['estado_entrenado'] = self._is_fitted
        return ok

    def save(self, filepath: str) -> None:
        state = {
            'pesos': self.pesos, 'sesgo': self.sesgo, 'nombre': self.nombre,
            'input_size': self.input_size, 'output_size': self.output_size,
            'max_depth': self._max_depth, 'min_samples_split': self._min_samples_split,
            'criterion': self._criterion, 'is_fitted': self._is_fitted,
            'tree': self._tree, 'feature_importances': self._feature_importances,
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
            'max_depth': self._max_depth, 'criterion': self._criterion,
        }

    def summary(self) -> str:
        cfg = self.get_config()
        stats = self.obtener_estadisticas()
        return "\n".join([
            f"=== {cfg['nombre']} ===",
            f"Depth: {cfg['max_depth']}, Criterion: {cfg['criterion']}",
            f"Input: {cfg['input_size']}, Output: {cfg['output_size']}",
            f"Fitted: {stats.get('is_fitted', False)}, Top feature: {stats.get('top_feature', 'N/A')}",
            f"Params: {self.params_count()}",
        ])

    def __str__(self) -> str:
        return f"{self.nombre}(ent={self.input_size},sal={self.output_size})"


class TreeSplitter:
    def __init__(self, criterion: str = "gini"):
        self.criterion = criterion

    def find_best_split(self, X: np.ndarray, y: np.ndarray) -> Tuple[int, float, float]:
        if self.criterion == "gini": score_fn = self._gini_score
        elif self.criterion == "entropy": score_fn = self._entropy_score
        else: score_fn = self._mse_score
        best_f, best_t, best_g = 0, 0.0, 0.0
        parent = score_fn(y)
        for f in range(X.shape[1]):
            for t in np.unique(X[:, f]):
                lm = X[:, f] <= t
                if lm.sum() == 0 or lm.sum() == len(y): continue
                g = parent - (lm.sum() * score_fn(y[lm]) + (~lm).sum() * score_fn(y[~lm])) / len(y)
                if g > best_g: best_g = g; best_f = f; best_t = float(t)
        return best_f, best_t, best_g

    @staticmethod
    def _gini_score(y: np.ndarray) -> float:
        p = np.bincount(y.astype(int)) / max(1, len(y)); return float(1.0 - np.sum(p ** 2))

    @staticmethod
    def _entropy_score(y: np.ndarray) -> float:
        p = np.bincount(y.astype(int)) / max(1, len(y)); p = p[p > 0]; return float(-np.sum(p * np.log2(p)))

    @staticmethod
    def _mse_score(y: np.ndarray) -> float: return float(np.var(y))


class TreePruner:
    def __init__(self, min_samples_split: int = 2):
        self.min_samples_split = min_samples_split

    def post_prune(self, tree: Dict, X: np.ndarray, y: np.ndarray) -> Dict:
        if tree.get('leaf'): return tree
        left_mask = X[:, tree['feature']] <= tree['threshold']
        tree['left'] = self.post_prune(tree['left'], X[left_mask], y[left_mask])
        tree['right'] = self.post_prune(tree['right'], X[~left_mask], y[~left_mask])
        if len(X) < self.min_samples_split: return {'leaf': True, 'value': float(np.bincount(y.astype(int)).argmax())}
        return tree


class FeatureImportance:
    def __init__(self, n_features: int):
        self.n_features = n_features
        self._importances = np.zeros(n_features)

    def add(self, features: np.ndarray, gains: np.ndarray) -> None:
        for f, g in zip(features, gains): self._importances[int(f)] += float(g)

    def get(self) -> np.ndarray: return self._importances.copy()

    def top_k(self, k: int = 3) -> np.ndarray: return np.argsort(self._importances)[-k:][::-1]

    def normalized(self) -> np.ndarray:
        total = self._importances.sum()
        return self._importances / max(1e-8, total)


class TreeSerializer:
    @staticmethod
    def save(tree: Dict, filepath: str) -> None:
        with open(filepath, 'wb') as f: pickle.dump(tree, f)

    @staticmethod
    def load(filepath: str) -> Dict:
        with open(filepath, 'rb') as f: return pickle.load(f)


class CriterionSelector:
    @staticmethod
    def get_criterion(name: str) -> str:
        valid = ["gini", "entropy", "mse"]
        return name if name in valid else "gini"

    @staticmethod
    def list_criteria() -> List[str]: return ["gini", "entropy", "mse"]


class TreeEnsemble:
    def __init__(self, n_trees: int = 10):
        self.n_trees = n_trees; self._trees: List[DecisionTreeOptimizer] = []

    def add_tree(self, tree: DecisionTreeOptimizer) -> None: self._trees.append(tree)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self._trees: return np.zeros(len(X))
        preds = np.zeros(len(X))
        for tree in self._trees: preds += tree.predict(X).flatten()
        return preds / len(self._trees)

    def get_n_trees(self) -> int: return len(self._trees)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> float:
        pred = self.predict(X); return float(np.mean(np.abs(pred - y.flatten())))

    def feature_importances_global(self, n_features: int) -> np.ndarray:
        imp = np.zeros(n_features)
        for tree in self._trees:
            if tree._feature_importances is not None: imp += tree._feature_importances
        return imp / max(1, len(self._trees))


def create_decision_tree_optimizer(input_size: int = 4, output_size: int = 8) -> DecisionTreeOptimizer:
    return DecisionTreeOptimizer(input_size=input_size, output_size=output_size)


if __name__ == "__main__":
    logger.info("RF_SL1_5.py cargado exitosamente")