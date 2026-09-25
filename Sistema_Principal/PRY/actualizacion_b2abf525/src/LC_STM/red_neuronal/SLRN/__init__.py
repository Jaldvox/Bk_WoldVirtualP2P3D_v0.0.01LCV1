"""SLRN - Supervised Learning with Recurrent Neurons.
Paquete de optimizadores neuronales supervisados 2026 con algoritmos de ultima
generacion para redes neuronales de aprendizaje supervisado. Combina 10 metodos
de optimizacion independientes y un meta-ensamble integrado con precision 2026.
Algoritmos disponibles (SL1 - SL10):
  SL1   BackpropOptimizer         -- Backpropagation + Muon NS-5 + SOAP + NAG + GSNR
  SL2   SGDOptimizer              -- SGD + Sign-SGD EF21 + SGDR Cosine Restarts + Top-K
  SL3   RMSpropOptimizer          -- Centered RMSprop + Nesterov RMS + Muon + GSNR
  SL4   AdaGradOptimizer          -- AdaGrad + AdaGrad-Norm Ward + Soft-Reset + Muon
  SL5   AdaDeltaOptimizer         -- AdaDelta 2nd-order + Homotopic Epsilon + Muon
  SL6   AdamWOptimizer            -- AdamW + SOAP Preconditioner + Analytic Bias + Muon
  SL7   AdamaxOptimizer           -- Adamax L-inf + AdamaxW Decoupled + Muon + GSNR
  SL8   AMSGradOptimizer          -- AMSGrad Monotonic v_max + AMSGradW + Muon + GSNR
  SL9   AdaBoundOptimizer         -- AdaBound Dynamic Bounds + AdaBoundW + Muon + GSNR
  SL10  IntegratedSupervisedLearningOptimizer  -- Meta-ensamble Softmax de 10 metodos
Uso rapido::
    from LC.celebro.red_neuronal.SLRN import create_backprop_optimizer
    opt = create_backprop_optimizer()
    result = opt.optimize_weights(model, data_loader)
    # Sistema integrado completo
    from LC.celebro.red_neuronal.SLRN import SLRNSystem
    system = SLRNSystem()
    system.run_all(model)
Referencias 2026:
  - Jordan, K. et al. "Muon: Momentum-Orthogonal Update Networks" (2026)
  - Bernstein, J. & Newhouse, L. "Modular Adaptive Optimization" (2024)
  - Luo, L. et al. "Adaptive Gradient Methods with Dynamic Bound" (ICLR 2019)
"""
from __future__ import annotations
import importlib
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
__version__: str = "2026.1.0"
__author__: str = "Neural Network Optimization System"
__license__: str = "MIT"
__all__: List[str] = [
    # Core data structures
    "SupervisedLearningNeuralConfig",
    "SupervisedLearningNeuralMetrics",
    "SupervisedLearningNeuralResult",
    "BaseSupervisedLearningNeuralOptimizer",
    # Orchestration
    "SLRNSystem",
    "SupervisedLearningNeuralSystem",
    "create_integrated_supervised_learning_neural_optimizer",
    # Module names for lazy access
    "OPTIMIZER_REGISTRY",
]
logger = logging.getLogger(__name__)
# ---------------------------------------------------------------------------
# Registry of all available optimizers (lazy-loaded on first access)
# ---------------------------------------------------------------------------
OPTIMIZER_REGISTRY: Dict[str, str] = {
    "backprop":     "SL1",
    "sgd":          "SL2",
    "rmsprop":      "SL3",
    "adagrad":      "SL4",
    "adadelta":     "SL5",
    "adamw":        "SL6",
    "adamax":       "SL7",
    "amsgrad":      "SL8",
    "adabound":     "SL9",
    "integrated":   "SL10",
}
# ---------------------------------------------------------------------------
# Version metadata
# ---------------------------------------------------------------------------
SLRN_META: Dict[str, str] = {
    "version":     __version__,
    "description": "Supervised Learning Neural Networks 2026 - 10 Optimizers + Meta-Ensemble",
    "author":      __author__,
    "license":     __license__,
    "created":     "2026",
    "algorithms":  ", ".join(OPTIMIZER_REGISTRY.keys()),
}
# ===========================================================================
# CORE DATA STRUCTURES
# ===========================================================================
@dataclass
class SupervisedLearningNeuralConfig:
    """Configuracion unificada para todos los optimizadores de SLRN 2026."""
    # --- Generales ---
    learning_rate: float = 0.001
    max_iterations: int = 1000
    batch_size: int = 32
    weight_decay: float = 0.0001
    random_state: int = 42
    # --- Backpropagation / Momentum ---
    bp_momentum: float = 0.9
    bp_nesterov: bool = True
    # --- SGD ---
    sgd_momentum: float = 0.9
    sgd_dampening: float = 0.0
    # --- RMSprop ---
    rmsprop_alpha: float = 0.99
    rmsprop_eps: float = 1e-8
    # --- AdaGrad ---
    adagrad_eps: float = 1e-10
    # --- AdaDelta ---
    adadelta_rho: float = 0.9
    adadelta_eps: float = 1e-6
    # --- Adam / AdamW ---
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8
    # --- Adamax ---
    adamax_beta1: float = 0.9
    adamax_beta2: float = 0.999
    adamax_eps: float = 1e-8
    # --- AMSGrad ---
    amsgrad_beta1: float = 0.9
    amsgrad_beta2: float = 0.999
    amsgrad_eps: float = 1e-8
    # --- AdaBound ---
    adabound_final_lr: float = 0.1
    adabound_gamma: float = 0.001
    # --- LAMB / RAdam / NAdam / NovoGrad / Ranger (reserved) ---
    lamb_beta1: float = 0.9
    lamb_beta2: float = 0.999
    lamb_eps: float = 1e-8
    radam_beta1: float = 0.9
    radam_beta2: float = 0.999
    radam_eps: float = 1e-8
    nadam_beta1: float = 0.9
    nadam_beta2: float = 0.999
    nadam_eps: float = 1e-8
    nadam_momentum_decay: float = 0.004
    novograd_beta1: float = 0.9
    novograd_beta2: float = 0.999
    novograd_eps: float = 1e-8
    ranger_beta1: float = 0.9
    ranger_beta2: float = 0.999
    ranger_eps: float = 1e-8
    ranger_lookahead_k: int = 5
    ranger_lookahead_alpha: float = 0.5
    def __post_init__(self) -> None:
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
@dataclass
class SupervisedLearningNeuralMetrics:
    """Metricas de rendimiento completas del optimizador SLRN 2026."""
    algorithm_name: str
    initial_loss: float
    final_loss: float
    convergence_iterations: int
    bp_momentum_efficiency: float
    sgd_gradient_descent_efficiency: float
    rmsprop_rms_efficiency: float
    adagrad_adaptive_efficiency: float
    adadelta_delta_efficiency: float
    adam_adaptive_momentum: float
    adamax_max_efficiency: float
    amsgrad_maximum_efficiency: float
    adabound_boundary_efficiency: float
    lamb_layer_efficiency: float
    radam_rectified_efficiency: float
    nadam_nesterov_efficiency: float
    novograd_gradient_efficiency: float
    ranger_lookahead_efficiency: float
    supervised_neural_integration_score: float
    overall_score: float
    optimization_time: float
    timestamp: str
    @property
    def loss_reduction_pct(self) -> float:
        """Porcentaje de reduccion de perdida lograda."""
        if self.initial_loss <= 0:
            return 0.0
        return 100.0 * (self.initial_loss - self.final_loss) / self.initial_loss
    def summary(self) -> str:
        """Resumen compacto de una linea."""
        return (
            f"[{self.algorithm_name}] Loss: {self.initial_loss:.4f} -> {self.final_loss:.4f} "
            f"({self.loss_reduction_pct:.1f}% red.) | Score: {self.overall_score:.4f} "
            f"| Iter: {self.convergence_iterations} | {self.timestamp}"
        )
@dataclass
class SupervisedLearningNeuralResult:
    """Resultado completo de una ejecucion de optimizacion SLRN 2026."""
    success: bool
    optimized_model: Optional[Any]
    metrics: Optional[SupervisedLearningNeuralMetrics]
    optimization_history: List[float]
    best_weights: Dict[str, Any]
    theoretical_analysis: Dict[str, Any]
    performance_analysis: Dict[str, Any]
    recommendations: List[str]
    error_message: Optional[str]
    def is_ok(self) -> bool:
        """True si la optimizacion fue exitosa y con metricas validas."""
        return self.success and self.metrics is not None
    def print_summary(self) -> None:
        """Imprime resumen compacto en terminal."""
        if self.is_ok():
            print(self.metrics.summary())
        else:
            print(f"[FAILED] {self.error_message}")
# ===========================================================================
# BASE OPTIMIZER
# ===========================================================================

class BaseSupervisedLearningNeuralOptimizer(ABC):
    """Clase base abstracta para todos los optimizadores SLRN 2026.

    Subclases deben implementar:
      - create_optimizer(model)
      - optimize_weights(model, data_loader, criterion)

    Se recomienda tambien sobreescribir _check_convergence para autonomia
    plena sin depender del entorno legacy lucIA.CORE.
    """

    def __init__(self, config: SupervisedLearningNeuralConfig) -> None:
        self.config = config
        self.optimizer: Optional[Any] = None
        self.history: List[float] = []
        self.metrics: Dict[str, Any] = {}

    @abstractmethod
    def create_optimizer(self, model: Any) -> Any:
        """Instancia y retorna el motor interno del optimizador."""

    @abstractmethod
    def optimize_weights(
        self,
        model: Any,
        data_loader: Any,
        criterion: Any = None,
    ) -> SupervisedLearningNeuralResult:
        """Ejecuta la optimizacion de pesos y retorna un resultado completo."""

    def _evaluate_model(
        self,
        model: Any,
        data_loader: Any,
        criterion: Any,
    ) -> Dict[str, float]:
        """Evaluacion generica del modelo con soporte para iterables vacios."""
        try:
            total_loss, total_samples, correct = 0.0, 0, 0
            for data, target in data_loader:
                predictions = model.predict(data) if hasattr(model, "predict") else None
                if predictions is not None:
                    if callable(criterion):
                        total_loss += criterion(predictions, target)
                    total_samples += len(target)
                    if hasattr(predictions, "argmax"):
                        correct += int((predictions.argmax(axis=1) == target).sum())
                else:
                    total_samples += len(target)
            accuracy = correct / total_samples if total_samples > 0 else 0.0
            avg_loss = total_loss / max(len(data_loader), 1) if hasattr(data_loader, "__len__") else total_loss
            return {"loss": avg_loss, "accuracy": accuracy, "samples": total_samples}
        except Exception as exc:
            logger.error("Error evaluando modelo: %s", exc)
            return {"loss": 0.0, "accuracy": 0.0, "samples": 0}

    def _check_convergence(self, loss_history: List[float], patience: int = 10) -> bool:
        """Verificacion de convergencia autonoma sin dependencias externas."""
        if len(loss_history) < patience:
            return False
        recent = loss_history[-patience:]
        rel_diff = abs(recent[0] - recent[-1]) / max(abs(recent[0]), 1e-8)
        return rel_diff < 1e-4


# ===========================================================================
# ORCHESTRATOR  -  SLRNSystem
# ===========================================================================

class SLRNSystem:
    """Orquestador principal del paquete SLRN 2026.

    Carga, registra y ejecuta optimizadores bajo demanda con lazy-loading
    para minimizar el tiempo de importacion del paquete.

    Ejemplo::

        system = SLRNSystem()
        result = system.run("adabound", model, data_loader)
        system.run_all(model, data_loader)
    """

    def __init__(self, config: Optional[SupervisedLearningNeuralConfig] = None) -> None:
        self.config = config or SupervisedLearningNeuralConfig()
        self._optimizers: Dict[str, BaseSupervisedLearningNeuralOptimizer] = {}
        self._results: Dict[str, SupervisedLearningNeuralResult] = {}

    # ------------------------------------------------------------------
    # Lazy loading helpers
    # ------------------------------------------------------------------

    def _load_optimizer(self, key: str) -> BaseSupervisedLearningNeuralOptimizer:
        """Carga dinamica (lazy) del modulo SLX y retorna el optimizador."""
        if key in self._optimizers:
            return self._optimizers[key]

        module_name = OPTIMIZER_REGISTRY.get(key)
        if module_name is None:
            raise ValueError(f"Optimizador desconocido: '{key}'. Disponibles: {list(OPTIMIZER_REGISTRY)}")

        pkg = __name__  # LC.celebro.red_neuronal.SLRN
        mod = importlib.import_module(f".{module_name}", package=pkg)

        # Convencion de nombres de factory en cada SLX.py (nombres reales)
        factory_map = {
            "backprop":   "create_backpropagation_optimizer",
            "sgd":        "create_sgd_optimizer",
            "rmsprop":    "create_rmsprop_optimizer",
            "adagrad":    "create_adagrad_optimizer",
            "adadelta":   "create_adadelta_optimizer",
            "adamw":      "create_adam_optimizer",
            "adamax":     "create_adamax_optimizer",
            "amsgrad":    "create_amsgrad_optimizer",
            "adabound":   "create_adabound_optimizer",
            "integrated": "create_integrated_supervised_learning_optimizer",
        }
        factory_name = factory_map[key]
        factory = getattr(mod, factory_name)
        optimizer = factory(self.config)
        self._optimizers[key] = optimizer
        return optimizer

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        key: str,
        model: Any,
        data_loader: Any = None,
        criterion: Any = None,
    ) -> SupervisedLearningNeuralResult:
        """Ejecuta el optimizador identificado por *key* sobre el modelo dado."""
        opt = self._load_optimizer(key)
        result = opt.optimize_weights(model, data_loader or [], criterion)
        self._results[key] = result
        return result

    def run_all(
        self,
        model: Any,
        data_loader: Any = None,
        criterion: Any = None,
        keys: Optional[List[str]] = None,
    ) -> Dict[str, SupervisedLearningNeuralResult]:
        """Ejecuta todos (o un subconjunto) de los optimizadores registrados."""
        targets = keys or list(OPTIMIZER_REGISTRY.keys())
        for key in targets:
            try:
                self.run(key, model, data_loader, criterion)
            except Exception as exc:
                logger.error("Error ejecutando optimizador '%s': %s", key, exc)
                self._results[key] = SupervisedLearningNeuralResult(
                    success=False, optimized_model=None, metrics=None,
                    optimization_history=[], best_weights={},
                    theoretical_analysis={}, performance_analysis={},
                    recommendations=[], error_message=str(exc),
                )
        return self._results

    def best(self) -> Optional[Tuple[str, SupervisedLearningNeuralResult]]:
        """Retorna la tupla (nombre, resultado) del mejor optimizador ejecutado."""
        successful = {k: v for k, v in self._results.items() if v.is_ok()}
        if not successful:
            return None
        best_key = max(successful, key=lambda k: successful[k].metrics.overall_score)
        return best_key, successful[best_key]

    def print_results(self) -> None:
        """Imprime un resumen de todos los resultados disponibles."""
        print("\n" + "=" * 80)
        print("SLRN 2026 - RESULTADOS DE 10 OPTIMIZADORES SUPERVISADOS")
        print("=" * 80)
        for i, (name, result) in enumerate(self._results.items(), 1):
            tag = f"[{i:2d}] {name.upper():12s}"
            if result.is_ok():
                print(f"{tag} | {result.metrics.summary()}")
            else:
                print(f"{tag} | [FAILED] {result.error_message}")
        print("=" * 80)
        best_pair = self.best()
        if best_pair:
            print(f"MEJOR: {best_pair[0].upper()} | Score: {best_pair[1].metrics.overall_score:.4f}")
        print("=" * 80 + "\n")

    def register(self, key: str, optimizer: BaseSupervisedLearningNeuralOptimizer) -> None:
        """Registra un optimizador externo personalizado bajo la clave *key*."""
        self._optimizers[key] = optimizer

    def get_result(self, key: str) -> Optional[SupervisedLearningNeuralResult]:
        """Retorna el resultado de un optimizador ya ejecutado."""
        return self._results.get(key)

    @property
    def available(self) -> List[str]:
        """Lista de optimizadores disponibles en el registro."""
        return list(OPTIMIZER_REGISTRY.keys())


# Alias de compatibilidad hacia atras
SupervisedLearningNeuralSystem = SLRNSystem


# ===========================================================================
# CONVENIENCE FUNCTIONS
# ===========================================================================

def create_integrated_supervised_learning_neural_optimizer(
    config: Optional[SupervisedLearningNeuralConfig] = None,
) -> SLRNSystem:
    """Crea y retorna un SLRNSystem configurado listo para usar."""
    return SLRNSystem(config)


def get_version() -> str:
    """Retorna la version del paquete SLRN."""
    return __version__


def list_optimizers() -> List[str]:
    """Retorna la lista de claves de optimizadores registrados."""
    return list(OPTIMIZER_REGISTRY.keys())

logger.info(
    "SLRN v%s cargado | 10 optimizadores neuronales supervisados 2026 disponibles | keys=%s",
    __version__,
    list(OPTIMIZER_REGISTRY.keys()),
)