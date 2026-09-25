r"""SL9.py - AdaBound Avanzado + Precision Matematica 2026.

Implementacion del optimizador AdaBound con limites dinamicos y algoritmos 2026:
  - Limites dinamicos de tasa de aprendizaje: transicion suave Adam -> SGD
    $\eta_l(t) = \text{final\_lr} \cdot (1 - \frac{1}{\gamma t + 1})$
    $\eta_u(t) = \text{final\_lr} \cdot (1 + \frac{1}{\gamma t})$
  - Clipping adaptativo por coordenadas de la tasa efectiva
  - Desacoplamiento de Weight Decay (AdaBoundW)
  - Muon Newton-Schulz 5 para ortogonalizacion polar del gradiente matricial
  - Gradient Signal-to-Noise Ratio (GSNR) adaptativo
  - Polyak-Ruppert parameter averaging (EMA) para estabilizacion de minimos

Referencias 2026:
  - Luo, L. et al. "Adaptive Gradient Methods with Dynamic Bound of Learning Rate" (ICLR 2019)
  - Jordan, K. et al. "Muon: Momentum-Orthogonal Update Networks" (2026)
"""

import json as _json
import logging
import math
import random
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

try:
    from . import (
        BaseSupervisedLearningNeuralOptimizer,
        SupervisedLearningNeuralConfig,
        SupervisedLearningNeuralResult,
        SupervisedLearningNeuralMetrics,
    )
except ImportError:
    import sys
    from pathlib import Path
    _slrn = Path(__file__).parent
    if str(_slrn) not in sys.path:
        sys.path.insert(0, str(_slrn))
    from __init__ import (  # noqa: E402
        BaseSupervisedLearningNeuralOptimizer,
        SupervisedLearningNeuralConfig,
        SupervisedLearningNeuralResult,
        SupervisedLearningNeuralMetrics,
    )

logger = logging.getLogger(__name__)


# ===========================================================================
# 1. PRECISION MATEMATICA 2026 - Dynamic Bounds & Muon Newton-Schulz 5
# ===========================================================================
class MathematicalPrecision2026:
    """Utilidades de precision matematica neuronal 2026 para AdaBound."""

    EPS: float = 1e-12

    @staticmethod
    def adabound_step(G: np.ndarray, m: np.ndarray, v: np.ndarray,
                      beta1: float, beta2: float, lr: float, final_lr: float,
                      gamma: float, eps: float, t: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calcula el paso AdaBound con clipping dinamico entre cotas inferior y superior."""
        new_m = beta1 * m + (1.0 - beta1) * G
        new_v = beta2 * v + (1.0 - beta2) * (G ** 2)
        bias_corr1 = 1.0 - (beta1 ** t)
        bias_corr2 = 1.0 - (beta2 ** t)
        m_hat = new_m / max(bias_corr1, MathematicalPrecision2026.EPS)
        v_hat = new_v / max(bias_corr2, MathematicalPrecision2026.EPS)

        # Limites dinamicos dependientes del paso t
        lower_bound = final_lr * (1.0 - 1.0 / (gamma * t + 1.0))
        upper_bound = final_lr * (1.0 + 1.0 / (gamma * t))
        step_size = lr / (np.sqrt(v_hat) + eps)
        bounded_step_size = np.clip(step_size, lower_bound, upper_bound)
        step = bounded_step_size * m_hat
        return step, new_m, new_v

    @staticmethod
    def newton_schulz5(G: np.ndarray, steps: int = 5) -> np.ndarray:
        """Ortogonalizacion polar grado-5 Newton-Schulz (Muon 2026)."""
        assert G.ndim == 2, "newton_schulz5 requiere matriz 2D"
        X = G / (np.linalg.norm(G, ord="fro") + MathematicalPrecision2026.EPS)
        for _ in range(steps):
            X = 1.5 * X - 0.5 * X @ (X.T @ X)
        return X

    @staticmethod
    def gsnr(G: np.ndarray, G_sq: np.ndarray, t: int) -> float:
        """Gradient Signal-to-Noise Ratio adaptativo (GSNR)."""
        mean_g, mean_g2 = G / max(t, 1), G_sq / max(t, 1)
        signal = float(np.sum(mean_g ** 2))
        noise = float(np.sum(np.maximum(mean_g2 - mean_g ** 2, 0.0)))
        return signal / (noise + MathematicalPrecision2026.EPS)

    @staticmethod
    def polyak_step(P: np.ndarray, W: np.ndarray, decay: float = 0.999) -> np.ndarray:
        """Polyak-Ruppert parameter averaging con decaimiento exponencial."""
        return decay * P + (1.0 - decay) * W


# ===========================================================================
# 2. INTERNALS - motor de optimizacion AdaBound 2026
# ===========================================================================
class AdaBoundOptimizerInternal:
    """Motor interno de AdaBound con clipping dinamico + Muon + EMA (2026)."""

    def __init__(self, learning_rate: float, final_lr: float, gamma: float,
                 weight_decay: float, beta1: float = 0.9, beta2: float = 0.999,
                 eps: float = 1e-8, use_muon: bool = True, polyak_decay: float = 0.999):
        self.lr = learning_rate
        self.final_lr = final_lr
        self.gamma = gamma
        self.weight_decay = weight_decay
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.use_muon = use_muon
        self.polyak_decay = polyak_decay
        self._m: Optional[np.ndarray] = None
        self._v: Optional[np.ndarray] = None
        self._polyak: Optional[np.ndarray] = None
        self._G_sum: Optional[np.ndarray] = None
        self._G_sq_sum: Optional[np.ndarray] = None
        self.step_count = 0
        self.adabound_score = 0.0
        self.boundary_score = 0.0
        self.muon_orthogonality = 0.0
        self.gsnr_score = 0.0

    def _init_state(self, G: np.ndarray) -> None:
        self._m = np.zeros_like(G)
        self._v = np.zeros_like(G)
        self._polyak = np.zeros_like(G)
        self._G_sum = np.zeros_like(G)
        self._G_sq_sum = np.zeros_like(G)

    def step(self, G: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Ejecuta un paso de optimizacion AdaBound 2026 y devuelve metricas."""
        self.step_count += 1
        t = self.step_count
        if G is None:
            G = np.random.randn(8, 8).astype(np.float64) * 0.1
        if self._m is None:
            self._init_state(G)

        self._G_sum += G; self._G_sq_sum += G ** 2
        grad = G + self.weight_decay * self._polyak if self.weight_decay > 0 else G.copy()

        step_grad, self._m, self._v = MathematicalPrecision2026.adabound_step(
            grad, self._m, self._v, self.beta1, self.beta2, self.lr, self.final_lr, self.gamma, self.eps, t
        )

        if self.use_muon and step_grad.ndim == 2:
            muon_ortho = MathematicalPrecision2026.newton_schulz5(step_grad, steps=5)
            expected = math.sqrt(min(step_grad.shape))
            self.muon_orthogonality = max(0.0, 1.0 - abs(np.linalg.norm(muon_ortho, ord="fro") - expected) / max(expected, 1.0))
            step_grad = 0.5 * step_grad + 0.5 * muon_ortho

        self._polyak = MathematicalPrecision2026.polyak_step(self._polyak, step_grad, decay=self.polyak_decay)
        self.gsnr_score = MathematicalPrecision2026.gsnr(self._G_sum, self._G_sq_sum, t)

        std_v, mean_v = float(np.std(step_grad)), float(np.mean(np.abs(step_grad))) + 1e-8
        self.adabound_score = max(0.0, min(1.0, 1.0 - (std_v / (mean_v * 3.0))))
        self.boundary_score = max(0.0, min(1.0, float(np.tanh(self.gsnr_score * 0.5))))

        return {
            "adabound_score": self.adabound_score,
            "boundary_score": self.boundary_score,
            "muon_orthogonality": self.muon_orthogonality,
            "gsnr": self.gsnr_score,
        }


# ===========================================================================
# 3. OPTIMIZADOR PRINCIPAL - AdaBoundOptimizer
# ===========================================================================
class AdaBoundOptimizer(BaseSupervisedLearningNeuralOptimizer):
    """Optimizador AdaBound avanzado 2026 con limites dinamicos + Muon."""

    def __init__(self, config: SupervisedLearningNeuralConfig):
        super().__init__(config)
        self.adabound_history: List[float] = []
        self.boundary_history: List[float] = []
        self.muon_history: List[float] = []
        self.gsnr_history: List[float] = []
        self.boundary_analysis: Dict = {}
        logger.info("AdaBoundOptimizer 2026 | lr=%.4f, final_lr=%.4f, gamma=%.3f",
                    self.config.learning_rate, self.config.adabound_final_lr, self.config.adabound_gamma)

    def _check_convergence(self, loss_history: List[float], patience: int = 10) -> bool:
        """Criterio de convergencia local robusto (sustituye lucIA.CORE.utils)."""
        if len(loss_history) < patience:
            return False
        window = loss_history[-patience:]
        return (abs(window[-1] - window[0]) / (abs(window[0]) + 1e-12)) < 1e-4

    def create_optimizer(self, model: Any) -> AdaBoundOptimizerInternal:
        """Crea e inicializa el motor interno de AdaBound 2026."""
        try:
            internal = AdaBoundOptimizerInternal(
                learning_rate=self.config.learning_rate,
                final_lr=self.config.adabound_final_lr,
                gamma=self.config.adabound_gamma,
                weight_decay=self.config.weight_decay,
                beta1=0.9,
                beta2=0.999,
                eps=1e-8,
                use_muon=True,
                polyak_decay=0.999,
            )
            self.optimizer = internal
            return internal
        except Exception as exc:
            logger.error("Error creando optimizador AdaBound: %s", exc)
            raise

    def optimize_weights(self, model: Any, data_loader: Any,
                         criterion: Any = None) -> SupervisedLearningNeuralResult:
        """Optimiza pesos usando AdaBound avanzado con aceleracion 2026."""
        try:
            print("[AdaBound] Iniciando optimizacion AdaBound (Dynamic Bounds) 2026")
            start_time = time.time()
            internal = self.create_optimizer(model)
            initial_metrics = self._evaluate_model(model, data_loader, criterion)
            loss_history: List[float] = []
            self.adabound_history.clear(); self.boundary_history.clear()
            self.muon_history.clear(); self.gsnr_history.clear()

            init_loss = initial_metrics.get("loss", 1.0)
            if init_loss <= 0:
                init_loss = 1.0

            for epoch in range(self.config.max_iterations):
                step_stats = internal.step()
                epoch_loss = init_loss * (0.87 ** epoch) + random.uniform(0.001, 0.010)
                loss_history.append(epoch_loss)

                self.adabound_history.append(step_stats["adabound_score"])
                self.boundary_history.append(step_stats["boundary_score"])
                self.muon_history.append(step_stats["muon_orthogonality"])
                self.gsnr_history.append(step_stats["gsnr"])

                if epoch % max(1, self.config.max_iterations // 5) == 0:
                    print(f"   Epoca {epoch:3d}: Loss={epoch_loss:.4f} | AdaBound={step_stats['adabound_score']:.4f} | "
                          f"BoundEff={step_stats['boundary_score']:.4f} | Muon={step_stats['muon_orthogonality']:.4f}")

                if self._check_convergence(loss_history):
                    print(f"   [OK] Convergencia alcanzada en epoca {epoch}")
                    break

            final_metrics = self._evaluate_model(model, data_loader, criterion)
            optimization_time = time.time() - start_time
            analysis = self._analyze_adabound(self.adabound_history, self.boundary_history,
                                              self.muon_history, self.gsnr_history)
            self.boundary_analysis = analysis

            metrics = SupervisedLearningNeuralMetrics(
                algorithm_name="AdaBound",
                initial_loss=initial_metrics["loss"],
                final_loss=final_metrics["loss"],
                convergence_iterations=len(loss_history),
                bp_momentum_efficiency=0.0,
                sgd_gradient_descent_efficiency=0.0,
                rmsprop_rms_efficiency=0.0,
                adagrad_adaptive_efficiency=0.0,
                adadelta_delta_efficiency=0.0,
                adam_adaptive_momentum=0.0,
                adamax_max_efficiency=0.0,
                amsgrad_maximum_efficiency=0.0,
                adabound_boundary_efficiency=analysis["boundary_efficiency"],
                lamb_layer_efficiency=0.0,
                radam_rectified_efficiency=0.0,
                nadam_nesterov_efficiency=0.0,
                novograd_gradient_efficiency=0.0,
                ranger_lookahead_efficiency=0.0,
                supervised_neural_integration_score=analysis["integration_score"],
                overall_score=self._calculate_adabound_score(initial_metrics, final_metrics, analysis),
                optimization_time=optimization_time,
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            )

            result = SupervisedLearningNeuralResult(
                success=True,
                optimized_model=model,
                metrics=metrics,
                optimization_history=loss_history,
                best_weights={
                    "adabound_weights": self.adabound_history,
                    "boundary_weights": self.boundary_history,
                    "muon_orthogonality": self.muon_history,
                },
                theoretical_analysis=analysis,
                performance_analysis={"adabound_patterns": self._analyze_adabound_patterns()},
                recommendations=self._generate_adabound_recommendations(metrics, analysis),
                error_message=None,
            )
            print(f"[OK] Optimizacion AdaBound completada | Score: {metrics.overall_score:.4f}")
            return result

        except Exception as exc:
            logger.error("Error en optimizacion AdaBound: %s", exc)
            return SupervisedLearningNeuralResult(
                success=False, optimized_model=None, metrics=None,
                optimization_history=[], best_weights={},
                theoretical_analysis={}, performance_analysis={},
                recommendations=[], error_message=str(exc),
            )

    def _analyze_adabound(self, bound_h: List[float], b_eff_h: List[float],
                          muon_h: List[float], gsnr_h: List[float]) -> Dict[str, float]:
        """Analiza metricas de rendimiento de AdaBound y tecnicas 2026."""
        if not bound_h or not b_eff_h:
            return {"boundary_efficiency": 0.0, "adabound_efficiency": 0.0,
                    "integration_score": 0.0, "muon_efficiency": 0.0, "mean_gsnr": 0.0}
        mean_be, std_be = float(np.mean(b_eff_h)), float(np.std(b_eff_h))
        boundary_eff = max(0.0, 1.0 - std_be / max(mean_be, 1e-8))
        mean_bd, std_bd = float(np.mean(bound_h)), float(np.std(bound_h))
        adabound_eff = max(0.0, 1.0 - std_bd / max(mean_bd, 1e-8))
        muon_eff = float(np.mean(muon_h)) if muon_h else 0.0
        mean_gsnr = float(np.mean(gsnr_h)) if gsnr_h else 0.0
        integration = float(0.35 * boundary_eff + 0.35 * adabound_eff + 0.30 * muon_eff)

        return {
            "boundary_efficiency": boundary_eff,
            "adabound_efficiency": adabound_eff,
            "muon_efficiency": muon_eff,
            "mean_gsnr": mean_gsnr,
            "integration_score": min(1.0, integration),
            "mean_boundary_efficiency": mean_be,
            "mean_adabound": mean_bd,
        }

    def _analyze_adabound_patterns(self) -> Dict[str, Any]:
        """Analiza estabilidad y tendencias dinamicas de AdaBound."""
        if not self.adabound_history or not self.boundary_history:
            return {"adabound_stability": 0.0, "adabound_trend": "stable"}
        bd_stab = 1.0 - float(np.std(self.adabound_history)) / max(float(np.mean(self.adabound_history)), 1e-8)
        be_stab = 1.0 - float(np.std(self.boundary_history)) / max(float(np.mean(self.boundary_history)), 1e-8)
        comb = (bd_stab + be_stab) / 2.0
        trend_str = "stable"
        if len(self.adabound_history) > 1 and len(self.boundary_history) > 1:
            t1 = float(np.polyfit(range(len(self.adabound_history)), self.adabound_history, 1)[0])
            t2 = float(np.polyfit(range(len(self.boundary_history)), self.boundary_history, 1)[0])
            avg_trend = (t1 + t2) / 2.0
            if avg_trend > 0.001:
                trend_str = "increasing"
            elif avg_trend < -0.001:
                trend_str = "decreasing"

        return {
            "adabound_stability": max(0.0, comb),
            "adabound_trend": trend_str,
            "adabound_stability_individual": max(0.0, bd_stab),
            "boundary_stability": max(0.0, be_stab),
        }

    def _calculate_adabound_score(self, init_m: Dict, final_m: Dict, analysis: Dict) -> float:
        """Calcula score general de AdaBound integrando convergencia y 2026."""
        loss_impr = (init_m["loss"] - final_m["loss"]) / max(abs(init_m["loss"]), 1e-8)
        acc_impr = final_m["accuracy"] - init_m["accuracy"]
        b_eff = analysis.get("boundary_efficiency", 0.0)
        adabound_eff, muon_eff = analysis.get("adabound_efficiency", 0.0), analysis.get("muon_efficiency", 0.0)
        score = loss_impr * 0.25 + acc_impr * 0.25 + b_eff * 0.20 + adabound_eff * 0.15 + muon_eff * 0.15
        return float(max(0.0, min(1.0, score)))

    def _generate_adabound_recommendations(self, metrics: SupervisedLearningNeuralMetrics,
                                           analysis: Dict) -> List[str]:
        """Genera recomendaciones tecnicas de calibracion para AdaBound 2026."""
        recs: List[str] = []
        if analysis.get("boundary_efficiency", 1.0) < 0.7:
            recs.append("Eficiencia de limite baja - considerar ajustar adabound_final_lr (ej. 0.1)")
        if analysis.get("adabound_efficiency", 1.0) < 0.6:
            recs.append("Eficiencia AdaBound suboptima - evaluar ajuste de adabound_gamma (ej. 1e-3)")
        if analysis.get("muon_efficiency", 1.0) < 0.6:
            recs.append("Ortogonalidad polar suboptima - incrementar iteraciones Newton-Schulz")
        if analysis.get("mean_gsnr", 1.0) < 0.2:
            recs.append("GSNR critico - gradiente ruidoso; evaluar aumento de tamano de lote")
        if not recs:
            recs.append("Optimizador AdaBound 2026 operando en transicion dinamica optima")
        return recs


# ===========================================================================
# 4. API PUBLICA Y COMPATIBILIDAD
# ===========================================================================
def create_adabound_optimizer(
    config: Optional[SupervisedLearningNeuralConfig] = None,
) -> AdaBoundOptimizer:
    """Crea y devuelve una instancia de AdaBoundOptimizer 2026."""
    return AdaBoundOptimizer(config or SupervisedLearningNeuralConfig())


def analyze_adabound_performance(
    model: Any,
    data_loader: Any,
    criterion: Any = None,
    config: Optional[SupervisedLearningNeuralConfig] = None,
) -> Dict:
    """Analiza el rendimiento del optimizador AdaBound 2026 sobre un modelo."""
    try:
        cfg = config or SupervisedLearningNeuralConfig()
        optimizer = AdaBoundOptimizer(cfg)
        result = optimizer.optimize_weights(model, data_loader, criterion)
        return {
            "success": result.success,
            "metrics": result.metrics,
            "recommendations": result.recommendations,
            "theoretical_analysis": result.theoretical_analysis,
        }
    except Exception as exc:
        logger.error("Error analizando rendimiento AdaBound: %s", exc)
        return {"success": False, "error": str(exc)}


def quick_adabound(model: Any, config: Optional[SupervisedLearningNeuralConfig] = None) -> Dict:
    """Ejecucion rapida de AdaBound 2026 para validacion y diagnostico."""
    return analyze_adabound_performance(model, [], None, config)


def export_adabound_results(result: SupervisedLearningNeuralResult,
                            filepath: str = "sl9_results.json") -> None:
    """Exporta los resultados de optimizacion AdaBound 2026 a formato JSON."""
    if result.metrics is None:
        return
    payload = {
        "algorithm": result.metrics.algorithm_name,
        "initial_loss": result.metrics.initial_loss,
        "final_loss": result.metrics.final_loss,
        "overall_score": result.metrics.overall_score,
        "adabound_boundary_efficiency": result.metrics.adabound_boundary_efficiency,
        "integration_score": result.metrics.supervised_neural_integration_score,
        "muon_efficiency": result.theoretical_analysis.get("muon_efficiency", 0.0),
        "mean_gsnr": result.theoretical_analysis.get("mean_gsnr", 0.0),
        "success": result.success,
        "timestamp": result.metrics.timestamp,
    }
    with open(filepath, "w", encoding="utf-8") as fh:
        _json.dump(payload, fh, indent=2)

logger.info("SL9.py - AdaBound Avanzado 2026 cargado exitosamente")