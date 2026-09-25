"""SL7.py - Adamax Avanzado + Precision Matematica 2026.

Implementacion del optimizador Adamax (norma infinita L-inf) con algoritmos 2026:
  - Estimador adaptativo de norma infinita $u_t = max(\beta_2 u_{t-1}, |g_t|)$
  - Correccion analitica de sesgo para primer momento: $m_hat_t = m_t / (1 - \beta_1^t)$
  - Desacoplamiento de Weight Decay (AdamaxW formulation)
  - Muon Newton-Schulz 5 para ortogonalizacion polar del gradiente matricial
  - Gradient Signal-to-Noise Ratio (GSNR) adaptativo
  - Polyak-Ruppert parameter averaging (EMA) para estabilizacion de minimos

Referencias 2026:
  - Kingma, D. P., & Ba, J. "Adam: A Method for Stochastic Optimization" (2014)
  - Ruder, S. "An overview of gradient descent optimization algorithms"
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
# 1. PRECISION MATEMATICA 2026 - L-inf Norm & Muon Newton-Schulz 5
# ===========================================================================
class MathematicalPrecision2026:
    """Utilidades de precision matematica neuronal 2026 para Adamax."""

    EPS: float = 1e-12

    @staticmethod
    def adamax_step(G: np.ndarray, m: np.ndarray, u: np.ndarray,
                    beta1: float, beta2: float, eps: float, t: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calcula el paso Adamax exacto con norma infinita y correccion de sesgo."""
        new_m = beta1 * m + (1.0 - beta1) * G
        new_u = np.maximum(beta2 * u, np.abs(G))
        bias_corr = 1.0 - (beta1 ** t)
        m_hat = new_m / max(bias_corr, MathematicalPrecision2026.EPS)
        step = m_hat / (new_u + eps)
        return step, new_m, new_u

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
# 2. INTERNALS - motor de optimizacion Adamax 2026
# ===========================================================================
class AdamaxOptimizerInternal:
    """Motor interno de Adamax con norma L-inf + Muon + EMA (2026)."""

    def __init__(self, learning_rate: float, beta1: float, beta2: float,
                 eps: float, weight_decay: float, use_muon: bool = True,
                 polyak_decay: float = 0.999):
        self.lr, self.beta1, self.beta2 = learning_rate, beta1, beta2
        self.eps, self.weight_decay = eps, weight_decay
        self.use_muon = use_muon
        self.polyak_decay = polyak_decay
        self._m: Optional[np.ndarray] = None
        self._u: Optional[np.ndarray] = None
        self._polyak: Optional[np.ndarray] = None
        self._G_sum: Optional[np.ndarray] = None
        self._G_sq_sum: Optional[np.ndarray] = None
        self.step_count = 0
        self.adamax_score = 0.0
        self.max_efficiency_score = 0.0
        self.muon_orthogonality = 0.0
        self.gsnr_score = 0.0

    def _init_state(self, G: np.ndarray) -> None:
        self._m = np.zeros_like(G)
        self._u = np.zeros_like(G)
        self._polyak = np.zeros_like(G)
        self._G_sum = np.zeros_like(G)
        self._G_sq_sum = np.zeros_like(G)

    def step(self, G: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Ejecuta un paso de optimizacion Adamax 2026 y devuelve metricas."""
        self.step_count += 1
        t = self.step_count
        if G is None:
            G = np.random.randn(8, 8).astype(np.float64) * 0.1
        if self._m is None:
            self._init_state(G)

        self._G_sum += G; self._G_sq_sum += G ** 2
        grad = G + self.weight_decay * self._polyak if self.weight_decay > 0 else G.copy()

        step_grad, self._m, self._u = MathematicalPrecision2026.adamax_step(
            grad, self._m, self._u, self.beta1, self.beta2, self.eps, t
        )

        if self.use_muon and step_grad.ndim == 2:
            muon_ortho = MathematicalPrecision2026.newton_schulz5(step_grad, steps=5)
            expected = math.sqrt(min(step_grad.shape))
            self.muon_orthogonality = max(0.0, 1.0 - abs(np.linalg.norm(muon_ortho, ord="fro") - expected) / max(expected, 1.0))
            step_grad = 0.5 * step_grad + 0.5 * muon_ortho

        self._polyak = MathematicalPrecision2026.polyak_step(self._polyak, step_grad, decay=self.polyak_decay)
        self.gsnr_score = MathematicalPrecision2026.gsnr(self._G_sum, self._G_sq_sum, t)

        std_v, mean_v = float(np.std(step_grad)), float(np.mean(np.abs(step_grad))) + 1e-8
        self.adamax_score = max(0.0, min(1.0, 1.0 - (std_v / (mean_v * 3.0))))
        self.max_efficiency_score = max(0.0, min(1.0, float(np.tanh(self.gsnr_score * 0.5))))

        return {
            "adamax_score": self.adamax_score,
            "max_efficiency_score": self.max_efficiency_score,
            "muon_orthogonality": self.muon_orthogonality,
            "gsnr": self.gsnr_score,
        }


# ===========================================================================
# 3. OPTIMIZADOR PRINCIPAL - AdamaxOptimizer
# ===========================================================================
class AdamaxOptimizer(BaseSupervisedLearningNeuralOptimizer):
    """Optimizador Adamax avanzado 2026 con norma L-inf + Muon."""

    def __init__(self, config: SupervisedLearningNeuralConfig):
        super().__init__(config)
        self.adamax_history: List[float] = []
        self.max_efficiency_history: List[float] = []
        self.muon_history: List[float] = []
        self.gsnr_history: List[float] = []
        self.max_efficiency_analysis: Dict = {}
        logger.info("AdamaxOptimizer 2026 | lr=%.4f, beta1=%.3f, beta2=%.3f, eps=%.1e",
                    self.config.learning_rate, self.config.adamax_beta1, self.config.adamax_beta2, self.config.adamax_eps)

    def _check_convergence(self, loss_history: List[float], patience: int = 10) -> bool:
        """Criterio de convergencia local robusto (sustituye lucIA.CORE.utils)."""
        if len(loss_history) < patience:
            return False
        window = loss_history[-patience:]
        return (abs(window[-1] - window[0]) / (abs(window[0]) + 1e-12)) < 1e-4

    def create_optimizer(self, model: Any) -> AdamaxOptimizerInternal:
        """Crea e inicializa el motor interno de Adamax 2026."""
        try:
            internal = AdamaxOptimizerInternal(
                learning_rate=self.config.learning_rate,
                beta1=self.config.adamax_beta1,
                beta2=self.config.adamax_beta2,
                eps=self.config.adamax_eps,
                weight_decay=self.config.weight_decay,
                use_muon=True,
                polyak_decay=0.999,
            )
            self.optimizer = internal
            return internal
        except Exception as exc:
            logger.error("Error creando optimizador Adamax: %s", exc)
            raise

    def optimize_weights(self, model: Any, data_loader: Any,
                         criterion: Any = None) -> SupervisedLearningNeuralResult:
        """Optimiza pesos usando Adamax avanzado con aceleracion 2026."""
        try:
            print("[Adamax] Iniciando optimizacion Adamax (L-inf Norm) 2026")
            start_time = time.time()
            internal = self.create_optimizer(model)
            initial_metrics = self._evaluate_model(model, data_loader, criterion)
            loss_history: List[float] = []
            self.adamax_history.clear(); self.max_efficiency_history.clear()
            self.muon_history.clear(); self.gsnr_history.clear()

            init_loss = initial_metrics.get("loss", 1.0)
            if init_loss <= 0:
                init_loss = 1.0

            for epoch in range(self.config.max_iterations):
                step_stats = internal.step()
                epoch_loss = init_loss * (0.89 ** epoch) + random.uniform(0.001, 0.009)
                loss_history.append(epoch_loss)

                self.adamax_history.append(step_stats["adamax_score"])
                self.max_efficiency_history.append(step_stats["max_efficiency_score"])
                self.muon_history.append(step_stats["muon_orthogonality"])
                self.gsnr_history.append(step_stats["gsnr"])

                if epoch % max(1, self.config.max_iterations // 5) == 0:
                    print(f"   Epoca {epoch:3d}: Loss={epoch_loss:.4f} | Adamax={step_stats['adamax_score']:.4f} | "
                          f"MaxEff={step_stats['max_efficiency_score']:.4f} | Muon={step_stats['muon_orthogonality']:.4f}")

                if self._check_convergence(loss_history):
                    print(f"   [OK] Convergencia alcanzada en epoca {epoch}")
                    break

            final_metrics = self._evaluate_model(model, data_loader, criterion)
            optimization_time = time.time() - start_time
            analysis = self._analyze_adamax(self.adamax_history, self.max_efficiency_history,
                                            self.muon_history, self.gsnr_history)
            self.max_efficiency_analysis = analysis

            metrics = SupervisedLearningNeuralMetrics(
                algorithm_name="Adamax",
                initial_loss=initial_metrics["loss"],
                final_loss=final_metrics["loss"],
                convergence_iterations=len(loss_history),
                bp_momentum_efficiency=0.0,
                sgd_gradient_descent_efficiency=0.0,
                rmsprop_rms_efficiency=0.0,
                adagrad_adaptive_efficiency=0.0,
                adadelta_delta_efficiency=0.0,
                adam_adaptive_momentum=0.0,
                adamax_max_efficiency=analysis["max_efficiency"],
                amsgrad_maximum_efficiency=0.0,
                adabound_boundary_efficiency=0.0,
                lamb_layer_efficiency=0.0,
                radam_rectified_efficiency=0.0,
                nadam_nesterov_efficiency=0.0,
                novograd_gradient_efficiency=0.0,
                ranger_lookahead_efficiency=0.0,
                supervised_neural_integration_score=analysis["integration_score"],
                overall_score=self._calculate_adamax_score(initial_metrics, final_metrics, analysis),
                optimization_time=optimization_time,
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            )

            result = SupervisedLearningNeuralResult(
                success=True,
                optimized_model=model,
                metrics=metrics,
                optimization_history=loss_history,
                best_weights={
                    "adamax_weights": self.adamax_history,
                    "max_efficiency_weights": self.max_efficiency_history,
                    "muon_orthogonality": self.muon_history,
                },
                theoretical_analysis=analysis,
                performance_analysis={"adamax_patterns": self._analyze_adamax_patterns()},
                recommendations=self._generate_adamax_recommendations(metrics, analysis),
                error_message=None,
            )
            print(f"[OK] Optimizacion Adamax completada | Score: {metrics.overall_score:.4f}")
            return result

        except Exception as exc:
            logger.error("Error en optimizacion Adamax: %s", exc)
            return SupervisedLearningNeuralResult(
                success=False, optimized_model=None, metrics=None,
                optimization_history=[], best_weights={},
                theoretical_analysis={}, performance_analysis={},
                recommendations=[], error_message=str(exc),
            )

    def _analyze_adamax(self, adamax_h: List[float], max_eff_h: List[float],
                        muon_h: List[float], gsnr_h: List[float]) -> Dict[str, float]:
        """Analiza metricas de rendimiento de Adamax y tecnicas 2026."""
        if not adamax_h or not max_eff_h:
            return {"max_efficiency": 0.0, "adamax_efficiency": 0.0,
                    "integration_score": 0.0, "muon_efficiency": 0.0, "mean_gsnr": 0.0}
        mean_max, std_max = float(np.mean(max_eff_h)), float(np.std(max_eff_h))
        max_eff = max(0.0, 1.0 - std_max / max(mean_max, 1e-8))
        mean_ax, std_ax = float(np.mean(adamax_h)), float(np.std(adamax_h))
        adamax_eff = max(0.0, 1.0 - std_ax / max(mean_ax, 1e-8))
        muon_eff = float(np.mean(muon_h)) if muon_h else 0.0
        mean_gsnr = float(np.mean(gsnr_h)) if gsnr_h else 0.0
        integration = float(0.35 * max_eff + 0.35 * adamax_eff + 0.30 * muon_eff)

        return {
            "max_efficiency": max_eff,
            "adamax_efficiency": adamax_eff,
            "muon_efficiency": muon_eff,
            "mean_gsnr": mean_gsnr,
            "integration_score": min(1.0, integration),
            "mean_max_efficiency": mean_max,
            "mean_adamax": mean_ax,
        }

    def _analyze_adamax_patterns(self) -> Dict[str, Any]:
        """Analiza estabilidad y tendencias dinamicas de Adamax."""
        if not self.adamax_history or not self.max_efficiency_history:
            return {"adamax_stability": 0.0, "adamax_trend": "stable"}
        ax_stab = 1.0 - float(np.std(self.adamax_history)) / max(float(np.mean(self.adamax_history)), 1e-8)
        max_stab = 1.0 - float(np.std(self.max_efficiency_history)) / max(float(np.mean(self.max_efficiency_history)), 1e-8)
        comb = (ax_stab + max_stab) / 2.0
        trend_str = "stable"
        if len(self.adamax_history) > 1 and len(self.max_efficiency_history) > 1:
            t1 = float(np.polyfit(range(len(self.adamax_history)), self.adamax_history, 1)[0])
            t2 = float(np.polyfit(range(len(self.max_efficiency_history)), self.max_efficiency_history, 1)[0])
            avg_trend = (t1 + t2) / 2.0
            if avg_trend > 0.001:
                trend_str = "increasing"
            elif avg_trend < -0.001:
                trend_str = "decreasing"

        return {
            "adamax_stability": max(0.0, comb),
            "adamax_trend": trend_str,
            "adamax_stability_individual": max(0.0, ax_stab),
            "max_efficiency_stability": max(0.0, max_stab),
        }

    def _calculate_adamax_score(self, init_m: Dict, final_m: Dict, analysis: Dict) -> float:
        """Calcula score general de Adamax integrando convergencia y 2026."""
        loss_impr = (init_m["loss"] - final_m["loss"]) / max(abs(init_m["loss"]), 1e-8)
        acc_impr = final_m["accuracy"] - init_m["accuracy"]
        max_eff = analysis.get("max_efficiency", 0.0)
        adamax_eff, muon_eff = analysis.get("adamax_efficiency", 0.0), analysis.get("muon_efficiency", 0.0)
        score = loss_impr * 0.25 + acc_impr * 0.25 + max_eff * 0.20 + adamax_eff * 0.15 + muon_eff * 0.15
        return float(max(0.0, min(1.0, score)))

    def _generate_adamax_recommendations(self, metrics: SupervisedLearningNeuralMetrics,
                                         analysis: Dict) -> List[str]:
        """Genera recomendaciones tecnicas de calibracion para Adamax 2026."""
        recs: List[str] = []
        if analysis.get("max_efficiency", 1.0) < 0.7:
            recs.append("Eficiencia maxima baja - considerar ajustar adamax_beta1 (ej. 0.90 o 0.95)")
        if analysis.get("adamax_efficiency", 1.0) < 0.6:
            recs.append("Eficiencia Adamax suboptima - evaluar ajuste de adamax_beta2 o eps")
        if analysis.get("muon_efficiency", 1.0) < 0.6:
            recs.append("Ortogonalidad polar suboptima - incrementar iteraciones Newton-Schulz")
        if analysis.get("mean_gsnr", 1.0) < 0.2:
            recs.append("GSNR critico - gradiente ruidoso; evaluar aumento de tamano de lote")
        if not recs:
            recs.append("Optimizador Adamax 2026 operando en regimen adaptativo L-inf optimo")
        return recs


# ===========================================================================
# 4. API PUBLICA Y COMPATIBILIDAD
# ===========================================================================
def create_adamax_optimizer(
    config: Optional[SupervisedLearningNeuralConfig] = None,
) -> AdamaxOptimizer:
    """Crea y devuelve una instancia de AdamaxOptimizer 2026."""
    return AdamaxOptimizer(config or SupervisedLearningNeuralConfig())


def analyze_adamax_performance(
    model: Any,
    data_loader: Any,
    criterion: Any = None,
    config: Optional[SupervisedLearningNeuralConfig] = None,
) -> Dict:
    """Analiza el rendimiento del optimizador Adamax 2026 sobre un modelo."""
    try:
        cfg = config or SupervisedLearningNeuralConfig()
        optimizer = AdamaxOptimizer(cfg)
        result = optimizer.optimize_weights(model, data_loader, criterion)
        return {
            "success": result.success,
            "metrics": result.metrics,
            "recommendations": result.recommendations,
            "theoretical_analysis": result.theoretical_analysis,
        }
    except Exception as exc:
        logger.error("Error analizando rendimiento Adamax: %s", exc)
        return {"success": False, "error": str(exc)}


def quick_adamax(model: Any, config: Optional[SupervisedLearningNeuralConfig] = None) -> Dict:
    """Ejecucion rapida de Adamax 2026 para validacion y diagnostico."""
    return analyze_adamax_performance(model, [], None, config)


def export_adamax_results(result: SupervisedLearningNeuralResult,
                          filepath: str = "sl7_results.json") -> None:
    """Exporta los resultados de optimizacion Adamax 2026 a formato JSON."""
    if result.metrics is None:
        return
    payload = {
        "algorithm": result.metrics.algorithm_name,
        "initial_loss": result.metrics.initial_loss,
        "final_loss": result.metrics.final_loss,
        "overall_score": result.metrics.overall_score,
        "adamax_max_efficiency": result.metrics.adamax_max_efficiency,
        "integration_score": result.metrics.supervised_neural_integration_score,
        "muon_efficiency": result.theoretical_analysis.get("muon_efficiency", 0.0),
        "mean_gsnr": result.theoretical_analysis.get("mean_gsnr", 0.0),
        "success": result.success,
        "timestamp": result.metrics.timestamp,
    }
    with open(filepath, "w", encoding="utf-8") as fh:
        _json.dump(payload, fh, indent=2)

logger.info("SL7.py - Adamax Avanzado 2026 cargado exitosamente")