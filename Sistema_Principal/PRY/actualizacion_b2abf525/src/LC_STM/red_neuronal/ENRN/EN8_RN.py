import numpy as np
import math
import time
import json
from typing import Tuple, Optional, Dict, Any, List, Callable
try:
    from . import NeuronaEntradaBase, LUCIA_CONFIG
except ImportError:
    import numpy as _np_alias  # noqa: F401
    from typing import Tuple as _Tuple, Dict as _Dict, Any as _Any

    LUCIA_CONFIG = {
        "precision": "float32",
        "random_seed": 42,
        "default_learning_rate": 0.001,
    }

    class NeuronaEntradaBase:
        """Base autocontenida (fallback sin ciclo de import)."""

        def __init__(self, input_size: int, output_size: int, nombre: str = "NeuronaEntrada"):
            self.input_size = input_size
            self.output_size = output_size
            self.nombre = nombre
            self.pesos = None
            self.sesgo = None
            self.historial_activaciones = []
            self.historial_gradientes = []
            self.llm_connector = None

        def inicializar_pesos(self) -> None:
            raise NotImplementedError

        def forward(self, entrada):
            raise NotImplementedError

        def backward(self, g, e):
            raise NotImplementedError

        def obtener_estadisticas(self):
            return {"nombre": self.nombre, "input_size": self.input_size,
                    "output_size": self.output_size,
                    "activaciones": len(self.historial_activaciones)}

        def obtener_estadisticas_completas(self):
            return self.obtener_estadisticas()

        def verificar_estabilidad(self):
            return {"pesos_inicializados": self.pesos is not None}

        def __str__(self) -> str:
            return f"{self.nombre}(entrada={self.input_size}, salida={self.output_size})"


_DTYPE_MAP = {"float32": np.float32, "float64": np.float64, "float16": np.float16}


def _dtype() -> np.dtype:
    return np.dtype(_DTYPE_MAP.get(LUCIA_CONFIG.get("precision", "float32"), np.float32))


def _now() -> float:
    return time.perf_counter()


class _RingBuffer:
    """Buffer circular eficiente para historiales."""

    def __init__(self, capacidad: int = 256):
        self.capacidad = int(capacidad)
        self._datos: List[Any] = []

    def append(self, item: Any) -> None:
        self._datos.append(item)
        if len(self._datos) > self.capacidad:
            del self._datos[0:len(self._datos) - self.capacidad]

    def __len__(self) -> int:
        return len(self._datos)

    def to_list(self) -> List[Any]:
        return list(self._datos)

    def clear(self) -> None:
        self._datos.clear()


class NeuronaEntradaReLU(NeuronaEntradaBase):
    """EN8 ReLU eficiente: relu/leaky/elu/silu vectorizados, Adam/SGD.

    Preserva API: tipo_relu, alpha, factor_escala, tasa_activacion,
    _activar, _gradiente_activacion, inicializar_pesos, forward, backward.
    """

    TIPOS = ("relu", "leaky_relu", "elu", "silu", "gelu_aprox")

    def __init__(self, input_size: int, output_size: int,
                 nombre: str = "NeuronaEntradaReLU",
                 tipo_relu: str = "relu",
                 alpha: float = 0.01,
                 factor_escala: float = 1.0,
                 learning_rate: float = 0.001,
                 optimizador: str = "adam",
                 beta1: float = 0.9,
                 beta2: float = 0.999,
                 epsilon: float = 1e-8,
                 weight_decay: float = 0.0,
                 grad_clip: float = 5.0,
                 lr_decay: float = 0.0,
                 lr_min: float = 1e-6,
                 usar_cache_forward: bool = True,
                 semilla: Optional[int] = None,
                 capacidad_historial: int = 256):
        super().__init__(input_size, output_size, nombre)
        self.tipo_relu = str(tipo_relu) if tipo_relu in self.TIPOS else "relu"
        self.alpha = float(alpha)
        self.factor_escala = float(factor_escala)
        self.varianza_real = 0.0
        self.varianza_esperada = 1.0
        self.tasa_activacion = 0.0
        self.tasa_muerta = 0.0
        self.lr_base = float(learning_rate or LUCIA_CONFIG.get("default_learning_rate", 0.001))
        self.lr_actual = float(self.lr_base)
        self.optimizador = str(optimizador).lower()
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.epsilon = float(epsilon)
        self.weight_decay = float(weight_decay)
        self.grad_clip = float(grad_clip)
        self.lr_decay = float(lr_decay)
        self.lr_min = float(lr_min)
        self.usar_cache_forward = bool(usar_cache_forward)
        self.semilla = semilla if semilla is not None else int(LUCIA_CONFIG.get("random_seed", 42))
        self.rng = np.random.default_rng(self.semilla)
        self._m_pesos: Optional[np.ndarray] = None
        self._v_pesos: Optional[np.ndarray] = None
        self._m_sesgo: Optional[np.ndarray] = None
        self._v_sesgo: Optional[np.ndarray] = None
        self._paso = 0
        self._cache: Dict[str, Any] = {}
        self.historial_activaciones = _RingBuffer(capacidad_historial)
        self.historial_gradientes = _RingBuffer(capacidad_historial)
        self.historial_loss: List[float] = []
        self.historial_lr: List[float] = []
        self.historial_tasa: List[float] = []
        self.contador_forward = 0
        self.contador_backward = 0
        self.tiempo_forward_acum = 0.0
        self.tiempo_backward_acum = 0.0
        self.mejor_loss = float("inf")
        self.callbacks: List[Callable[[str, Dict[str, Any]], None]] = []

    # ------------------------------------------------------------------
    def _emit(self, evento: str, info: Optional[Dict[str, Any]] = None) -> None:
        for cb in self.callbacks:
            try:
                cb(evento, info or {})
            except Exception:
                continue

    def registrar_callback(self, fn: Callable[[str, Dict[str, Any]], None]) -> None:
        self.callbacks.append(fn)

    def fijar_activacion(self, tipo: str, alpha: Optional[float] = None) -> None:
        if tipo not in self.TIPOS:
            raise ValueError(f"tipo {tipo} no en {self.TIPOS}")
        self.tipo_relu = tipo
        if alpha is not None:
            self.alpha = float(alpha)

    def _asegurar_adam(self) -> None:
        if self._m_pesos is None:
            assert self.pesos is not None and self.sesgo is not None
            self._m_pesos = np.zeros_like(self.pesos, dtype=np.float64)
            self._v_pesos = np.zeros_like(self.pesos, dtype=np.float64)
            self._m_sesgo = np.zeros_like(self.sesgo, dtype=np.float64)
            self._v_sesgo = np.zeros_like(self.sesgo, dtype=np.float64)

    def _decaer_lr(self) -> None:
        if self.lr_decay > 0.0:
            self.lr_actual = max(self.lr_min, self.lr_base / (1.0 + self.lr_decay * self._paso))
        self.historial_lr.append(self.lr_actual)

    def _clip(self, g: np.ndarray) -> np.ndarray:
        if self.grad_clip and self.grad_clip > 0:
            n = float(np.linalg.norm(g))
            if n > self.grad_clip:
                g = g * (self.grad_clip / (n + 1e-12))
        return g

    def _preparar(self, entrada: np.ndarray) -> np.ndarray:
        x = np.asarray(entrada, dtype=_dtype())
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if x.shape[1] != self.input_size:
            raise ValueError(f"entrada dim {x.shape[1]} != input_size {self.input_size}")
        return np.ascontiguousarray(x, dtype=_dtype())

    def _activar(self, x: np.ndarray) -> np.ndarray:
        if self.tipo_relu == "relu":
            return np.maximum(0, x)
        elif self.tipo_relu == "leaky_relu":
            return np.where(x > 0, x, self.alpha * x)
        elif self.tipo_relu == "elu":
            return np.where(x > 0, x, self.alpha * (np.exp(np.clip(x, -20, 20)) - 1))
        elif self.tipo_relu == "silu":
            return x / (1.0 + np.exp(-np.clip(x, -20, 20)))
        elif self.tipo_relu == "gelu_aprox":
            return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))
        return np.maximum(0, x)

    def _gradiente_activacion(self, x: np.ndarray) -> np.ndarray:
        if self.tipo_relu == "relu":
            return (x > 0).astype(np.float64)
        elif self.tipo_relu == "leaky_relu":
            return np.where(x > 0, 1.0, self.alpha).astype(np.float64)
        elif self.tipo_relu == "elu":
            return np.where(x > 0, 1.0, self.alpha * np.exp(np.clip(x, -20, 20))).astype(np.float64)
        elif self.tipo_relu == "silu":
            s = 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))
            return (s + x * s * (1 - s)).astype(np.float64)
        elif self.tipo_relu == "gelu_aprox":
            # derivada aproximada eficiente por diferencias centrales evitada: formula cerrada
            c = np.sqrt(2.0 / np.pi)
            t = np.tanh(c * (x + 0.044715 * x ** 3))
            return (0.5 * (1 + t) + 0.5 * x * (1 - t * t) * c * (1 + 3 * 0.044715 * x * x)).astype(np.float64)
        return (x > 0).astype(np.float64)

    # ------------------------------------------------------------------
    def inicializar_pesos(self, modo: str = "he") -> None:
        dt = _dtype()
        modo = str(modo).lower()
        if modo == "he":
            std = math.sqrt(2.0 / self.input_size) * self.factor_escala
            self.pesos = self.rng.normal(0, std, (self.input_size, self.output_size)).astype(dt)
        elif modo == "xavier":
            lim = math.sqrt(6.0 / (self.input_size + self.output_size)) * self.factor_escala
            self.pesos = self.rng.uniform(-lim, lim, (self.input_size, self.output_size)).astype(dt)
        else:
            raise ValueError(f"modo {modo} desconocido")
        self.sesgo = np.zeros((1, self.output_size), dtype=dt)
        self.varianza_real = float(np.var(self.pesos))
        self._m_pesos = self._v_pesos = self._m_sesgo = self._v_sesgo = None
        self._paso = 0
        self._emit("init", {"modo": modo})

    def forward(self, entrada: np.ndarray) -> np.ndarray:
        t0 = _now()
        if self.pesos is None:
            self.inicializar_pesos()
        assert self.pesos is not None and self.sesgo is not None
        x = self._preparar(entrada)
        suma = np.dot(x, self.pesos) + self.sesgo
        salida = np.ascontiguousarray(self._activar(suma).astype(_dtype()), dtype=_dtype())
        tasa = float(np.sum(np.asarray(salida) > 0)) / max(1, salida.size)
        n = len(self.historial_activaciones)
        self.tasa_activacion = 0.9 * self.tasa_activacion + 0.1 * tasa if n else tasa
        self.tasa_muerta = 1.0 - self.tasa_activacion
        v = float(np.var(salida)) if salida.size else 0.0
        self.varianza_real = 0.9 * self.varianza_real + 0.1 * v if n else v
        self.historial_activaciones.append(salida.copy())
        self.historial_tasa.append(self.tasa_activacion)
        if self.usar_cache_forward:
            self._cache = {"x": x, "pre": suma.astype(_dtype())}
        self.contador_forward += 1
        self.tiempo_forward_acum += _now() - t0
        return salida

    def forward_batch(self, lote: List[np.ndarray]) -> List[np.ndarray]:
        return [self.forward(x) for x in lote]

    def backward(self, gradiente_salida: np.ndarray, entrada: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        t0 = _now()
        g = np.asarray(gradiente_salida, dtype=np.float64)
        if g.ndim == 1:
            g = g.reshape(1, -1)
        pre = self._cache.get("pre") if entrada is None else None
        if pre is None:
            # recomputa pre-activacion de forma eficiente si no hay cache
            x_tmp = self._preparar(entrada) if entrada is not None else self._cache.get("x")
            if x_tmp is None:
                raise ValueError("sin entrada ni cache forward")
            assert self.pesos is not None and self.sesgo is not None
            pre = (np.dot(x_tmp, self.pesos) + self.sesgo).astype(np.float64)
        grad_act = self._gradiente_activacion(np.asarray(pre, dtype=np.float64))
        g = g * grad_act
        x = self._preparar(entrada) if entrada is not None else self._cache.get("x")
        if x is None:
            raise ValueError("sin entrada ni cache forward")
        b = max(1, x.shape[0])
        gp = self._clip(np.asarray(np.dot(x.T.astype(np.float64), g) / b))
        gb = self._clip(np.asarray(np.sum(g, axis=0, keepdims=True) / b))
        if self.weight_decay > 0.0 and self.pesos is not None:
            gp = gp + self.weight_decay * self.pesos.astype(np.float64)
        self.historial_gradientes.append({'pesos': np.asarray(gp, dtype=_dtype()).copy(),
                                          'sesgo': np.asarray(gb, dtype=_dtype()).copy(),
                                          'norma_pesos': float(np.linalg.norm(gp)),
                                          'norma_sesgo': float(np.linalg.norm(gb))})
        self.contador_backward += 1
        self.tiempo_backward_acum += _now() - t0
        return np.asarray(gp, dtype=_dtype()), np.asarray(gb, dtype=_dtype())

    def grad_entrada(self, gradiente_salida: np.ndarray) -> np.ndarray:
        g = np.asarray(gradiente_salida, dtype=np.float64)
        if g.ndim == 1:
            g = g.reshape(1, -1)
        pre = self._cache.get("pre")
        if pre is not None:
            g = g * self._gradiente_activacion(np.asarray(pre, dtype=np.float64))
        assert self.pesos is not None
        return np.dot(g, self.pesos.T.astype(np.float64)).astype(_dtype())

    def paso_optimizador(self, grad_pesos: np.ndarray, grad_sesgo: np.ndarray) -> Dict[str, float]:
        if self.pesos is None or self.sesgo is None:
            raise RuntimeError("pesos no inicializados")
        self._paso += 1
        self._decaer_lr()
        gp = np.asarray(grad_pesos, dtype=np.float64)
        gb = np.asarray(grad_sesgo, dtype=np.float64)
        if self.optimizador == "sgd":
            self.pesos = (self.pesos.astype(np.float64) - self.lr_actual * gp).astype(_dtype())
            self.sesgo = (self.sesgo.astype(np.float64) - self.lr_actual * gb).astype(_dtype())
        elif self.optimizador == "sgd_momentum":
            self._asegurar_adam()
            assert self._m_pesos is not None and self._m_sesgo is not None
            self._m_pesos[:] = self.beta1 * self._m_pesos + (1 - self.beta1) * gp
            self._m_sesgo[:] = self.beta1 * self._m_sesgo + (1 - self.beta1) * gb
            self.pesos = (self.pesos.astype(np.float64) - self.lr_actual * self._m_pesos).astype(_dtype())
            self.sesgo = (self.sesgo.astype(np.float64) - self.lr_actual * self._m_sesgo).astype(_dtype())
        else:
            self._asegurar_adam()
            assert self._m_pesos is not None and self._v_pesos is not None
            assert self._m_sesgo is not None and self._v_sesgo is not None
            self._m_pesos[:] = self.beta1 * self._m_pesos + (1 - self.beta1) * gp
            self._v_pesos[:] = self.beta2 * self._v_pesos + (1 - self.beta2) * (gp * gp)
            self._m_sesgo[:] = self.beta1 * self._m_sesgo + (1 - self.beta1) * gb
            self._v_sesgo[:] = self.beta2 * self._v_sesgo + (1 - self.beta2) * (gb * gb)
            mh_p = self._m_pesos / (1 - self.beta1 ** self._paso)
            vh_p = self._v_pesos / (1 - self.beta2 ** self._paso)
            mh_b = self._m_sesgo / (1 - self.beta1 ** self._paso)
            vh_b = self._v_sesgo / (1 - self.beta2 ** self._paso)
            self.pesos = (self.pesos.astype(np.float64) - self.lr_actual * mh_p / (np.sqrt(vh_p) + self.epsilon)).astype(_dtype())
            self.sesgo = (self.sesgo.astype(np.float64) - self.lr_actual * mh_b / (np.sqrt(vh_b) + self.epsilon)).astype(_dtype())
        info = {"lr": self.lr_actual, "paso": float(self._paso)}
        self._emit("paso", info)
        return info

    def entrenar_paso(self, entrada: np.ndarray, objetivo: np.ndarray) -> Dict[str, float]:
        pred = self.forward(entrada)
        y = np.asarray(objetivo, dtype=_dtype())
        if y.ndim == 1:
            y = y.reshape(1, -1)
        # para ReLU la salida ya esta activada: MSE directo eficiente
        err = (pred.astype(np.float64) - y.astype(np.float64)) / max(1, pred.shape[0])
        loss = float(np.mean((pred.astype(np.float64) - y.astype(np.float64)) ** 2))
        gp, gb = self.backward(err.astype(_dtype()), entrada)
        opt = self.paso_optimizador(gp, gb)
        self.historial_loss.append(loss)
        if loss < self.mejor_loss:
            self.mejor_loss = loss
        return {"loss": loss, **opt}

    # ------------------------------------------------------------------
    def obtener_estadisticas(self) -> Dict[str, Any]:
        return {"nombre": self.nombre, "input_size": self.input_size,
                "output_size": self.output_size,
                "activaciones": len(self.historial_activaciones),
                "forwards": self.contador_forward, "backwards": self.contador_backward,
                "lr_actual": self.lr_actual, "paso": self._paso,
                "optimizador": self.optimizador}

    def obtener_estadisticas_completas(self) -> Dict[str, Any]:
        stats = self.obtener_estadisticas()
        stats.update({'tipo_relu': self.tipo_relu, 'alpha': self.alpha,
                      'tasa_activacion': self.tasa_activacion, 'tasa_muerta': self.tasa_muerta,
                      'varianza_real': self.varianza_real,
                      'mejor_loss': self.mejor_loss,
                      'ultima_loss': self.historial_loss[-1] if self.historial_loss else None,
                      't_medio_forward_ms': (self.tiempo_forward_acum / max(1, self.contador_forward)) * 1000.0,
                      't_medio_backward_ms': (self.tiempo_backward_acum / max(1, self.contador_backward)) * 1000.0})
        return stats

    def verificar_estabilidad(self) -> Dict[str, bool]:
        est = {'pesos_inicializados': self.pesos is not None, 'sesgo_inicializado': self.sesgo is not None}
        if self.pesos is not None:
            pmax = float(np.max(np.abs(self.pesos)))
            pmin = max(float(np.min(np.abs(self.pesos))), 1e-12)
            est['pesos_no_explosivos'] = pmax < 5.0
            est['pesos_no_desaparecen'] = pmin > 1e-6
            est['neuronas_muertas'] = self.tasa_activacion > 0.01
            est['sin_nan'] = bool(not np.isnan(self.pesos).any())
            est['sin_inf'] = bool(not np.isinf(self.pesos).any())
        return est

    def resumen_rendimiento(self) -> Dict[str, Any]:
        return {"forwards": self.contador_forward, "backwards": self.contador_backward,
                "t_forward_total_s": self.tiempo_forward_acum,
                "t_backward_total_s": self.tiempo_backward_acum}

    def reiniciar_metricas(self) -> None:
        self.historial_activaciones.clear()
        self.historial_gradientes.clear()
        self.historial_loss.clear()
        self.contador_forward = 0
        self.contador_backward = 0
        self.tiempo_forward_acum = 0.0
        self.tiempo_backward_acum = 0.0

    def a_dict(self) -> Dict[str, Any]:
        return {"input_size": self.input_size, "output_size": self.output_size,
                "nombre": self.nombre, "tipo_relu": self.tipo_relu,
                "alpha": self.alpha, "lr_base": self.lr_base,
                "optimizador": self.optimizador,
                "pesos": self.pesos.tolist() if self.pesos is not None else None,
                "sesgo": self.sesgo.tolist() if self.sesgo is not None else None}

    def guardar(self, ruta: str) -> None:
        with open(ruta, "w", encoding="utf-8") as f:
            json.dump(self.a_dict(), f)

    @classmethod
    def cargar(cls, ruta: str) -> "NeuronaEntradaReLU":
        with open(ruta, "r", encoding="utf-8") as f:
            d = json.load(f)
        obj = cls(d["input_size"], d["output_size"], d.get("nombre", "NeuronaEntradaReLU"),
                  tipo_relu=d.get("tipo_relu", "relu"), alpha=d.get("alpha", 0.01))
        if d.get("pesos") is not None:
            obj.pesos = np.asarray(d["pesos"], dtype=_dtype())
            obj.sesgo = np.asarray(d["sesgo"], dtype=_dtype())
        return obj

    def __str__(self) -> str:
        return f"{self.nombre}(ent={self.input_size},sal={self.output_size},relu={self.tipo_relu},opt={self.optimizador})"
