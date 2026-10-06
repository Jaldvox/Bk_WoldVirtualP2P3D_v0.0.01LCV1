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
    """Buffer circular eficiente para historiales (evita crecimiento infinito)."""

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


class NeuronaEntradaBasica(NeuronaEntradaBase):
    """Neurona de entrada densa con init He, Adam/SGD, clipping y metricas.

    Mantiene API original: inicializar_pesos, forward, backward,
    obtener_estadisticas_completas, verificar_estabilidad, __str__.
    """

    def __init__(self, input_size: int, output_size: int,
                 nombre: str = "NeuronaEntradaBasica",
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
                 normalizar_entrada: bool = False,
                 semilla: Optional[int] = None,
                 capacidad_historial: int = 256):
        super().__init__(input_size, output_size, nombre)
        self.factor_escala = float(factor_escala)
        self.varianza_esperada = 0.0
        self.varianza_real = 0.0
        # Hiperparametros eficientes
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
        self.normalizar_entrada = bool(normalizar_entrada)
        self.semilla = semilla if semilla is not None else int(LUCIA_CONFIG.get("random_seed", 42))
        self.rng = np.random.default_rng(self.semilla)
        # Estado Adam
        self._m_pesos: Optional[np.ndarray] = None
        self._v_pesos: Optional[np.ndarray] = None
        self._m_sesgo: Optional[np.ndarray] = None
        self._v_sesgo: Optional[np.ndarray] = None
        self._paso = 0
        # Cache y estadisticas de rendimiento
        self._cache_entrada: Optional[np.ndarray] = None
        self._cache_salida: Optional[np.ndarray] = None
        self._media_entrada: Optional[np.ndarray] = None
        self._std_entrada: Optional[np.ndarray] = None
        self.historial_activaciones = _RingBuffer(capacidad_historial)
        self.historial_gradientes = _RingBuffer(capacidad_historial)
        self.historial_loss: List[float] = []
        self.historial_lr: List[float] = []
        self.historial_tiempo: List[float] = []
        self.contador_forward = 0
        self.contador_backward = 0
        self.tiempo_forward_acum = 0.0
        self.tiempo_backward_acum = 0.0
        self.mejor_loss: float = float("inf")
        self.callbacks: List[Callable[[str, Dict[str, Any]], None]] = []

    # ------------------------------------------------------------------
    # Utilidades internas
    # ------------------------------------------------------------------
    def _emit(self, evento: str, info: Optional[Dict[str, Any]] = None) -> None:
        for cb in self.callbacks:
            try:
                cb(evento, info or {})
            except Exception:
                continue

    def registrar_callback(self, fn: Callable[[str, Dict[str, Any]], None]) -> None:
        self.callbacks.append(fn)

    def _asegurar_estado_adam(self) -> None:
        dt = _dtype()
        if self._m_pesos is None:
            self._m_pesos = np.zeros_like(self.pesos, dtype=np.float64)
            self._v_pesos = np.zeros_like(self.pesos, dtype=np.float64)
            self._m_sesgo = np.zeros_like(self.sesgo, dtype=np.float64)
            self._v_sesgo = np.zeros_like(self.sesgo, dtype=np.float64)

    def _aplicar_decaimiento_lr(self) -> None:
        if self.lr_decay > 0.0:
            self.lr_actual = max(self.lr_min, self.lr_base / (1.0 + self.lr_decay * self._paso))
        self.historial_lr.append(self.lr_actual)

    def _clip(self, g: np.ndarray) -> np.ndarray:
        if self.grad_clip and self.grad_clip > 0:
            n = float(np.linalg.norm(g))
            if n > self.grad_clip:
                g = g * (self.grad_clip / (n + 1e-12))
        return g

    def _preparar_entrada(self, entrada: np.ndarray) -> np.ndarray:
        x = np.asarray(entrada, dtype=_dtype())
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if x.shape[1] != self.input_size:
            raise ValueError(f"entrada dim {x.shape[1]} != input_size {self.input_size}")
        if self.normalizar_entrada:
            mu = x.mean(axis=0, keepdims=True)
            sd = x.std(axis=0, keepdims=True) + 1e-8
            self._media_entrada = mu.astype(_dtype())
            self._std_entrada = sd.astype(_dtype())
            x = ((x - mu) / sd).astype(_dtype(), copy=False)
        x = np.ascontiguousarray(x, dtype=_dtype())
        return x

    # ------------------------------------------------------------------
    # API principal (estructura original preservada)
    # ------------------------------------------------------------------
    def inicializar_pesos(self, modo: str = "he") -> None:
        dt = _dtype()
        modo = str(modo).lower()
        if modo == "he":
            std = math.sqrt(2.0 / self.input_size) * self.factor_escala
            self.pesos = self.rng.normal(0.0, std, (self.input_size, self.output_size)).astype(dt)
        elif modo == "xavier":
            lim = math.sqrt(6.0 / (self.input_size + self.output_size)) * self.factor_escala
            self.pesos = self.rng.uniform(-lim, lim, (self.input_size, self.output_size)).astype(dt)
        elif modo == "ortogonal":
            a = self.rng.normal(0, 1, (self.input_size, self.output_size))
            q, _ = np.linalg.qr(a) if self.input_size >= self.output_size else np.linalg.qr(a.T)
            q = q if self.input_size >= self.output_size else q.T
            self.pesos = (q[:, :self.output_size] * self.factor_escala).astype(dt)
        elif modo == "ceros":
            self.pesos = np.zeros((self.input_size, self.output_size), dtype=dt)
        else:
            raise ValueError(f"modo desconocido: {modo}")
        self.sesgo = np.zeros((1, self.output_size), dtype=dt)
        self.varianza_esperada = 2.0 / self.input_size
        self.varianza_real = float(np.var(self.pesos))
        self._m_pesos = self._v_pesos = self._m_sesgo = self._v_sesgo = None
        self._paso = 0
        self._emit("init", {"modo": modo, "var": self.varianza_real})

    def forward(self, entrada: np.ndarray) -> np.ndarray:
        t0 = _now()
        if self.pesos is None:
            self.inicializar_pesos()
        x = self._preparar_entrada(entrada)
        # BLAS eficiente: dot sobre arrays contiguos
        salida = np.dot(x, self.pesos) + self.sesgo
        salida = np.ascontiguousarray(salida, dtype=_dtype())
        if self.usar_cache_forward:
            self._cache_entrada = x
            self._cache_salida = salida
        v = float(np.var(salida)) if salida.size else 0.0
        self.varianza_real = 0.9 * self.varianza_real + 0.1 * v if len(self.historial_activaciones) else v
        self.historial_activaciones.append(salida.copy())
        self.contador_forward += 1
        dt = _now() - t0
        self.tiempo_forward_acum += dt
        self.historial_tiempo.append(dt)
        return salida

    def forward_batch(self, lote: List[np.ndarray]) -> List[np.ndarray]:
        return [self.forward(x) for x in lote]

    def backward(self, gradiente_salida: np.ndarray, entrada: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        t0 = _now()
        g = np.ascontiguousarray(np.asarray(gradiente_salida, dtype=_dtype()))
        if g.ndim == 1:
            g = g.reshape(1, -1)
        x = self._cache_entrada if (entrada is None and self._cache_entrada is not None) else self._preparar_entrada(entrada if entrada is not None else self._cache_entrada)
        # Gradientes vectorizados: X^T @ G / batch (media para estabilidad)
        b = max(1, x.shape[0])
        grad_pesos = np.dot(x.T, g) / b
        grad_sesgo = np.sum(g, axis=0, keepdims=True) / b
        grad_pesos = self._clip(np.asarray(grad_pesos, dtype=np.float64))
        grad_sesgo = self._clip(np.asarray(grad_sesgo, dtype=np.float64))
        # Decaimiento L2 desacoplado
        if self.weight_decay > 0.0 and self.pesos is not None:
            grad_pesos = grad_pesos + self.weight_decay * self.pesos.astype(np.float64)
        self.historial_gradientes.append({'pesos': np.asarray(grad_pesos, dtype=_dtype()).copy(),
                                          'sesgo': np.asarray(grad_sesgo, dtype=_dtype()).copy(),
                                          'norma_pesos': float(np.linalg.norm(grad_pesos)),
                                          'norma_sesgo': float(np.linalg.norm(grad_sesgo))})
        self.contador_backward += 1
        self.tiempo_backward_acum += _now() - t0
        return np.asarray(grad_pesos, dtype=_dtype()), np.asarray(grad_sesgo, dtype=_dtype())

    def grad_entrada(self, gradiente_salida: np.ndarray) -> np.ndarray:
        g = np.asarray(gradiente_salida, dtype=_dtype())
        if g.ndim == 1:
            g = g.reshape(1, -1)
        return np.dot(g, self.pesos.T)

    def paso_optimizador(self, grad_pesos: np.ndarray, grad_sesgo: np.ndarray) -> Dict[str, float]:
        if self.pesos is None:
            raise RuntimeError("pesos no inicializados")
        self._paso += 1
        self._aplicar_decaimiento_lr()
        gp = np.asarray(grad_pesos, dtype=np.float64)
        gb = np.asarray(grad_sesgo, dtype=np.float64)
        if self.optimizador == "sgd":
            self.pesos = (self.pesos.astype(np.float64) - self.lr_actual * gp).astype(_dtype())
            self.sesgo = (self.sesgo.astype(np.float64) - self.lr_actual * gb).astype(_dtype())
        elif self.optimizador == "sgd_momentum":
            self._asegurar_estado_adam()
            assert self._m_pesos is not None and self._m_sesgo is not None
            self._m_pesos[:] = self.beta1 * self._m_pesos + (1 - self.beta1) * gp
            self._m_sesgo[:] = self.beta1 * self._m_sesgo + (1 - self.beta1) * gb
            self.pesos = (self.pesos.astype(np.float64) - self.lr_actual * self._m_pesos).astype(_dtype())
            self.sesgo = (self.sesgo.astype(np.float64) - self.lr_actual * self._m_sesgo).astype(_dtype())
        else:  # adam por defecto
            self._asegurar_estado_adam()
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
        err = (pred - y) / max(1, pred.shape[0])
        loss = float(np.mean((pred - y) ** 2))
        gp, gb = self.backward(err, entrada)
        opt = self.paso_optimizador(gp, gb)
        self.historial_loss.append(loss)
        if loss < self.mejor_loss:
            self.mejor_loss = loss
        return {"loss": loss, **opt}

    # ------------------------------------------------------------------
    # Estadisticas, estabilidad, persistencia y rendimiento
    # ------------------------------------------------------------------
    def obtener_estadisticas(self) -> Dict[str, Any]:
        base = {"nombre": self.nombre, "input_size": self.input_size,
                "output_size": self.output_size,
                "activaciones": len(self.historial_activaciones)}
        base.update({"forwards": self.contador_forward, "backwards": self.contador_backward,
                     "lr_actual": self.lr_actual, "paso": self._paso,
                     "optimizador": self.optimizador})
        return base

    def obtener_estadisticas_completas(self) -> Dict[str, Any]:
        stats = self.obtener_estadisticas()
        pw = float(np.mean(np.abs(self.pesos))) if self.pesos is not None else 0.0
        nw = float(np.linalg.norm(self.pesos)) if self.pesos is not None else 0.0
        tf = self.tiempo_forward_acum / max(1, self.contador_forward)
        tb = self.tiempo_backward_acum / max(1, self.contador_backward)
        stats.update({'varianza_esperada': self.varianza_esperada, 'varianza_real': self.varianza_real,
                      'media_abs_pesos': pw, 'norma_pesos': nw,
                      'mejor_loss': self.mejor_loss,
                      'ultima_loss': self.historial_loss[-1] if self.historial_loss else None,
                      't_medio_forward_ms': tf * 1000.0, 't_medio_backward_ms': tb * 1000.0})
        return stats

    def verificar_estabilidad(self) -> Dict[str, bool]:
        est = {'pesos_inicializados': self.pesos is not None, 'sesgo_inicializado': self.sesgo is not None}
        if self.pesos is not None:
            pmax = float(np.max(np.abs(self.pesos)))
            pmin = max(float(np.min(np.abs(self.pesos))), 1e-12)
            est['pesos_no_explosivos'] = pmax < 5.0
            est['pesos_no_desaparecen'] = pmin > 1e-6
            est['pesos_balanceados'] = pmax / pmin < 1000.0
            est['sin_nan'] = bool(not np.isnan(self.pesos).any())
            est['sin_inf'] = bool(not np.isinf(self.pesos).any())
            est['sesgo_acotado'] = bool(float(np.max(np.abs(self.sesgo))) < 5.0)
        return est

    def resumen_rendimiento(self) -> Dict[str, Any]:
        return {"forwards": self.contador_forward, "backwards": self.contador_backward,
                "t_forward_total_s": self.tiempo_forward_acum, "t_backward_total_s": self.tiempo_backward_acum,
                "t_medio_forward_ms": (self.tiempo_forward_acum / max(1, self.contador_forward)) * 1000.0,
                "t_medio_backward_ms": (self.tiempo_backward_acum / max(1, self.contador_backward)) * 1000.0}

    def reiniciar_metricas(self) -> None:
        self.historial_activaciones.clear()
        self.historial_gradientes.clear()
        self.historial_loss.clear()
        self.historial_lr.clear()
        self.historial_tiempo.clear()
        self.contador_forward = 0
        self.contador_backward = 0
        self.tiempo_forward_acum = 0.0
        self.tiempo_backward_acum = 0.0

    def a_dict(self) -> Dict[str, Any]:
        return {"input_size": self.input_size, "output_size": self.output_size, "nombre": self.nombre,
                "factor_escala": self.factor_escala, "lr_base": self.lr_base, "optimizador": self.optimizador,
                "pesos": self.pesos.tolist() if self.pesos is not None else None,
                "sesgo": self.sesgo.tolist() if self.sesgo is not None else None}

    def guardar(self, ruta: str) -> None:
        with open(ruta, "w", encoding="utf-8") as f:
            json.dump(self.a_dict(), f)

    @classmethod
    def cargar(cls, ruta: str) -> "NeuronaEntradaBasica":
        with open(ruta, "r", encoding="utf-8") as f:
            d = json.load(f)
        obj = cls(d["input_size"], d["output_size"], d.get("nombre", "NeuronaEntradaBasica"),
                  factor_escala=d.get("factor_escala", 1.0),
                  learning_rate=d.get("lr_base", 0.001),
                  optimizador=d.get("optimizador", "adam"))
        if d.get("pesos") is not None:
            obj.pesos = np.asarray(d["pesos"], dtype=_dtype())
            obj.sesgo = np.asarray(d["sesgo"], dtype=_dtype())
        return obj

    def __str__(self) -> str:
        return f"{self.nombre}(ent={self.input_size},sal={self.output_size},init=He,opt={self.optimizador},lr={self.lr_actual:.5f})"