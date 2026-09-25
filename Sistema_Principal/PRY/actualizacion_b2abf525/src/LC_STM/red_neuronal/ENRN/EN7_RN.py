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


class NeuronaEntradaBatchNorm(NeuronaEntradaBase):
    """EN7 BatchNorm eficiente: forward/backward BN correctos, Adam/SGD.

    Preserva API: momentum, epsilon, factor_escala, modo_entrenamiento,
    gamma, beta, running_mean/var, inicializar_pesos, forward, backward.
    """

    def __init__(self, input_size: int, output_size: int,
                 nombre: str = "NeuronaEntradaBatchNorm",
                 momentum: float = 0.9,
                 epsilon: float = 1e-5,
                 factor_escala: float = 1.0,
                 modo_entrenamiento: bool = True,
                 learning_rate: float = 0.001,
                 optimizador: str = "adam",
                 beta1: float = 0.9,
                 beta2: float = 0.999,
                 eps_adam: float = 1e-8,
                 weight_decay: float = 0.0,
                 grad_clip: float = 5.0,
                 lr_decay: float = 0.0,
                 lr_min: float = 1e-6,
                 usar_cache_forward: bool = True,
                 semilla: Optional[int] = None,
                 capacidad_historial: int = 256):
        super().__init__(input_size, output_size, nombre)
        self.momentum = float(momentum)
        self.epsilon = float(epsilon)
        self.factor_escala = float(factor_escala)
        self.modo_entrenamiento = bool(modo_entrenamiento)
        dt = _dtype()
        self.gamma: Optional[np.ndarray] = None
        self.beta: Optional[np.ndarray] = None
        self.running_mean: Optional[np.ndarray] = None
        self.running_var: Optional[np.ndarray] = None
        self.varianza_real = 0.0
        self.varianza_esperada = 1.0
        self.lr_base = float(learning_rate or LUCIA_CONFIG.get("default_learning_rate", 0.001))
        self.lr_actual = float(self.lr_base)
        self.optimizador = str(optimizador).lower()
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.eps_adam = float(eps_adam)
        self.weight_decay = float(weight_decay)
        self.grad_clip = float(grad_clip)
        self.lr_decay = float(lr_decay)
        self.lr_min = float(lr_min)
        self.usar_cache_forward = bool(usar_cache_forward)
        self.semilla = semilla if semilla is not None else int(LUCIA_CONFIG.get("random_seed", 42))
        self.rng = np.random.default_rng(self.semilla)
        self._m: Dict[str, np.ndarray] = {}
        self._v: Dict[str, np.ndarray] = {}
        self._paso = 0
        self._cache: Dict[str, Any] = {}
        self.historial_activaciones = _RingBuffer(capacidad_historial)
        self.historial_gradientes = _RingBuffer(capacidad_historial)
        self.historial_loss: List[float] = []
        self.historial_lr: List[float] = []
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

    def entrenar(self) -> None:
        self.modo_entrenamiento = True

    def evaluar(self) -> None:
        self.modo_entrenamiento = False

    def _asegurar_opt(self, clave: str, ref: np.ndarray) -> None:
        if clave not in self._m:
            self._m[clave] = np.zeros_like(ref, dtype=np.float64)
            self._v[clave] = np.zeros_like(ref, dtype=np.float64)

    def _asegurar_todo(self) -> None:
        assert self.pesos is not None and self.sesgo is not None
        assert self.gamma is not None and self.beta is not None
        for k, r in (("w", self.pesos), ("b", self.sesgo), ("g", self.gamma), ("bt", self.beta)):
            self._asegurar_opt(k, r)

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

    def _bn_forward(self, z: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        assert self.gamma is not None and self.beta is not None
        z64 = z.astype(np.float64)
        if self.modo_entrenamiento:
            mu = z64.mean(axis=0, keepdims=True)
            var = z64.var(axis=0, keepdims=True)
            if self.running_mean is None:
                self.running_mean, self.running_var = mu.astype(_dtype()), var.astype(_dtype())
            else:
                m = self.momentum
                self.running_mean = (m * self.running_mean.astype(np.float64) + (1 - m) * mu).astype(_dtype())
                self.running_var = (m * self.running_var.astype(np.float64) + (1 - m) * var).astype(_dtype())
        else:
            mu = self.running_mean.astype(np.float64) if self.running_mean is not None else np.zeros((1, z.shape[1]))
            var = self.running_var.astype(np.float64) if self.running_var is not None else np.ones((1, z.shape[1]))
        std_inv = 1.0 / np.sqrt(var + self.epsilon)
        zn = (z64 - mu) * std_inv
        out = zn * self.gamma.astype(np.float64) + self.beta.astype(np.float64)
        cache = {"zn": zn, "std_inv": std_inv, "gamma": self.gamma.astype(np.float64), "m": z.shape[0]}
        return out.astype(_dtype(), copy=False), cache

    def _bn_backward(self, g: np.ndarray, cache: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        zn, std_inv, gamma, m = cache["zn"], cache["std_inv"], cache["gamma"], cache["m"]
        g64 = np.asarray(g, dtype=np.float64)
        g_gamma = np.sum(g64 * zn, axis=0, keepdims=True)
        g_beta = np.sum(g64, axis=0, keepdims=True)
        gz = (1.0 / m) * gamma * std_inv * (m * g64 - np.sum(g64, axis=0, keepdims=True)
                                            - zn * np.sum(g64 * zn, axis=0, keepdims=True))
        return gz.astype(_dtype()), g_gamma.astype(_dtype()), g_beta.astype(_dtype())

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
        self.gamma = np.ones((1, self.output_size), dtype=dt)
        self.beta = np.zeros((1, self.output_size), dtype=dt)
        self.varianza_real = float(np.var(self.pesos))
        self._m.clear(); self._v.clear(); self._paso = 0
        self._emit("init", {"modo": modo})

    def forward(self, entrada: np.ndarray, entrenamiento: Optional[bool] = None) -> np.ndarray:
        t0 = _now()
        if self.pesos is None:
            self.inicializar_pesos()
        assert self.pesos is not None and self.sesgo is not None
        if entrenamiento is not None:
            self.modo_entrenamiento = bool(entrenamiento)
        x = self._preparar(entrada)
        z = np.dot(x, self.pesos) + self.sesgo
        salida, cache_bn = self._bn_forward(z)
        salida = np.ascontiguousarray(salida, dtype=_dtype())
        if self.usar_cache_forward:
            self._cache = {"x": x, "z": z, "bn": cache_bn}
        v = float(np.var(salida)) if salida.size else 0.0
        self.varianza_real = 0.9 * self.varianza_real + 0.1 * v if len(self.historial_activaciones) else v
        self.historial_activaciones.append(salida.copy())
        self.contador_forward += 1
        self.tiempo_forward_acum += _now() - t0
        return salida

    def forward_batch(self, lote: List[np.ndarray]) -> List[np.ndarray]:
        return [self.forward(x) for x in lote]

    def backward(self, gradiente_salida: np.ndarray, entrada: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        t0 = _now()
        g = np.ascontiguousarray(np.asarray(gradiente_salida, dtype=_dtype()))
        if g.ndim == 1:
            g = g.reshape(1, -1)
        x = self._preparar(entrada) if entrada is not None else self._cache.get("x")
        if x is None:
            raise ValueError("sin entrada ni cache forward")
        bn = self._cache.get("bn")
        if bn is None:
            gz = g
            g_gamma = np.zeros_like(self.gamma); g_beta = np.zeros_like(self.beta)
        else:
            gz, g_gamma, g_beta = self._bn_backward(g, bn)
        b = max(1, x.shape[0])
        gp = self._clip(np.asarray(np.dot(x.T, gz.astype(np.float64)) / b, dtype=np.float64))
        gb = self._clip(np.asarray(np.sum(gz.astype(np.float64), axis=0, keepdims=True) / b, dtype=np.float64))
        if self.weight_decay > 0.0 and self.pesos is not None:
            gp = gp + self.weight_decay * self.pesos.astype(np.float64)
        self._cache["g_gamma"] = np.asarray(g_gamma, dtype=_dtype())
        self._cache["g_beta"] = np.asarray(g_beta, dtype=_dtype())
        self.historial_gradientes.append({'pesos': np.asarray(gp, dtype=_dtype()).copy(),
                                          'sesgo': np.asarray(gb, dtype=_dtype()).copy(),
                                          'norma_pesos': float(np.linalg.norm(gp)),
                                          'norma_sesgo': float(np.linalg.norm(gb))})
        self.contador_backward += 1
        self.tiempo_backward_acum += _now() - t0
        return np.asarray(gp, dtype=_dtype()), np.asarray(gb, dtype=_dtype())

    def grad_entrada(self, gradiente_salida: np.ndarray) -> np.ndarray:
        g = np.asarray(gradiente_salida, dtype=_dtype())
        if g.ndim == 1:
            g = g.reshape(1, -1)
        assert self.pesos is not None
        return np.dot(g, self.pesos.T)

    def _adam(self, p: np.ndarray, gr: np.ndarray, k: str) -> np.ndarray:
        self._asegurar_opt(k, p)
        m, v = self._m[k], self._v[k]
        m[:] = self.beta1 * m + (1 - self.beta1) * gr
        v[:] = self.beta2 * v + (1 - self.beta2) * (gr * gr)
        mh = m / (1 - self.beta1 ** self._paso)
        vh = v / (1 - self.beta2 ** self._paso)
        return (p.astype(np.float64) - self.lr_actual * mh / (np.sqrt(vh) + self.eps_adam)).astype(_dtype())

    def paso_optimizador(self, grad_pesos: np.ndarray, grad_sesgo: np.ndarray) -> Dict[str, float]:
        if self.pesos is None or self.sesgo is None or self.gamma is None or self.beta is None:
            raise RuntimeError("pesos no inicializados")
        self._paso += 1
        self._decaer_lr()
        gp = np.asarray(grad_pesos, dtype=np.float64)
        gb = np.asarray(grad_sesgo, dtype=np.float64)
        gg = np.asarray(self._cache.get("g_gamma", np.zeros_like(self.gamma)), dtype=np.float64)
        gbt = np.asarray(self._cache.get("g_beta", np.zeros_like(self.beta)), dtype=np.float64)
        if self.optimizador == "sgd":
            self.pesos = (self.pesos.astype(np.float64) - self.lr_actual * gp).astype(_dtype())
            self.sesgo = (self.sesgo.astype(np.float64) - self.lr_actual * gb).astype(_dtype())
            self.gamma = (self.gamma.astype(np.float64) - self.lr_actual * gg).astype(_dtype())
            self.beta = (self.beta.astype(np.float64) - self.lr_actual * gbt).astype(_dtype())
        elif self.optimizador == "sgd_momentum":
            self._asegurar_todo()
            for k, gr in (("w", gp), ("b", gb), ("g", gg), ("bt", gbt)):
                self._m[k][:] = self.beta1 * self._m[k] + (1 - self.beta1) * gr
            self.pesos = (self.pesos.astype(np.float64) - self.lr_actual * self._m["w"]).astype(_dtype())
            self.sesgo = (self.sesgo.astype(np.float64) - self.lr_actual * self._m["b"]).astype(_dtype())
            self.gamma = (self.gamma.astype(np.float64) - self.lr_actual * self._m["g"]).astype(_dtype())
            self.beta = (self.beta.astype(np.float64) - self.lr_actual * self._m["bt"]).astype(_dtype())
        else:
            self.pesos = self._adam(self.pesos, gp, "w")
            self.sesgo = self._adam(self.sesgo, gb, "b")
            self.gamma = self._adam(self.gamma, gg, "g")
            self.beta = self._adam(self.beta, gbt, "bt")
        info = {"lr": self.lr_actual, "paso": float(self._paso)}
        self._emit("paso", info)
        return info

    def entrenar_paso(self, entrada: np.ndarray, objetivo: np.ndarray) -> Dict[str, float]:
        pred = self.forward(entrada, entrenamiento=True)
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
    def obtener_estadisticas(self) -> Dict[str, Any]:
        return {"nombre": self.nombre, "input_size": self.input_size,
                "output_size": self.output_size,
                "activaciones": len(self.historial_activaciones),
                "forwards": self.contador_forward, "backwards": self.contador_backward,
                "lr_actual": self.lr_actual, "paso": self._paso,
                "optimizador": self.optimizador}

    def obtener_estadisticas_completas(self) -> Dict[str, Any]:
        stats = self.obtener_estadisticas()
        stats.update({'varianza_real': self.varianza_real, 'momentum': self.momentum,
                      'running_mean_norm': float(np.linalg.norm(self.running_mean)) if self.running_mean is not None else 0.0,
                      'running_var_media': float(np.mean(self.running_var)) if self.running_var is not None else 0.0,
                      'gamma_media': float(np.mean(self.gamma)) if self.gamma is not None else 0.0,
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
            est['bn_inicializado'] = self.gamma is not None
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
                "nombre": self.nombre, "momentum": self.momentum,
                "lr_base": self.lr_base, "optimizador": self.optimizador,
                "pesos": self.pesos.tolist() if self.pesos is not None else None,
                "sesgo": self.sesgo.tolist() if self.sesgo is not None else None,
                "gamma": self.gamma.tolist() if self.gamma is not None else None,
                "beta": self.beta.tolist() if self.beta is not None else None}

    def guardar(self, ruta: str) -> None:
        with open(ruta, "w", encoding="utf-8") as f:
            json.dump(self.a_dict(), f)

    @classmethod
    def cargar(cls, ruta: str) -> "NeuronaEntradaBatchNorm":
        with open(ruta, "r", encoding="utf-8") as f:
            d = json.load(f)
        obj = cls(d["input_size"], d["output_size"], d.get("nombre", "NeuronaEntradaBatchNorm"),
                  momentum=d.get("momentum", 0.9))
        if d.get("pesos") is not None:
            obj.pesos = np.asarray(d["pesos"], dtype=_dtype())
            obj.sesgo = np.asarray(d["sesgo"], dtype=_dtype())
            obj.gamma = np.asarray(d["gamma"], dtype=_dtype())
            obj.beta = np.asarray(d["beta"], dtype=_dtype())
        return obj

    def __str__(self) -> str:
        return f"{self.nombre}(ent={self.input_size},sal={self.output_size},bn_mom={self.momentum},opt={self.optimizador})"
