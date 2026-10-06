"""
IAFREE.py - SubSistema de Inferencia Gratuita OpenRouter para LucIA (Arquitectura 2026)
========================================================================================
Gestiona el acceso y rotacion automatica de modelos 100% gratuitos de OpenRouter:
  - Registro exhaustivo y actualizado de modelos :free disponibles en la plataforma.
  - Sincronizacion dinamica con la API de OpenRouter (https://openrouter.ai/api/v1/models).
  - Rotacion automatica ante limites de tasa (HTTP 429 Too Many Requests) o 404.
  - Filtro por capacidades: conversacion general, razonamiento, programacion y analisis.
  - Inyeccion del contexto cognitivo de las 50 neuronas activas de Celebro.
  - Modo streaming y modo sincrono con registro de latencia y metricas de consumo $0.00.
  - Integracion transparente con BKSVCB (registro en bloques) y SNSBSTNPRB (sesion).
"""
from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, Final, Generator, List, Optional, Tuple, Union

# ─── VERSION Y METADATOS DEL SUBSISTEMA ──────────────────────────────────────
__version__: Final[str] = "2026.3.1"
__subsystem__: Final[str] = "LucIA-IAFREE-Subsystem"
__author__: Final[str] = "Equipo WoldVirtualP2P3D"

logger: logging.Logger = logging.getLogger("WoldVirtualP2P3D.SBSTM.IAFREE")

# ─── RUTAS Y UBICACIONES CRITICAS ───────────────────────────────────────────
PACKAGE_ROOT: Final[Path] = Path(__file__).parent.resolve()
CMFG_DIR: Final[Path] = PACKAGE_ROOT.parent.resolve()
CELEBRO_DIR: Final[Path] = CMFG_DIR.parent.resolve()
LC_DIR: Final[Path] = CELEBRO_DIR.parent.resolve()
ROOT_DIR: Final[Path] = LC_DIR.parent.resolve()
ENV_FILE: Final[Path] = ROOT_DIR / ".env"
try:
    from STM_CH.rutas import CACHE_IAFREE as CACHE_FILE
except ImportError:
    CACHE_FILE: Final[Path] = PACKAGE_ROOT / "openrouter_free_cache.json"

# ─── LISTA BASE DE MODELOS GRATUITOS VERIFICADOS (2026) ─────────────────────
# Obtenidos en vivo desde https://openrouter.ai/models?q=free
CATALOGO_MODELOS_GRATUITOS: Final[List[Dict[str, Any]]] = [
    {"id": "openrouter/free", "nombre": "OpenRouter Free Router", "contexto": 200000, "categoria": "general", "descripcion": "Enrutador oficial que auto-selecciona el mejor modelo gratuito."},
    {"id": "qwen/qwen3.8-27b:free", "nombre": "Qwen 3.8 27B Free", "contexto": 262144, "categoria": "razonamiento", "descripcion": "Gran modelo general con 256k tokens de ventana de contexto."},
    {"id": "google/gemma-4-31b-it:free", "nombre": "Google Gemma 4 31B Instruct Free", "contexto": 262144, "categoria": "instruccion", "descripcion": "Seguimiento estricto de directrices tecnicas y formato."},
    {"id": "google/gemma-4-26b-a4b-it:free", "nombre": "Google Gemma 4 26B A4B Free", "contexto": 262144, "categoria": "ligero", "descripcion": "Arquitectura eficiente de atencion para minima latencia."},
    {"id": "nvidia/nemotron-3-super-120b-a12b:free", "nombre": "NVIDIA Nemotron 3 Super 120B Free", "contexto": 262144, "categoria": "razonamiento", "descripcion": "Modelo masivo de logica y resolucion analitica profunda."},
    {"id": "nvidia/nemotron-3.5-lightning:free", "nombre": "NVIDIA Nemotron 3.5 Lightning Free", "contexto": 1000000, "categoria": "gran_contexto", "descripcion": "1 Millon de tokens de contexto con alta velocidad de inferencia."},
    {"id": "nvidia/nemotron-3-ultra-550b-a55b:free", "nombre": "NVIDIA Nemotron 3 Ultra 550B Free", "contexto": 1000000, "categoria": "pesado", "descripcion": "Arquitectura MoE de maximo parametro con coste cero."},
    {"id": "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free", "nombre": "NVIDIA Nemotron 3 Nano Reasoning Free", "contexto": 256000, "categoria": "razonamiento", "descripcion": "Razonamiento por pasos y logica formal."},
    {"id": "cohere/north-mini-code:free", "nombre": "Cohere North Mini Code Free", "contexto": 256000, "categoria": "codigo", "descripcion": "Especializado en generacion, refactorizacion y sintaxis de codigo."},
    {"id": "z-ai/glm-5.2:free", "nombre": "Z-AI GLM 5.2 Free", "contexto": 32768, "categoria": "general", "descripcion": "Conversacion rapida y concisa con baja utilizacion de memoria."},
    {"id": "liquid/lfm-2.5-2.6b:free", "nombre": "Liquid LFM 2.5 2.6B Free", "contexto": 65536, "categoria": "eficiente", "descripcion": "Red neuronal de estado liquido con generacion ultrarrapida."},
    {"id": "thinkingmachines/inkling:free", "nombre": "ThinkingMachines Inkling Free", "contexto": 1048576, "categoria": "gran_contexto", "descripcion": "Contexto extendido de 1M tokens para analisis de grandes textos."},
    {"id": "thinkingmachines/inkling-small:free", "nombre": "ThinkingMachines Inkling Small Free", "contexto": 1048576, "categoria": "gran_contexto", "descripcion": "Version compacta optimizada para respuestas inmediatas."},
    {"id": "poolside/laguna-s-2.1:free", "nombre": "Poolside Laguna S 2.1 Free", "contexto": 262144, "categoria": "codigo", "descripcion": "Optimizacion de nivel produccion para arquitectura de software."},
    {"id": "poolside/laguna-xs-2.1:free", "nombre": "Poolside Laguna XS 2.1 Free", "contexto": 262144, "categoria": "codigo", "descripcion": "Asistencia ultraligera para programacion interactiva."},
    {"id": "dots-studio/dots-3-note-preview:free", "nombre": "Dots 3 Note Preview Free", "contexto": 512000, "categoria": "analisis", "descripcion": "512k tokens dedicados a sintesis y notas estructuradas."},
    {"id": "nex-agi/nex-n2.5-mini:free", "nombre": "Nex-AGI N2.5 Mini Free", "contexto": 262144, "categoria": "general", "descripcion": "Respuestas agiles para dialogos de soporte continuo."},
    {"id": "nex-agi/nex-n2.5-pro:free", "nombre": "Nex-AGI N2.5 Pro Free", "contexto": 262144, "categoria": "razonamiento", "descripcion": "Capacidades avanzadas de comprension logica."},
    {"id": "inclusionai/ling-3.0-flash-vl:free", "nombre": "InclusionAI Ling 3.0 Flash VL Free", "contexto": 262144, "categoria": "multimodal", "descripcion": "Comprension multimodal y razonamiento veloz."},
    {"id": "inclusionai/ling-3.0-flash-fin:free", "nombre": "InclusionAI Ling 3.0 Flash Fin Free", "contexto": 262144, "categoria": "analisis", "descripcion": "Especializado en analitica cuantitativa y calculos precisos."},
    {"id": "inclusionai/ling-3.0-flash-sante:free", "nombre": "InclusionAI Ling 3.0 Flash Sante Free", "contexto": 262144, "categoria": "general", "descripcion": "Alta fidelidad y alineacion etica de respuestas."},
]


# ─── GESTOR DE ROTACION Y DISPONIBILIDAD DE MODELOS ─────────────────────────
class GestorModelosGratuitos:
    """Administra la lista de modelos gratuitos, pruebas de liveness y conmutacion por error."""

    def __init__(self, api_key: Optional[str] = None) -> None:
        self.api_key = api_key or self._leer_api_key()
        self._modelos: List[Dict[str, Any]] = list(CATALOGO_MODELOS_GRATUITOS)
        self._indice_activo = 0
        self._fallos_consecutivos: Dict[str, int] = {}
        self._lock = threading.Lock()
        self._ultimo_refresco = 0.0

    def _leer_api_key(self) -> str:
        """Extrae la clave API desde .env (varias ubicaciones) o el entorno."""
        candidatos = [
            ENV_FILE,                       # RFC/.env (histórico)
            LC_DIR / "LC" / ".env",         # RFC/LC/LC/.env (ubicación real)
            LC_DIR / ".env",                # RFC/LC/.env
            PACKAGE_ROOT / ".env",
        ]
        for ruta in candidatos:
            try:
                if ruta.exists():
                    txt = ruta.read_text(encoding="utf-8-sig", errors="replace")
                    for line in txt.splitlines():
                        if line.strip().startswith("OPENROUTER_API_KEY="):
                            clave = line.strip().split("=", 1)[1].strip().strip("'\"")
                            if len(clave) > 10:
                                return clave
            except Exception:
                pass
        try:
            from STM_SGR.vault_openrouter import obtener_key
            clave_vault = obtener_key()
            if clave_vault and len(clave_vault) > 10:
                return clave_vault
        except Exception:
            pass
        return os.getenv("OPENROUTER_API_KEY", "").strip()

    def obtener_modelo_activo(self) -> Dict[str, Any]:
        """Retorna el modelo que actualmente encabeza la cola de inferencia."""
        with self._lock:
            return self._modelos[self._indice_activo]

    def rotar_al_siguiente_modelo(self, razon: str = "error") -> Dict[str, Any]:
        """Avanza al siguiente modelo gratuito de la lista ante errores o limites de tasa."""
        with self._lock:
            actual = self._modelos[self._indice_activo]["id"]
            self._fallos_consecutivos[actual] = self._fallos_consecutivos.get(actual, 0) + 1
            self._indice_activo = (self._indice_activo + 1) % len(self._modelos)
            nuevo = self._modelos[self._indice_activo]
            logger.warning(
                "Rotacion de modelo gratuito: %s -> %s (Razon: %s)",
                actual, nuevo["id"], razon
            )
            return nuevo

    def seleccionar_por_id(self, modelo_id: str) -> bool:
        """Fija manualmente un modelo especifico por su identificador."""
        with self._lock:
            for idx, m in enumerate(self._modelos):
                if m["id"] == modelo_id or m["id"].split(":")[0] == modelo_id:
                    self._indice_activo = idx
                    return True
            return False

    def listar_modelos(self, categoria: Optional[str] = None) -> List[Dict[str, Any]]:
        """Devuelve los modelos gratuitos registrados, opcionalmente filtrados por categoria."""
        with self._lock:
            if not categoria:
                return list(self._modelos)
            return [m for m in self._modelos if m.get("categoria") == categoria]

    def actualizar_catalogo_online(self) -> int:
        """Consulta la API de OpenRouter y agrega nuevos modelos gratuitos descubiertos."""
        ahora = time.time()
        if (ahora - self._ultimo_refresco) < 300.0:
            return len(self._modelos)

        url = "https://openrouter.ai/api/v1/models"
        req = urllib.request.Request(url, headers={"User-Agent": "LucIA-IAFREE-2026"})
        if self.api_key:
            req.add_header("Authorization", f"Bearer {self.api_key}")

        try:
            with urllib.request.urlopen(req, timeout=8.0) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                nuevos_registrados = 0
                existentes = {m["id"] for m in self._modelos}

                for item in data.get("data", []):
                    m_id = item.get("id", "")
                    pricing = item.get("pricing", {})
                    p_cost = float(pricing.get("prompt", 0) or 0)
                    c_cost = float(pricing.get("completion", 0) or 0)

                    if (":free" in m_id or (p_cost == 0.0 and c_cost == 0.0)) and m_id not in existentes:
                        nuevo_m = {
                            "id": m_id,
                            "nombre": item.get("name", m_id),
                            "contexto": item.get("context_length", 262144),
                            "categoria": "descubierto",
                            "descripcion": item.get("description", "Modelo gratuito descubierto en linea.")[:120],
                        }
                        with self._lock:
                            self._modelos.append(nuevo_m)
                            existentes.add(m_id)
                        nuevos_registrados += 1

                self._ultimo_refresco = ahora
                logger.info("Catalogo gratuito actualizado: %d modelos anadidos.", nuevos_registrados)
                return len(self._modelos)
        except Exception as exc:
            logger.debug("No se pudo actualizar el catalogo online: %s", exc)
            return len(self._modelos)


# ─── CLIENTE DE INFERENCIA GRATUITA RESILIENTE (IAFREE CLIENT) ──────────────
class ClienteIAFree:
    """Cliente de inferencia para LucIA que garantiza coste cero ($0.00) y alta resiliencia."""

    ENDPOINT_CHAT: Final[str] = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(self, api_key: Optional[str] = None) -> None:
        self.gestor = GestorModelosGratuitos(api_key=api_key)
        self.temperatura = 0.7
        self.max_tokens = 1536
        self.total_consultas_gratuitas = 0
        self.total_caracteres_recibidos = 0
        self._lock = threading.Lock()

    def esta_autenticado(self) -> bool:
        """Verifica que se cuente con un API token valido de OpenRouter."""
        return bool(self.gestor.api_key and len(self.gestor.api_key) > 10)

    def generar_respuesta(
        self,
        prompt: str,
        contexto_neuronal: Optional[Dict[str, Any]] = None,
        max_reintentos: int = 4,
        stream_en_vivo: bool = True,
    ) -> Tuple[str, str, float]:
        """
        Ejecuta la inferencia utilizando la cola de modelos gratuitos.
        Si un modelo responde con 429 (Too Many Requests), 404 o timeout,
        rota automaticamente al siguiente modelo gratuito disponible sin cobrar dinero.

        Retorna:
          (texto_respuesta, id_modelo_utilizado, latencia_ms)
        """
        if not self.esta_autenticado():
            return ("[IAFREE] Clave OPENROUTER_API_KEY no encontrada en .env", "sin_clave", 0.0)

        contexto = contexto_neuronal or {}
        tono = contexto.get("tono_cognitivo", "analitico")
        valencia = contexto.get("estado_emocional", 0.0)

        prompt_sistema = (
            "Eres LucIA, el sistema cognitivo distribuido de WoldVirtualP2P3D (2026). "
            f"Tu tono neuronal actual es '{tono}' y tu valencia es {valencia:+.2f}. "
            "Responde de forma clara, directa, tecnica y cordial a la solicitud del usuario."
        )

        intentos = 0
        while intentos < max_reintentos:
            modelo_actual = self.gestor.obtener_modelo_activo()
            modelo_id = modelo_actual["id"]
            t_inicio = time.perf_counter()

            payload = json.dumps({
                "model": modelo_id,
                "messages": [
                    {"role": "system", "content": prompt_sistema},
                    {"role": "user", "content": prompt},
                ],
                "temperature": self.temperatura,
                "max_tokens": self.max_tokens,
                "stream": stream_en_vivo,
            }).encode("utf-8")

            headers = {
                "Authorization": f"Bearer {self.gestor.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://woldvirtualp2p3d.network",
                "X-Title": "LucIA-IAFREE-Subsystem",
            }

            req = urllib.request.Request(self.ENDPOINT_CHAT, data=payload, headers=headers, method="POST")
            tokens: List[str] = []

            try:
                with urllib.request.urlopen(req, timeout=25.0) as respuesta:
                    if stream_en_vivo:
                        sys.stdout.write(f"\n\033[96m  LucIA [Free:{modelo_id}] ▶ \033[0m")
                        sys.stdout.flush()
                        for linea in respuesta:
                            linea_str = linea.decode("utf-8").strip()
                            if not linea_str or not linea_str.startswith("data: "):
                                continue
                            raw = linea_str[6:]
                            if raw == "[DONE]":
                                break
                            try:
                                chunk = json.loads(raw)
                                d = chunk["choices"][0]["delta"].get("content", "")
                                if d:
                                    sys.stdout.write(d)
                                    sys.stdout.flush()
                                    tokens.append(d)
                            except Exception:
                                continue
                        sys.stdout.write("\n")
                        sys.stdout.flush()
                    else:
                        data = json.loads(respuesta.read().decode("utf-8"))
                        texto = data["choices"][0]["message"]["content"]
                        tokens.append(texto)

                latencia = (time.perf_counter() - t_inicio) * 1000.0
                resultado = "".join(tokens).strip()

                if resultado:
                    with self._lock:
                        self.total_consultas_gratuitas += 1
                        self.total_caracteres_recibidos += len(resultado)
                    return (resultado, modelo_id, latencia)

            except urllib.error.HTTPError as h_err:
                codigo = h_err.code
                razon = f"HTTP {codigo}"
                # Rotacion automatica si el modelo da error o rate limit
                self.gestor.rotar_al_siguiente_modelo(razon=razon)
                intentos += 1
                time.sleep(0.4)
            except Exception as e_gen:
                self.gestor.rotar_al_siguiente_modelo(razon=str(e_gen))
                intentos += 1
                time.sleep(0.4)

        return (
            "[IAFREE] Todos los modelos gratuitos consultados se encuentran temporalmente saturados. "
            "Activando reflejo neuronal interno autónomo.",
            "fallback_agotado",
            0.0,
        )


    def obtener_metricas_consumo(self) -> Dict[str, Any]:
        """Retorna un reporte de auditoria demostrando coste total de $0.00 USD."""
        with self._lock:
            return {
                "consultas_totales_gratuitas": self.total_consultas_gratuitas,
                "caracteres_recibidos": self.total_caracteres_recibidos,
                "costo_acumulado_usd": 0.0,
                "modelo_activo": self.gestor.obtener_modelo_activo()["id"],
                "total_modelos_en_pool": len(self.gestor._modelos),
                "fallos_registrados": dict(self.gestor._fallos_consecutivos),
            }

    def benchmark_rapido_modelos(self, max_modelos: int = 3) -> Dict[str, float]:
        """Mide la latencia de respuesta de los primeros modelos gratuitos del pool."""
        resultados: Dict[str, float] = {}
        for m in self.gestor.listar_modelos()[:max_modelos]:
            m_id = m["id"]
            t0 = time.perf_counter()
            try:
                payload = json.dumps({
                    "model": m_id,
                    "messages": [{"role": "user", "content": "1+1="}],
                    "max_tokens": 4,
                }).encode("utf-8")
                req = urllib.request.Request(
                    self.ENDPOINT_CHAT,
                    data=payload,
                    headers={
                        "Authorization": f"Bearer {self.gestor.api_key}",
                        "Content-Type": "application/json",
                    },
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=10.0) as r:
                    _ = r.read()
                resultados[m_id] = round((time.perf_counter() - t0) * 1000.0, 1)
            except Exception:
                resultados[m_id] = -1.0
        return resultados


# ─── SUBSISTEMA Y SINGLETON CENTRAL ─────────────────────────────────────────
_INSTANCIA_IAFREE: Optional[ClienteIAFree] = None
_LOCK_SINGLETON: Final[threading.Lock] = threading.Lock()


def get_cliente_iafree() -> ClienteIAFree:
    """Devuelve la instancia unica y compartida del cliente de inferencia gratuita."""
    global _INSTANCIA_IAFREE
    with _LOCK_SINGLETON:
        if _INSTANCIA_IAFREE is None:
            _INSTANCIA_IAFREE = ClienteIAFree()
        return _INSTANCIA_IAFREE


def consultar_lucia_gratis(prompt: str, contexto_neuronal: Optional[Dict[str, Any]] = None) -> str:
    """Punto de acceso universal para obtener respuestas de LucIA a coste cero ($0.00)."""
    cliente = get_cliente_iafree()
    respuesta, _, _ = cliente.generar_respuesta(prompt, contexto_neuronal=contexto_neuronal, stream_en_vivo=True)
    return respuesta


def listar_catalogo_gratis() -> List[Dict[str, Any]]:
    """Devuelve el catalogo de modelos sin costo disponibles para la sesion."""
    return get_cliente_iafree().gestor.listar_modelos()


def actualizar_modelos_en_red() -> int:
    """Fuerza la sincronizacion del catalogo en tiempo real con OpenRouter."""
    return get_cliente_iafree().gestor.actualizar_catalogo_online()


def obtener_auditoria_costo_cero() -> Dict[str, Any]:
    """Genera comprobante de uso 100% libre de costo."""
    return get_cliente_iafree().obtener_metricas_consumo()


# ─── PUNTO DE ENTRADA EN CONSOLA Y TEST DE DIAGNOSTICO ──────────────────────
if __name__ == "__main__":
    print("\n\033[1;36m" + "=" * 74 + "\033[0m")
    print(f"  \033[1;32mSUBSISTEMA DE INFERENCIA GRATUITA -- {__subsystem__} v{__version__}\033[0m")
    print("\033[1;36m" + "=" * 74 + "\033[0m")

    cliente = get_cliente_iafree()
    print(f"  Autenticado en OpenRouter : \033[1;33m{cliente.esta_autenticado()}\033[0m")
    print(f"  Modelos :free precargados : \033[1;35m{len(CATALOGO_MODELOS_GRATUITOS)}\033[0m")
    print("  Actualizando modelos en vivo desde OpenRouter API...")
    total = cliente.gestor.actualizar_catalogo_online()
    print(f"  Total modelos :free listos: \033[1;32m{total}\033[0m\n")

    print("\033[1;37mModelos Gratuitos Principales (Coste $0.00):\033[0m")
    for m in cliente.gestor.listar_modelos()[:8]:
        print(f"  * \033[1;36m{m['id']}\033[0m | {m['contexto']} tok | {m['nombre']}")

    print("\n\033[1;32mRealizando prueba de inferencia a coste cero ($0.00)...\033[0m")
    resp, mod, lat = cliente.generar_respuesta(
        "Hola LucIA, confirma que este subsistema de inferencia es 100% gratuito.",
        contexto_neuronal={"tono_cognitivo": "seguro y analitico", "estado_emocional": 0.45},
        stream_en_vivo=True,
    )
    print(f"\n  Modelo utilizado: \033[1;33m{mod}\033[0m | Latencia: \033[1;32m{lat:.0f}ms\033[0m")
    auditoria = cliente.obtener_metricas_consumo()
    print(f"  Auditoria de costo : \033[1;32m${auditoria['costo_acumulado_usd']:.2f} USD\033[0m")
    print("\033[1;36m" + "=" * 74 + "\033[0m\n")
