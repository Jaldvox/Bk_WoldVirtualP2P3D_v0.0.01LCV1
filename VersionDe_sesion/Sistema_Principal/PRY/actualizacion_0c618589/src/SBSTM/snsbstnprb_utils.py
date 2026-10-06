"""Utilidades extraidas de SNSBSTNPRB para mantener limite de 450 lineas."""
from __future__ import annotations

from typing import Any, Dict


def mini_pesos(turno: int, gsnr: float, deriva: float,
               neuronas: int, delta: float, tono: str) -> str:
    """Genera linea compacta de telemetria sinaptica para terminal."""
    BL = 12
    ratio = min(1.0, delta / max(0.01, deriva + 1e-9))
    llenos = int(round(ratio * BL))
    barra = "\033[96m" + "\u2588" * llenos + "\033[2m" + "\u2591" * (BL - llenos) + "\033[0m"
    ec = "\033[92m" if gsnr > 1.5 else ("\033[93m" if gsnr > 0.8 else "\033[91m")
    return (
        f"  \033[2m[T#{turno:04d}]\033[0m "
        f"GSNR:{ec}{gsnr:.2f}\033[0m "
        f"Delta:\033[95m{delta:.5f}\033[0m "
        f"Deriva:\033[93m{deriva:.4f}\033[0m "
        f"N:\033[97m{neuronas}\033[0m "
        f"{barra} \033[94m{tono[:22]}\033[0m"
    )


def mostrar_estado(bks: Any, conversor: Any, gestor_pv: Any,
                   turno: int, modelo: str = "") -> None:
    """Panel de estado compacto del sistema en tiempo real."""
    valida, err = bks.validar_cadena()
    tel = gestor_pv.obtener_telemetria_resumen() if gestor_pv else {}
    print("\033[96m" + "\n" + "-" * 60 + "\033[0m")
    print("\033[1m  ESTADO DEL SISTEMA\033[0m")
    print(f"  Bloques: \033[93m{len(bks.cadena)}\033[0m | "
          f"Cadena: \033[92mVALIDA\033[0m" if valida else f"\033[91mERROR {err}\033[0m")
    print(f"  Txs pendientes : \033[97m{len(bks.transacciones_pendientes)}\033[0m")
    print(f"  Neuronas activas: \033[95m{len(conversor.neuronas)}\033[0m")
    print(f"  Turnos sesion  : \033[96m{tel.get('total_turnos', turno)}\033[0m")
    _deriva = f"{tel.get('deriva_total', 0.0):.6f}"
    _gsnr = f"{tel.get('gsnr_medio', 0.0):.3f}"
    print(f"  Deriva total   : \033[93m{_deriva}\033[0m")
    print(f"  GSNR medio     : \033[92m{_gsnr}\033[0m")
    if modelo:
        print(f"  Modelo activo  : \033[96m{modelo}\033[0m")
    print("\033[96m" + "-" * 60 + "\033[0m")


def reflejo_neuronal(conversor: Any, prompt: str) -> str:
    """Respuesta interna de la red neuronal cuando Ollama no esta disponible."""
    try:
        info = conversor.procesar_consulta_a_pesos(prompt)
        return (
            f"[Reflejo Neuronal] Tono: {info.get('tono_cognitivo','reflexivo')} | "
            f"Valencia: {info.get('estado_emocional', 0.0):+.3f} | "
            f"Delta-Sinaptico: {info.get('norma_delta_aplicada', 0.0):.5f}"
        )
    except Exception:
        return "[Reflejo Neuronal] Estimulo procesado en modo offline."


def resolver_modelo(nombre: str, disponibles: list) -> str:
    """Devuelve el nombre si existe entre los disponibles, o cadena vacia."""
    nombre = (nombre or "").strip()
    if nombre and any(nombre in m for m in (disponibles or [])):
        return nombre
    return ""


def gestionar_comando(sesion: Any, prompt: str) -> str:
    """Despacha comandos de sesion. Devuelve el prompt a procesar o "".

    Comandos: estado | modelo <nombre> | saber <tema> | escuchar.
    """
    bajo = prompt.lower()
    if bajo in ("estado", "status"):
        mostrar_estado(sesion.bks, sesion.conversor, sesion.gestor_pv,
                       sesion._turno, sesion.modelo)
        return ""
    if bajo.startswith("modelo "):
        from SBSTM.SNSBSTNPRB import _listar_modelos_ollama  # noqa: E402
        nuevo = resolver_modelo(prompt[7:].strip(), _listar_modelos_ollama())
        if nuevo:
            sesion.modelo = nuevo
            print(f"  Modelo cambiado a: {nuevo}")
        else:
            print(f"  '{prompt[7:].strip()}' no disponible.")
        return ""
    if bajo.startswith("saber "):
        try:
            from STM_IA.conocimiento import contexto_para_prompt  # noqa: E402
            print(contexto_para_prompt(prompt[6:].strip()) or "  Sin resultados.")
        except Exception as ex:
            print(f"  [conocimiento] {ex}")
        return ""
    if bajo in ("escuchar", "voz", "hablar"):
        try:
            from SBSTM.escucha import escuchar_o_pedir  # noqa: E402
            return escuchar_o_pedir()
        except Exception:
            return ""
    return prompt


def cerrar_sesion(bks: Any, conversor: Any, gestor_pv: Any, sesion_id: str, turno: int) -> None:
    """Hook atexit: mina pendientes finales, checkpoint PSNRL, sube a IPFS."""
    if not bks:
        return
    print("\033[96m" + "\n" + "=" * 74 + "\033[0m")
    print("\033[1m  CONSOLIDANDO SESION NEURONAL...\033[0m")

    blk = bks.minar_transacciones_pendientes()
    if blk:
        print(f"  Bloque final #{blk.indice}: "
              f"\033[93m{blk.hash_bloque[:32]}...\033[0m")

    if gestor_pv and conversor:
        try:
            archs = gestor_pv.checkpoint_y_registrar(
                conversor=conversor, etiqueta="snsbstnprb_cierre",
            )
            print(f"  PSNRL: \033[92m{len(archs)}\033[0m archivos guardados")
        except Exception as ex:
            print(f"  \033[93m[PSNRL] {ex}\033[0m")

    try:
        res = bks.cerrar_sesion_y_subir_ipfs()
        cid_l = res.get("cid_ledger") or "--"
        n_psnrl = len(res.get("cids_psnrl", []))
        cid_str = cid_l[:36] + "..." if len(cid_l) > 36 else cid_l
        print(f"  IPFS Ledger CID: \033[96m{cid_str}\033[0m")
        print(f"  IPFS PSNRL     : \033[92m{n_psnrl}\033[0m archivos subidos")
    except Exception as ex:
        print(f"  \033[93m[IPFS] {ex}\033[0m")

    valida, err = bks.validar_cadena()
    print(f"  Cadena final: "
          f"\033[92mINTEGRA\033[0m" if valida else f"\033[91m{err}\033[0m")
    print(f"  Sesion: \033[97m{sesion_id}\033[0m | "
          f"Turnos: \033[96m{turno}\033[0m")
    print("\033[96m" + "=" * 74 + "\033[0m\n")


def postprocesar_turno(sesion: Any, prompt: str, respuesta: str,
                       info_neural: dict, emo: dict, dt: float,
                       cid: str) -> None:
    """Post-procesado del turno: memoria, perfil, voz, dashboard, rubrica."""
    tono = str(info_neural.get("tono_cognitivo", ""))
    try:
        val = float(info_neural.get("estado_emocional", 0.0))
    except (TypeError, ValueError):
        val = 0.0

    try:
        from SBSTM.memoria_sesion import get_memoria  # noqa: E402
        get_memoria(sesion._sesion_id).registrar_turno(
            prompt=prompt, respuesta=respuesta, tono=tono, valencia=val)
    except Exception:
        pass

    try:
        from SBSTM.perfil import get_perfil  # noqa: E402
        from STM_IA.conocimiento import palabras_clave  # noqa: E402
        get_perfil().aprender_turno(
            tono=tono, valencia=val, modelo=sesion.modelo,
            palabras_clave=palabras_clave(prompt))
    except Exception:
        pass

    try:
        from SBSTM.emocion import modular_prosodia  # noqa: E402
        from SBSTM.accesores import reproducir_voz_lucia  # noqa: E402
        emo_voz, _ = modular_prosodia(float(emo.get("valencia", 0.0)), val)
        reproducir_voz_lucia(respuesta, emocion=emo_voz)
    except Exception:
        pass

    try:
        res_pv = sesion.gestor_pv.inyectar_turno_en_vivo(
            conversor=sesion.conversor, pregunta=prompt,
            respuesta=respuesta, info_pesos=info_neural,
            mostrar_en_terminal=False)
        print(mini_pesos(
            sesion._turno, float(res_pv.get("gsnr_global", 1.0)),
            float(res_pv.get("deriva_sesion", 0.0)),
            len(sesion.conversor.neuronas),
            float(res_pv.get("delta_aplicada", 0.0)),
            str(info_neural.get("tono_cognitivo", "analitico"))))
    except Exception:
        pass

    print(f"  {dt:.0f}ms | CID: {cid[:22] if cid != '--' else '--'} | "
          f"Txs pendientes: {len(sesion.bks.transacciones_pendientes)}")

    try:
        from SBSTM.rubrica import evaluar_calidad  # noqa: E402
        cal = evaluar_calidad(respuesta, tono)
        print(f"  Calidad: {cal['puntuacion']:.2f} ({cal['veredicto']})")
    except Exception:
        pass
