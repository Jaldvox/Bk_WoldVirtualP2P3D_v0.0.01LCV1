"""Suite de tests minima para RFC/src — 7 tests."""
from __future__ import annotations

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC_DIR))


def test_gestor_errores_singleton():
    from SBSTM.gestor_errores import obtener_gestor_errores

    g1 = obtener_gestor_errores()
    g2 = obtener_gestor_errores()
    assert g1 is g2
    g1.limpiar()
    assert g1.resumen_errores()["total"] == 0


def test_gestor_errores_captura():
    from SBSTM.gestor_errores import obtener_gestor_errores

    g = obtener_gestor_errores()
    g.limpiar()
    try:
        raise ValueError("test error")
    except ValueError as e:
        entrada = g.capturar(e, contexto="test")
    assert entrada["tipo"] == "ValueError"
    assert entrada["mensaje"] == "test error"
    assert g.resumen_errores()["total"] == 1
    g.limpiar()


def test_blockchain_servidor():
    from STM_BKCH.BKSVCB import CelebroBlockchain

    bks = CelebroBlockchain()
    valida, err = bks.validar_cadena()
    assert valida is True
    assert err is None
    assert len(bks.cadena) >= 1


def test_vault_openrouter():
    from STM_SGR.vault_openrouter import cifrar_key, descifrar_blob, enmascarar

    key = "sk-test-12345-abcdef"
    blob = cifrar_key(key)
    assert blob != key
    assert descifrar_blob(blob) == key
    assert enmascarar(key) != key
    assert enmascarar("") == "****"


def test_ipfs_manager():
    from STM_BKCH.compat_local import get_ipfs_manager

    mgr = get_ipfs_manager()
    res = mgr.almacenar_pesos(datos={"test": True})
    assert res["ok"] is True
    assert res["cid"].startswith("QmLocal")


def test_compat_local_conversor():
    from STM_BKCH.compat_local import get_conversor_pesos

    conv = get_conversor_pesos()
    assert len(conv.neuronas) == 50
    info = conv.procesar_consulta_a_pesos("test prompt")
    assert "tono_cognitivo" in info
    assert "estado_emocional" in info
    assert "norma_delta_aplicada" in info


def test_snsbstnprb_importar_nucleo():
    from SBSTM.SNSBSTNPRB import _importar_nucleo

    get_bks, iniciar_daemon, get_conv, get_gpv = _importar_nucleo()
    assert callable(get_bks)
    assert callable(iniciar_daemon)
    assert callable(get_conv)


def test_autodiagnostico():
    from SBSTM.autodiagnostico import AutodiagnosticoLucIA, es_peticion_diagnostico, ejecutar_diagnostico

    assert es_peticion_diagnostico("puedes hacer otro diagnostico profundo sobre tu propio sistema?")
    assert es_peticion_diagnostico("revisa tu sistema para ver que funciona y que no")
    assert not es_peticion_diagnostico("cuanto es dos mas dos?")

    diag = AutodiagnosticoLucIA().ejecutar_diagnostico_completo()
    assert diag["blockchain"]["estado"] == "OPERATIVO"
    assert diag["red_neuronal"]["estado"] == "OPERATIVO"
    assert diag["integridad_codigo"]["limite_respetado"] is True
    resumen = ejecutar_diagnostico()
    assert "REPORTE DE AUTODIAGNÓSTICO" in resumen
