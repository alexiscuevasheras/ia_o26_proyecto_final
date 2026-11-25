"""Limpieza de trayectorias del avatar (src/lsm/palabras/animaciones.py)."""

import numpy as np

from lsm.palabras.animaciones import (CABEZA_CENTRO, CORONILLA, adelante_para_no_atravesar, con_rampas_al_reposo,
                                      en_espacio_de_senas, reposo, sin_rachas_cortas, sin_saltos)


def test_quita_detecciones_sueltas():
    presente = np.array([0, 1, 0, 1, 1, 1, 0, 1], bool)
    assert sin_rachas_cortas(presente).tolist() == [False, False, False, True, True, True, False, False]


def test_quita_un_salto_aislado():
    trayectoria = np.zeros((8, 3))
    trayectoria[4] = [0.5, 0.5, 0.0]
    assert sin_saltos(trayectoria, np.ones(8, bool)).tolist() == [True] * 4 + [False] + [True] * 3


def test_sale_del_reposo_y_vuelve_en_vez_de_congelarse():
    presente = np.zeros(20, bool)
    presente[8:12] = True
    arriba = np.tile([0.0, 0.0, 0.3], (20, 1))
    t = con_rampas_al_reposo(arriba, presente, 1)
    assert np.allclose(t[0], reposo(1)) and np.allclose(t[-1], reposo(1))
    assert np.allclose(t[8:12], arriba[8:12])


def test_la_muneca_no_pasa_de_la_coronilla():
    assert en_espacio_de_senas(np.array([0.0, 0.5, 0.3]))[1] == CORONILLA


def test_empuja_la_mano_fuera_de_la_cabeza_y_del_torso():
    assert adelante_para_no_atravesar(np.array([CABEZA_CENTRO])) > 0.1
    assert adelante_para_no_atravesar(np.array([[0.0, -0.2, 0.05]])) > 0
    assert adelante_para_no_atravesar(np.array([[0.0, 0.22, 0.3]])) == 0
