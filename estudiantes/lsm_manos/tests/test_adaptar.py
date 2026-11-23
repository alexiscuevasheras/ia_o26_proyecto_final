"""Adaptar solo la última capa: mismas cuentas que la red completa, capas ocultas intactas."""

import numpy as np

from lsm.adaptar import adaptar_ultima_capa, ejemplos_por_letra, representacion_oculta
from lsm.entrenar import K, una_caliente
from redneuronal.perdida import EntropiaCruzada
from redneuronal.red import Red, gradiente, perdida_promedio, red_inicial


def red_y_datos(semilla=0, m=63):
    rng = np.random.default_rng(semilla)
    red = red_inicial((60, 32, 16, K), ("relu", "relu", "softmax"), semilla=semilla)
    return red, rng.standard_normal((m, 60)), rng.integers(0, K, m)


def test_gradiente_de_la_capa_sola_es_el_bloque_de_la_red():
    red, X, letras = red_y_datos()
    Y = una_caliente(letras)
    grad_W, grad_b, _ = gradiente(red, X.T, Y, EntropiaCruzada())
    salida = Red(W=[None, red.W[3]], b=[None, red.b[3]], g=[None, "softmax"])
    g_W, g_b, _ = gradiente(salida, representacion_oculta(red, X), Y, EntropiaCruzada())
    np.testing.assert_allclose(g_W[1], grad_W[3], atol=1e-12)
    np.testing.assert_allclose(g_b[1], grad_b[3], atol=1e-12)


def test_adaptar_baja_la_perdida_y_no_toca_capas_ocultas():
    red, X, letras = red_y_datos()
    nueva = adaptar_ultima_capa(red, X, letras)
    Y = una_caliente(letras)
    assert perdida_promedio(nueva, X.T, Y, EntropiaCruzada()) < perdida_promedio(red, X.T, Y, EntropiaCruzada())
    for l in (1, 2):
        np.testing.assert_array_equal(nueva.W[l], red.W[l])
        np.testing.assert_array_equal(nueva.b[l], red.b[l])
    assert not np.allclose(nueva.W[3], red.W[3])


def test_ejemplos_por_letra():
    letras = np.repeat(np.arange(4), [5, 1, 3, 2])
    idx = ejemplos_por_letra(letras, 2, np.random.default_rng(0))
    assert np.bincount(letras[idx]).tolist() == [2, 1, 2, 2]
    assert len(set(idx)) == len(idx)
