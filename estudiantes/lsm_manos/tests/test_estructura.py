"""Conteo de parámetros, inicialización y colapso sin activaciones."""

import numpy as np
import pytest

from redneuronal.perdida import EntropiaCruzada, PerdidaCuadratica
from redneuronal.red import gradiente, predecir, red_inicial


@pytest.mark.parametrize("anchos", [(2, 3, 2, 1), (60, 128, 64, 21), (60, 21), (960, 64, 6)])
def test_conteo_de_parametros(anchos):
    activaciones = ["relu"] * (len(anchos) - 2) + ["identidad"]
    red = red_inicial(anchos, activaciones)
    esperado = sum(anchos[l] * (anchos[l - 1] + 1) for l in range(1, len(anchos)))
    assert red.num_parametros() == esperado
    assert red.anchos == tuple(anchos)


def test_modelo_principal_tiene_17429_parametros():
    red = red_inicial((60, 128, 64, 21), ("relu", "relu", "softmax"))
    assert red.num_parametros() == 17_429


def test_inicializacion_he():
    red = red_inicial((400, 300, 10), ("relu", "identidad"), semilla=0)
    assert np.std(red.W[1]) == pytest.approx(np.sqrt(2 / 400), rel=0.02)
    assert np.all(red.b[1] == 0) and np.all(red.b[2] == 0)


def test_colapso_sin_activacion():
    """Observación 10.1.2.3: sin activaciones, la red es una capa afín W′x + b′."""
    red = red_inicial((5, 7, 4, 3), ("identidad",) * 3, semilla=5)
    for l in (1, 2, 3):
        red.b[l] = np.random.default_rng(l).standard_normal(red.b[l].shape)
    W_prima = red.W[3] @ red.W[2] @ red.W[1]
    b_prima = red.W[3] @ (red.W[2] @ red.b[1] + red.b[2]) + red.b[3]
    X = np.random.default_rng(9).standard_normal((5, 10))
    np.testing.assert_allclose(predecir(red, X), W_prima @ X + b_prima[:, None], atol=1e-12)


def test_softmax_suma_uno_y_es_estable():
    red = red_inicial((3, 4, 5), ("relu", "softmax"))
    X = np.array([[1e3, -1e3], [0.0, 5.0], [2.0, 1.0]])
    Y_gorro = predecir(red, X)
    assert np.all(np.isfinite(Y_gorro))
    np.testing.assert_allclose(Y_gorro.sum(axis=0), 1.0)


def test_softmax_exige_entropia_cruzada():
    red = red_inicial((3, 4, 5), ("relu", "softmax"))
    with pytest.raises(ValueError):
        gradiente(red, np.ones(3), np.eye(5)[0], PerdidaCuadratica())
    gradiente(red, np.ones(3), np.eye(5)[0], EntropiaCruzada())
