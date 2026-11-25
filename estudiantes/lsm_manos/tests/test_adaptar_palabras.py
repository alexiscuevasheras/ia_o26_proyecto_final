"""Adaptar la salida de la GRU: aprende lo confirmado y no toca las capas recurrentes."""

import numpy as np

from lsm.palabras.adaptar import adaptar_salida
from lsm.palabras.modelos import ConfigGRU, ModeloGRU


def test_adaptar_sube_la_glosa_confirmada_y_deja_fija_la_gru():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 6, 126))
    X[:, :, 62] = X[:, :, 125] = 1
    y = rng.integers(0, 4, 40)
    base = ModeloGRU(ConfigGRU(oculta=8, epocas=2, aumentar=False)).ajustar(X, y, 4)
    nuevos = np.full(10, 3)
    antes = base.probabilidades(X[:10])[:, 3].mean()
    adaptado = adaptar_salida(base, X[:10], nuevos)
    assert adaptado.probabilidades(X[:10])[:, 3].mean() > antes + 0.1
    for k in base.red.p:
        if k not in ("V", "c"):
            np.testing.assert_array_equal(adaptado.red.p[k], base.red.p[k])
    assert not np.allclose(adaptado.red.p["V"], base.red.p["V"])
