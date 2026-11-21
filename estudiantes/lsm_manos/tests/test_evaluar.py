"""Las métricas en NumPy coinciden con las de scikit-learn."""

import numpy as np
import pytest

from lsm.evaluar import metricas
from lsm.features import vector_entrada
from test_features import a_mediapipe, mano_en_pixeles

sklearn_metrics = pytest.importorskip("sklearn.metrics")


@pytest.mark.parametrize("semilla", [0, 1, 2])
def test_metricas_como_scikit_learn(semilla):
    rng = np.random.default_rng(semilla)
    K = 21
    y = rng.integers(0, K, 500)
    y_gorro = np.where(rng.random(500) < 0.6, y, rng.integers(0, K, 500))
    m = metricas(y, y_gorro, K)
    assert m["f1_macro"] == pytest.approx(sklearn_metrics.f1_score(y, y_gorro, average="macro", labels=range(K)))
    assert m["mcc"] == pytest.approx(sklearn_metrics.matthews_corrcoef(y, y_gorro))
    assert m["exactitud"] == pytest.approx(sklearn_metrics.accuracy_score(y, y_gorro))
    assert np.array_equal(m["confusion"], sklearn_metrics.confusion_matrix(y, y_gorro, labels=range(K)))


def test_todo_correcto_y_constante():
    y = np.array([0, 1, 2, 2])
    assert metricas(y, y, 3)["mcc"] == pytest.approx(1.0)
    assert metricas(y, np.full(4, 2), 3)["mcc"] == pytest.approx(0.0)


def test_proporcion_distinta_por_mano():
    """Con una proporción por fila, cada mano da lo mismo que procesada sola."""
    manos = np.stack([a_mediapipe(mano_en_pixeles(s), 640, 480) for s in range(3)])
    aspectos = np.array([4 / 3, 1.0, 16 / 9])
    juntas = vector_entrada(manos, False, aspecto=aspectos)
    for i in range(3):
        np.testing.assert_allclose(juntas[i], vector_entrada(manos[i], False, aspecto=aspectos[i]))
