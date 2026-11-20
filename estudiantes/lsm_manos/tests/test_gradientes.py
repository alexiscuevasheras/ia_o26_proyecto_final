"""Retropropagación contra diferencias finitas centrales (h = 1e-6, float64)."""

import numpy as np
import pytest

from redneuronal.gradcheck import errores_por_bloque
from redneuronal.perdida import Castigo, EntropiaCruzada, PerdidaCuadratica
from redneuronal.red import red_inicial
from test_hilo_curso import red_del_hilo


class PerdidaSinElDos(PerdidaCuadratica):
    """El descuido del ejercicio 10.1.4.5: δ(L) = ŷ − y, sin el factor 2."""

    def gradiente_salida(self, Y_gorro, Y):
        return Y_gorro - Y


# El curso reporta < 1e-10 para el hilo. Con lasso, la suma de |w| añade
# redondeo y el peor bloque queda alrededor de 1e-10; por eso 1e-9 ahí.
@pytest.mark.parametrize("castigo, tope", [
    (Castigo(0.0), 1e-10),
    (Castigo(0.01), 1e-10),
    (Castigo(0.01, "lasso"), 1e-9),
])
def test_hilo_coincide(castigo, tope):
    red, x, y = red_del_hilo()
    errores = errores_por_bloque(red, x, y, PerdidaCuadratica(), castigo)
    assert max(errores.values()) < tope, errores


def test_hilo_sin_el_dos_da_un_tercio():
    red, x, y = red_del_hilo()
    errores = errores_por_bloque(red, x, y, PerdidaSinElDos())
    for bloque, e in errores.items():
        if bloque != "b1" or e > 0:
            assert e == pytest.approx(1 / 3, abs=1e-8), (bloque, e)


def lote_aleatorio(anchos, m, semilla):
    rng = np.random.default_rng(semilla)
    X = rng.standard_normal((anchos[0], m))
    clases = rng.integers(0, anchos[-1], m)
    Y = np.eye(anchos[-1])[:, clases]  # one-hot en columnas
    return X, Y


def test_red_del_tamano_real_softmax():
    """60–128–64–21 con softmax y entropía cruzada (la extensión): δ(L) = ŷ − y."""
    anchos = (60, 128, 64, 21)
    red = red_inicial(anchos, ("relu", "relu", "softmax"), semilla=1)
    X, Y = lote_aleatorio(anchos, m=4, semilla=2)
    errores = errores_por_bloque(red, X, Y, EntropiaCruzada(), Castigo(1e-3))
    assert max(errores.values()) < 1e-6, errores


def test_red_mediana_cuadratica_sigmoide():
    """La versión «del curso» para clasificar: salida sigmoide y pérdida cuadrática."""
    anchos = (12, 16, 8, 5)
    red = red_inicial(anchos, ("relu", "relu", "sigmoide"), semilla=3)
    X, Y = lote_aleatorio(anchos, m=6, semilla=4)
    errores = errores_por_bloque(red, X, Y, PerdidaCuadratica(), Castigo(1e-2, "lasso", capas={1}))
    assert max(errores.values()) < 1e-6, errores
