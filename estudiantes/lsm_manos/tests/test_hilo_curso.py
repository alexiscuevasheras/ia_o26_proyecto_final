"""El hilo 2–3–2–1 de las páginas de la unidad 10, cifra por cifra.

Las cifras del curso vienen redondeadas a 3 decimales: tolerancia 1e-3.
"""

import numpy as np
import pytest

from redneuronal.optimizador import paso
from redneuronal.perdida import Castigo, PerdidaCuadratica
from redneuronal.red import Red, adelante, gradiente, omega, perdida_promedio

TOL = 1e-3
CUADRATICA = PerdidaCuadratica()
RIDGE = Castigo(1.0)  # μ = 1 solo para leer Ω(θ) sin escalar


def red_del_hilo():
    red = Red(
        W=[None,
           np.array([[0.5, -0.2], [-0.4, 0.1], [0.3, 0.2]]),
           np.array([[0.4, -0.3, 0.5], [0.2, 0.6, -0.1]]),
           np.array([[0.7, -0.5]])],
        b=[None, np.array([0.1, 0.0, -0.1]), np.array([0.1, 0.2]), np.array([0.3])],
        g=[None, "relu", "relu", "identidad"],
    )
    x = np.array([1.0, 2.0])
    y = np.array([1.0])
    return red, x, y


def cerca(obtenido, esperado, tol=TOL):
    np.testing.assert_allclose(np.ravel(obtenido), np.ravel(esperado), atol=tol, rtol=0)


def test_adelante():
    red, x, _ = red_del_hilo()
    Z, A = adelante(red, x)
    cerca(Z[1], [0.2, -0.2, 0.6])
    cerca(A[1], [0.2, 0.0, 0.6])
    cerca(Z[2], [0.48, 0.18])
    cerca(A[2], [0.48, 0.18])
    cerca(A[3], [0.546])


def test_perdida_y_omega():
    red, x, y = red_del_hilo()
    cerca(perdida_promedio(red, x, y, CUADRATICA), 0.206)
    normas = [RIDGE.omega(red.W[l], l) for l in (1, 2, 3)]
    cerca(normas, [0.59, 0.91, 0.74])
    cerca(omega(red, RIDGE), 2.24)


def test_parametros():
    red, _, _ = red_del_hilo()
    assert sum(red.W[l].size for l in (1, 2, 3)) == 14
    assert sum(red.b[l].size for l in (1, 2, 3)) == 6
    assert red.num_parametros() == 20


def test_retropropagacion():
    red, x, y = red_del_hilo()
    grad_W, grad_b, deltas = gradiente(red, x, y, CUADRATICA)
    cerca(deltas[3], [-0.908])
    cerca(deltas[2], [-0.636, 0.454])
    cerca(deltas[1], [-0.163, 0.0, -0.363])
    assert deltas[1][1, 0] == 0.0, "la neurona 2 de la capa 1 está apagada"
    assert np.all(grad_W[1][1] == 0.0), "su fila del gradiente es cero"
    cerca(grad_W[3], [-0.436, -0.163])
    cerca(grad_b[3], [-0.908])


def test_un_paso_sin_mu():
    red, x, y = red_del_hilo()
    nueva = paso(red, x, y, eta=0.1, perdida=CUADRATICA)
    cerca(adelante(nueva, x)[1][-1], [0.879])
    cerca(perdida_promedio(nueva, x, y, CUADRATICA), 0.015)
    cerca(nueva.W[3][0, 0], 0.744)


def test_un_paso_con_mu():
    red, x, y = red_del_hilo()
    nueva = paso(red, x, y, eta=0.1, perdida=CUADRATICA, castigo=Castigo(0.01))
    cerca(nueva.W[3][0, 0], 0.742)
    cerca(nueva.b[3][0], 0.391)


def test_paso_no_modifica_la_red_original():
    red, x, y = red_del_hilo()
    paso(red, x, y, eta=0.1, perdida=CUADRATICA)
    assert red.W[3][0, 0] == 0.7


@pytest.mark.parametrize("peso, ridge, lasso", [(0.7, 0.0014, 0.001), (0.001, 0.000002, 0.001)])
def test_ejercicio_lasso_cuanto_encoge(peso, ridge, lasso):
    """Ejercicio 10.1.5.4: con η = 0.1 y μ = 0.01, sin contar ∇L_B."""
    eta, W = 0.1, np.array([[peso]])
    assert eta * Castigo(0.01, "ridge").gradiente(W, 1)[0, 0] == pytest.approx(ridge)
    assert eta * Castigo(0.01, "lasso").gradiente(W, 1)[0, 0] == pytest.approx(lasso)
    assert Castigo(0.01, "lasso").gradiente(np.zeros((1, 1)), 1)[0, 0] == 0.0
