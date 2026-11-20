"""Descenso de gradiente por minibatch con decaimiento de pesos.

W(l) ← W(l) − η (∇W(l) L_B + ∇W(l) μΩ),   b(l) ← b(l) − η ∇b(l) L_B.
Con ridge, el término 2μW(l) equivale a encoger cada peso por (1 − 2ημ).
"""

import numpy as np

from .perdida import SIN_CASTIGO
from .red import columnas, gradiente, objetivo


def paso(red, X, Y, eta, perdida, castigo=SIN_CASTIGO):
    """Un paso de descenso sobre el lote (X, Y). Devuelve una red nueva."""
    grad_W, grad_b, _ = gradiente(red, X, Y, perdida, castigo)
    nueva = red.copia()
    for l in range(1, red.L + 1):
        nueva.W[l] -= eta * grad_W[l]
        nueva.b[l] -= eta * grad_b[l]
    return nueva


def entrenar(red, X, Y, eta, perdida, castigo=SIN_CASTIGO, epocas=10, lote=32,
             semilla=0, al_terminar_epoca=None):
    """Una época = una pasada barajada por todos los datos, un paso por lote.

    `al_terminar_epoca(epoca, red)` permite registrar curvas (pérdida de
    entrenamiento y de validación) sin que este módulo sepa de validación.
    Devuelve la red final y la historia de L(θ) + μΩ en entrenamiento.
    """
    X, Y = columnas(X), columnas(Y)
    rng = np.random.default_rng(semilla)
    historia = [objetivo(red, X, Y, perdida, castigo)]
    for epoca in range(1, epocas + 1):
        orden = rng.permutation(X.shape[1])
        for inicio in range(0, len(orden), lote):
            k = orden[inicio:inicio + lote]
            red = paso(red, X[:, k], Y[:, k], eta, perdida, castigo)
        historia.append(objetivo(red, X, Y, perdida, castigo))
        if al_terminar_epoca is not None:
            al_terminar_epoca(epoca, red)
    return red, historia
