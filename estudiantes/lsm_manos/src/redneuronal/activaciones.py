"""Activaciones g(l) y sus derivadas g(l)'.

Cada g se aplica componente a componente, salvo softmax, que mezcla las
componentes de una columna y por eso solo se permite en la capa de salida
junto con la entropía cruzada (ver `perdida.py`).
"""

import numpy as np


def relu(z):
    return np.maximum(z, 0.0)


def relu_prima(z):
    # Convención del curso: g'(0) = 0.
    return (z > 0).astype(z.dtype)


def sigmoide(z):
    return 1.0 / (1.0 + np.exp(-z))


def sigmoide_prima(z):
    s = sigmoide(z)
    return s * (1.0 - s)


def identidad(z):
    return z


def identidad_prima(z):
    return np.ones_like(z)


def softmax(z):
    """Softmax por columna. Restar el máximo no cambia el resultado y evita
    que exp se desborde."""
    e = np.exp(z - z.max(axis=0, keepdims=True))
    return e / e.sum(axis=0, keepdims=True)


ACTIVACIONES = {
    "relu": (relu, relu_prima),
    "sigmoide": (sigmoide, sigmoide_prima),
    "identidad": (identidad, identidad_prima),
    "softmax": (softmax, None),
}
