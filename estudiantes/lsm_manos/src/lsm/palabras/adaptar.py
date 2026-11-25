"""Adaptación de la capa de salida de la GRU a una persona (igual que web/adaptar.js).

Las capas recurrentes quedan fijas; con el resumen [h_T; h̄] fijo, reentrenar
V y c es una regresión logística multinomial (convexa). Mismos valores que en
las letras (E8): lote completo, η = 0.05, 100 pasos, sin castigo.
"""

import numpy as np

from redneuronal.recurrente import adelante

from ..adaptar import ETA, PASOS


def resumenes(modelo, X):
    """[h_T; h̄] de cada seña: (2n, m)."""
    Xe = np.transpose(modelo.estandar(X), (1, 2, 0))
    return adelante(modelo.red, Xe)[1][5]


def adaptar_salida(modelo, X, y, eta=ETA, pasos=PASOS):
    """Devuelve una copia del modelo con V y c reentrenados en (X, y)."""
    R = resumenes(modelo, X)
    red = modelo.red.copia()
    V, c = red.p["V"], red.p["c"]
    Y = np.eye(red.K)[:, y]
    m = len(y)
    for _ in range(pasos):
        Z = V @ R + c[:, None]
        Z -= Z.max(axis=0)
        P = np.exp(Z) / np.exp(Z).sum(axis=0)
        delta = (P - Y) / m  # softmax con entropía cruzada: ŷ − y
        V -= eta * delta @ R.T
        c -= eta * delta.sum(axis=1)
    nuevo = type(modelo)(modelo.config)
    nuevo.red, nuevo.estandar = red, modelo.estandar
    return nuevo
