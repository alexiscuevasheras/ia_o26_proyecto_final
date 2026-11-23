"""Adaptar la red a una persona nueva con pocos ejemplos suyos.

Solo se reentrena la última capa: W(L) y b(L). Las capas ocultas quedan
fijas, así que a(L−1) de cada ejemplo se calcula una vez y el problema es una
regresión softmax con entrada a(L−1), que empieza desde los pesos ya
entrenados. Es lo bastante barato para hacerlo en el navegador.

Los hiperparámetros se fijaron antes de ver resultados: descenso de gradiente
con todo el lote, η = 0.05, 100 pasos, sin castigo.
"""

import numpy as np

from redneuronal.optimizador import entrenar
from redneuronal.perdida import EntropiaCruzada
from redneuronal.red import Red, adelante

from .entrenar import una_caliente

ETA = 0.05
PASOS = 100


def representacion_oculta(red, X):
    """a(L−1) de cada fila de X, en columnas."""
    return adelante(red, X.T)[1][red.L - 1]


def adaptar_ultima_capa(red, X, letras, eta=ETA, pasos=PASOS):
    """Devuelve una red nueva con W(L) y b(L) ajustados a (X, letras)."""
    L = red.L
    salida = Red(W=[None, red.W[L].copy()], b=[None, red.b[L].copy()], g=[None, red.g[L]])
    H = representacion_oculta(red, X)
    salida, _ = entrenar(salida, H, una_caliente(letras), eta, EntropiaCruzada(),
                         epocas=pasos, lote=H.shape[1])
    nueva = red.copia()
    nueva.W[L], nueva.b[L] = salida.W[1], salida.b[1]
    return nueva


def ejemplos_por_letra(letras, k, rng):
    """Índices de k ejemplos al azar de cada letra (o todos, si hay menos)."""
    elegidos = [rng.choice(np.flatnonzero(letras == c), min(k, np.sum(letras == c)), replace=False)
                for c in np.unique(letras)]
    return np.concatenate(elegidos)
