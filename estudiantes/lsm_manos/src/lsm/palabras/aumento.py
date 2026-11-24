"""Aumento de datos para secuencias: variaciones que una persona nueva o
una cámara distinta producirían sin cambiar la seña.

- Tiempo: unos hacen la seña más rápido al inicio o al final, así que se
  deforma el eje del tiempo con una curva monótona al azar.
- Posición: el cuerpo de otra persona, o la cámara a otra altura, mueve y
  escala la muñeca en el marco de los hombros (±10 %).
- Forma: un giro pequeño de la mano en el plano de la imagen (±10°) más un
  ruido chico en cada punto, como el error de MediaPipe.

Una mano ausente (presencia 0) se queda en ceros.
"""

import numpy as np

from .secuencias import POR_MANO

GIRO_MAXIMO = np.deg2rad(10)


def deformar_tiempo(X, rng):
    """t' = t + a·sen(πt): los extremos quedan fijos y el medio se adelanta o atrasa."""
    T = X.shape[0]
    t = np.linspace(0, 1, T)
    nuevo = np.clip(t + rng.uniform(-0.15, 0.15) * np.sin(np.pi * t), 0, 1)
    return np.stack([np.interp(nuevo, t, X[:, j]) for j in range(X.shape[1])], axis=1)


def aumentar_una(x, rng):
    x = deformar_tiempo(x, rng)
    for mano in range(2):
        b = mano * POR_MANO
        presente = x[:, b + 62] > 0.5
        x[:, b:b + 2] = x[:, b:b + 2] * rng.uniform(0.9, 1.1) + rng.uniform(-0.1, 0.1, 2)
        forma = x[:, b + 2:b + 62].reshape(-1, 20, 3)
        a = rng.uniform(-GIRO_MAXIMO, GIRO_MAXIMO)
        c, s = np.cos(a), np.sin(a)
        xy = forma[..., :2] @ np.array([[c, s], [-s, c]])
        forma = np.concatenate([xy, forma[..., 2:]], axis=-1) + rng.normal(0, 0.02, forma.shape)
        x[:, b + 2:b + 62] = forma.reshape(-1, 60)
        x[~presente, b:b + POR_MANO] = 0
        x[presente, b + 62] = 1
    return x


def aumentar(X, rng):
    """X: (n, T, D). Devuelve una copia aumentada."""
    return np.stack([aumentar_una(x.copy(), rng) for x in X])
