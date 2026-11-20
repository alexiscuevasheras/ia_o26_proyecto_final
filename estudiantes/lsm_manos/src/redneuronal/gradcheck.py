"""Comprobación de la retropropagación por diferencias finitas centrales.

∂/∂w ≈ (f(w + h) − f(w − h)) / (2h), con h = 1e-6, y por bloque
el error relativo ‖retro − difer‖ / (‖retro‖ + ‖difer‖).
"""

import numpy as np

from .perdida import SIN_CASTIGO
from .red import gradiente, objetivo

H = 1e-6


def gradiente_numerico(red, X, Y, perdida, castigo=SIN_CASTIGO, h=H):
    """Diferencias finitas sobre cada parámetro, una entrada a la vez."""
    numerico = {"W": [None], "b": [None]}
    for nombre in ("W", "b"):
        for l in range(1, red.L + 1):
            bloque = getattr(red, nombre)[l]
            G = np.zeros_like(bloque)
            for idx in np.ndindex(bloque.shape):
                original = bloque[idx]
                bloque[idx] = original + h
                arriba = objetivo(red, X, Y, perdida, castigo)
                bloque[idx] = original - h
                abajo = objetivo(red, X, Y, perdida, castigo)
                bloque[idx] = original
                G[idx] = (arriba - abajo) / (2 * h)
            numerico[nombre].append(G)
    return numerico["W"], numerico["b"]


def error_relativo(retro, difer):
    escala = np.linalg.norm(retro) + np.linalg.norm(difer)
    return 0.0 if escala == 0 else float(np.linalg.norm(retro - difer) / escala)


def errores_por_bloque(red, X, Y, perdida, castigo=SIN_CASTIGO, h=H):
    """{"W1": e, "b1": e, …}: error relativo de cada bloque W(l) y b(l)."""
    grad_W, grad_b, _ = gradiente(red, X, Y, perdida, castigo)
    num_W, num_b = gradiente_numerico(red, X, Y, perdida, castigo, h)
    errores = {}
    for l in range(1, red.L + 1):
        errores[f"W{l}"] = error_relativo(grad_W[l], num_W[l])
        errores[f"b{l}"] = error_relativo(grad_b[l], num_b[l])
    return errores
