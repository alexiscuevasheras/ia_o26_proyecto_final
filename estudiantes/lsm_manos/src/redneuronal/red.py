"""La red d_0–d_1–…–d_L con la notación de la unidad 10.

- W[l] es W(l), de d_l × d_{l−1} (filas = destino i, columnas = origen j).
- b[l] es b(l), de d_l entradas. La posición 0 de cada lista es None:
  la capa 0 es la entrada x = a(0) y no tiene parámetros.
- Las observaciones van en columnas: X es d_0 × m, y
  Z(l) = W(l) A(l−1) + b(l), con el sesgo difundido a cada columna.
"""

from dataclasses import dataclass

import numpy as np

from .activaciones import ACTIVACIONES
from .perdida import SIN_CASTIGO, EntropiaCruzada


def columnas(v):
    """Un vector suelto es una sola observación: una columna."""
    v = np.asarray(v, dtype=float)
    return v[:, None] if v.ndim == 1 else v


@dataclass
class Red:
    W: list
    b: list
    g: list

    @property
    def L(self):
        """Número de capas con parámetros."""
        return len(self.W) - 1

    @property
    def anchos(self):
        """(d_0, d_1, …, d_L)."""
        return (self.W[1].shape[1],) + tuple(self.W[l].shape[0] for l in range(1, self.L + 1))

    def num_parametros(self):
        """Σ_l d_l (d_{l−1} + 1)."""
        return sum(self.W[l].size + self.b[l].size for l in range(1, self.L + 1))

    def copia(self):
        return Red(
            W=[None] + [W.copy() for W in self.W[1:]],
            b=[None] + [b.copy() for b in self.b[1:]],
            g=list(self.g),
        )


def red_inicial(anchos, activaciones, semilla=0):
    """Pesos normales escalados por √(2/d_{l−1}) (He) y sesgos en cero,
    como en el notebook del curso."""
    if len(activaciones) != len(anchos) - 1:
        raise ValueError("hace falta una activación por capa con parámetros")
    rng = np.random.default_rng(semilla)
    W, b = [None], [None]
    for l in range(1, len(anchos)):
        escala = np.sqrt(2.0 / anchos[l - 1])
        W.append(rng.standard_normal((anchos[l], anchos[l - 1])) * escala)
        b.append(np.zeros(anchos[l]))
    return Red(W=W, b=b, g=[None, *activaciones])


def adelante(red, X):
    """Pase hacia adelante. Devuelve las listas Z (Z[0] = None) y A (A[0] = X).
    Ŷ = A[L]."""
    X = columnas(X)
    Z, A = [None], [X]
    for l in range(1, red.L + 1):
        g, _ = ACTIVACIONES[red.g[l]]
        Z.append(red.W[l] @ A[l - 1] + red.b[l][:, None])
        A.append(g(Z[l]))
    return Z, A


def predecir(red, X):
    """Ŷ = f_θ(X), una columna por observación."""
    return adelante(red, X)[1][-1]


def perdida_promedio(red, X, Y, perdida):
    """L(θ) = (1/n) Σ_k ℓ(f_θ(x_k), y_k)."""
    return float(np.mean(perdida.valor(predecir(red, X), columnas(Y))))


def omega(red, castigo):
    """Ω(θ) sumado sobre las capas que el castigo toca."""
    return sum(castigo.omega(red.W[l], l) for l in range(1, red.L + 1))


def objetivo(red, X, Y, perdida, castigo=SIN_CASTIGO):
    """L(θ) + μΩ(θ): el lagrangeano sin el término −μd, que no cambia el
    gradiente ni dónde está el mínimo."""
    return perdida_promedio(red, X, Y, perdida) + castigo.mu * omega(red, castigo)


def delta_salida(red, Z, Y_gorro, Y, perdida):
    """δ(L) = ∇_{a(L)} ℓ ⊙ g(L)'(z(L)); con softmax y entropía cruzada, ŷ − y."""
    if red.g[red.L] == "softmax":
        if not isinstance(perdida, EntropiaCruzada):
            raise ValueError("softmax en la salida solo se usa con entropía cruzada")
        return perdida.delta_salida_softmax(Y_gorro, Y)
    _, g_prima = ACTIVACIONES[red.g[red.L]]
    return perdida.gradiente_salida(Y_gorro, Y) * g_prima(Z[red.L])


def gradiente(red, X, Y, perdida, castigo=SIN_CASTIGO):
    """Retropropagación de un lote. Devuelve (grad_W, grad_b, deltas):

    δ(l) = (W(l+1)ᵀ δ(l+1)) ⊙ g(l)'(z(l)), de d_l × m;
    ∇W(l) L_B = (1/m) δ(l) A(l−1)ᵀ + ∇W(l) μΩ;   ∇b(l) L_B = promedio de δ(l).
    """
    X, Y = columnas(X), columnas(Y)
    Z, A = adelante(red, X)
    m = X.shape[1]
    L = red.L
    deltas = [None] * (L + 1)
    grad_W, grad_b = [None] * (L + 1), [None] * (L + 1)
    deltas[L] = delta_salida(red, Z, A[L], Y, perdida)
    for l in range(L, 0, -1):
        if l < L:
            _, g_prima = ACTIVACIONES[red.g[l]]
            deltas[l] = (red.W[l + 1].T @ deltas[l + 1]) * g_prima(Z[l])
        grad_W[l] = deltas[l] @ A[l - 1].T / m + castigo.gradiente(red.W[l], l)
        grad_b[l] = deltas[l].mean(axis=1)
    return grad_W, grad_b, deltas
