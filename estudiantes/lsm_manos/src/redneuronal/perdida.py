"""Pérdidas ℓ(ŷ, y) de una observación y castigos Ω(θ).

Convención: Ŷ y Y son matrices d_L × m, una columna por observación k.
"""

import numpy as np

EPSILON_LOG = 1e-12


class PerdidaCuadratica:
    """ℓ(ŷ, y) = Σ_i (ŷ_i − y_i)², sin el 1/2, como en el curso."""

    nombre = "cuadratica"

    def valor(self, Y_gorro, Y):
        """ℓ_k de cada observación: un vector de longitud m."""
        return np.sum((Y_gorro - Y) ** 2, axis=0)

    def gradiente_salida(self, Y_gorro, Y):
        """∇_{a(L)} ℓ = 2(ŷ − y)."""
        return 2.0 * (Y_gorro - Y)


class EntropiaCruzada:
    """ℓ(ŷ, y) = −Σ_i y_i log ŷ_i, con y one-hot y ŷ = softmax(z(L)).

    Extensión: no está en las páginas de la unidad 10. Junto con softmax,
    el error de la salida se simplifica a δ(L) = ŷ − y (derivación en
    docs/NOTAS_ESTUDIO.md).
    """

    nombre = "entropia_cruzada"

    def valor(self, Y_gorro, Y):
        return -np.sum(Y * np.log(Y_gorro + EPSILON_LOG), axis=0)

    def delta_salida_softmax(self, Y_gorro, Y):
        return Y_gorro - Y


class Castigo:
    """Castigo μ·Ω(θ) sobre los pesos; los sesgos nunca se castigan.

    tipo "ridge": Ω = Σ_l ‖W(l)‖²_F, gradiente 2μW(l).
    tipo "lasso": Ω₁ = Σ_l Σ_ij |w(l)_ij|, gradiente μ·sign(W(l)) (ej. 10.1.5.4).
    capas: qué capas l se castigan; None significa todas. Para la pregunta Q1
    se castiga solo W(1).
    """

    def __init__(self, mu=0.0, tipo="ridge", capas=None):
        if tipo not in ("ridge", "lasso"):
            raise ValueError(f"castigo desconocido: {tipo}")
        self.mu = mu
        self.tipo = tipo
        self.capas = capas

    def aplica_a(self, l):
        return self.mu > 0 and (self.capas is None or l in self.capas)

    def omega(self, W, l):
        if not self.aplica_a(l):
            return 0.0
        if self.tipo == "ridge":
            return float(np.sum(W**2))
        return float(np.sum(np.abs(W)))

    def gradiente(self, W, l):
        if not self.aplica_a(l):
            return np.zeros_like(W)
        if self.tipo == "ridge":
            return 2.0 * self.mu * W
        return self.mu * np.sign(W)  # np.sign(0) = 0, como pide el curso


SIN_CASTIGO = Castigo(0.0)
