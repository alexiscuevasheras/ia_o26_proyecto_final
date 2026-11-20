"""Mismos pesos y mismo lote dan los mismos gradientes que PyTorch.

PyTorch es solo dependencia de desarrollo (no va en la imagen de Docker):
si no está instalado, la prueba se salta.
"""

import numpy as np
import pytest

from redneuronal.perdida import Castigo, EntropiaCruzada
from redneuronal.red import gradiente, red_inicial
from test_gradientes import lote_aleatorio

torch = pytest.importorskip("torch")


def test_gradientes_iguales_a_pytorch():
    anchos = (60, 128, 64, 21)
    mu = 1e-3
    red = red_inicial(anchos, ("relu", "relu", "softmax"), semilla=7)
    X, Y = lote_aleatorio(anchos, m=16, semilla=8)

    W = [torch.tensor(red.W[l], dtype=torch.float64, requires_grad=True) for l in (1, 2, 3)]
    b = [torch.tensor(red.b[l], dtype=torch.float64, requires_grad=True) for l in (1, 2, 3)]
    A = torch.tensor(X)
    for l in range(3):
        Z = W[l] @ A + b[l][:, None]
        A = torch.relu(Z) if l < 2 else Z
    log_probs = torch.log_softmax(A, dim=0)
    perdida = -(torch.tensor(Y) * log_probs).sum(dim=0).mean()
    perdida = perdida + mu * sum((w**2).sum() for w in W)
    perdida.backward()

    grad_W, grad_b, _ = gradiente(red, X, Y, EntropiaCruzada(), Castigo(mu))
    for l in (1, 2, 3):
        np.testing.assert_allclose(grad_W[l], W[l - 1].grad.numpy(), atol=1e-6)
        np.testing.assert_allclose(grad_b[l], b[l - 1].grad.numpy(), atol=1e-6)
