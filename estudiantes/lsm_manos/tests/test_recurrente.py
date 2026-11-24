"""La BPTT de la GRU contra diferencias finitas, y que aprende algo simple."""

import numpy as np

from redneuronal.recurrente import entrenar, gradiente, gru_inicial, predecir


def datos(T=5, d=3, m=4, K=3, semilla=1):
    rng = np.random.default_rng(semilla)
    X = rng.standard_normal((T, d, m))
    Y = np.eye(K)[:, rng.integers(0, K, m)]
    return X, Y


def test_gradiente_coincide_con_diferencias_finitas():
    X, Y = datos()
    red = gru_inicial(3, 4, 3, semilla=2)
    for p in red.p.values():  # sesgos distintos de cero para probar todas las ramas
        p += np.random.default_rng(3).normal(0, 0.3, p.shape)
    _, g = gradiente(red, X, Y, mu=1e-2)
    h = 1e-6
    for nombre, P in red.p.items():
        for idx in [tuple(np.unravel_index(i, P.shape)) for i in range(0, P.size, max(1, P.size // 7))]:
            original = P[idx]
            P[idx] = original + h
            mas, _ = gradiente(red, X, Y, mu=1e-2)
            P[idx] = original - h
            menos, _ = gradiente(red, X, Y, mu=1e-2)
            P[idx] = original
            numerico = (mas - menos) / (2 * h)
            assert abs(numerico - g[nombre][idx]) < 1e-6 * max(1.0, abs(numerico)), (nombre, idx)


def test_aprende_cual_entrada_se_encendio_primero():
    """Clase = canal que se enciende en el primer paso; después todo es ruido.
    Solo se resuelve recordando a lo largo de la secuencia."""
    rng = np.random.default_rng(0)
    T, d, n = 8, 3, 300
    clase = rng.integers(0, d, n)
    X = rng.normal(0, 0.3, (T, d, n))
    X[0, clase, np.arange(n)] += 3.0
    Y = np.eye(d)[:, clase]
    red = entrenar(gru_inicial(d, 16, d), X, Y, epocas=30, eta=1e-2)
    exactitud = np.mean(predecir(red, X).argmax(axis=0) == clase)
    assert exactitud > 0.95
