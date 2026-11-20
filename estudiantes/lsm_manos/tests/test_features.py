"""El vector de entrada no debe cambiar con lo que no es la seña: dónde está la
mano, qué tan cerca está, si es izquierda o derecha y la forma de la imagen."""

import numpy as np
import pytest

from lsm.features import normalizar, vector_entrada


def mano_en_pixeles(semilla=0):
    """Una mano fija en píxeles de una imagen de 640 × 480: muñeca abajo y
    los demás puntos arriba de ella, con el punto 9 justo encima."""
    rng = np.random.default_rng(semilla)
    P = np.zeros((21, 3))
    P[0] = (320, 400, 0)
    P[1:, 0] = 320 + rng.uniform(-80, 80, 20)
    P[1:, 1] = 400 - rng.uniform(40, 200, 20)
    P[1:, 2] = rng.uniform(-30, 30, 20)
    P[9] = (330, 280, -5)
    return P


def a_mediapipe(P, ancho, alto):
    """Coordenadas como las da MediaPipe: x/ancho, y/alto; z en la escala de x."""
    return np.column_stack([P[:, 0] / ancho, P[:, 1] / alto, P[:, 2] / ancho])


def girar(P, angulo, centro):
    c, s = np.cos(angulo), np.sin(angulo)
    Q = P.copy()
    dx, dy = P[:, 0] - centro[0], P[:, 1] - centro[1]
    Q[:, 0] = centro[0] + c * dx - s * dy
    Q[:, 1] = centro[1] + s * dx + c * dy
    return Q


def features(P, ancho=640, alto=480, izquierda=False, **opciones):
    return vector_entrada(a_mediapipe(P, ancho, alto), izquierda, aspecto=ancho / alto, **opciones)


def test_valores_conocidos():
    P = np.zeros((21, 3))
    P[0] = (0.5, 0.5, 0.0)
    P[9] = (0.5, 0.3, 0.1)
    P[8] = (0.6, 0.3, 0.0)
    N = normalizar(P, es_izquierda=False)
    np.testing.assert_allclose(N[0], [0, 0, 0])
    np.testing.assert_allclose(N[9], [0, -1, 0.5])  # s = 0.2
    np.testing.assert_allclose(N[8], [0.5, -1, 0])


def test_tamano_del_vector():
    P = mano_en_pixeles()
    assert features(P).shape == (60,)
    assert features(P, usar_z=False).shape == (40,)
    lote = np.stack([a_mediapipe(mano_en_pixeles(s), 640, 480) for s in range(5)])
    assert vector_entrada(lote, False, aspecto=4 / 3).shape == (5, 60)


def test_no_depende_de_posicion_ni_tamano():
    P = mano_en_pixeles()
    movida = P * 0.6 + np.array([150, -60, 0])
    np.testing.assert_allclose(features(movida), features(P), atol=1e-12)


def test_no_depende_de_la_proporcion_de_la_imagen():
    """La misma mano física en una foto 4:3 y en una cámara 16:9."""
    P = mano_en_pixeles()
    ancha = P + np.array([320, 0, 0])  # misma mano, imagen 1280 × 720 con la mano corrida
    ancha[:, :2] *= 720 / 480
    ancha[:, 2] *= 720 / 480
    np.testing.assert_allclose(features(ancha, 1280, 720), features(P, 640, 480), atol=1e-12)


def test_mano_izquierda_se_vuelve_derecha():
    """Una mano izquierda es el reflejo de una derecha: con el espejo canónico
    dan el mismo vector."""
    derecha = mano_en_pixeles()
    izquierda = derecha.copy()
    izquierda[:, 0] = 640 - derecha[:, 0]
    np.testing.assert_allclose(features(izquierda, izquierda=True), features(derecha), atol=1e-12)
    assert not np.allclose(features(izquierda, izquierda=False), features(derecha))


def test_lateralidad_por_mano_en_un_lote():
    derecha = a_mediapipe(mano_en_pixeles(), 640, 480)
    izquierda = derecha.copy()
    izquierda[:, 0] = 1 - derecha[:, 0]
    X = vector_entrada(np.stack([derecha, izquierda]), [False, True], aspecto=4 / 3)
    np.testing.assert_allclose(X[0], X[1], atol=1e-12)


@pytest.mark.parametrize("angulo", [0.3, -1.0, 2.5])
def test_rotacion_opcional(angulo):
    P = mano_en_pixeles()
    girada = girar(P, angulo, centro=(200, 100))
    np.testing.assert_allclose(features(girada, rotar=True), features(P, rotar=True), atol=1e-10)
    assert not np.allclose(features(girada), features(P))
    N = normalizar(a_mediapipe(girada, 640, 480), False, aspecto=4 / 3, rotar=True)
    np.testing.assert_allclose(N[9, :2], [0, -1], atol=1e-12)


def test_no_modifica_la_entrada():
    P = a_mediapipe(mano_en_pixeles(), 640, 480)
    copia = P.copy()
    vector_entrada(P, True, aspecto=4 / 3, rotar=True)
    np.testing.assert_array_equal(P, copia)


def test_palma_de_tamano_cero():
    with pytest.raises(ValueError):
        normalizar(np.zeros((21, 3)), es_izquierda=False)
