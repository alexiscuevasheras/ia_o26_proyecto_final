"""De los 21 puntos de MediaPipe al vector de entrada x = a(0) de la red.

La demo en JavaScript (web/features.js) debe hacer exactamente lo mismo; una
prueba compara los dos con landmarks fijos.

Pasos, para cada mano:
1. Proporción: MediaPipe divide x entre el ancho e y entre el alto, y da z
   en la escala de x. Se multiplican x y z por ancho/alto para que todo
   quede en la escala de y (si no, una foto 4:3 y una cámara 16:9 deforman
   la mano distinto).
2. Espejo canónico: si la mano es izquierda, x → −x; todas parecen derechas.
3. Centrar en la muñeca (punto 0).
4. Escalar por el tamaño de la palma, s = ‖p9 − p0‖ en el plano x-y (z de
   MediaPipe es más ruidosa, así que no entra en s).
5. Opcional (ablación Q3): rotar en el plano x-y para que p9 − p0 apunte
   hacia arriba.
6. Quitar la muñeca (queda en cero) y aplanar punto por punto:
   (x1, y1, z1, x2, y2, z2, …) → 60 números, o 40 si solo se usan x e y.
"""

import numpy as np

MUNECA = 0
BASE_MEDIO = 9
NUM_PUNTOS = 21


def normalizar(puntos, es_izquierda, aspecto=1.0, rotar=False):
    """puntos: arreglo (N, 21, 3) o (21, 3) con x, y, z de MediaPipe.
    es_izquierda: bool o arreglo de N bools (lateralidad de la mano tal como
    se ve en la imagen sin espejo). aspecto: ancho/alto de la imagen (un
    número o uno por mano).
    Devuelve los puntos normalizados con la misma forma."""
    P = np.array(puntos, dtype=float)
    una_sola = P.ndim == 2
    if una_sola:
        P = P[None]
    if P.shape[1:] != (NUM_PUNTOS, 3):
        raise ValueError(f"se esperaban (N, 21, 3) puntos, llegó {P.shape}")

    izquierda = np.broadcast_to(np.asarray(es_izquierda, dtype=bool), (P.shape[0],))
    aspectos = np.broadcast_to(np.asarray(aspecto, dtype=float), (P.shape[0],))
    P[:, :, [0, 2]] *= aspectos[:, None, None]  # z de MediaPipe viene en la escala de x
    P[izquierda, :, 0] *= -1.0
    P -= P[:, MUNECA:MUNECA + 1, :]

    palma = P[:, BASE_MEDIO, :2]
    s = np.linalg.norm(palma, axis=1)
    if np.any(s < 1e-9):
        raise ValueError("palma de tamaño cero: la muñeca y el punto 9 coinciden")
    P /= s[:, None, None]

    if rotar:
        P = _alinear_palma_hacia_arriba(P)
    return P[0] if una_sola else P


def _alinear_palma_hacia_arriba(P):
    """Rota x-y para que p9 − p0 quede en (0, −1): en la imagen, y crece
    hacia abajo, así que «arriba» es −y. z no cambia."""
    angulo_palma = np.arctan2(P[:, BASE_MEDIO, 1], P[:, BASE_MEDIO, 0])
    giro = (-np.pi / 2 - angulo_palma)[:, None]
    c, s = np.cos(giro), np.sin(giro)
    x, y = P[:, :, 0], P[:, :, 1]
    R = P.copy()
    R[:, :, 0] = c * x - s * y
    R[:, :, 1] = s * x + c * y
    return R


def vector_entrada(puntos, es_izquierda, aspecto=1.0, rotar=False, usar_z=True):
    """El vector de entrada: (N, 60) con x, y, z, o (N, 40) solo con x, y.
    Con una sola mano (21, 3) devuelve un vector de 60 o 40."""
    P = normalizar(puntos, es_izquierda, aspecto, rotar)
    una_sola = P.ndim == 2
    if una_sola:
        P = P[None]
    sin_muneca = np.delete(P, MUNECA, axis=1)
    if not usar_z:
        sin_muneca = sin_muneca[:, :, :2]
    X = sin_muneca.reshape(P.shape[0], -1)
    return X[0] if una_sola else X
