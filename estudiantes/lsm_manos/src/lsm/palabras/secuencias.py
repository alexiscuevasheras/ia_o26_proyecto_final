"""De los puntos crudos de una seña a la entrada de la red: T cuadros × D números.

La demo (web/secuencias.js) hace exactamente lo mismo; una prueba los compara.

1. Proporción: x (y la z de las manos) se multiplican por ancho/alto, como en
   las letras, para que todo quede en la escala de y.
2. Marco del cuerpo: centro = punto medio de los hombros, escala = ancho de
   hombros. Se usa la mediana de la seña porque el cuerpo casi no se mueve y
   el detector lo pierde en algunos cuadros. Sin cuerpo en toda la seña, se
   estima con las manos (ver `marco_sin_cuerpo`).
3. Mano dominante: es un rasgo de la persona, no de la seña, así que se
   decide por mayoría entre todas sus señas (en cada una, la mano que más se
   mueve). En la demo se va decidiendo con las señas que lleva la persona. Si
   es la izquierda, todo se espeja (x → −x y se intercambian las manos): así
   una persona zurda y una diestra producen la misma entrada.
4. Por cuadro y por mano (dominante primero): la muñeca en el marco del
   cuerpo (x, y), la forma de la mano igual que en las letras (60 números) y
   1 si la mano está presente. Huecos cortos se interpolan; una mano ausente
   vale 0 con presencia 0.
5. Se remuestrea en el tiempo a T cuadros con interpolación lineal.
"""

import numpy as np

from ..features import normalizar

T = 16
HOMBRO_IZQ, HOMBRO_DER = 11, 12
POR_MANO = 2 + 60 + 1
D = 2 * POR_MANO
HUECO_MAXIMO = 3  # cuadros seguidos sin mano que todavía se interpolan


def corregir_aspecto(cuerpo, manos, aspecto):
    cuerpo, manos = cuerpo.copy(), manos.copy()
    cuerpo[..., 0] *= aspecto
    manos[..., 0] *= aspecto
    manos[..., 2] *= aspecto
    return cuerpo, manos


def marco_del_cuerpo(cuerpo, manos):
    """(centro x-y, escala) de la seña completa."""
    hombros = cuerpo[:, [HOMBRO_IZQ, HOMBRO_DER], :2]
    validos = ~np.isnan(hombros).any(axis=(1, 2))
    if validos.any():
        h = np.median(hombros[validos], axis=0)
        return h.mean(axis=0), max(float(np.linalg.norm(h[0] - h[1])), 1e-3)
    return marco_sin_cuerpo(manos)


def marco_sin_cuerpo(manos):
    """Sin hombros: la palma mide en promedio ~0.4 anchos de hombro, y el centro
    del pecho queda ~1.5 anchos debajo de la mano más alta (supuesto burdo,
    solo para no perder la muestra)."""
    palmas = np.linalg.norm(manos[:, :, 9, :2] - manos[:, :, 0, :2], axis=-1)
    escala = float(np.nanmedian(palmas)) / 0.4 if np.isfinite(palmas).any() else 1.0
    munecas = manos[:, :, 0, :2].reshape(-1, 2)
    munecas = munecas[~np.isnan(munecas).any(axis=1)]
    if len(munecas) == 0:
        return np.array([0.5, 0.5]), escala
    return np.array([np.median(munecas[:, 0]), munecas[:, 1].min() + 1.5 * escala]), escala


def movimiento(manos, i):
    """Recorrido total de la muñeca i (en unidades de imagen) más su presencia:
    una mano que casi nunca aparece no puede ser la dominante."""
    m = manos[:, i, 0, :2]
    presente = ~np.isnan(m[:, 0])
    if presente.sum() < 2:
        return presente.sum() * 1e-3
    pasos = np.diff(m[presente], axis=0)
    return float(np.linalg.norm(pasos, axis=1).sum()) + presente.mean() * 1e-3


def rellenar_huecos(x, presente):
    """Interpola linealmente huecos de hasta HUECO_MAXIMO cuadros, por columna."""
    x, presente = x.copy(), presente.copy()
    idx = np.flatnonzero(presente)
    for a, b in zip(idx[:-1], idx[1:]):
        if 1 < b - a <= HUECO_MAXIMO + 1:
            t = (np.arange(a + 1, b) - a) / (b - a)
            x[a + 1:b] = x[a] + t[:, None] * (x[b] - x[a])
            presente[a + 1:b] = True
    return x, presente


def mano_que_mas_se_mueve(manos):
    """0 (brazo izquierdo) o 1 (derecho) en una seña."""
    return int(movimiento(manos, 1) >= movimiento(manos, 0))


def cuadros(cuerpo, manos, aspecto, dominante=None):
    """(F, D) sin remuestrear, más el lado dominante usado (0 izquierda, 1 derecha).
    Sin `dominante`, se usa la mano que más se mueve en esta seña."""
    cuerpo, manos = corregir_aspecto(cuerpo, manos, aspecto)
    centro, escala = marco_del_cuerpo(cuerpo, manos)
    if dominante is None:
        dominante = mano_que_mas_se_mueve(manos)
    espejo = dominante == 0
    F = len(manos)
    salida = np.zeros((F, D))
    for posicion, brazo in enumerate((dominante, 1 - dominante)):
        mano = manos[:, brazo]
        presente = ~np.isnan(mano[:, 0, 0])
        bloque = np.zeros((F, POR_MANO))
        if presente.any():
            muneca = (mano[presente, 0, :2] - centro) / escala
            if espejo:
                muneca[:, 0] *= -1
            # La forma es la misma entrada que la red de letras: MediaPipe supone
            # imagen espejada, así que en un cuadro sin espejo etiqueta «Left» a la
            # mano derecha de la persona, y esa es la que las letras espejan.
            forma = np.delete(normalizar(mano[presente], brazo == 1), 0, axis=1).reshape(-1, 60)
            bloque[presente, :2] = muneca
            bloque[presente, 2:62] = forma
            bloque[presente, 62] = 1
            bloque, presente = rellenar_huecos(bloque, presente)
            bloque[~presente] = 0
        salida[:, posicion * POR_MANO:(posicion + 1) * POR_MANO] = bloque
    return salida, dominante


def remuestrear(x, t=T):
    """(F, D) → (t, D) por interpolación lineal en el tiempo."""
    if len(x) == 1:
        return np.repeat(x, t, axis=0)
    viejo = np.linspace(0, 1, len(x))
    nuevo = np.linspace(0, 1, t)
    return np.stack([np.interp(nuevo, viejo, x[:, j]) for j in range(x.shape[1])], axis=1)


def secuencia(cuerpo, manos, aspecto, dominante=None, t=T):
    """La entrada de la red para una seña: (t, D)."""
    return remuestrear(cuadros(cuerpo, manos, aspecto, dominante)[0], t)
