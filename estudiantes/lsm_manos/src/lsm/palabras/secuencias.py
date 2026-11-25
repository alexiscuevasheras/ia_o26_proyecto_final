"""De los puntos crudos de una seña a la entrada de la red: T cuadros × D números.

La demo (web/secuencias.js) hace exactamente lo mismo; una prueba los compara.

1. Proporción: x (y la z de las manos) se multiplican por ancho/alto, como en
   las letras, para que todo quede en la escala de y.
2. Marco del cuerpo, medido en «anchos de hombro». En UAEMex (ropa negra
   sobre fondo negro) el detector de cuerpo a veces pone los hombros casi
   encimados: su ancho sale bimodal (≈0.29 real contra 0.01–0.07 falso). La
   palma, en cambio, sale estable. Por eso:
   - escala = mediana del largo de la palma ÷ 0.24 (en la persona 01, con
     hombros bien detectados, la palma mide 0.072 / 0.293 ≈ 0.24 hombros);
   - centro = mediana del punto medio de los hombros, solo en cuadros con un
     ancho de hombros plausible (0.6 a 1.6 veces la escala). Si la seña no
     tiene ninguno, el centro de respaldo (el de la persona en todas sus
     señas; en vivo, el último bueno) o, sin respaldo, una estimación con
     las manos (`centro_sin_cuerpo`).
3. Qué mano es de qué brazo: con el cuerpo mal detectado, la asignación por
   muñeca más cercana de extraer.py falla y revuelve las manos entre cuadros.
   Se rehace por seguimiento: cada seña empieza en reposo, con la mano del
   brazo izquierdo de la persona a la derecha de la imagen; después cada mano
   detectada va al brazo cuya última posición conocida queda más cerca (así
   aguanta cuando las manos se cruzan). Ver `seguir_manos`.
4. Mano dominante: es un rasgo de la persona, no de la seña, así que se
   decide por mayoría entre todas sus señas (en cada una, la mano que más se
   mueve). En la demo se va decidiendo con las señas que lleva la persona. Si
   es la izquierda, todo se espeja (x → −x y se intercambian las manos): así
   una persona zurda y una diestra producen la misma entrada.
5. Por cuadro y por mano (dominante primero): la muñeca en el marco del
   cuerpo (x, y), la forma de la mano igual que en las letras (60 números) y
   1 si la mano está presente. Huecos cortos se interpolan; una mano ausente
   vale 0 con presencia 0.
6. Se remuestrea en el tiempo a T cuadros con interpolación lineal.
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


PALMA_POR_HOMBRO = 0.24
HOMBROS_PLAUSIBLES = (0.6, 1.6)


def escala_de_palmas(manos):
    palmas = np.linalg.norm(manos[:, :, 9, :2] - manos[:, :, 0, :2], axis=-1)
    if not np.isfinite(palmas).any():
        return None
    return max(float(np.nanmedian(palmas)) / PALMA_POR_HOMBRO, 1e-3)


def centro_de_hombros(cuerpo, escala):
    """Mediana del punto medio de los hombros en cuadros con ancho plausible, o None."""
    hombros = cuerpo[:, [HOMBRO_IZQ, HOMBRO_DER], :2]
    ancho = np.linalg.norm(hombros[:, 0] - hombros[:, 1], axis=1)
    bajo, alto = HOMBROS_PLAUSIBLES
    buenos = np.isfinite(ancho) & (ancho >= bajo * escala) & (ancho <= alto * escala)
    if not buenos.any():
        return None
    return np.median(hombros[buenos].mean(axis=1), axis=0)


def centro_sin_cuerpo(manos, escala):
    """Supuesto burdo, solo para no perder la seña: el centro de los hombros
    queda 1.5 anchos arriba de la muñeca más baja (las manos en reposo)."""
    munecas = manos[:, :, 0, :2].reshape(-1, 2)
    munecas = munecas[~np.isnan(munecas).any(axis=1)]
    if len(munecas) == 0:
        return np.array([0.5, 0.5])
    return np.array([np.median(munecas[:, 0]), munecas[:, 1].max() - 1.5 * escala])


def marco_del_cuerpo(cuerpo, manos, centro_respaldo=None):
    """(centro x-y, escala) de la seña completa; ver el paso 2 del docstring."""
    escala = escala_de_palmas(manos) or 1.0
    centro = centro_de_hombros(cuerpo, escala)
    if centro is None:
        centro = centro_respaldo if centro_respaldo is not None else centro_sin_cuerpo(manos, escala)
    return np.asarray(centro, float), escala


def seguir_manos(manos, centro, escala):
    """Asignación por continuidad (paso 3): (F, 2) con, para cada brazo, el
    índice de la mano original que le toca, o −1. Antes de ver una mano, su
    «última posición» es su lugar de reposo."""
    ultima = np.array([[centro[0] + 0.8 * escala, centro[1] + 1.6 * escala],
                       [centro[0] - 0.8 * escala, centro[1] + 1.6 * escala]])
    origen = np.full((len(manos), 2), -1)
    for f in range(len(manos)):
        vistas = [i for i in (0, 1) if not np.isnan(manos[f, i, 0, 0])]
        if len(vistas) == 2:
            directo = sum(np.linalg.norm(manos[f, i, 0, :2] - ultima[i]) for i in (0, 1))
            cruzado = sum(np.linalg.norm(manos[f, 1 - i, 0, :2] - ultima[i]) for i in (0, 1))
            origen[f] = (0, 1) if directo <= cruzado else (1, 0)
        elif len(vistas) == 1:
            origen[f, int(np.argmin(np.linalg.norm(ultima - manos[f, vistas[0], 0, :2], axis=1)))] = vistas[0]
        for brazo in (0, 1):
            if origen[f, brazo] >= 0:
                ultima[brazo] = manos[f, origen[f, brazo], 0, :2]
    return origen


def reordenar(manos, origen):
    """Aplica la asignación de `seguir_manos` a cualquier arreglo (F, 2, …) de manos."""
    salida = np.full_like(manos, np.nan)
    for brazo in (0, 1):
        f = np.flatnonzero(origen[:, brazo] >= 0)
        salida[f, brazo] = manos[f, origen[f, brazo]]
    return salida


def preparar_sena(cuerpo, manos, aspecto, centro_respaldo=None):
    """Proporción corregida, marco y manos seguidas: (manos, centro, escala, origen)."""
    cuerpo, manos = corregir_aspecto(cuerpo, manos, aspecto)
    centro, escala = marco_del_cuerpo(cuerpo, manos, centro_respaldo)
    origen = seguir_manos(manos, centro, escala)
    return reordenar(manos, origen), centro, escala, origen


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


def cuadros(cuerpo, manos, aspecto, dominante=None, centro_respaldo=None):
    """(F, D) sin remuestrear, más el lado dominante usado (0 izquierda, 1 derecha).
    Sin `dominante`, se usa la mano que más se mueve en esta seña.
    `centro_respaldo` va en coordenadas ya corregidas por la proporción."""
    manos, centro, escala, _ = preparar_sena(cuerpo, manos, aspecto, centro_respaldo)
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


def secuencia(cuerpo, manos, aspecto, dominante=None, t=T, centro_respaldo=None):
    """La entrada de la red para una seña: (t, D)."""
    return remuestrear(cuadros(cuerpo, manos, aspecto, dominante, centro_respaldo)[0], t)
