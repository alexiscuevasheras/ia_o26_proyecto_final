"""Genera modelos/animaciones.json para el avatar (formato en docs/FORMATOS.md).

Por glosa se elige el medoide: la repetición cuya secuencia de features
está, en promedio, más cerca de las demás de esa glosa. No se promedian
repeticiones porque el promedio de dos movimientos emborrona los dos. Se
prefiere ITESO (video continuo a 15 fps) sobre UAEMex (~14 fotos sueltas
por seña) y ambos sobre el dataset propio.

Cómo se arma cada cuadro. La pose 3D del cuerpo que da MediaPipe en estos
videos es poco confiable (en UAEMex, ropa negra sobre fondo negro), pero las
manos sí se detectan bien. Por eso:
- el torso y la cabeza son un esqueleto fijo de proporciones humanas (el
  torso de quien seña casi no se mueve, y la pose detectada solo metía ruido);
- la muñeca sale de la trayectoria 2D de la mano en el marco de los hombros
  (la misma de las features), pasada a metros con un ancho de hombros de
  0.36 m; la profundidad se estima: más adelante cuanto más sube la mano;
- el codo se resuelve con cinemática inversa de dos huesos;
- la forma de la mano son sus coordenadas 3D («world», en metros), giradas a
  los ejes del avatar y apoyadas en la muñeca;
- si quien seña es zurdo, se espeja: el avatar siempre seña con la derecha.

Ejes del avatar: metros, origen entre los hombros, x hacia la izquierda de la
persona (en una cámara sin espejo, la derecha de la imagen), y hacia arriba,
z hacia la cámara. Las «world» de MediaPipe tienen y hacia abajo y z
alejándose, así que se multiplican por (1, −1, −1).

Uso:
    python -m lsm.palabras.animaciones modelos/animaciones.json \
        --secuencias datos/secuencias_uaemex.npz datos/secuencias_iteso.npz \
        --letras datos/landmarks_d1.csv
"""

import argparse
import json
from pathlib import Path

import numpy as np

from ..datos import LETRAS_ESTATICAS
from ..entrenar import cargar_landmarks
from ..features import normalizar
from .conjunto import (centros_por_persona, cortes, dominancia_por_persona, glosa_de, leer_secuencias,
                       leer_vocabulario, personas_de)
from .secuencias import preparar_sena, reordenar, secuencia

FPS = 15
CUERPO_INDICES = [0, 2, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 23, 24]
PALMA_METROS = 0.09
CUADROS_UAEMEX = 30  # sus ~14 fotos no tienen tiempo real; 30 cuadros a 15 fps = 2 s (con 1.2 s se veía apresurado)
LETRAS_CON_MOVIMIENTO = ["J", "K", "Ñ", "Q", "X", "Z"]
EJES = np.array([1.0, -1.0, -1.0])
# Video continuo de señantes primero; el dataset propio (estudiantes que aprenden LSM)
# solo para glosas que no estén en los otros.
PRIORIDAD = {"iteso": 2, "uaemex": 1, "propio": 0}
# En UAEMex el detector pierde la mano justo cuando sube frente a la cara (caras
# pixeladas y fotos movidas): una repetición sirve para el avatar solo si la mano
# dominante se vio levantada (sobre ARRIBA anchos de hombro) en varios cuadros.
ARRIBA = 1.0
MINIMO_ARRIBA = 0.3  # fracción de cuadros

# Esqueleto fijo (metros, ejes del avatar), con proporciones de un adulto promedio.
ANCHO_HOMBROS = 0.36
BRAZO, ANTEBRAZO = 0.28, 0.25
ESQUELETO = {
    0: (0.0, 0.21, 0.10),  # nariz
    2: (0.032, 0.235, 0.085), 5: (-0.032, 0.235, 0.085),  # ojos
    7: (0.075, 0.215, 0.0), 8: (-0.075, 0.215, 0.0),  # orejas
    9: (0.025, 0.165, 0.085), 10: (-0.025, 0.165, 0.085),  # boca
    11: (ANCHO_HOMBROS / 2, 0.0, 0.0), 12: (-ANCHO_HOMBROS / 2, 0.0, 0.0),  # hombros
    23: (0.13, -0.50, 0.0), 24: (-0.13, -0.50, 0.0),  # caderas
}
HOMBRO = {0: np.array(ESQUELETO[11]), 1: np.array(ESQUELETO[12])}  # brazo 0 = izquierdo
POLO_CODO = {0: np.array([0.5, -1.0, -0.3]), 1: np.array([-0.5, -1.0, -0.3])}  # abajo, afuera y un poco atrás


def interpolar_faltantes(a):
    """Rellena NaN a lo largo del tiempo (eje 0) con interpolación lineal por
    coordenada; antes del primer valor y después del último se repite el extremo."""
    a = a.copy()
    plano = a.reshape(len(a), -1)
    t = np.arange(len(a))
    for j in range(plano.shape[1]):
        bien = np.isfinite(plano[:, j])
        if bien.any() and not bien.all():
            plano[:, j] = np.interp(t, t[bien], plano[bien, j])
    return plano.reshape(a.shape)


def suavizar(a, ventana=3):
    """Promedio móvil en el tiempo: quita el temblor de MediaPipe sin mover los extremos."""
    if len(a) < ventana:
        return a
    relleno = np.concatenate([a[:1]] * (ventana // 2) + [a] + [a[-1:]] * (ventana // 2))
    nucleo = np.ones(ventana) / ventana
    return np.apply_along_axis(lambda v: np.convolve(v, nucleo, mode="valid"), 0, relleno)


def remuestrear_en_tiempo(a, n):
    t = np.linspace(0, 1, len(a))
    nuevo = np.linspace(0, 1, n)
    plano = a.reshape(len(a), -1)
    return np.stack([np.interp(nuevo, t, plano[:, j]) for j in range(plano.shape[1])], 1).reshape((n,) + a.shape[1:])


def reposo(brazo):
    """Muñeca colgando junto a la cadera."""
    return HOMBRO[brazo] + np.array([0.03 if brazo == 0 else -0.03, -(BRAZO + ANTEBRAZO) * 0.97, 0.04])


def profundidad(altura):
    """Las manos en reposo cuelgan junto al cuerpo (z ≈ 0.04); al subir al
    espacio de señas quedan ~0.3 m al frente. `altura` en anchos de hombro
    bajo los hombros (como en las features)."""
    return 0.04 + 0.26 * np.clip((1.6 - altura) / 1.6, 0, 1)


def munecas_desde_imagen(manos, centro, escala, espejo):
    """(F, 2, 3) muñecas en metros del avatar a partir de las manos 2D seguidas."""
    munecas = np.full((len(manos), 2, 3), np.nan)
    for brazo in (0, 1):
        destino = 1 - brazo if espejo else brazo
        m = manos[:, brazo, 0, :2]
        x = (m[:, 0] - centro[0]) / escala
        y = (m[:, 1] - centro[1]) / escala
        if espejo:
            x = -x
        munecas[:, destino] = np.stack([x * ANCHO_HOMBROS, -y * ANCHO_HOMBROS, profundidad(y)], axis=1)
    for brazo in (0, 1):
        if np.isnan(munecas[:, brazo, 0]).all():
            munecas[:, brazo] = reposo(brazo)
    return interpolar_faltantes(munecas)


def alcanzable(hombro, muneca):
    """Acerca la muñeca al hombro si queda más lejos de lo que mide el brazo."""
    d = muneca - hombro
    largo = np.linalg.norm(d)
    maximo = 0.98 * (BRAZO + ANTEBRAZO)
    return hombro + d * min(1.0, maximo / max(largo, 1e-9))


def codo_ik(hombro, muneca, polo):
    """Cinemática inversa de dos huesos: el codo queda en el plano del polo,
    con los largos BRAZO y ANTEBRAZO."""
    d = muneca - hombro
    largo = max(np.linalg.norm(d), 1e-9)
    u = d / largo
    v = polo - np.dot(polo, u) * u
    v /= max(np.linalg.norm(v), 1e-9)
    coseno = np.clip((BRAZO ** 2 + largo ** 2 - ANTEBRAZO ** 2) / (2 * BRAZO * largo), -1, 1)
    return hombro + BRAZO * (coseno * u + np.sqrt(1 - coseno ** 2) * v)


def forma_3d(mano_3d, espejo):
    """Puntos 3D de una mano relativos a su muñeca, en ejes del avatar."""
    m = mano_3d * EJES
    if espejo:
        m = m * np.array([-1.0, 1.0, 1.0])
    return m - m[0]


def clip_de_muestra(manos, manos_3d, centro, escala, espejo, n_cuadros=None):
    """Lista de cuadros {cuerpo, manos} en el sistema del avatar (ver el docstring)."""
    munecas = suavizar(munecas_desde_imagen(manos, centro, escala, espejo))
    formas = np.full_like(manos_3d, np.nan)
    for brazo in (0, 1):
        formas[:, 1 - brazo if espejo else brazo] = [
            forma_3d(m, espejo) if np.isfinite(m).all() else np.full((21, 3), np.nan) for m in manos_3d[:, brazo]]
    presente = np.isfinite(formas[:, :, 0, 0])
    formas = interpolar_faltantes(formas)
    if n_cuadros:
        munecas, formas = remuestrear_en_tiempo(munecas, n_cuadros), remuestrear_en_tiempo(formas, n_cuadros)
        presente = remuestrear_en_tiempo(presente.astype(float), n_cuadros) > 0.5
    cuadros = []
    for muneca, forma, visto in zip(munecas, formas, presente):
        puntos = {k: np.array(v) for k, v in ESQUELETO.items()}
        lista_manos = []
        for brazo in (0, 1):
            w = alcanzable(HOMBRO[brazo], muneca[brazo])
            puntos[(15, 16)[brazo]] = w
            puntos[(13, 14)[brazo]] = codo_ik(HOMBRO[brazo], w, POLO_CODO[brazo])
            ok = visto[brazo] and np.isfinite(forma[brazo]).all()
            lista_manos.append((forma[brazo] + w).round(4).tolist() if ok else None)
        cuadros.append({"cuerpo": np.array([puntos[k] for k in CUERPO_INDICES]).round(4).tolist(),
                        "manos": lista_manos})
    return cuadros


def medoide(X):
    plano = X.reshape(len(X), -1)
    plano = (plano - plano.mean(0)) / (plano.std(0) + 1e-6)
    d = np.linalg.norm(plano[:, None] - plano[None], axis=-1)
    return int(d.sum(axis=1).argmin())


def animaciones_de_glosas(rutas):
    vocabulario = leer_vocabulario()
    candidatos = {}  # glosa → [(prioridad, X, datos para el clip, origen)]
    for ruta in rutas:
        z = leer_secuencias(ruta)
        personas = personas_de(z)
        respaldo = centros_por_persona(z, personas)
        dominante = dominancia_por_persona(z, personas, respaldo)
        es_iteso = z["dataset"][0] == "iteso"
        for i, (a, b) in enumerate(cortes(z)):
            p = personas[i]
            aspecto = float(z["aspecto"][i])
            manos, centro, escala, origen = preparar_sena(z["cuerpo"][a:b], z["manos"][a:b], aspecto, respaldo.get(p))
            altura = (manos[:, dominante[p], 0, 1] - centro[1]) / escala
            if np.mean(altura < ARRIBA) < MINIMO_ARRIBA:  # NaN cuenta como no vista
                continue
            X = secuencia(z["cuerpo"][a:b], z["manos"][a:b], aspecto, dominante[p], 16, respaldo.get(p))
            manos_3d = reordenar(z["manos_3d"][a:b], origen)  # mismas manos por brazo que en las features
            datos = (manos, manos_3d, centro, escala, dominante[p] == 0, None if es_iteso else CUADROS_UAEMEX)
            glosa = glosa_de(vocabulario, z["dataset"][i], z["clase"][i])
            candidatos.setdefault(glosa, []).append((PRIORIDAD[z["dataset"][i]], X, datos, f"{p}/{z['ident'][i]}"))
    glosas = {}
    for glosa, lista in sorted(candidatos.items()):
        mejor = max(c[0] for c in lista)
        lista = [c for c in lista if c[0] == mejor]
        _, _, datos, origen = lista[medoide(np.stack([c[1] for c in lista]))]
        glosas[glosa] = {"origen": origen, "cuadros": clip_de_muestra(*datos)}
    return glosas


def formas_de_letras(ruta_csv, por_letra=400, semilla=0):
    """Medoide de la forma normalizada de cada letra en D1 (partición train),
    como mano derecha vista de frente, con la palma de PALMA_METROS."""
    d1 = cargar_landmarks(ruta_csv)
    d1 = d1.filtrar(d1.particion == "train")
    rng = np.random.default_rng(semilla)
    letras = {}
    for k, letra in enumerate(LETRAS_ESTATICAS):
        idx = np.flatnonzero(d1.letra == k)
        idx = rng.choice(idx, min(por_letra, len(idx)), replace=False)
        P = normalizar(d1.puntos[idx], d1.es_izquierda[idx], d1.aspecto[idx])
        mano = P[medoide(P)]
        # normalizar deja la mano como la izquierda vista en la imagen; espejada
        # queda la derecha, y luego pasa a los ejes del avatar.
        mano = mano * np.array([-1.0, 1.0, 1.0]) * EJES * PALMA_METROS
        letras[letra] = {"mano": mano.round(4).tolist()}
    return letras


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("salida", type=Path)
    ap.add_argument("--secuencias", type=Path, nargs="+", required=True)
    ap.add_argument("--letras", type=Path, required=True)
    args = ap.parse_args()

    datos = {
        "fps": FPS,
        "cuerpo_indices": CUERPO_INDICES,
        "glosas": animaciones_de_glosas(args.secuencias),
        "letras": formas_de_letras(args.letras),
        "letras_con_movimiento": LETRAS_CON_MOVIMIENTO,
    }
    args.salida.parent.mkdir(parents=True, exist_ok=True)
    args.salida.write_text(json.dumps(datos, ensure_ascii=False, separators=(",", ":")))
    print(f"escrito {args.salida}: {len(datos['glosas'])} glosas, {len(datos['letras'])} letras, "
          f"{args.salida.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
