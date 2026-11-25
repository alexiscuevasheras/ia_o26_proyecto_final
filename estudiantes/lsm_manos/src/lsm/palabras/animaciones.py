"""Genera modelos/animaciones.json para el avatar (formato en docs/FORMATOS.md).

Por glosa se elige el medoide: la repetición cuya secuencia de features
está, en promedio, más cerca de las demás de esa glosa. No se promedian
repeticiones porque el promedio de dos movimientos emborrona los dos. Se
prefiere ITESO (video continuo a 15 fps) sobre UAEMex (~14 fotos sueltas
por seña) y ambos sobre el dataset propio, y solo se eligen repeticiones donde casi siempre se vio el cuerpo.

Coordenadas: las «world» de MediaPipe (metros, x a la derecha de la imagen,
y hacia abajo, z alejándose de la cámara). En una cámara sin espejo la
derecha de la imagen es la izquierda de la persona, así que el avatar usa
(x, −y, −z) con origen entre los hombros, y cada mano se coloca con su
muñeca sobre la muñeca del cuerpo.

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
MUNECAS = (15, 16)
PALMA_METROS = 0.09
CUADROS_UAEMEX = 18  # sus ~14 fotos no tienen tiempo real; 18 cuadros a 15 fps ≈ 1.2 s
MINIMO_CUERPO = 0.8
LETRAS_CON_MOVIMIENTO = ["J", "K", "Ñ", "Q", "X", "Z"]
EJES = np.array([1.0, -1.0, -1.0])
# Video continuo de señantes primero; el dataset propio (estudiantes que aprenden LSM)
# solo para glosas que no estén en los otros.
PRIORIDAD = {"iteso": 2, "uaemex": 1, "propio": 0}


def interpolar_faltantes(a):
    """Rellena NaN a lo largo del tiempo (eje 0) con interpolación lineal por coordenada."""
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


def clip_de_muestra(cuerpo_3d, manos_3d, n_cuadros=None):
    """Lista de cuadros {cuerpo, manos} en el sistema del avatar."""
    cuerpo = suavizar(interpolar_faltantes(cuerpo_3d)) * EJES
    cuerpo -= cuerpo[:, [11, 12]].mean(axis=1, keepdims=True)
    presente = ~np.isnan(manos_3d[:, :, 0, 0])
    manos = suavizar(interpolar_faltantes(manos_3d)) * EJES
    if n_cuadros:
        cuerpo, manos = remuestrear_en_tiempo(cuerpo, n_cuadros), remuestrear_en_tiempo(manos, n_cuadros)
        presente = remuestrear_en_tiempo(presente.astype(float), n_cuadros) > 0.5
    cuadros = []
    for c, m, p in zip(cuerpo, manos, presente):
        lista_manos = []
        for lado in (0, 1):
            if not p[lado] or np.isnan(m[lado]).any():
                lista_manos.append(None)
                continue
            colocada = m[lado] - m[lado, 0] + c[MUNECAS[lado]]
            lista_manos.append(colocada.round(4).tolist())
        cuadros.append({"cuerpo": c[CUERPO_INDICES].round(4).tolist(), "manos": lista_manos})
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
            cuerpo_visto = np.isfinite(z["cuerpo_3d"][a:b, 11, 0]).mean()
            if cuerpo_visto < MINIMO_CUERPO:
                continue
            glosa = glosa_de(vocabulario, z["dataset"][i], z["clase"][i])
            centro = respaldo.get(personas[i])
            X = secuencia(z["cuerpo"][a:b], z["manos"][a:b], float(z["aspecto"][i]), dominante[personas[i]], 16, centro)
            *_, origen = preparar_sena(z["cuerpo"][a:b], z["manos"][a:b], float(z["aspecto"][i]), centro)
            manos_3d = reordenar(z["manos_3d"][a:b], origen)  # mismas manos por brazo que en las features
            candidatos.setdefault(glosa, []).append(
                (PRIORIDAD[z["dataset"][i]], X, (z["cuerpo_3d"][a:b], manos_3d, None if es_iteso else CUADROS_UAEMEX),
                 f"{personas[i]}/{z['ident'][i]}"))
    glosas = {}
    for glosa, lista in sorted(candidatos.items()):
        mejor = max(p for p, *_ in lista)
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
