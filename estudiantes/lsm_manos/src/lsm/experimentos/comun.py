"""Lo que comparten los experimentos: los datos de entrenamiento y los pliegues.

Solo se usa la partición oficial `train` (18 participantes); la prueba oficial
(participantes 18 y 20) no se toca. Las fotos de cada participante son
fotogramas casi repetidos de un video, así que cada pliegue entrena con una
muestra al azar de FRACCION de sus fotos (la validación usa todas).
"""

import platform

import numpy as np

from ..datos import pliegues_por_participante, verificar_sin_fugas
from ..entrenar import cargar_landmarks

PLIEGUES = 6
FRACCION = 0.25
SEMILLA = 0


def datos_de_entrenamiento(ruta_csv):
    datos = cargar_landmarks(ruta_csv)
    return datos.filtrar(datos.particion == "train"), datos.n_sin_mano


def indices_por_pliegue(persona):
    """[(validacion, idx_ent, idx_val), …]: los mismos pliegues y muestras
    para todos los experimentos."""
    pliegues = pliegues_por_participante(persona, PLIEGUES, semilla=SEMILLA)
    rng = np.random.default_rng(SEMILLA)
    resultado = []
    for validacion in pliegues:
        en_val = np.isin(persona, list(validacion))
        verificar_sin_fugas({"entrenamiento": persona[~en_val], "validacion": persona[en_val]})
        idx_ent = np.flatnonzero(~en_val)
        idx_ent = rng.choice(idx_ent, int(FRACCION * len(idx_ent)), replace=False)
        resultado.append((sorted(int(p) for p in validacion), idx_ent, np.flatnonzero(en_val)))
    return resultado


def resumir(valores):
    return {"media": float(np.mean(valores)), "desviacion": float(np.std(valores)), "por_pliegue": valores}


def entorno():
    return {"python": platform.python_version(), "numpy": np.__version__}
