"""Las secuencias extraídas, listas para entrenar: X (n, T, D), glosas y personas.

Prueba final (fijada antes de ver ningún resultado, como en las letras):
las dos personas de UAEMex con número más alto. No se usan para elegir nada.

ITESO no dice quién es quién. Los nombres de archivo «12», «12_1», «12_2»
sugieren que el sufijo es la persona (la página dice que son 3); es una
hipótesis que se revisa con las proporciones del cuerpo (ver DATOS.md).
"""

import csv
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .secuencias import (T, centro_de_hombros, corregir_aspecto, escala_de_palmas, mano_que_mas_se_mueve,
                         preparar_sena, secuencia)

VOCABULARIO = Path(__file__).parent / "vocabulario.csv"


def leer_vocabulario(ruta=VOCABULARIO):
    """{(dataset, clase): fila} con glosa, sinónimos y tema."""
    with open(ruta, newline="", encoding="utf-8") as f:
        return {(fila["dataset"], fila["clase"]): fila for fila in csv.DictReader(f)}


def persona_iteso(ident):
    """'Comer/12_1' → 'iteso-1'; 'Comer/12' → 'iteso-0'; 'Cansado/IMG_5922' → 'iteso-img'."""
    nombre = ident.rsplit("/", 1)[-1]
    if nombre.upper().startswith("IMG"):
        return "iteso-img"
    m = re.fullmatch(r"\s*\d+(?:_(\d+))?", nombre)
    return f"iteso-{m[1] or 0}" if m else "iteso-?"


@dataclass
class Secuencias:
    X: np.ndarray  # (n, T, D)
    glosa: np.ndarray  # (n,) texto
    persona: np.ndarray  # (n,) texto, con prefijo del dataset
    dataset: np.ndarray
    ident: np.ndarray

    def filtrar(self, mascara):
        return Secuencias(*(getattr(self, c)[mascara] for c in ("X", "glosa", "persona", "dataset", "ident")))

    def __len__(self):
        return len(self.glosa)


def leer_secuencias(ruta):
    """El .npz de extraer.py ya en memoria: indexar un .npz comprimido lo
    descomprime completo en cada acceso (cargar UAEMex tardaba 4 minutos)."""
    with np.load(ruta) as z:
        return {k: z[k] for k in z.files}


def cortes(z):
    inicio = z["inicio"]
    return [(inicio[i], inicio[i + 1]) for i in range(len(inicio) - 1)]


def personas_de(z):
    if z["dataset"][0] == "uaemex":
        return np.array([f"uaemex-{p}" for p in z["persona"]])
    return np.array([persona_iteso(i) for i in z["ident"]])


def dominancia_por_persona(z, personas, centros=None):
    """{persona: 0 o 1} por mayoría de la mano que más se mueve en sus señas
    (con las manos ya seguidas, ver secuencias.seguir_manos)."""
    centros = centros or {}
    votos = {}
    for i, ((a, b), p) in enumerate(zip(cortes(z), personas)):
        manos, *_ = preparar_sena(z["cuerpo"][a:b], z["manos"][a:b], float(z["aspecto"][i]), centros.get(p))
        votos.setdefault(p, Counter())[mano_que_mas_se_mueve(manos)] += 1
    return {p: c.most_common(1)[0][0] for p, c in votos.items()}


def centros_por_persona(z, personas):
    """{persona: centro de hombros en todas sus señas} para UAEMex, donde cada
    persona se grabó con la cámara fija; sirve de respaldo a las señas en que
    no hubo ningún cuadro con hombros plausibles. En ITESO (teléfono en mano)
    no aplica y se devuelve {}."""
    if z["dataset"][0] != "uaemex":
        return {}
    centros = {}
    todos = cortes(z)
    for p in set(personas):
        partes = [corregir_aspecto(z["cuerpo"][a:b], z["manos"][a:b], float(z["aspecto"][i]))
                  for i, (a, b) in enumerate(todos) if personas[i] == p]
        cuerpo = np.concatenate([c for c, _ in partes])
        manos = np.concatenate([m for _, m in partes])
        centro = centro_de_hombros(cuerpo, escala_de_palmas(manos) or 1.0)
        if centro is not None:
            centros[p] = centro
    return centros


def cargar(rutas, t=T):
    """Une uno o más .npz de extraer.py en un solo conjunto."""
    vocabulario = leer_vocabulario()
    partes = []
    for ruta in rutas:
        z = leer_secuencias(ruta)
        personas = personas_de(z)
        respaldo = centros_por_persona(z, personas)
        dominante = dominancia_por_persona(z, personas, respaldo)
        X = np.stack([secuencia(z["cuerpo"][a:b], z["manos"][a:b], float(z["aspecto"][i]), dominante[personas[i]], t,
                                respaldo.get(personas[i])) for i, (a, b) in enumerate(cortes(z))])
        glosas = np.array([vocabulario[(d, c)]["glosa"] for d, c in zip(z["dataset"], z["clase"])])
        partes.append(Secuencias(X, glosas, personas, z["dataset"], z["ident"]))
    return Secuencias(*(np.concatenate([getattr(p, c) for p in partes])
                        for c in ("X", "glosa", "persona", "dataset", "ident")))


def personas_de_prueba(personas):
    """Las dos personas de UAEMex con número más alto."""
    uaemex = sorted({p for p in personas if p.startswith("uaemex-")}, key=lambda p: int(p.split("-")[1]))
    return set(uaemex[-2:])
