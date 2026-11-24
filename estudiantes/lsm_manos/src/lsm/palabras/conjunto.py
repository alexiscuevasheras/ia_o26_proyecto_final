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

from .secuencias import T, mano_que_mas_se_mueve, secuencia

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


def cortes(z):
    inicio = z["inicio"]
    return [(inicio[i], inicio[i + 1]) for i in range(len(inicio) - 1)]


def personas_de(z):
    if z["dataset"][0] == "uaemex":
        return np.array([f"uaemex-{p}" for p in z["persona"]])
    return np.array([persona_iteso(i) for i in z["ident"]])


def dominancia_por_persona(z, personas):
    """{persona: 0 o 1} por mayoría de la mano que más se mueve en sus señas."""
    votos = {}
    for (a, b), p in zip(cortes(z), personas):
        votos.setdefault(p, Counter())[mano_que_mas_se_mueve(z["manos"][a:b])] += 1
    return {p: c.most_common(1)[0][0] for p, c in votos.items()}


def cargar(rutas, t=T):
    """Une uno o más .npz de extraer.py en un solo conjunto."""
    vocabulario = leer_vocabulario()
    partes = []
    for ruta in rutas:
        z = np.load(ruta)
        personas = personas_de(z)
        dominante = dominancia_por_persona(z, personas)
        X = np.stack([secuencia(z["cuerpo"][a:b], z["manos"][a:b], float(z["aspecto"][i]), dominante[personas[i]], t)
                      for i, (a, b) in enumerate(cortes(z))])
        glosas = np.array([vocabulario[(d, c)]["glosa"] for d, c in zip(z["dataset"], z["clase"])])
        partes.append(Secuencias(X, glosas, personas, z["dataset"], z["ident"]))
    return Secuencias(*(np.concatenate([getattr(p, c) for p in partes])
                        for c in ("X", "glosa", "persona", "dataset", "ident")))


def personas_de_prueba(personas):
    """Las dos personas de UAEMex con número más alto."""
    uaemex = sorted({p for p in personas if p.startswith("uaemex-")}, key=lambda p: int(p.split("-")[1]))
    return set(uaemex[-2:])
