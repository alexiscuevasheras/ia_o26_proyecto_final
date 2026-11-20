"""Nombres de archivo de D1 y particiones por participante.

Ruta en el 7z: MSL-ABC/<grupo>/<train|test>/<Letra>/S<id>-<Letra>-<código>-<n>.jpg
Todas las fotos de un participante van al mismo conjunto: si no, la red
podría reconocer a la persona y no la seña, y la validación saldría inflada.
"""

import re
from dataclasses import dataclass

import numpy as np

PATRON_D1 = re.compile(
    r"(?P<grupo>lsm-abc-[ABC])/(?P<particion>train|test)/(?P<letra>[A-Z])/"
    r"S(?P<participante>\d+)-(?P=letra)-(?P<codigo>\d+)-(?P<n>\d+)\.jpg$"
)
LETRAS_ESTATICAS = tuple("ABCDEFGHILMNOPRSTUVWY")
PARTICIPANTES_PRUEBA = frozenset({18, 20})


@dataclass(frozen=True)
class FotoD1:
    ruta: str
    grupo: str  # "A", "B" o "C"
    particion: str  # "train" o "test", la oficial del dataset
    letra: str
    participante: int
    codigo: int
    n: int


def leer_nombre(ruta):
    """Interpreta una ruta de D1; lanza ValueError si no tiene el formato."""
    m = PATRON_D1.search(ruta)
    if m is None:
        raise ValueError(f"ruta fuera del formato de D1: {ruta}")
    return FotoD1(
        ruta=ruta,
        grupo=m["grupo"][-1],
        particion=m["particion"],
        letra=m["letra"],
        participante=int(m["participante"]),
        codigo=int(m["codigo"]),
        n=int(m["n"]),
    )


def pliegues_por_participante(participantes, k, semilla=0):
    """Reparte los participantes (sin repetir) en k pliegues.
    Devuelve una lista de k conjuntos de ids."""
    ids = np.array(sorted(set(participantes)))
    if k < 2 or k > len(ids):
        raise ValueError(f"k = {k} no sirve con {len(ids)} participantes")
    np.random.default_rng(semilla).shuffle(ids)
    return [set(map(int, parte)) for parte in np.array_split(ids, k)]


def verificar_sin_fugas(conjuntos):
    """conjuntos: {nombre: iterable de ids de participante}. Lanza
    ValueError si un participante aparece en dos conjuntos."""
    visto_en = {}
    for nombre, ids in conjuntos.items():
        for p in set(ids):
            if p in visto_en:
                raise ValueError(f"el participante {p} está en «{visto_en[p]}» y en «{nombre}»")
            visto_en[p] = nombre
