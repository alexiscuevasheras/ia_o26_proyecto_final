"""Revisa modelos/animaciones.json en busca de poses imposibles del avatar.

Cuenta los clips con la muñeca más arriba que la coronilla, con algún punto
de la mano dentro de la cabeza, o con una mano que casi no se ve (menos de
30 % de los cuadros), y lista los peores con la grabación de la que salen.
Sirve para comprobar que un cambio en animaciones.py no rompe el avatar
antes de mirarlo en web/voz.html.

Uso:
    python -m lsm.palabras.revisar_animaciones modelos/animaciones.json
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from .animaciones import CABEZA_CENTRO, CABEZA_RADIOS, CORONILLA

POCA_PRESENCIA = 0.3


def revisar(clips, indices):
    munecas = [indices.index(15), indices.index(16)]
    problemas = []
    for glosa, clip in clips.items():
        cuerpo = np.array([c["cuerpo"] for c in clip["cuadros"]])
        altura = float(cuerpo[:, munecas, 1].max())
        presencia = np.array([[m is not None for m in c["manos"]] for c in clip["cuadros"]]).mean(axis=0)
        puntos = [np.array(m) for c in clip["cuadros"] for m in c["manos"] if m is not None]
        en_cabeza = any((((p - CABEZA_CENTRO) / CABEZA_RADIOS) ** 2).sum(axis=1).min() < 1 for p in puntos)
        problemas.append({
            "glosa": glosa, "origen": clip["origen"],
            "sobre_la_coronilla": altura > CORONILLA - 0.02,
            "mano_en_la_cabeza": bool(en_cabeza),
            "mano_casi_no_vista": bool(presencia.min() < POCA_PRESENCIA and presencia.max() > 0),
            "altura_maxima": round(altura, 3),
        })
    return problemas


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("animaciones", type=Path)
    args = ap.parse_args()

    datos = json.loads(args.animaciones.read_text())
    problemas = revisar(datos["glosas"], datos["cuerpo_indices"])
    for clave in ("sobre_la_coronilla", "mano_en_la_cabeza", "mano_casi_no_vista"):
        malos = [p for p in problemas if p[clave]]
        por_dataset = Counter(p["origen"].split("-")[0] for p in malos)
        print(f"{clave}: {len(malos)} de {len(problemas)} {dict(por_dataset)}")
        for p in sorted(malos, key=lambda p: -p["altura_maxima"])[:5]:
            print(f"   {p['glosa']:<14} {p['origen']}  muñeca hasta {p['altura_maxima']} m")


if __name__ == "__main__":
    main()
