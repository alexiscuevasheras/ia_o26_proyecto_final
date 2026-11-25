"""Convierte las sesiones de web/grabar.html al formato de extraer.py.

Uso:
    python -m lsm.palabras.importar_propios datos/propios/*.json --salida datos/secuencias_propias.npz

Cada sesión trae el alias de quien grabó (no su nombre) y, por repetición,
la glosa y sus cuadros: cuerpo (33 puntos con visibilidad), dos manos
(ya asignadas a los brazos) y sus coordenadas 3D. Solo puntos, nunca video.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from .extraer import Muestra, guardar


def a_arreglo(valor, forma):
    return np.full(forma, np.nan, np.float32) if valor is None else np.asarray(valor, np.float32)


def resultados_de_sesion(ruta):
    sesion = json.loads(Path(ruta).read_text())
    alias = sesion["alias"]
    resultados = []
    for n, muestra in enumerate(sesion["muestras"]):
        cuadros = muestra["cuadros"]
        por_cuadro = [
            np.stack([a_arreglo(c["cuerpo"], (33, 4)) for c in cuadros]),
            np.stack([[a_arreglo(m, (21, 3)) for m in c["manos"]] for c in cuadros]),
            np.stack([a_arreglo(c["cuerpo3d"], (33, 3)) for c in cuadros]),
            np.stack([[a_arreglo(m, (21, 3)) for m in c["manos3d"]] for c in cuadros]),
        ]
        ident = f"{Path(ruta).stem}/{n:04d}"
        resultados.append((Muestra("propio", muestra["glosa"], alias, ident, ()), (por_cuadro, sesion["aspecto"])))
    return resultados


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("sesiones", type=Path, nargs="+")
    ap.add_argument("--salida", type=Path, default=Path("datos/secuencias_propias.npz"))
    args = ap.parse_args()
    resultados = [r for ruta in args.sesiones for r in resultados_de_sesion(ruta)]
    ok, _ = guardar(args.salida, resultados)
    print(f"escrito {args.salida}: {ok} repeticiones de {len({m.persona for m, _ in resultados})} personas")


if __name__ == "__main__":
    main()
