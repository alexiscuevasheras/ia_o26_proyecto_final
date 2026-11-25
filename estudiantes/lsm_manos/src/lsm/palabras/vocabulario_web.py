"""Genera web/vocabulario.json (texto → glosas) a partir de src/lsm/palabras/vocabulario.csv.

Uso:
    python -m lsm.palabras.vocabulario_web web/vocabulario.json

Una glosa puede venir de los dos datasets (AVION, AYUDAR…): se juntan sus
sinónimos sin repetir y se queda el tema de la primera fila. El orden del CSV
se conserva porque la página resuelve los sinónimos repetidos («alto» es ALTO
y PARAR) con la primera glosa que los trae; aquí solo se avisa.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from .conjunto import leer_vocabulario


def vocabulario_web(filas):
    """{"glosas": {GLOSA: {"tema", "sinonimos"}}} en el orden del CSV."""
    glosas = {}
    for fila in filas:
        entrada = glosas.setdefault(fila["glosa"], {"tema": fila["tema"], "sinonimos": []})
        for sinonimo in fila["sinonimos"].split("|"):
            sinonimo = sinonimo.strip().lower()
            if sinonimo and sinonimo not in entrada["sinonimos"]:
                entrada["sinonimos"].append(sinonimo)
    return {"glosas": glosas}


def sinonimos_repetidos(vocabulario):
    """{sinónimo: [glosas]} de los que aparecen en más de una glosa."""
    duenos = defaultdict(list)
    for glosa, entrada in vocabulario["glosas"].items():
        for sinonimo in entrada["sinonimos"]:
            duenos[sinonimo].append(glosa)
    return {s: g for s, g in duenos.items() if len(g) > 1}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("salida", type=Path)
    args = parser.parse_args()

    vocabulario = vocabulario_web(leer_vocabulario().values())
    args.salida.write_text(json.dumps(vocabulario, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"{len(vocabulario['glosas'])} glosas → {args.salida}")
    for sinonimo, glosas in sinonimos_repetidos(vocabulario).items():
        print(f"  aviso: «{sinonimo}» está en {', '.join(glosas)}; la página usa {glosas[0]}")


if __name__ == "__main__":
    main()
