"""P3 · ¿Sirve lo aprendido en un dataset para otro grabado distinto?

UAEMex (fondo negro, cámara fija, fotos sueltas, Estado de México) e ITESO
(videos de teléfono, condiciones no controladas, Jalisco) comparten algunas
glosas. Se entrena la GRU con uno y se evalúa en el otro, solo en las glosas
compartidas:

- «vocabulario completo»: la red elige entre todas sus glosas (lo que pasa en
  la página);
- «solo compartidas»: la red elige solo entre las compartidas (separa el
  cambio de dominio de la confusión con glosas que el otro no tiene).

Advertencia: la misma glosa puede tener variantes regionales distintas en
los dos datasets; un error aquí puede ser una seña diferente y no un fallo.

Uso:
    python -m lsm.palabras.experimentos.p3_entre_datasets datos/secuencias_uaemex.npz datos/secuencias_iteso.npz
"""

import argparse
import json
from pathlib import Path

import numpy as np

from ...experimentos.comun import entorno
from ..conjunto import cargar, personas_de_prueba
from ..modelos import ModeloGRU


def evaluar_direccion(origen, destino, compartidas):
    glosas = sorted(set(origen.glosa))
    indice = {g: k for k, g in enumerate(glosas)}
    modelo = ModeloGRU().ajustar(origen.X, np.array([indice[g] for g in origen.glosa]), len(glosas))
    destino = destino.filtrar(np.isin(destino.glosa, compartidas))
    P = modelo.probabilidades(destino.X)
    y = np.array([indice[g] for g in destino.glosa])
    columnas = [indice[g] for g in compartidas]
    restringida = np.array(columnas)[P[:, columnas].argmax(axis=1)]
    return {
        "secuencias_evaluadas": len(destino),
        "top1_vocabulario_completo": float(np.mean(P.argmax(axis=1) == y)),
        "top1_solo_compartidas": float(np.mean(restringida == y)),
        "azar_solo_compartidas": 1 / len(compartidas),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("uaemex", type=Path)
    ap.add_argument("iteso", type=Path)
    ap.add_argument("--salida", type=Path, default=Path("resultados/p3_entre_datasets.json"))
    args = ap.parse_args()

    uaemex, iteso = cargar([args.uaemex]), cargar([args.iteso])
    uaemex = uaemex.filtrar(~np.isin(uaemex.persona, list(personas_de_prueba(uaemex.persona))))
    compartidas = sorted(set(uaemex.glosa) & set(iteso.glosa))
    print(f"glosas compartidas ({len(compartidas)}): {compartidas}", flush=True)
    resultados = {
        "uaemex_a_iteso": evaluar_direccion(uaemex, iteso, compartidas),
        "iteso_a_uaemex": evaluar_direccion(iteso, uaemex, compartidas),
    }
    for k, v in resultados.items():
        print(k, v, flush=True)
    args.salida.parent.mkdir(parents=True, exist_ok=True)
    args.salida.write_text(json.dumps({
        "experimento": "P3 de un dataset al otro, en las glosas compartidas",
        "glosas_compartidas": compartidas,
        "resultados": resultados,
        "entorno": entorno(),
    }, indent=2, ensure_ascii=False))
    print(f"escrito {args.salida}")


if __name__ == "__main__":
    main()
