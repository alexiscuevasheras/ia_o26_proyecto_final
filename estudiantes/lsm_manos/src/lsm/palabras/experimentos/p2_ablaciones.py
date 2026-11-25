"""P2 · ¿Qué parte de la entrada y del entrenamiento importa?

Misma validación que P1 (dejando una persona fuera), solo con la GRU. Cada
variante quita una cosa:

- sin_aumento: entrenar sin el aumento de datos.
- solo_dominante: la mano no dominante en ceros.
- sin_posicion: sin la muñeca en el marco de los hombros (solo la forma).
- sin_forma: sin la forma de la mano (solo dónde está la muñeca).
- T=8 y T=32: otra longitud del remuestreo (16 por omisión).

Uso:
    python -m lsm.palabras.experimentos.p2_ablaciones datos/secuencias_uaemex.npz
"""

import argparse
import json
from functools import partial
from pathlib import Path

from ...experimentos.comun import entorno
from ..modelos import ConfigGRU, ModeloGRU
from ..secuencias import POR_MANO
from .comun import dejando_una_persona_fuera, linea, preparar, resumir_metricas


def sin_columnas(X, columnas):
    X = X.copy()
    X[:, :, columnas] = 0
    return X


NO_DOMINANTE = list(range(POR_MANO, 2 * POR_MANO))
POSICION = [0, 1, POR_MANO, POR_MANO + 1]
FORMA = list(range(2, 62)) + list(range(POR_MANO + 2, POR_MANO + 62))

VARIANTES = {
    "completo": (16, None, ConfigGRU()),
    "sin_aumento": (16, None, ConfigGRU(aumentar=False)),
    "solo_dominante": (16, NO_DOMINANTE, ConfigGRU()),
    "sin_posicion": (16, POSICION, ConfigGRU()),
    "sin_forma": (16, FORMA, ConfigGRU()),
    "T=8": (8, None, ConfigGRU()),
    "T=32": (32, None, ConfigGRU()),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("uaemex", type=Path)
    ap.add_argument("--variantes", nargs="+", default=list(VARIANTES))
    ap.add_argument("--salida", type=Path, default=Path("resultados/p2_ablaciones.json"))
    ap.add_argument("--procesos", type=int, default=6)
    args = ap.parse_args()

    por_T = {}
    resultados = {}
    for nombre in args.variantes:
        t, columnas, config = VARIANTES[nombre]
        if t not in por_T:
            por_T[t] = preparar([args.uaemex], t)
        datos, y, glosas, _ = por_T[t]
        X = datos.X if columnas is None else sin_columnas(datos.X, columnas)
        por_persona, _ = dejando_una_persona_fuera(partial(ModeloGRU, config), datos, X, y, len(glosas), args.procesos)
        resultados[nombre] = {"resumen": resumir_metricas(por_persona), "por_persona": por_persona}
        print(linea(nombre, resultados[nombre]["resumen"]), flush=True)

    args.salida.parent.mkdir(parents=True, exist_ok=True)
    args.salida.write_text(json.dumps({
        "experimento": "P2 ablaciones de la GRU, dejando una persona fuera (UAEMex)",
        "variantes": {k: {"T": VARIANTES[k][0], "config": vars(VARIANTES[k][2])} for k in args.variantes},
        "resultados": resultados,
        "entorno": entorno(),
    }, indent=2, ensure_ascii=False))
    print(f"escrito {args.salida}")


if __name__ == "__main__":
    main()
