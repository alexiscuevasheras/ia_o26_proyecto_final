"""P4 · ¿Cuánto mejora la GRU con k ejemplos por palabra de la persona nueva?

UAEMex tiene una sola repetición por persona y palabra, así que no permite
adaptar y evaluar en la misma persona. ITESO tiene ~17 por persona y palabra
(si el sufijo del archivo es la persona; ver conjunto.py). Para cada persona
de ITESO:

- red base: UAEMex (sin las personas de prueba) + las otras personas de ITESO;
- adaptación: k repeticiones al azar por glosa de esa persona;
- evaluación: el resto de sus repeticiones.

Se repite con 3 semillas y se promedia. k = 0 es la red base en las mismas
señas de evaluación.

Uso:
    python -m lsm.palabras.experimentos.p4_adaptacion datos/secuencias_uaemex.npz datos/secuencias_iteso.npz
"""

import argparse
import json
from pathlib import Path

import numpy as np

from ...experimentos.comun import entorno
from ..adaptar import adaptar_salida
from ..modelos import ModeloGRU
from .comun import preparar

KS = [0, 1, 2, 5]
SEMILLAS = [0, 1, 2]
K_MAXIMO = max(KS)


def particion(y, k, rng):
    """k índices por glosa para adaptar (de las que tienen más de K_MAXIMO) y el resto."""
    adaptar, evaluar = [], []
    for g in np.unique(y):
        idx = rng.permutation(np.flatnonzero(y == g))
        if len(idx) <= K_MAXIMO:
            continue  # sin repeticiones suficientes no se puede evaluar igual para todo k
        adaptar.extend(idx[:k])
        evaluar.extend(idx[K_MAXIMO:])  # misma evaluación para todo k
    return np.array(adaptar, int), np.array(evaluar, int)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("uaemex", type=Path)
    ap.add_argument("iteso", type=Path)
    ap.add_argument("--salida", type=Path, default=Path("resultados/p4_adaptacion.json"))
    args = ap.parse_args()

    datos, y, glosas, _ = preparar([args.uaemex, args.iteso])
    personas = sorted({p for p in datos.persona if p.startswith("iteso-") and p not in ("iteso-img", "iteso-?")})
    resultados = {}
    for p in personas:
        suyas = datos.persona == p
        base = ModeloGRU().ajustar(datos.X[~suyas], y[~suyas], len(glosas))
        Xp, yp = datos.X[suyas], y[suyas]
        por_k = {k: [] for k in KS}
        for semilla in SEMILLAS:
            for k in KS:
                idx_a, idx_e = particion(yp, k, np.random.default_rng(semilla))
                modelo = base if k == 0 else adaptar_salida(base, Xp[idx_a], yp[idx_a])
                por_k[k].append(float(np.mean(modelo.probabilidades(Xp[idx_e]).argmax(axis=1) == yp[idx_e])))
        resultados[p] = {str(k): float(np.mean(v)) for k, v in por_k.items()}
        print(p, resultados[p], flush=True)

    args.salida.parent.mkdir(parents=True, exist_ok=True)
    args.salida.write_text(json.dumps({
        "experimento": "P4 adaptación de la capa de salida de la GRU con k ejemplos por glosa (ITESO)",
        "supuesto": "el sufijo del nombre de archivo de ITESO identifica a la persona (sin confirmar)",
        "ks": KS, "semillas": SEMILLAS,
        "top1_por_persona": resultados,
        "top1_medio_por_k": {str(k): float(np.mean([r[str(k)] for r in resultados.values()])) for k in KS},
        "entorno": entorno(),
    }, indent=2, ensure_ascii=False))
    print(f"escrito {args.salida}")


if __name__ == "__main__":
    main()
