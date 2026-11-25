"""P4 · ¿Cuánto mejora la GRU con k ejemplos por palabra de la persona nueva?

Hace falta que cada persona tenga varias repeticiones de cada palabra y que
se sepa quién es quién. UAEMex tiene una sola repetición por persona y
palabra, e ITESO no identifica a las personas (ver docs/DATOS.md), así que
P4 usa el dataset propio grabado con web/grabar.html. Para cada persona del
dataset propio:

- red base: UAEMex (sin las personas de prueba), ITESO si se da, y las otras
  personas del dataset propio;
- adaptación: k repeticiones al azar por glosa de esa persona (solo V y c,
  igual que la página);
- evaluación: las repeticiones que quedan después de las K_MAXIMO primeras,
  las mismas para todo k.

Se repite con 3 semillas y se promedia. k = 0 es la red base en las mismas
señas de evaluación.

Uso:
    python -m lsm.palabras.experimentos.p4_adaptacion datos/secuencias_uaemex.npz \\
        datos/secuencias_propias.npz --iteso datos/secuencias_iteso.npz
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
        evaluar.extend(idx[K_MAXIMO:])
    return np.array(adaptar, int), np.array(evaluar, int)


def curva_de_persona(base, X, y):
    """Top-1 medio por k sobre las semillas."""
    por_k = {k: [] for k in KS}
    for semilla in SEMILLAS:
        for k in KS:
            idx_a, idx_e = particion(y, k, np.random.default_rng(semilla))
            modelo = base if k == 0 else adaptar_salida(base, X[idx_a], y[idx_a])
            por_k[k].append(float(np.mean(modelo.probabilidades(X[idx_e]).argmax(axis=1) == y[idx_e])))
    return {str(k): float(np.mean(v)) for k, v in por_k.items()}, len(idx_e)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("uaemex", type=Path)
    ap.add_argument("propios", type=Path)
    ap.add_argument("--iteso", type=Path)
    ap.add_argument("--salida", type=Path, default=Path("resultados/p4_adaptacion.json"))
    args = ap.parse_args()

    rutas = [args.uaemex, args.propios] + ([args.iteso] if args.iteso else [])
    datos, y, glosas, _ = preparar(rutas)
    personas = sorted({p for p in datos.persona if p.startswith("propio-")})
    resultados = {}
    for p in personas:
        suyas = datos.persona == p
        base = ModeloGRU().ajustar(datos.X[~suyas], y[~suyas], len(glosas))
        curva, evaluadas = curva_de_persona(base, datos.X[suyas], y[suyas])
        resultados[p] = {"top1_por_k": curva, "senas_evaluadas": evaluadas}
        print(p, resultados[p], flush=True)

    args.salida.parent.mkdir(parents=True, exist_ok=True)
    args.salida.write_text(json.dumps({
        "experimento": "P4 adaptación de la capa de salida de la GRU con k ejemplos por glosa (dataset propio)",
        "datos": [str(r) for r in rutas],
        "ks": KS, "semillas": SEMILLAS, "glosas": len(glosas),
        "por_persona": resultados,
        "top1_medio_por_k": {str(k): float(np.mean([r["top1_por_k"][str(k)] for r in resultados.values()]))
                             for k in KS},
        "entorno": entorno(),
    }, indent=2, ensure_ascii=False))
    print(f"escrito {args.salida}")


if __name__ == "__main__":
    main()
