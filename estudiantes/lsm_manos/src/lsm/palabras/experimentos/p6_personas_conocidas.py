"""P6 · ¿Cuánto cae la GRU cuando la persona es nueva?

Diagnóstico, no resultado de generalización: se divide cada dataset al azar
(80/20, estratificado por glosa), así que las mismas personas aparecen en
entrenamiento y en evaluación. Comparado con P1 (dejando una persona fuera)
separa dos causas de error:

- si aquí la red acierta mucho, la red y la entrada sirven, y lo que falla
  es generalizar a personas que no ha visto;
- si aquí también falla, el problema está en los datos o en la entrada.

UAEMex tiene ~9 ejemplos por glosa (uno por persona) e ITESO ~50. Para
separar la cantidad de la calidad, «iteso_pocos» entrena ITESO con tantos
ejemplos por glosa como UAEMex (POCOS) y evalúa en la misma división.

En UAEMex se excluyen las personas de prueba, como en P1. ITESO se usa
completo: no se sabe quién es quién, que es justo por lo que no entra en P1.

Uso:
    python -m lsm.palabras.experimentos.p6_personas_conocidas datos/secuencias_uaemex.npz datos/secuencias_iteso.npz
"""

import argparse
import json
from pathlib import Path

import numpy as np

from ...experimentos.comun import entorno
from ..conjunto import cargar, personas_de_prueba
from ..modelos import ModeloGRU
from .comun import evaluar, resumen_con_peor

SEMILLAS = (0, 1, 2)
POCOS = 7  # ejemplos de entrenamiento por glosa en UAEMex con la división 80/20


def division_estratificada(y, fraccion, semilla):
    """Máscara de evaluación con `fraccion` de cada clase (al menos una si la
    clase tiene dos o más ejemplos)."""
    rng = np.random.default_rng(semilla)
    evaluacion = np.zeros(len(y), bool)
    for clase in np.unique(y):
        indices = rng.permutation(np.flatnonzero(y == clase))
        if len(indices) >= 2:
            evaluacion[indices[:max(1, round(fraccion * len(indices)))]] = True
    return evaluacion


def recortar(entrenar, y, maximo, semilla):
    """Deja a lo más `maximo` ejemplos de entrenamiento por clase."""
    rng = np.random.default_rng(semilla)
    recortada = np.zeros_like(entrenar)
    for clase in np.unique(y[entrenar]):
        recortada[rng.permutation(np.flatnonzero(entrenar & (y == clase)))[:maximo]] = True
    return recortada


def evaluar_dataset(datos, maximo_por_glosa=None):
    glosas = sorted(set(datos.glosa))
    indice = {g: k for k, g in enumerate(glosas)}
    y = np.array([indice[g] for g in datos.glosa])
    por_semilla = {}
    for semilla in SEMILLAS:
        es_eval = division_estratificada(y, 0.2, semilla)
        entrenar = ~es_eval
        if maximo_por_glosa:
            entrenar = recortar(entrenar, y, maximo_por_glosa, semilla)
        modelo = ModeloGRU().ajustar(datos.X[entrenar], y[entrenar], len(glosas))
        por_semilla[semilla] = evaluar(modelo.probabilidades(datos.X[es_eval]), y[es_eval], len(glosas))
        print(f"  semilla {semilla}: {por_semilla[semilla]}", flush=True)
    return {
        "glosas": len(glosas),
        "secuencias": len(datos),
        "maximo_por_glosa_al_entrenar": maximo_por_glosa,
        "resumen": {m: resumen_con_peor([r[m] for r in por_semilla.values()]) for m in ("top1", "top5", "f1_macro")},
        "por_semilla": por_semilla,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("uaemex", type=Path)
    ap.add_argument("iteso", type=Path)
    ap.add_argument("--salida", type=Path, default=Path("resultados/p6_personas_conocidas.json"))
    args = ap.parse_args()

    uaemex = cargar([args.uaemex])
    uaemex = uaemex.filtrar(~np.isin(uaemex.persona, list(personas_de_prueba(uaemex.persona))))
    iteso = cargar([args.iteso])
    resultados = {}
    for nombre, datos, maximo in (("uaemex", uaemex, None), ("iteso", iteso, None), ("iteso_pocos", iteso, POCOS)):
        print(nombre, flush=True)
        resultados[nombre] = evaluar_dataset(datos, maximo)
    args.salida.parent.mkdir(parents=True, exist_ok=True)
    args.salida.write_text(json.dumps({
        "experimento": "P6 diagnóstico con personas conocidas: división al azar 80/20 por glosa",
        "advertencia": "las mismas personas están en entrenamiento y evaluación; no mide generalización",
        "resultados": resultados,
        "entorno": entorno(),
    }, indent=2, ensure_ascii=False))
    print(f"escrito {args.salida}")


if __name__ == "__main__":
    main()
