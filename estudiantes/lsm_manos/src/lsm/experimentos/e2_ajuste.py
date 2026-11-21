"""E2 · Ajuste de anchos, η y μ, y curvas por época con y sin castigo (Q4).

Mismos 6 pliegues por participante que E1. Para cada configuración se guarda,
época por época y promediado en los pliegues, el F1 macro de validación y la
entropía cruzada de entrenamiento y de validación. La configuración elegida es
la de mayor F1 de validación medio (incluida la época). Ese número está
sesgado hacia arriba, porque se eligió mirando esos mismos pliegues; la
estimación sin sesgo será la prueba oficial, que se abre una sola vez al final.

Uso:
    python -m lsm.experimentos.e2_ajuste datos/landmarks_d1.csv
"""

import argparse
import itertools
import json
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np

from redneuronal.perdida import EntropiaCruzada
from redneuronal.red import predecir

from ..entrenar import K, ConfigRed, entrenar_red, una_caliente
from ..evaluar import metricas
from .comun import FRACCION, datos_de_entrenamiento, entorno, indices_por_pliegue

ANCHOS = [(64,), (128, 64), (256, 128)]
ETAS = [0.02, 0.05, 0.1]
MUS = [0.0, 1e-4, 1e-3]
EPOCAS = 40
BASE = ConfigRed(epocas=EPOCAS, lote=64, semilla=0)

_X = _y = None


def _cargar(ruta_csv):
    global _X, _y
    entrenamiento, _ = datos_de_entrenamiento(ruta_csv)
    _X, _y = entrenamiento.features(), entrenamiento.letra


def entropia(red, X, y):
    return float(np.mean(EntropiaCruzada().valor(predecir(red, X.T), una_caliente(y))))


def _curvas(trabajo):
    """Entrena una configuración en un pliegue y mide cada época."""
    config, idx_ent, idx_val = trabajo
    X_ent, y_ent, X_val, y_val = _X[idx_ent], _y[idx_ent], _X[idx_val], _y[idx_val]
    curva = {"f1_validacion": [], "perdida_entrenamiento": [], "perdida_validacion": []}

    def medir(epoca, red):
        y_gorro = predecir(red, X_val.T).argmax(axis=0)
        curva["f1_validacion"].append(metricas(y_val, y_gorro, K)["f1_macro"])
        curva["perdida_entrenamiento"].append(entropia(red, X_ent, y_ent))
        curva["perdida_validacion"].append(entropia(red, X_val, y_val))

    entrenar_red(X_ent, y_ent, config, al_terminar_epoca=medir)
    return curva


def nombre(config):
    capas = "–".join(map(str, (60, *config.anchos_ocultos, K)))
    return f"{capas} η={config.eta} μ={config.mu}"


def resumen(curvas_por_pliegue):
    """Curvas medias y la mejor época según el F1 de validación medio."""
    medias = {clave: np.mean([c[clave] for c in curvas_por_pliegue], axis=0)
              for clave in curvas_por_pliegue[0]}
    mejor = int(np.argmax(medias["f1_validacion"]))
    f1_mejor = [c["f1_validacion"][mejor] for c in curvas_por_pliegue]
    return {
        "mejor_epoca": mejor + 1,
        "f1_mejor_epoca": {"media": float(np.mean(f1_mejor)), "desviacion": float(np.std(f1_mejor)),
                           "por_pliegue": f1_mejor},
        "f1_ultima_epoca": float(medias["f1_validacion"][-1]),
        "curvas_medias": {clave: [round(float(v), 5) for v in valores] for clave, valores in medias.items()},
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("csv", type=Path)
    ap.add_argument("--salida", type=Path, default=Path("resultados/e2_ajuste.json"))
    ap.add_argument("--procesos", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = ap.parse_args()

    entrenamiento, _ = datos_de_entrenamiento(args.csv)
    pliegues = indices_por_pliegue(entrenamiento.participante)
    configs = [replace(BASE, anchos_ocultos=a, eta=e, mu=m) for a, e, m in itertools.product(ANCHOS, ETAS, MUS)]
    trabajos = [(c, idx_ent, idx_val) for c in configs for _, idx_ent, idx_val in pliegues]
    print(f"{len(configs)} configuraciones × {len(pliegues)} pliegues = {len(trabajos)} entrenamientos", flush=True)

    with ProcessPoolExecutor(args.procesos, initializer=_cargar, initargs=(args.csv,)) as ejecutor:
        curvas = list(ejecutor.map(_curvas, trabajos))

    por_config = {}
    for i, config in enumerate(configs):
        r = resumen(curvas[i * len(pliegues):(i + 1) * len(pliegues)])
        por_config[nombre(config)] = {"config": vars(config), **r}
        print(f"{nombre(config)}: F1 {r['f1_mejor_epoca']['media']:.3f} en la época {r['mejor_epoca']}", flush=True)

    elegida = max(por_config, key=lambda n: por_config[n]["f1_mejor_epoca"]["media"])
    resultado = {
        "experimento": "E2 ajuste de hiperparámetros y curvas con y sin μ",
        "advertencia": "el F1 de la configuración elegida está sesgado hacia arriba: se eligió con estos mismos pliegues",
        "pliegues": [v for v, _, _ in pliegues],
        "fraccion_entrenamiento": FRACCION,
        "rejilla": {"anchos_ocultos": ANCHOS, "eta": ETAS, "mu": MUS, "epocas": EPOCAS, "lote": BASE.lote},
        "elegida": {"nombre": elegida, **{k: por_config[elegida][k] for k in ("config", "mejor_epoca", "f1_mejor_epoca")}},
        "configuraciones": por_config,
        "entorno": entorno(),
    }
    args.salida.parent.mkdir(parents=True, exist_ok=True)
    args.salida.write_text(json.dumps(resultado, indent=2, ensure_ascii=False))
    print(f"elegida: {elegida}\nescrito {args.salida}")


if __name__ == "__main__":
    main()
