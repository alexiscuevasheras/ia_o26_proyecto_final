"""E8 · ¿Cuánto mejora la red con k ejemplos por letra de la persona nueva?

Mismos 6 pliegues por participante. En cada pliegue se entrena la red base
con la configuración de E2 sin ver a los participantes de validación. Para cada
uno de ellos, y para cada una de sus 5 grabaciones (códigos):

- adaptación: k fotos al azar por letra, solo de esa grabación;
- evaluación: todas sus fotos de las otras 4 grabaciones.

Así ninguna foto de evaluación es un cuadro vecino de una foto de adaptación.
k = 0 es la red base evaluada en las mismas fotos. Se reporta el F1 macro por
participante (promedio de sus 5 grabaciones) y la media sobre los 18.

Uso:
    python -m lsm.experimentos.e8_adaptacion datos/landmarks_d1.csv
"""

import argparse
import json
from pathlib import Path

import numpy as np

from ..adaptar import ETA, PASOS, adaptar_ultima_capa, ejemplos_por_letra
from ..entrenar import K, entrenar_red, predecir_letras
from ..evaluar import metricas
from .comun import FRACCION, SEMILLA, datos_de_entrenamiento, entorno, indices_por_pliegue, resumir
from .e3_ablaciones import config_elegida

KS = [0, 1, 2, 5, 10, 20]


def f1(red, X, y):
    return metricas(y, predecir_letras(red, X), K)["f1_macro"]


def curva_de_un_participante(red, X, y, codigo, rng):
    """F1 para cada k, promediado sobre las grabaciones usadas para adaptar."""
    por_k = {k: [] for k in KS}
    for c in np.unique(codigo):
        adaptacion, evaluacion = codigo == c, codigo != c
        for k in KS:
            if k == 0:
                por_k[k].append(f1(red, X[evaluacion], y[evaluacion]))
                continue
            idx = np.flatnonzero(adaptacion)[ejemplos_por_letra(y[adaptacion], k, rng)]
            adaptada = adaptar_ultima_capa(red, X[idx], y[idx])
            por_k[k].append(f1(adaptada, X[evaluacion], y[evaluacion]))
    return {k: float(np.mean(v)) for k, v in por_k.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("csv", type=Path)
    ap.add_argument("--ajuste", type=Path, default=Path("resultados/e2_ajuste.json"))
    ap.add_argument("--salida", type=Path, default=Path("resultados/e8_adaptacion.json"))
    args = ap.parse_args()

    config = config_elegida(args.ajuste)
    entrenamiento, _ = datos_de_entrenamiento(args.csv)
    X, y = entrenamiento.features(), entrenamiento.letra
    persona, codigo = entrenamiento.participante, entrenamiento.codigo
    rng = np.random.default_rng(SEMILLA)

    por_participante = {}
    for validacion, idx_ent, _ in indices_por_pliegue(persona):
        red = entrenar_red(X[idx_ent], y[idx_ent], config)
        for p in validacion:
            suyas = persona == p
            curva = curva_de_un_participante(red, X[suyas], y[suyas], codigo[suyas], rng)
            por_participante[p] = curva
            print(f"participante {p}: " + ", ".join(f"k={k} {v:.3f}" for k, v in curva.items()), flush=True)

    participantes = sorted(por_participante)
    resultado = {
        "experimento": "E8 adaptación de la última capa con k ejemplos por letra",
        "diseno": "adaptar con k fotos por letra de una grabación; evaluar con las otras 4 grabaciones de la misma persona",
        "adaptacion": {"capas": "solo W(L), b(L)", "eta": ETA, "pasos": PASOS, "lote": "completo", "castigo": 0},
        "config_red_base": {**vars(config), "anchos_ocultos": list(config.anchos_ocultos)},
        "fraccion_entrenamiento_base": FRACCION,
        "ks": KS,
        "f1_macro_por_k": {str(k): resumir([por_participante[p][k] for p in participantes]) for k in KS},
        "participantes": participantes,
        "entorno": entorno(),
    }
    args.salida.parent.mkdir(parents=True, exist_ok=True)
    args.salida.write_text(json.dumps(resultado, indent=2, ensure_ascii=False))
    print(f"escrito {args.salida}")


if __name__ == "__main__":
    main()
