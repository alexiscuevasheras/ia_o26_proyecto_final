"""E3 · Ablaciones de features (Q3): ¿ayuda rotar la palma? ¿ayuda z?

Usa la configuración que eligió E2 (anchos, η, μ y número de épocas) y los
mismos 6 pliegues por participante; solo cambia el vector de entrada.

Uso:
    python -m lsm.experimentos.e3_ablaciones datos/landmarks_d1.csv
"""

import argparse
import json
from pathlib import Path

from ..entrenar import K, ConfigRed, entrenar_red, predecir_letras
from ..evaluar import metricas
from .comun import FRACCION, datos_de_entrenamiento, entorno, indices_por_pliegue, resumir

VARIANTES = {
    "x, y, z sin rotar (60)": {"rotar": False, "usar_z": True},
    "x, y, z rotado (60)": {"rotar": True, "usar_z": True},
    "x, y sin rotar (40)": {"rotar": False, "usar_z": False},
    "x, y rotado (40)": {"rotar": True, "usar_z": False},
}


def config_elegida(ruta_e2):
    elegida = json.loads(Path(ruta_e2).read_text())["elegida"]
    config = ConfigRed(**elegida["config"])
    config.anchos_ocultos = tuple(config.anchos_ocultos)
    config.epocas = elegida["mejor_epoca"]
    return config


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("csv", type=Path)
    ap.add_argument("--ajuste", type=Path, default=Path("resultados/e2_ajuste.json"))
    ap.add_argument("--salida", type=Path, default=Path("resultados/e3_ablaciones.json"))
    args = ap.parse_args()

    config = config_elegida(args.ajuste)
    entrenamiento, _ = datos_de_entrenamiento(args.csv)
    y = entrenamiento.letra
    pliegues = indices_por_pliegue(entrenamiento.participante)

    por_variante = {}
    for nombre, opciones in VARIANTES.items():
        X = entrenamiento.features(**opciones)
        f1, mcc = [], []
        for _, idx_ent, idx_val in pliegues:
            red = entrenar_red(X[idx_ent], y[idx_ent], config)
            m = metricas(y[idx_val], predecir_letras(red, X[idx_val]), K)
            f1.append(m["f1_macro"])
            mcc.append(m["mcc"])
        por_variante[nombre] = {"opciones": opciones, "f1_macro": resumir(f1), "mcc": resumir(mcc)}
        print(f"{nombre}: F1 macro {por_variante[nombre]['f1_macro']['media']:.3f}", flush=True)

    resultado = {
        "experimento": "E3 ablaciones de features",
        "config_red": {**vars(config), "anchos_ocultos": list(config.anchos_ocultos)},
        "pliegues": [v for v, _, _ in pliegues],
        "fraccion_entrenamiento": FRACCION,
        "variantes": por_variante,
        "entorno": entorno(),
    }
    args.salida.parent.mkdir(parents=True, exist_ok=True)
    args.salida.write_text(json.dumps(resultado, indent=2, ensure_ascii=False))
    print(f"escrito {args.salida}")


if __name__ == "__main__":
    main()
