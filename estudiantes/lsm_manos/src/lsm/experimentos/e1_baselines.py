"""E1 · Baselines contra la red, con validación cruzada agrupada por participante.

Solo usa la partición oficial `train` (18 participantes); la prueba oficial
(participantes 18 y 20) no se toca. Escribe resultados/e1_baselines.json.

Uso:
    python -m lsm.experimentos.e1_baselines datos/landmarks_d1.csv

Las fotos de cada participante son fotogramas casi repetidos de un video, así
que se usa una muestra al azar de FRACCION de las fotos de entrenamiento de
cada pliegue (la validación usa todas). Así cada modelo se entrena en minutos.
"""

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from ..datos import pliegues_por_participante, verificar_sin_fugas
from ..entrenar import K, ConfigRed, cargar_landmarks, entrenar_red, predecir_letras
from ..evaluar import metricas

PLIEGUES = 6
FRACCION = 0.25
MUESTRA_SVM = 15_000  # el SVM con kernel RBF crece como n²
SEMILLA = 0
RED = ConfigRed(anchos_ocultos=(128, 64), eta=0.05, mu=1e-4, epocas=20, lote=64)
LOGISTICA = ConfigRed(anchos_ocultos=(), eta=0.05, mu=1e-4, epocas=20, lote=64)


def mayoritaria(X_ent, y_ent, X_val):
    return np.full(len(X_val), np.bincount(y_ent, minlength=K).argmax())


def knn(X_ent, y_ent, X_val):
    return KNeighborsClassifier(n_neighbors=5).fit(X_ent, y_ent).predict(X_val)


def svm(X_ent, y_ent, X_val):
    idx = np.random.default_rng(SEMILLA).choice(len(X_ent), min(MUESTRA_SVM, len(X_ent)), replace=False)
    return SVC(kernel="rbf", C=10, gamma="scale").fit(X_ent[idx], y_ent[idx]).predict(X_val)


def red_numpy(config):
    def modelo(X_ent, y_ent, X_val):
        red = entrenar_red(X_ent, y_ent, config)
        modelo.entrenamiento = predecir_letras(red, X_ent)
        return predecir_letras(red, X_val)
    return modelo


MODELOS = {
    "mayoritaria": mayoritaria,
    "knn_5": knn,
    "svm_rbf": svm,
    "logistica_numpy (60–21)": red_numpy(LOGISTICA),
    "red_numpy (60–128–64–21)": red_numpy(RED),
}


def resumir(valores):
    return {"media": float(np.mean(valores)), "desviacion": float(np.std(valores)), "por_pliegue": valores}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("csv", type=Path)
    ap.add_argument("--salida", type=Path, default=Path("resultados/e1_baselines.json"))
    args = ap.parse_args()

    datos = cargar_landmarks(args.csv)
    entrenamiento = datos.filtrar(datos.particion == "train")
    X, y, persona = entrenamiento.features(), entrenamiento.letra, entrenamiento.participante
    pliegues = pliegues_por_participante(persona, PLIEGUES, semilla=SEMILLA)
    rng = np.random.default_rng(SEMILLA)

    por_modelo = {nombre: {"f1_macro": [], "mcc": [], "exactitud": [], "segundos": []} for nombre in MODELOS}
    brecha_red = []
    for i, validacion in enumerate(pliegues):
        en_val = np.isin(persona, list(validacion))
        verificar_sin_fugas({"entrenamiento": persona[~en_val], "validacion": persona[en_val]})
        idx_ent = np.flatnonzero(~en_val)
        idx_ent = rng.choice(idx_ent, int(FRACCION * len(idx_ent)), replace=False)
        X_ent, y_ent, X_val, y_val = X[idx_ent], y[idx_ent], X[en_val], y[en_val]
        print(f"pliegue {i + 1}: validación {sorted(validacion)}, {len(X_ent)} ent. / {len(X_val)} val.", flush=True)
        for nombre, modelo in MODELOS.items():
            t0 = time.time()
            m = metricas(y_val, modelo(X_ent, y_ent, X_val), K)
            for clave in ("f1_macro", "mcc", "exactitud"):
                por_modelo[nombre][clave].append(m[clave])
            por_modelo[nombre]["segundos"].append(round(time.time() - t0, 1))
            print(f"  {nombre}: F1 macro {m['f1_macro']:.3f}", flush=True)
            if nombre.startswith("red_numpy"):
                f1_ent = metricas(y_ent, modelo.entrenamiento, K)["f1_macro"]
                brecha_red.append({"f1_entrenamiento": f1_ent, "f1_validacion": m["f1_macro"]})

    resultado = {
        "experimento": "E1 baselines contra la red",
        "datos": {"fotos_con_mano_train": int(len(X)), "fotos_sin_mano_total": datos.n_sin_mano,
                  "participantes_train": sorted(map(int, set(persona))), "pliegues": [sorted(p) for p in pliegues],
                  "fraccion_entrenamiento": FRACCION, "muestra_svm": MUESTRA_SVM,
                  "features": "60 (x, y, z de 20 puntos; espejo, centrado, escala; sin rotación)"},
        "config_red": vars(RED), "config_logistica": vars(LOGISTICA),
        "modelos": {nombre: {clave: resumir(v) for clave, v in valores.items()}
                    for nombre, valores in por_modelo.items()},
        "brecha_red_por_pliegue": brecha_red,
        "entorno": {"python": platform.python_version(), "numpy": np.__version__},
    }
    args.salida.parent.mkdir(parents=True, exist_ok=True)
    args.salida.write_text(json.dumps(resultado, indent=2, ensure_ascii=False))
    print(f"escrito {args.salida}")


if __name__ == "__main__":
    main()
