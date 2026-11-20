"""Exporta la red a JSON para la demo y genera el archivo de paridad Python–JS.

Uso:
    python -m lsm.exportar paridad web/fixtures/paridad.json

Los pesos se redondean a float32 (la mitad de tamaño); Python calcula las
salidas de referencia con esos mismos pesos redondeados, así que JavaScript
debe coincidir casi exacto.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from redneuronal.red import Red, adelante, red_inicial

from .datos import LETRAS_ESTATICAS
from .features import vector_entrada


def _a_float32(arreglo):
    return np.asarray(arreglo, dtype=np.float32).astype(float).tolist()


def red_a_json(red, letras):
    return {
        "anchos": list(red.anchos),
        "activaciones": red.g[1:],
        "letras": list(letras),
        "W": [_a_float32(red.W[l]) for l in range(1, red.L + 1)],
        "b": [_a_float32(red.b[l]) for l in range(1, red.L + 1)],
    }


def red_desde_json(datos):
    return Red(
        W=[None] + [np.array(W, dtype=float) for W in datos["W"]],
        b=[None] + [np.array(b, dtype=float) for b in datos["b"]],
        g=[None, *datos["activaciones"]],
    )


def mano_de_prueba(semilla):
    """Puntos sintéticos con forma de mano (muñeca abajo, dedos arriba), en
    coordenadas de MediaPipe. Solo para comprobar que Python y JS coinciden;
    nunca se usan para entrenar."""
    rng = np.random.default_rng(semilla)
    P = np.zeros((21, 3))
    P[0] = (0.5, 0.8, 0.0)
    P[1:, 0] = 0.5 + rng.uniform(-0.15, 0.15, 20)
    P[1:, 1] = 0.8 - rng.uniform(0.1, 0.45, 20)
    P[1:, 2] = rng.uniform(-0.05, 0.05, 20)
    return P


def generar_paridad(semilla=0):
    """Casos de features y de red con sus resultados en Python."""
    casos_features = []
    for i, (izquierda, aspecto, rotar, usar_z) in enumerate([
        (False, 4 / 3, False, True), (True, 16 / 9, False, True),
        (False, 1.0, True, True), (True, 4 / 3, True, False),
    ]):
        puntos = mano_de_prueba(semilla + i)
        casos_features.append({
            "puntos": puntos.tolist(), "es_izquierda": izquierda, "aspecto": aspecto,
            "rotar": rotar, "usar_z": usar_z,
            "vector": vector_entrada(puntos, izquierda, aspecto, rotar, usar_z).tolist(),
        })

    # Red chica para que el archivo pese poco; el código es el mismo para la grande.
    red = red_desde_json(red_a_json(
        red_inicial((60, 12, 8, len(LETRAS_ESTATICAS)), ("relu", "relu", "softmax"), semilla),
        LETRAS_ESTATICAS,
    ))
    X = np.array([c["vector"] for c in casos_features if c["usar_z"]]).T
    _, A = adelante(red, X)
    casos_red = [
        {"entrada": X[:, k].tolist(), "activaciones": [A[l][:, k].tolist() for l in range(1, red.L + 1)]}
        for k in range(X.shape[1])
    ]
    return {"features": casos_features, "red": red_a_json(red, LETRAS_ESTATICAS), "casos_red": casos_red}


def main():
    ap = argparse.ArgumentParser(description="Exporta archivos para la demo web.")
    ap.add_argument("que", choices=["paridad"])
    ap.add_argument("salida", type=Path)
    args = ap.parse_args()
    args.salida.parent.mkdir(parents=True, exist_ok=True)
    args.salida.write_text(json.dumps(generar_paridad(), separators=(",", ":")))
    print(f"escrito {args.salida} ({args.salida.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
