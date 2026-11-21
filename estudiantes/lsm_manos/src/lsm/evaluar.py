"""Métricas de clasificación multiclase, escritas en NumPy.

F1 macro es la métrica principal (se fijó antes de ver resultados); también
se reportan exactitud, MCC multiclase y la matriz de confusión.
"""

import numpy as np


def matriz_confusion(y, y_gorro, K):
    """C[i, j] = número de observaciones de la clase i predichas como j."""
    C = np.zeros((K, K), dtype=np.int64)
    np.add.at(C, (y, y_gorro), 1)
    return C


def f1_macro(C):
    """Promedio sin pesos del F1 de cada clase; una clase sin predicciones ni
    casos cuenta como F1 = 0, como en scikit-learn."""
    verdaderos = np.diag(C).astype(float)
    predichos, reales = C.sum(axis=0), C.sum(axis=1)
    denominador = predichos + reales
    f1 = np.divide(2 * verdaderos, denominador, out=np.zeros_like(verdaderos), where=denominador > 0)
    return float(f1.mean())


def mcc_multiclase(C):
    """Coeficiente de correlación de Matthews para K clases (Gorodkin, 2004).
    Vale 1 si todo es correcto, cerca de 0 si es como adivinar."""
    C = C.astype(float)
    n, correctos = C.sum(), np.trace(C)
    predichos, reales = C.sum(axis=0), C.sum(axis=1)
    numerador = correctos * n - reales @ predichos
    denominador = np.sqrt((n**2 - predichos @ predichos) * (n**2 - reales @ reales))
    return 0.0 if denominador == 0 else float(numerador / denominador)


def metricas(y, y_gorro, K):
    C = matriz_confusion(np.asarray(y), np.asarray(y_gorro), K)
    return {
        "f1_macro": f1_macro(C),
        "mcc": mcc_multiclase(C),
        "exactitud": float(np.trace(C) / C.sum()),
        "confusion": C.tolist(),
    }
