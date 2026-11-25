"""P5 · ¿Puede la red decir «no sé» cuando se va a equivocar?

La red responde solo si su probabilidad más alta pasa un umbral u. Con las
probabilidades de la validación dejando una persona fuera (personas nunca
vistas) se mide, para cada u:

- cobertura: fracción de señas a las que responde;
- exactitud de lo que responde.

También la calibración (ECE: diferencia promedio entre la confianza y la
exactitud real, en 10 cubetas) y una prueba de señas desconocidas: se quita
del entrenamiento un 20 % de las glosas, elegidas al azar con semilla fija, y
se mide cuántas de esas señas rechaza el umbral contra las conocidas (AUROC de
la probabilidad máxima: 0.5 = no las distingue, 1 = perfecto).

El umbral de la página se elige aquí: el menor u con exactitud de lo
respondido ≥ 0.9 en validación.

Uso:
    python -m lsm.palabras.experimentos.p5_no_se datos/secuencias_uaemex.npz
"""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from ...experimentos.comun import entorno
from ..modelos import ModeloGRU
from .comun import dejando_una_persona_fuera, personas_de_validacion, preparar

UMBRALES = np.round(np.arange(0.0, 1.0, 0.05), 2)
EXACTITUD_OBJETIVO = 0.9
FRACCION_DESCONOCIDAS = 0.2


def curva(confianza, acierto):
    puntos = []
    for u in UMBRALES:
        responde = confianza >= u
        puntos.append({"umbral": float(u), "cobertura": float(responde.mean()),
                       "exactitud": float(acierto[responde].mean()) if responde.any() else None})
    return puntos


def ece(confianza, acierto, cubetas=10):
    bordes = np.linspace(0, 1, cubetas + 1)
    total = 0.0
    for a, b in zip(bordes[:-1], bordes[1:]):
        dentro = (confianza > a) & (confianza <= b)
        if dentro.any():
            total += dentro.mean() * abs(confianza[dentro].mean() - acierto[dentro].mean())
    return float(total)


def auroc(positivos, negativos):
    """P(un puntaje positivo > uno negativo), contando empates como 1/2."""
    p, n = np.asarray(positivos)[:, None], np.asarray(negativos)[None, :]
    return float(np.mean((p > n) + 0.5 * (p == n)))


def _pliegue_abierto(tarea):
    """Entrena sin la persona y sin las glosas desconocidas; devuelve la confianza
    en sus señas conocidas y en sus desconocidas."""
    X, y, entrenar, conocidas_val, desconocidas_val, K = tarea
    modelo = ModeloGRU().ajustar(X[entrenar], y[entrenar], K)
    return modelo.probabilidades(X[conocidas_val]).max(axis=1), modelo.probabilidades(X[desconocidas_val]).max(axis=1)


def senas_desconocidas(datos, y, K, procesos):
    rng = np.random.default_rng(0)
    desconocidas = rng.choice(K, int(FRACCION_DESCONOCIDAS * K), replace=False)
    conocida = ~np.isin(y, desconocidas)
    restantes = np.unique(y[conocida])
    y_nueva = np.searchsorted(restantes, y)  # índices 0..K'−1 para las conocidas
    tareas = []
    for p in personas_de_validacion(datos):
        suyas = (datos.persona == p) & (datos.dataset == "uaemex")
        tareas.append((datos.X, y_nueva, (datos.persona != p) & conocida, suyas & conocida, suyas & ~conocida,
                       len(restantes)))
    with ProcessPoolExecutor(procesos) as ejecutor:
        pares = list(ejecutor.map(_pliegue_abierto, tareas))
    return np.concatenate([a for a, _ in pares]), np.concatenate([b for _, b in pares]), desconocidas


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("uaemex", type=Path)
    ap.add_argument("--salida", type=Path, default=Path("resultados/p5_no_se.json"))
    ap.add_argument("--procesos", type=int, default=6)
    args = ap.parse_args()

    datos, y, glosas, _ = preparar([args.uaemex])
    K = len(glosas)
    _, salidas = dejando_una_persona_fuera(ModeloGRU, datos, datos.X, y, K, args.procesos)
    P = np.concatenate([s[0] for s in salidas.values()])
    yv = np.concatenate([s[1] for s in salidas.values()])
    confianza, acierto = P.max(axis=1), P.argmax(axis=1) == yv
    puntos = curva(confianza, acierto)
    elegido = next((p for p in puntos if p["exactitud"] is not None and p["exactitud"] >= EXACTITUD_OBJETIVO), None)
    print(f"ECE {ece(confianza, acierto):.3f}; umbral elegido: {elegido}", flush=True)

    confianza_conocidas, confianza_desconocidas, desconocidas = senas_desconocidas(datos, y, K, args.procesos)
    u = elegido["umbral"] if elegido else 0.5
    abierto = {
        "glosas_desconocidas": [glosas[k] for k in sorted(desconocidas)],
        "auroc_conocidas_vs_desconocidas": auroc(confianza_conocidas, confianza_desconocidas),
        "umbral": u,
        "rechazo_desconocidas": float(np.mean(confianza_desconocidas < u)),
        "rechazo_conocidas": float(np.mean(confianza_conocidas < u)),
    }
    print(abierto, flush=True)

    args.salida.parent.mkdir(parents=True, exist_ok=True)
    args.salida.write_text(json.dumps({
        "experimento": "P5 «no sé»: cobertura contra exactitud, calibración y señas desconocidas",
        "ece": ece(confianza, acierto),
        "curva": puntos,
        "umbral_elegido": elegido,
        "criterio": f"menor umbral con exactitud de lo respondido >= {EXACTITUD_OBJETIVO}",
        "senas_desconocidas": abierto,
        "entorno": entorno(),
    }, indent=2, ensure_ascii=False))
    print(f"escrito {args.salida}")


if __name__ == "__main__":
    main()
