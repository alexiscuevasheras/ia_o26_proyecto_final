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

Calibración por temperatura: ŷ = softmax(z/τ), con un solo número τ que
minimiza la entropía cruzada. Para no hacer trampa, el τ con que se evalúa a
cada persona se ajusta con las salidas de las otras personas de validación.
Se reporta todo sin calibrar y calibrado.

El umbral de la página se elige aquí: el menor u con exactitud de lo
respondido ≥ 0.9 en validación que todavía responda al menos a COBERTURA_MINIMA
de las señas (si ninguno llega, se reporta y la página usa el que maximiza
exactitud × cobertura). Sin esa cobertura mínima el criterio llegó a elegir
un umbral que respondía al 0.7 % de las señas (14 de 15 bien): una página que
casi siempre dice «no sé», con una exactitud estimada sobre 15 señas.

Con --iteso o --propios esos datos entran al entrenamiento de cada pliegue
(se valida igual, con las personas de UAEMex y las del dataset propio): así
se calibra el modelo de la página con los mismos datos con que se entrena.

Uso:
    python -m lsm.palabras.experimentos.p5_no_se datos/secuencias_uaemex.npz
    python -m lsm.palabras.experimentos.p5_no_se datos/secuencias_uaemex.npz \\
        --iteso datos/secuencias_iteso.npz --salida resultados/p5_no_se_con_iteso.json
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
COBERTURA_MINIMA = 0.2
FRACCION_DESCONOCIDAS = 0.2


TEMPERATURAS = np.exp(np.linspace(np.log(0.25), np.log(8), 60))


def temperatura(P, y):
    """τ que minimiza la entropía cruzada de softmax(log P / τ), por búsqueda en rejilla.
    log P sirve como z: difiere de los logits solo por una constante por fila."""
    z = np.log(np.clip(P, 1e-12, None))
    mejor, mejor_perdida = 1.0, np.inf
    for tau in TEMPERATURAS:
        perdida = -np.mean(np.log(np.clip(calibrar(z, tau)[np.arange(len(y)), y], 1e-12, None)))
        if perdida < mejor_perdida:
            mejor, mejor_perdida = float(tau), perdida
    return mejor


def calibrar(z, tau):
    e = np.exp((z - z.max(axis=1, keepdims=True)) / tau)
    return e / e.sum(axis=1, keepdims=True)


def calibrado_sin_trampa(salidas):
    """Para cada persona, τ ajustada con las otras; devuelve (P calibradas, y, τ por persona)."""
    P_cal, ys, taus = [], [], {}
    for p, (P, y) in salidas.items():
        otras = [salidas[q] for q in salidas if q != p]
        taus[p] = temperatura(np.concatenate([o[0] for o in otras]), np.concatenate([o[1] for o in otras]))
        P_cal.append(calibrar(np.log(np.clip(P, 1e-12, None)), taus[p]))
        ys.append(y)
    return np.concatenate(P_cal), np.concatenate(ys), taus


def resumen_de_confianza(P, y):
    confianza, acierto = P.max(axis=1), P.argmax(axis=1) == y
    puntos = curva(confianza, acierto)
    elegido = next((p for p in puntos if p["exactitud"] is not None and p["exactitud"] >= EXACTITUD_OBJETIVO
                    and p["cobertura"] >= COBERTURA_MINIMA), None)
    respaldo = max(puntos, key=lambda p: (p["exactitud"] or 0) * p["cobertura"])
    return {"ece": ece(confianza, acierto), "curva": puntos, "umbral_objetivo": elegido,
            "umbral_exactitud_por_cobertura": respaldo}


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
    """Entrena sin la persona y sin las glosas desconocidas; devuelve las
    probabilidades en sus señas conocidas y en sus desconocidas."""
    X, y, entrenar, conocidas_val, desconocidas_val, K = tarea
    modelo = ModeloGRU().ajustar(X[entrenar], y[entrenar], K)
    return modelo.probabilidades(X[conocidas_val]), modelo.probabilidades(X[desconocidas_val])


def senas_desconocidas(datos, y, K, procesos):
    rng = np.random.default_rng(0)
    desconocidas = rng.choice(K, int(FRACCION_DESCONOCIDAS * K), replace=False)
    conocida = ~np.isin(y, desconocidas)
    restantes = np.unique(y[conocida])
    y_nueva = np.searchsorted(restantes, y)  # índices 0..K'−1 para las conocidas
    tareas = []
    for p in personas_de_validacion(datos):
        suyas = datos.persona == p
        tareas.append((datos.X, y_nueva, (datos.persona != p) & conocida, suyas & conocida, suyas & ~conocida,
                       len(restantes)))
    with ProcessPoolExecutor(procesos) as ejecutor:
        pares = list(ejecutor.map(_pliegue_abierto, tareas))
    return np.concatenate([a for a, _ in pares]), np.concatenate([b for _, b in pares]), desconocidas


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("uaemex", type=Path)
    ap.add_argument("--iteso", type=Path)
    ap.add_argument("--propios", type=Path)
    ap.add_argument("--salida", type=Path, default=Path("resultados/p5_no_se.json"))
    ap.add_argument("--procesos", type=int, default=6)
    args = ap.parse_args()

    rutas = [args.uaemex] + [r for r in (args.iteso, args.propios) if r]
    datos, y, glosas, _ = preparar(rutas)
    K = len(glosas)
    _, salidas = dejando_una_persona_fuera(ModeloGRU, datos, datos.X, y, K, args.procesos)
    P = np.concatenate([s[0] for s in salidas.values()])
    yv = np.concatenate([s[1] for s in salidas.values()])
    sin_calibrar = resumen_de_confianza(P, yv)
    P_cal, y_cal, taus = calibrado_sin_trampa(salidas)
    calibrado = resumen_de_confianza(P_cal, y_cal)
    tau_final = temperatura(P, yv)  # para la página: con todas las personas de validación
    print(f"ECE sin calibrar {sin_calibrar['ece']:.3f}, calibrado {calibrado['ece']:.3f}; τ = {tau_final:.2f}", flush=True)
    elegido = calibrado["umbral_objetivo"] or calibrado["umbral_exactitud_por_cobertura"]

    P_conocidas, P_desconocidas, desconocidas = senas_desconocidas(datos, y, K, args.procesos)
    confianza_conocidas = calibrar(np.log(np.clip(P_conocidas, 1e-12, None)), tau_final).max(axis=1)
    confianza_desconocidas = calibrar(np.log(np.clip(P_desconocidas, 1e-12, None)), tau_final).max(axis=1)
    u = elegido["umbral"]
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
        "datos": [str(r) for r in rutas],
        "sin_calibrar": sin_calibrar,
        "calibrado": {**calibrado, "temperatura_por_persona": taus},
        "temperatura_final": tau_final,
        "umbral_elegido": elegido,
        "criterio": f"menor umbral con exactitud de lo respondido >= {EXACTITUD_OBJETIVO} y cobertura >= "
                    f"{COBERTURA_MINIMA}; si ninguno, "
                    "el que maximiza exactitud × cobertura (curva calibrada)",
        "senas_desconocidas": abierto,
        "entorno": entorno(),
    }, indent=2, ensure_ascii=False))
    print(f"escrito {args.salida}")


if __name__ == "__main__":
    main()
