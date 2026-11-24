"""P1 · ¿Qué tan bien se reconocen palabras de personas que el modelo nunca vio?

Validación dejando una persona fuera, sobre las personas de UAEMex que no son
de prueba (las de prueba no se tocan). En cada pliegue se entrena con las
demás personas (y con ITESO si se pasa su archivo) y se evalúa en la persona
fuera, solo con sus señas de UAEMex.

Métricas por persona: exactitud top-1 y top-5 y F1 macro; se reporta la
media, la desviación y la peor persona.

Uso:
    python -m lsm.palabras.experimentos.p1_reconocimiento datos/secuencias_uaemex.npz \
        [--iteso datos/secuencias_iteso.npz] [--modelos knn logistica svm mlp gru]
"""

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from ...evaluar import matriz_confusion
from ...experimentos.comun import entorno, resumir
from ..conjunto import cargar, personas_de_prueba
from ..modelos import MODELOS


def top_k(P, y, k):
    return float(np.mean([y[i] in np.argsort(-P[i])[:k] for i in range(len(y))]))


def evaluar(P, y, K):
    """El F1 macro se promedia solo sobre las glosas que la persona tiene: las
    que solo están en ITESO valdrían 0 y lo bajarían sin decir nada."""
    C = matriz_confusion(y, P.argmax(axis=1), K)
    verdaderos = np.diag(C).astype(float)
    denominador = C.sum(axis=0) + C.sum(axis=1)
    f1 = np.divide(2 * verdaderos, denominador, out=np.zeros_like(verdaderos), where=denominador > 0)
    return {"top1": float(np.mean(P.argmax(axis=1) == y)), "top5": top_k(P, y, 5),
            "f1_macro": float(f1[np.unique(y)].mean())}


def resumen_con_peor(valores):
    return {**resumir(valores), "peor": float(np.min(valores))}


def un_pliegue(args):
    nombre, X, y, persona, es_val, K = args
    inicio = time.time()
    modelo = MODELOS[nombre]().ajustar(X[~es_val], y[~es_val], K)
    resultado = evaluar(modelo.probabilidades(X[es_val]), y[es_val], K)
    return {**resultado, "segundos": round(time.time() - inicio, 1)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("uaemex", type=Path)
    ap.add_argument("--iteso", type=Path)
    ap.add_argument("--modelos", nargs="+", default=list(MODELOS))
    ap.add_argument("--salida", type=Path, default=Path("resultados/p1_palabras.json"))
    ap.add_argument("--procesos", type=int, default=6)
    args = ap.parse_args()

    datos = cargar([args.uaemex] + ([args.iteso] if args.iteso else []))
    prueba = personas_de_prueba(datos.persona)
    datos = datos.filtrar(~np.isin(datos.persona, list(prueba)))
    glosas = sorted(set(datos.glosa))
    indice = {g: k for k, g in enumerate(glosas)}
    y = np.array([indice[g] for g in datos.glosa])
    K = len(glosas)
    validacion = sorted({p for p in datos.persona if p.startswith("uaemex-")})
    print(f"{len(datos)} secuencias, {K} glosas, validación: {validacion}, prueba (sin tocar): {sorted(prueba)}", flush=True)

    resultados = {}
    for nombre in args.modelos:
        tareas = [(nombre, datos.X, y, datos.persona, (datos.persona == p) & (datos.dataset == "uaemex"), K)
                  for p in validacion]
        with ProcessPoolExecutor(args.procesos) as ejecutor:
            por_persona = dict(zip(validacion, ejecutor.map(un_pliegue, tareas)))
        resumen = {m: resumen_con_peor([r[m] for r in por_persona.values()]) for m in ("top1", "top5", "f1_macro")}
        resultados[nombre] = {"resumen": resumen, "por_persona": por_persona}
        print(f"{nombre}: top1 {resumen['top1']['media']:.3f} ± {resumen['top1']['desviacion']:.3f} "
              f"(peor {resumen['top1']['peor']:.3f}), top5 {resumen['top5']['media']:.3f}, "
              f"F1 {resumen['f1_macro']['media']:.3f}", flush=True)

    salida = {
        "experimento": "P1 reconocimiento de palabras aisladas, dejando una persona fuera",
        "datos": {"uaemex": str(args.uaemex), "iteso": str(args.iteso) if args.iteso else None,
                  "secuencias": len(datos), "glosas": K, "personas_validacion": validacion,
                  "personas_prueba_sin_tocar": sorted(prueba)},
        "modelos": resultados,
        "entorno": entorno(),
    }
    args.salida.parent.mkdir(parents=True, exist_ok=True)
    args.salida.write_text(json.dumps(salida, indent=2, ensure_ascii=False))
    print(f"escrito {args.salida}")


if __name__ == "__main__":
    main()
