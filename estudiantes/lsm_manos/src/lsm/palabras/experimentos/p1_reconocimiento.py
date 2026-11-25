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
from pathlib import Path

from ...experimentos.comun import entorno
from ..modelos import MODELOS
from .comun import dejando_una_persona_fuera, linea, personas_de_validacion, preparar, resumir_metricas


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("uaemex", type=Path)
    ap.add_argument("--iteso", type=Path)
    ap.add_argument("--modelos", nargs="+", default=list(MODELOS))
    ap.add_argument("--salida", type=Path, default=Path("resultados/p1_palabras.json"))
    ap.add_argument("--procesos", type=int, default=6)
    args = ap.parse_args()

    datos, y, glosas, prueba = preparar([args.uaemex] + ([args.iteso] if args.iteso else []))
    K = len(glosas)
    print(f"{len(datos)} secuencias, {K} glosas, prueba sin tocar: {prueba}", flush=True)

    resultados = {}
    for nombre in args.modelos:
        por_persona, _ = dejando_una_persona_fuera(MODELOS[nombre], datos, datos.X, y, K, args.procesos)
        resultados[nombre] = {"resumen": resumir_metricas(por_persona), "por_persona": por_persona}
        print(linea(nombre, resultados[nombre]["resumen"]), flush=True)

    salida = {
        "experimento": "P1 reconocimiento de palabras aisladas, dejando una persona fuera",
        "datos": {"uaemex": str(args.uaemex), "iteso": str(args.iteso) if args.iteso else None,
                  "secuencias": len(datos), "glosas": K, "personas_validacion": personas_de_validacion(datos),
                  "personas_prueba_sin_tocar": prueba},
        "modelos": resultados,
        "entorno": entorno(),
    }
    args.salida.parent.mkdir(parents=True, exist_ok=True)
    args.salida.write_text(json.dumps(salida, indent=2, ensure_ascii=False))
    print(f"escrito {args.salida}")


if __name__ == "__main__":
    main()
