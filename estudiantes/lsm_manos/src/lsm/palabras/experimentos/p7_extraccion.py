"""P7 · ¿Mejora el reconocimiento si MediaPipe pierde menos manos?

P6 mostró que el límite en UAEMex son sus datos: con ropa negra sobre fondo
negro, MediaPipe no ve mano en ~27 % de los cuadros. Aquí se comparan dos
extracciones de las mismas fotos que solo difieren en el umbral de confianza
de las manos (--confianza-manos de extraer.py):

- detección: fracción de cuadros sin mano y con las dos manos;
- reconocimiento: P1 (dejando una persona fuera) con la GRU y la regresión
  logística en cada extracción, y la diferencia pareada por persona con la
  prueba de signos y la de Wilcoxon (9 personas: poca potencia, se reportan
  igual).

El umbral se eligió mirando solo la detección en 400 fotos al azar y una
revisión a ojo de las detecciones nuevas, no la exactitud; las personas de
prueba siguen sin tocarse.

Uso:
    python -m lsm.palabras.experimentos.p7_extraccion \\
        datos/secuencias_uaemex_c03.npz datos/secuencias_uaemex.npz --nombres 0.3 0.1
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest, wilcoxon

from ...experimentos.comun import entorno
from ..conjunto import leer_secuencias
from ..modelos import MODELOS
from .comun import dejando_una_persona_fuera, linea, preparar, resumir_metricas

MODELOS_P7 = ("logistica", "gru")


def deteccion(ruta):
    vistas = np.isfinite(leer_secuencias(ruta)["manos"][:, :, 0, 0])
    return {"cuadros": int(len(vistas)), "sin_mano": float((~vistas.any(axis=1)).mean()),
            "dos_manos": float(vistas.all(axis=1).mean())}


def reconocimiento(ruta, procesos):
    datos, y, glosas, _ = preparar([ruta])
    resultados = {}
    for nombre in MODELOS_P7:
        por_persona, _ = dejando_una_persona_fuera(MODELOS[nombre], datos, datos.X, y, len(glosas), procesos)
        resultados[nombre] = {"resumen": resumir_metricas(por_persona), "por_persona": por_persona}
        print(linea(f"  {nombre}", resultados[nombre]["resumen"]), flush=True)
    return resultados


def comparacion_pareada(antes, despues):
    personas = sorted(antes)
    d = np.array([despues[p]["top1"] - antes[p]["top1"] for p in personas])
    mejoran = int((d > 0).sum())
    distintas = int((d != 0).sum())
    return {
        "diferencia_top1_media": float(d.mean()),
        "diferencia_top1_desviacion": float(d.std(ddof=1)),
        "personas_que_mejoran": mejoran,
        "personas": len(personas),
        "p_signos": float(binomtest(mejoran, distintas, 0.5).pvalue),
        "p_wilcoxon": float(wilcoxon(d).pvalue),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("antes", type=Path)
    ap.add_argument("despues", type=Path)
    ap.add_argument("--nombres", nargs=2, default=["antes", "despues"])
    ap.add_argument("--salida", type=Path, default=Path("resultados/p7_extraccion.json"))
    ap.add_argument("--procesos", type=int, default=6)
    args = ap.parse_args()

    extracciones = {}
    for nombre, ruta in zip(args.nombres, (args.antes, args.despues)):
        print(f"confianza {nombre}: {ruta}", flush=True)
        extracciones[nombre] = {"archivo": str(ruta), "deteccion": deteccion(ruta),
                                "reconocimiento": reconocimiento(ruta, args.procesos)}
        print(f"  {extracciones[nombre]['deteccion']}", flush=True)
    a, b = (extracciones[n]["reconocimiento"] for n in args.nombres)
    pareada = {m: comparacion_pareada(a[m]["por_persona"], b[m]["por_persona"]) for m in MODELOS_P7}
    print(pareada, flush=True)

    args.salida.parent.mkdir(parents=True, exist_ok=True)
    args.salida.write_text(json.dumps({
        "experimento": "P7 umbral de confianza de manos de MediaPipe en UAEMex: detección y reconocimiento",
        "extracciones": extracciones,
        "comparacion_pareada": pareada,
        "entorno": entorno(),
    }, indent=2, ensure_ascii=False))
    print(f"escrito {args.salida}")


if __name__ == "__main__":
    main()
