"""P7 (tamiz) · ¿Qué preprocesamiento o umbral recupera más manos en UAEMex?

Antes de volver a extraer todo, se prueba en 400 fotos al azar (semilla 0)
cuántas manos detecta MediaPipe con cada variante: el umbral de confianza
por omisión (0.3) o uno más bajo (0.1), y cambios de imagen (gamma, CLAHE,
doble tamaño). Solo mide detección: la exactitud se mide después, en P7,
con la extracción completa.

Resultado guardado (2026-11-25): con 0.1 los cuadros con al menos una mano
pasan de 72 % a 82 %; gamma, CLAHE y doble tamaño no ayudan o empeoran.

Uso:
    python -m lsm.palabras.experimentos.p7_tamiz datos/crudos/uaemex
"""

import argparse
import json
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

from ...experimentos.comun import entorno
from ..extraer import MODELO_MANOS

FOTOS = 400


def detector(confianza):
    return vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODELO_MANOS)), num_hands=2,
        min_hand_detection_confidence=confianza, min_hand_presence_confidence=confianza))


def gamma(rgb, g=0.6):
    return (255 * (rgb / 255.0) ** g).astype(np.uint8)


def clahe(rgb):
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    lab[..., 0] = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def doble(rgb):
    return cv2.resize(rgb, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)


VARIANTES = {
    "base (0.3)": (0.3, lambda x: x),
    "confianza 0.1": (0.1, lambda x: x),
    "gamma 0.6": (0.3, gamma),
    "CLAHE": (0.3, clahe),
    "doble tamaño": (0.3, doble),
    "CLAHE + gamma": (0.3, lambda x: gamma(clahe(x))),
    "CLAHE + 0.1": (0.1, clahe),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("raiz", type=Path, help="carpeta de UAEMex (datos/crudos/uaemex)")
    ap.add_argument("--salida", type=Path, default=Path("resultados/p7_tamiz.json"))
    args = ap.parse_args()

    fotos = sorted(args.raiz.rglob("*.jpg"))
    elegidas = [fotos[i] for i in np.random.default_rng(0).choice(len(fotos), FOTOS, replace=False)]
    imagenes = [cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2RGB) for f in elegidas]
    resultados = {}
    for nombre, (confianza, preparar) in VARIANTES.items():
        det = detector(confianza)
        n = np.array([len(det.detect(mp.Image(image_format=mp.ImageFormat.SRGB,
                                              data=np.ascontiguousarray(preparar(im)))).hand_landmarks)
                      for im in imagenes])
        resultados[nombre] = {"manos_por_cuadro": float(n.mean()), "al_menos_una": float(np.mean(n >= 1)),
                              "dos_manos": float(np.mean(n == 2))}
        print(nombre, resultados[nombre], flush=True)
    args.salida.parent.mkdir(parents=True, exist_ok=True)
    args.salida.write_text(json.dumps({
        "experimento": "P7 tamiz de detección de manos en UAEMex (400 fotos al azar, semilla 0)",
        "resultados": resultados,
        "entorno": entorno(),
    }, indent=2, ensure_ascii=False))
    print(f"escrito {args.salida}")


if __name__ == "__main__":
    main()
