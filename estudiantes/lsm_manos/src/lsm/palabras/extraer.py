"""Saca de cada seña la secuencia de puntos de las dos manos y del cuerpo.

Por cada cuadro: 33 puntos del cuerpo (PoseLandmarker) y hasta 2 manos de
21 puntos (HandLandmarker). Las manos se asignan al brazo del cuerpo cuya
muñeca queda más cerca, no a la etiqueta Left/Right de MediaPipe: con dos
manos esa etiqueta se confunde a menudo y el cuerpo es más estable.

Solo se guardan puntos. Las fotos y los videos (que muestran caras) se
quedan en datos/crudos/ y nunca se suben.

Uso:
    python -m lsm.palabras.extraer uaemex datos/crudos/uaemex datos/secuencias_uaemex.npz
    python -m lsm.palabras.extraer iteso datos/crudos/iteso datos/secuencias_iteso.npz
"""

import argparse
import os
import re
import unicodedata
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

MODELO_MANOS = Path("modelos/hand_landmarker.task")
MODELO_CUERPO = Path("modelos/pose_landmarker_full.task")
MUNECA_IZQ, MUNECA_DER = 15, 16  # en el cuerpo: lados anatómicos de la persona
FPS_ITESO = 15  # los videos van a 30 o 60 fps; 15 basta para el movimiento de una seña


@dataclass(frozen=True)
class Muestra:
    """Una repetición de una seña: sus cuadros vienen de fotos o de un video."""
    dataset: str
    clase: str
    persona: str
    ident: str
    archivos: tuple  # fotos en orden, o un solo video


def muestras_uaemex(raiz):
    """.../MSLwords1/<clase>/<persona><clase>/<cuadros>.jpg (el zip de Mendeley
    agrega carpetas arriba de MSLwords1)."""
    muestras = []
    base = next(raiz.rglob("MSLwords1"))
    for carpeta in sorted(base.glob("*/*")):
        fotos = tuple(sorted(carpeta.glob("*.jpg")))
        if carpeta.is_dir() and fotos:
            clase = carpeta.parent.name
            muestras.append(Muestra("uaemex", clase, carpeta.name[:-len(clase)] or "?", carpeta.name, fotos))
    return muestras


def sin_acentos(texto):
    return unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode().upper()


def muestras_iteso(raiz):
    """<Palabra>/<video>. La página no dice quién es quién: el nombre del
    archivo («12», «12_1», «12_2», «IMG_5922») es la única pista y se guarda."""
    muestras = []
    for video in sorted(p for p in raiz.glob("*/*") if p.suffix.lower() in (".mp4", ".mov")):
        clase = sin_acentos(re.sub(r"\s+\d+$", "", video.parent.name))
        muestras.append(Muestra("iteso", clase, "?", f"{video.parent.name}/{video.stem}", (video,)))
    return muestras


# Con 0.3, en UAEMex no se ve mano en 27 % de los cuadros; con 0.1, en 15 %, y la GRU
# reconoce mejor (P7). La página usa el mismo valor (web/camara.js).
CONFIANZA_MANOS = 0.1
_detectores = None
_confianza_manos = CONFIANZA_MANOS


def _iniciar(modo, confianza_manos=CONFIANZA_MANOS):
    global _detectores, _confianza_manos
    _confianza_manos = confianza_manos
    corriendo = vision.RunningMode.VIDEO if modo == "video" else vision.RunningMode.IMAGE
    manos = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODELO_MANOS)), running_mode=corriendo,
        num_hands=2, min_hand_detection_confidence=confianza_manos, min_hand_presence_confidence=confianza_manos))
    # Con 0.5 (el valor por defecto) el cuerpo sale en 47 % de los cuadros de UAEMex
    # (ropa negra sobre fondo negro y caras pixeladas); con 0.2, en 74 %.
    cuerpo = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODELO_CUERPO)), running_mode=corriendo,
        min_pose_detection_confidence=0.2, min_pose_presence_confidence=0.2))
    _detectores = (manos, cuerpo)


def puntos_de_cuadro(rgb, tiempo_ms=None):
    """Puntos de un cuadro. Mano 0 = la del brazo izquierdo de la persona,
    mano 1 = la del derecho; NaN si falta. Además de las coordenadas de imagen
    se guardan las «world» de MediaPipe (metros, origen en la cadera o en el
    centro de la mano): con ellas se anima el avatar en 3D."""
    manos_det, cuerpo_det = _detectores
    imagen = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
    if tiempo_ms is None:
        r_manos, r_cuerpo = manos_det.detect(imagen), cuerpo_det.detect(imagen)
    else:
        r_manos = manos_det.detect_for_video(imagen, tiempo_ms)
        r_cuerpo = cuerpo_det.detect_for_video(imagen, tiempo_ms)

    cuerpo = np.full((33, 4), np.nan, np.float32)
    cuerpo_3d = np.full((33, 3), np.nan, np.float32)
    if r_cuerpo.pose_landmarks:
        cuerpo[:] = [(p.x, p.y, p.z, p.visibility) for p in r_cuerpo.pose_landmarks[0]]
        cuerpo_3d[:] = [(p.x, p.y, p.z) for p in r_cuerpo.pose_world_landmarks[0]]
    manos = np.full((2, 21, 3), np.nan, np.float32)
    manos_3d = np.full((2, 21, 3), np.nan, np.float32)
    for lm, mundo in zip(r_manos.hand_landmarks, r_manos.hand_world_landmarks):
        mano = np.array([(p.x, p.y, p.z) for p in lm], np.float32)
        lado = lado_de_mano(mano, cuerpo, manos)
        manos[lado] = mano
        manos_3d[lado] = [(p.x, p.y, p.z) for p in mundo]
    return cuerpo, manos, cuerpo_3d, manos_3d


def lado_de_mano(mano, cuerpo, ya):
    """0 o 1 según la muñeca del cuerpo más cercana; si no hay cuerpo, por
    posición en la imagen (sin espejo, el brazo izquierdo sale a la derecha)."""
    libres = [i for i in (0, 1) if np.isnan(ya[i, 0, 0])] or [0, 1]
    if not np.isnan(cuerpo[MUNECA_IZQ, 0]):
        d = [np.linalg.norm(mano[0, :2] - cuerpo[(MUNECA_IZQ, MUNECA_DER)[i], :2]) for i in (0, 1)]
        orden = np.argsort(d)
    else:
        orden = [0, 1] if mano[0, 0] > 0.5 else [1, 0]
    return next(int(i) for i in orden if i in libres)


def procesar(muestra):
    cuadros, aspecto = [], 1.0
    if muestra.dataset == "uaemex":
        for foto in muestra.archivos:
            bgr = cv2.imread(str(foto))
            if bgr is None:
                continue
            aspecto = bgr.shape[1] / bgr.shape[0]
            cuadros.append(puntos_de_cuadro(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
    else:
        _iniciar("video", _confianza_manos)  # VIDEO exige tiempos crecientes: detectores nuevos por video
        cap = cv2.VideoCapture(str(muestra.archivos[0]))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        paso = max(1, round(fps / FPS_ITESO))
        i = 0
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            if i % paso == 0:
                bgr = reducir(bgr)
                aspecto = bgr.shape[1] / bgr.shape[0]
                cuadros.append(puntos_de_cuadro(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), int(1000 * i / fps)))
            i += 1
        cap.release()
    if not cuadros:
        return muestra, None
    return muestra, ([np.stack(c) for c in zip(*cuadros)], aspecto)


def reducir(bgr, lado_mayor=960):
    """Los videos de teléfono vienen en 1080p o 4K; MediaPipe trabaja a 256 px de todos modos."""
    escala = lado_mayor / max(bgr.shape[:2])
    return cv2.resize(bgr, None, fx=escala, fy=escala, interpolation=cv2.INTER_AREA) if escala < 1 else bgr


CAMPOS_POR_CUADRO = ("cuerpo", "manos", "cuerpo_3d", "manos_3d")


def guardar(salida, resultados):
    """Secuencias de largo variable concatenadas; `inicio[i]:inicio[i+1]` es la i-ésima."""
    validos = [(m, r) for m, r in resultados if r is not None]
    largos = np.array([len(r[0][0]) for _, r in validos])
    por_cuadro = {campo: np.concatenate([r[0][j] for _, r in validos])
                  for j, campo in enumerate(CAMPOS_POR_CUADRO)}
    np.savez_compressed(
        salida,
        **por_cuadro,
        inicio=np.concatenate([[0], np.cumsum(largos)]),
        aspecto=np.array([r[1] for _, r in validos], np.float32),
        dataset=np.array([m.dataset for m, _ in validos]),
        clase=np.array([m.clase for m, _ in validos]),
        persona=np.array([m.persona for m, _ in validos]),
        ident=np.array([m.ident for m, _ in validos]),
    )
    return len(validos), len(resultados) - len(validos)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("dataset", choices=("uaemex", "iteso"))
    ap.add_argument("raiz", type=Path)
    ap.add_argument("salida", type=Path)
    ap.add_argument("--procesos", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--confianza-manos", type=float, default=CONFIANZA_MANOS,
                    help="umbral de detección y presencia de manos de MediaPipe")
    args = ap.parse_args()

    muestras = (muestras_uaemex if args.dataset == "uaemex" else muestras_iteso)(args.raiz)
    print(f"{args.dataset}: {len(muestras)} muestras", flush=True)
    resultados = []
    with ProcessPoolExecutor(args.procesos, initializer=_iniciar, initargs=("imagen", args.confianza_manos)) as ejecutor:
        for i, r in enumerate(ejecutor.map(procesar, muestras, chunksize=4), 1):
            resultados.append(r)
            if i % 200 == 0:
                print(f"  {i} de {len(muestras)}", flush=True)
    ok, vacias = guardar(args.salida, resultados)
    print(f"escrito {args.salida}: {ok} secuencias, {vacias} sin cuadros", flush=True)


if __name__ == "__main__":
    main()
