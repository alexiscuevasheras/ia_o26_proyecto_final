"""Extrae los 21 puntos de la mano de las fotos de D1 con MediaPipe.

El 7z de D1 pesa 4.8 GB y descomprimido ocupa 5 GB más, así que se procesa
por partes: se extrae un grupo y partición a la vez (p. ej. lsm-abc-A/train),
se sacan los puntos, se agregan al CSV y se borran las fotos.

Uso:
    python -m lsm.landmarks --archivo datos/d1_zip/MSL-ABC.7z \
        --salida datos/landmarks_d1.csv --modelo modelos/hand_landmarker.task

El CSV guarda una fila por foto (también las que no detectan mano, con
detectada = 0) para poder reportar la tasa de detección. Las coordenadas son
las crudas de MediaPipe; la normalización vive en features.py.
"""

import argparse
import csv
import os
import shutil
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision

from .datos import leer_nombre

PARTES_D1 = [f"lsm-abc-{g}/{p}" for g in "ABC" for p in ("train", "test")]
COORDENADAS = [f"{eje}{i}" for i in range(21) for eje in "xyz"]
COLUMNAS = ["ruta", "grupo", "particion", "letra", "participante", "codigo", "n",
            "detectada", "lateralidad", "confianza", "ancho", "alto", *COORDENADAS]


def crear_detector(modelo):
    opciones = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(modelo)),
        running_mode=vision.RunningMode.IMAGE,
        num_hands=1,
    )
    return vision.HandLandmarker.create_from_options(opciones)


def fila_de_foto(detector, archivo, ruta_relativa):
    """Una fila del CSV. `lateralidad` es la etiqueta cruda de MediaPipe
    ("Left"/"Right"); qué significa en estas fotos se decide al analizar."""
    foto = leer_nombre(ruta_relativa)
    imagen = mp.Image.create_from_file(str(archivo))
    resultado = detector.detect(imagen)
    fila = {
        "ruta": ruta_relativa, "grupo": foto.grupo, "particion": foto.particion,
        "letra": foto.letra, "participante": foto.participante, "codigo": foto.codigo,
        "n": foto.n, "ancho": imagen.width, "alto": imagen.height,
    }
    if not resultado.hand_landmarks:
        return {**fila, "detectada": 0, "lateralidad": "", "confianza": ""}
    categoria = resultado.handedness[0][0]
    puntos = resultado.hand_landmarks[0]
    coordenadas = {f"{eje}{i}": getattr(p, eje) for i, p in enumerate(puntos) for eje in "xyz"}
    return {**fila, "detectada": 1, "lateralidad": categoria.category_name,
            "confianza": round(categoria.score, 4), **coordenadas}


def rutas_ya_procesadas(salida):
    if not salida.exists():
        return set()
    with salida.open(newline="") as f:
        return {fila["ruta"] for fila in csv.DictReader(f)}


_detector_del_proceso = None


def _iniciar_proceso(modelo):
    """Cada proceso tiene su propio detector: MediaPipe no se comparte."""
    global _detector_del_proceso
    _detector_del_proceso = crear_detector(modelo)


def _procesar(tarea):
    archivo, ruta_relativa = tarea
    return fila_de_foto(_detector_del_proceso, archivo, ruta_relativa)


def procesar_carpeta(raiz, salida, hechas, modelo, procesos):
    """Agrega al CSV las fotos de `raiz` que no estén en `hechas`, repartidas
    entre varios procesos (la detección usa un núcleo por proceso)."""
    tareas = [(a, a.relative_to(raiz).as_posix()) for a in sorted(raiz.rglob("*.jpg"))]
    tareas = [t for t in tareas if t[1] not in hechas]
    nuevo = not salida.exists()
    with salida.open("a", newline="") as f, \
            ProcessPoolExecutor(procesos, initializer=_iniciar_proceso, initargs=(modelo,)) as ejecutor:
        escritor = csv.DictWriter(f, fieldnames=COLUMNAS)
        if nuevo:
            escritor.writeheader()
        for contador, fila in enumerate(ejecutor.map(_procesar, tareas, chunksize=64), 1):
            escritor.writerow(fila)
            if contador % 10000 == 0:
                print(f"  {contador} de {len(tareas)} fotos", flush=True)
    return len(tareas)


def extraer_parte(archivo_7z, parte, destino):
    """Extrae solo MSL-ABC/<parte>/ del 7z (bsdtar viene con macOS)."""
    subprocess.run(["bsdtar", "-xf", str(archivo_7z), "-C", str(destino),
                    f"MSL-ABC/{parte}/*"], check=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--archivo", type=Path, required=True)
    ap.add_argument("--salida", type=Path, required=True)
    ap.add_argument("--modelo", type=Path, required=True)
    ap.add_argument("--partes", nargs="*", default=PARTES_D1)
    ap.add_argument("--procesos", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = ap.parse_args()

    hechas = rutas_ya_procesadas(args.salida)
    for parte in args.partes:
        temporal = Path(tempfile.mkdtemp(prefix="d1_", dir=args.salida.parent))
        try:
            print(f"extrayendo {parte}", flush=True)
            extraer_parte(args.archivo, parte, temporal)
            n = procesar_carpeta(temporal, args.salida, hechas, args.modelo, args.procesos)
            print(f"{parte}: {n} fotos nuevas", flush=True)
        finally:
            shutil.rmtree(temporal)


if __name__ == "__main__":
    main()
