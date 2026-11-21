"""Carga los landmarks de D1 y entrena la red de NumPy para clasificar letras."""

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from redneuronal.optimizador import entrenar
from redneuronal.perdida import Castigo, EntropiaCruzada
from redneuronal.red import predecir, red_inicial

from .datos import LETRAS_ESTATICAS
from .features import vector_entrada
from .landmarks import COORDENADAS

INDICE_LETRA = {letra: k for k, letra in enumerate(LETRAS_ESTATICAS)}
K = len(LETRAS_ESTATICAS)


@dataclass
class Conjunto:
    """Una fila por foto con mano detectada."""
    puntos: np.ndarray  # (N, 21, 3) crudos de MediaPipe
    es_izquierda: np.ndarray  # (N,) bool, según la etiqueta de MediaPipe
    aspecto: np.ndarray  # (N,) ancho / alto
    letra: np.ndarray  # (N,) índice 0..K−1
    participante: np.ndarray
    grupo: np.ndarray  # "A", "B", "C"
    particion: np.ndarray  # "train" o "test", la oficial
    n_sin_mano: int

    def filtrar(self, mascara):
        campos = {c: getattr(self, c)[mascara] for c in
                  ("puntos", "es_izquierda", "aspecto", "letra", "participante", "grupo", "particion")}
        return Conjunto(**campos, n_sin_mano=0)

    def features(self, rotar=False, usar_z=True):
        """(N, 60) o (N, 40). La red espera observaciones en columnas: usar .T."""
        return vector_entrada(self.puntos, self.es_izquierda, aspecto=self.aspecto,
                              rotar=rotar, usar_z=usar_z)


def cargar_landmarks(ruta_csv):
    """Lee el CSV de landmarks.py; guarda una copia .npz al lado para que la
    siguiente carga tarde segundos y no un minuto."""
    ruta_csv = Path(ruta_csv)
    cache = ruta_csv.with_suffix(".npz")
    if cache.exists() and cache.stat().st_mtime > ruta_csv.stat().st_mtime:
        d = np.load(cache, allow_pickle=False)
        return Conjunto(**{c: d[c] for c in d.files if c != "n_sin_mano"}, n_sin_mano=int(d["n_sin_mano"]))

    filas, sin_mano = [], 0
    with ruta_csv.open(newline="") as f:
        for r in csv.DictReader(f):
            if r["detectada"] != "1":
                sin_mano += 1
                continue
            filas.append(r)
    datos = {
        "puntos": np.array([[float(r[c]) for c in COORDENADAS] for r in filas]).reshape(-1, 21, 3),
        "es_izquierda": np.array([r["lateralidad"] == "Left" for r in filas]),
        "aspecto": np.array([int(r["ancho"]) / int(r["alto"]) for r in filas]),
        "letra": np.array([INDICE_LETRA[r["letra"]] for r in filas]),
        "participante": np.array([int(r["participante"]) for r in filas]),
        "grupo": np.array([r["grupo"] for r in filas]),
        "particion": np.array([r["particion"] for r in filas]),
    }
    np.savez(cache, **datos, n_sin_mano=sin_mano)
    return Conjunto(**datos, n_sin_mano=sin_mano)


def una_caliente(letras):
    """Y one-hot en columnas: K × n."""
    return np.eye(K)[:, letras]


@dataclass
class ConfigRed:
    anchos_ocultos: tuple = (128, 64)
    eta: float = 0.05
    mu: float = 1e-4
    epocas: int = 20
    lote: int = 64
    semilla: int = 0


def entrenar_red(X, letras, config, al_terminar_epoca=None):
    """X: (n, d_0), una fila por observación. Devuelve la red entrenada."""
    anchos = (X.shape[1], *config.anchos_ocultos, K)
    activaciones = ("relu",) * len(config.anchos_ocultos) + ("softmax",)
    red = red_inicial(anchos, activaciones, semilla=config.semilla)
    red, _ = entrenar(red, X.T, una_caliente(letras), config.eta, EntropiaCruzada(),
                      Castigo(config.mu), config.epocas, config.lote, config.semilla,
                      al_terminar_epoca)
    return red


def predecir_letras(red, X):
    return predecir(red, X.T).argmax(axis=0)
