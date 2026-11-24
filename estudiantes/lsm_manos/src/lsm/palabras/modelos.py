"""Los clasificadores de palabras: todos reciben X (n, T, D) y glosas como índices.

- `gru`: la GRU desde cero de redneuronal/recurrente.py.
- `mlp`: la red densa de la unidad 10 sobre la secuencia aplanada (T·D).
- Líneas base de scikit-learn (solo para comparar): kNN, regresión logística, SVM.

Las features se estandarizan con media y desviación del entrenamiento.
"""

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from redneuronal import recurrente
from redneuronal.optimizador import entrenar as entrenar_densa
from redneuronal.perdida import Castigo, EntropiaCruzada
from redneuronal.red import predecir as predecir_densa
from redneuronal.red import red_inicial

from .aumento import aumentar


@dataclass
class Estandar:
    media: np.ndarray
    desv: np.ndarray

    @staticmethod
    def ajustar(X):
        plano = X.reshape(-1, X.shape[-1])
        return Estandar(plano.mean(axis=0), plano.std(axis=0) + 1e-6)

    def __call__(self, X):
        return (X - self.media) / self.desv


@dataclass
class ConfigGRU:
    oculta: int = 128
    epocas: int = 60
    lote: int = 32
    eta: float = 2e-3
    mu: float = 1e-5
    aumentar: bool = True
    semilla: int = 0


@dataclass
class ConfigMLP:
    anchos_ocultos: tuple = (256, 128)
    epocas: int = 60
    lote: int = 32
    eta: float = 0.02
    mu: float = 1e-4
    aumentar: bool = True
    semilla: int = 0


class ModeloGRU:
    def __init__(self, config=ConfigGRU()):
        self.config = config

    def ajustar(self, X, y, K, al_terminar_epoca=None):
        c = self.config
        self.estandar = Estandar.ajustar(X)
        rng = np.random.default_rng(c.semilla)
        red = recurrente.gru_inicial(X.shape[2], c.oculta, K, c.semilla)
        opt = recurrente.Adam(eta=c.eta)
        for epoca in range(c.epocas):
            Xe = aumentar(X, rng) if c.aumentar else X
            Xe = np.transpose(self.estandar(Xe), (1, 2, 0))  # T × D × n
            Y = np.eye(K)[:, y]
            orden = rng.permutation(len(y))
            for i in range(0, len(y), c.lote):
                idx = orden[i:i + c.lote]
                _, g = recurrente.gradiente(red, Xe[:, :, idx], Y[:, idx], c.mu)
                opt.paso(red.p, g)
            if al_terminar_epoca:
                self.red = red
                al_terminar_epoca(epoca, self)
        self.red = red
        return self

    def probabilidades(self, X):
        """(n, K)."""
        Xe = np.transpose(self.estandar(X), (1, 2, 0))
        return np.concatenate([recurrente.predecir(self.red, Xe[:, :, i:i + 512]).T
                               for i in range(0, Xe.shape[2], 512)])


class ModeloMLP:
    def __init__(self, config=ConfigMLP()):
        self.config = config

    def ajustar(self, X, y, K, al_terminar_epoca=None):
        c = self.config
        self.estandar = Estandar.ajustar(X)
        rng = np.random.default_rng(c.semilla)
        anchos = (X.shape[1] * X.shape[2], *c.anchos_ocultos, K)
        red = red_inicial(anchos, ("relu",) * len(c.anchos_ocultos) + ("softmax",), c.semilla)
        Y = np.eye(K)[:, y]
        for epoca in range(c.epocas):
            Xe = aumentar(X, rng) if c.aumentar else X
            plano = self.estandar(Xe).reshape(len(y), -1).T
            red, _ = entrenar_densa(red, plano, Y, c.eta, EntropiaCruzada(), Castigo(c.mu), 1, c.lote,
                                    c.semilla + epoca)
            if al_terminar_epoca:
                self.red = red
                al_terminar_epoca(epoca, self)
        self.red = red
        return self

    def probabilidades(self, X):
        return predecir_densa(self.red, self.estandar(X).reshape(len(X), -1).T).T


class ModeloSklearn:
    """Envuelve un clasificador de scikit-learn sobre la secuencia aplanada."""

    def __init__(self, fabrica):
        self.fabrica = fabrica

    def ajustar(self, X, y, K, al_terminar_epoca=None):
        self.estandar = Estandar.ajustar(X)
        self.K = K
        self.modelo = self.fabrica().fit(self.estandar(X).reshape(len(X), -1), y)
        return self

    def probabilidades(self, X):
        plano = self.estandar(X).reshape(len(X), -1)
        P = np.zeros((len(X), self.K))
        if hasattr(self.modelo, "predict_proba"):
            P[:, self.modelo.classes_] = self.modelo.predict_proba(plano)
        else:
            P[np.arange(len(X)), self.modelo.predict(plano)] = 1
        return P


MODELOS = {
    "knn": lambda: ModeloSklearn(lambda: KNeighborsClassifier(1)),
    "logistica": lambda: ModeloSklearn(lambda: LogisticRegression(C=0.1, max_iter=2000)),
    "svm": lambda: ModeloSklearn(lambda: SVC(C=10, gamma="scale")),
    "mlp": lambda: ModeloMLP(),
    "gru": lambda: ModeloGRU(),
}
