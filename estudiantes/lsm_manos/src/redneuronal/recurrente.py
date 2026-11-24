"""Red recurrente GRU para secuencias, escrita desde cero (extensión del curso).

La unidad 10 cubre redes de capas densas. Una seña es una secuencia
x_1, …, x_T, así que se agrega una celda recurrente con compuertas (GRU, Cho
et al. 2014) y la retropropagación en el tiempo (BPTT). Misma convención que
red.py: observaciones en columnas, X de forma T × d × m.

    z_t = σ(W_z x_t + U_z h_{t−1} + b_z)            compuerta de actualización
    r_t = σ(W_r x_t + U_r h_{t−1} + b_r)            compuerta de reinicio
    h̃_t = tanh(W_h x_t + U_h (r_t ⊙ h_{t−1}) + b_h)  candidato
    h_t = (1 − z_t) ⊙ h_{t−1} + z_t ⊙ h̃_t,  h_0 = 0
    ŷ  = softmax(V [h_T; h̄] + c),  h̄ = (1/T) Σ_t h_t

La salida ve el último estado y el promedio de todos: el promedio deja que el
gradiente llegue directo a cada t y hace el entrenamiento más estable.
"""

from dataclasses import dataclass, field

import numpy as np

from .activaciones import ACTIVACIONES

COMPUERTAS = ("z", "r", "h")
_softmax = ACTIVACIONES["softmax"][0]


def sigma(x):
    return 0.5 * (1.0 + np.tanh(0.5 * x))  # σ(x) estable para |x| grande


@dataclass
class GRU:
    """Parámetros en un diccionario: W_z, U_z, b_z, …, V, c."""
    p: dict = field(default_factory=dict)

    @property
    def d_entrada(self):
        return self.p["W_z"].shape[1]

    @property
    def d_oculta(self):
        return self.p["U_z"].shape[0]

    @property
    def K(self):
        return self.p["V"].shape[0]

    def copia(self):
        return GRU({k: v.copy() for k, v in self.p.items()})

    def num_parametros(self):
        return sum(v.size for v in self.p.values())


def gru_inicial(d, n, K, semilla=0):
    """W de entrada con escala √(1/d), U ortogonales (estables en el tiempo),
    sesgo de la compuerta z en −1: al inicio la celda tiende a recordar."""
    rng = np.random.default_rng(semilla)
    p = {}
    for g in COMPUERTAS:
        p[f"W_{g}"] = rng.standard_normal((n, d)) * np.sqrt(1.0 / d)
        p[f"U_{g}"] = np.linalg.qr(rng.standard_normal((n, n)))[0]
        p[f"b_{g}"] = np.zeros(n)
    p["b_z"][:] = -1.0
    p["V"] = rng.standard_normal((K, 2 * n)) * np.sqrt(1.0 / (2 * n))
    p["c"] = np.zeros(K)
    return GRU(p)


def adelante(red, X):
    """X: T × d × m. Devuelve (Ŷ de K × m, caché para la retropropagación)."""
    p = red.p
    T, _, m = X.shape
    n = red.d_oculta
    H = np.zeros((T + 1, n, m))
    Zs, Rs, Cs = (np.zeros((T, n, m)) for _ in range(3))
    for t in range(T):
        h = H[t]
        Zs[t] = sigma(p["W_z"] @ X[t] + p["U_z"] @ h + p["b_z"][:, None])
        Rs[t] = sigma(p["W_r"] @ X[t] + p["U_r"] @ h + p["b_r"][:, None])
        Cs[t] = np.tanh(p["W_h"] @ X[t] + p["U_h"] @ (Rs[t] * h) + p["b_h"][:, None])
        H[t + 1] = (1 - Zs[t]) * h + Zs[t] * Cs[t]
    resumen = np.vstack([H[T], H[1:].mean(axis=0)])
    Y_gorro = _softmax(p["V"] @ resumen + p["c"][:, None])
    return Y_gorro, (X, H, Zs, Rs, Cs, resumen)


def predecir(red, X):
    return adelante(red, X)[0]


def entropia_cruzada(Y_gorro, Y):
    return float(-np.mean(np.sum(Y * np.log(np.clip(Y_gorro, 1e-12, None)), axis=0)))


def gradiente(red, X, Y, mu=0.0):
    """BPTT de un lote. Devuelve (pérdida, {nombre: ∇}). Con softmax y entropía
    cruzada el error de salida es ŷ − y, igual que en red.py."""
    p = red.p
    Y_gorro, (X, H, Zs, Rs, Cs, resumen) = adelante(red, X)
    T, _, m = X.shape
    n = red.d_oculta
    g = {k: np.zeros_like(v) for k, v in p.items()}

    delta = (Y_gorro - Y) / m
    g["V"] = delta @ resumen.T
    g["c"] = delta.sum(axis=1)
    d_resumen = p["V"].T @ delta
    d_promedio = d_resumen[n:] / T  # cada h_t recibe 1/T del promedio
    dh = d_resumen[:n] + d_promedio
    for t in range(T - 1, -1, -1):
        h_prev = H[t]
        dz = dh * (Cs[t] - h_prev)
        dc = dh * Zs[t]
        dh_prev = dh * (1 - Zs[t])
        dac = dc * (1 - Cs[t] ** 2)  # tanh' = 1 − tanh²
        g["W_h"] += dac @ X[t].T
        g["U_h"] += dac @ (Rs[t] * h_prev).T
        g["b_h"] += dac.sum(axis=1)
        d_rh = p["U_h"].T @ dac
        dr = d_rh * h_prev
        dh_prev += d_rh * Rs[t]
        daz = dz * Zs[t] * (1 - Zs[t])  # σ' = σ(1 − σ)
        dar = dr * Rs[t] * (1 - Rs[t])
        for nombre, da in (("z", daz), ("r", dar)):
            g[f"W_{nombre}"] += da @ X[t].T
            g[f"U_{nombre}"] += da @ h_prev.T
            g[f"b_{nombre}"] += da.sum(axis=1)
            dh_prev += p[f"U_{nombre}"].T @ da
        dh = dh_prev + (d_promedio if t > 0 else 0)
    perdida = entropia_cruzada(Y_gorro, Y)
    if mu:
        for k in p:
            if k[0] in "WUV":  # ridge solo en pesos, no en sesgos, como en el curso
                perdida += mu * float(np.sum(p[k] ** 2))
                g[k] += 2 * mu * p[k]
    return perdida, g


@dataclass
class Adam:
    """Adam (Kingma y Ba, 2015): descenso con promedios móviles del gradiente
    (m) y de su cuadrado (v), corregidos por sesgo. Extensión del curso: con
    SGD simple la GRU tarda mucho más en entrenar."""
    eta: float = 2e-3
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    recorte: float = 5.0  # norma máxima del gradiente: evita saltos por gradientes que explotan
    t: int = 0
    m: dict = field(default_factory=dict)
    v: dict = field(default_factory=dict)

    def paso(self, params, grads):
        norma = np.sqrt(sum(float(np.sum(gr ** 2)) for gr in grads.values()))
        factor = min(1.0, self.recorte / (norma + 1e-12))
        self.t += 1
        for k, gr in grads.items():
            gr = gr * factor
            self.m[k] = self.beta1 * self.m.get(k, 0) + (1 - self.beta1) * gr
            self.v[k] = self.beta2 * self.v.get(k, 0) + (1 - self.beta2) * gr ** 2
            m_hat = self.m[k] / (1 - self.beta1 ** self.t)
            v_hat = self.v[k] / (1 - self.beta2 ** self.t)
            params[k] -= self.eta * m_hat / (np.sqrt(v_hat) + self.eps)


def entrenar(red, X, Y, epocas, lote=32, eta=2e-3, mu=0.0, semilla=0, al_terminar_epoca=None):
    """X: T × d × n, Y: K × n. Devuelve la red entrenada (una copia)."""
    red = red.copia()
    rng = np.random.default_rng(semilla)
    opt = Adam(eta=eta)
    n = X.shape[2]
    for epoca in range(epocas):
        orden = rng.permutation(n)
        for inicio in range(0, n, lote):
            idx = orden[inicio:inicio + lote]
            _, g = gradiente(red, X[:, :, idx], Y[:, idx], mu)
            opt.paso(red.p, g)
        if al_terminar_epoca is not None:
            al_terminar_epoca(epoca, red)
    return red
