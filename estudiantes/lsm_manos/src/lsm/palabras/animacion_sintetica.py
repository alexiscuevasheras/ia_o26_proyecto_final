"""Animaciones SINTÉTICAS para probar web/voz.html sin datos reales.

Nada de este archivo sale de una grabación: el cuerpo, las manos y las cuatro
señas (ADIOS, YO, COMER, GRACIAS) están hechos a mano con geometría simple, y
las 21 letras son formas de mano aproximadas, no una referencia de la LSM.
Sirven solo para que la página se pueda probar mientras no exista
modelos/animaciones.json. El JSON lleva "sintetico": true.

Sigue el formato de docs/FORMATOS.md: metros, origen entre los hombros,
x a la izquierda de la persona, y hacia arriba, z hacia la cámara.

Uso:
    python -m lsm.palabras.animacion_sintetica web/fixtures/animaciones_prueba.json
"""

import argparse
import json
from pathlib import Path

import numpy as np

FPS = 15
CUERPO_INDICES = [0, 2, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 23, 24]
LETRAS_CON_MOVIMIENTO = ["J", "K", "Ñ", "Q", "X", "Z"]

CABEZA = {
    0: (0.0, 0.22, 0.10), 2: (0.035, 0.25, 0.08), 5: (-0.035, 0.25, 0.08),
    7: (0.075, 0.23, 0.0), 8: (-0.075, 0.23, 0.0), 9: (0.025, 0.18, 0.085), 10: (-0.025, 0.18, 0.085),
}
HOMBROS = {"izq": np.array([0.18, 0.0, 0.0]), "der": np.array([-0.18, 0.0, 0.0])}
CADERAS = {23: (0.13, -0.50, 0.0), 24: (-0.13, -0.50, 0.0)}
BRAZO, ANTEBRAZO = 0.28, 0.25
MUNECA_REPOSO = {"izq": np.array([0.21, -0.52, 0.04]), "der": np.array([-0.21, -0.52, 0.04])}

# ---------- La mano: modelo de 21 puntos con ángulos de flexión ----------
# Marco local de la mano derecha: dedos hacia +y, palma hacia +z, pulgar hacia +x.

NUDILLOS = {  # punto MCP de cada dedo; ‖p9‖ = 0.09 m, como pide FORMATOS.md
    "indice": (5, (0.025, 0.085, 0.0), (0.040, 0.024, 0.020), 0.08),
    "medio": (9, (0.0, 0.09, 0.0), (0.045, 0.028, 0.022), 0.0),
    "anular": (13, (-0.020, 0.084, 0.0), (0.042, 0.026, 0.021), -0.07),
    "menique": (17, (-0.038, 0.074, 0.0), (0.032, 0.019, 0.018), -0.15),
}
# Cuánto se abre cada dedo con la «abertura» de la letra; negativa cruza índice y medio (R).
ABRE = {"indice": 1.0, "medio": -0.3, "anular": -0.8, "menique": -1.2}
PULGAR_BASE = np.array([0.022, 0.022, 0.008])
PULGAR_LARGOS = (0.040, 0.032, 0.027)

ESTIRADO, CURVO, DOBLADO, PUNO = (0, 0, 0), (0.6, 0.8, 0.4), (1.2, 1.4, 0.8), (1.5, 1.7, 0.9)

# Pulgar: (dirección inicial, hacia dónde se dobla, ángulos acumulados por falange)
PULGAR_AFUERA = ((1.0, 0.6, 0.15), (-0.3, 0.3, 1.0), (0.0, 0.1, 0.1))
PULGAR_AL_LADO = ((0.3, 1.0, 0.35), (-0.3, 0.2, 1.0), (0.0, 0.1, 0.1))
PULGAR_CRUZADO = ((0.3, 0.8, 0.6), (-1.0, 0.0, 0.3), (0.4, 0.9, 0.5))
PULGAR_OPUESTO = ((0.3, 0.7, 0.8), (-0.4, 0.6, 0.1), (0.2, 0.5, 0.4))
PULGAR_ARRIBA = ((0.5, 1.0, 0.4), (-0.6, 0.2, 0.8), (0.0, 0.2, 0.2))

LETRAS = {  # (índice, medio, anular, meñique, pulgar, abertura extra de los dedos)
    "A": (PUNO, PUNO, PUNO, PUNO, PULGAR_AL_LADO, 0.0),
    "B": (ESTIRADO, ESTIRADO, ESTIRADO, ESTIRADO, PULGAR_CRUZADO, 0.0),
    "C": (CURVO, CURVO, CURVO, CURVO, PULGAR_OPUESTO, 0.0),
    "D": (ESTIRADO, DOBLADO, DOBLADO, DOBLADO, PULGAR_OPUESTO, 0.0),
    "E": ((0.3, 1.8, 1.0),) * 4 + (PULGAR_CRUZADO, 0.0),
    "F": ((0.9, 1.0, 0.6), ESTIRADO, ESTIRADO, ESTIRADO, PULGAR_OPUESTO, 0.08),
    "G": (ESTIRADO, PUNO, PUNO, PUNO, PULGAR_ARRIBA, 0.0),
    "H": (ESTIRADO, ESTIRADO, PUNO, PUNO, PULGAR_CRUZADO, 0.0),
    "I": (PUNO, PUNO, PUNO, ESTIRADO, PULGAR_CRUZADO, 0.0),
    "L": (ESTIRADO, PUNO, PUNO, PUNO, PULGAR_AFUERA, 0.0),
    "M": ((1.4, 1.2, 0.6),) * 3 + (PUNO, PULGAR_CRUZADO, 0.0),
    "N": ((1.4, 1.2, 0.6),) * 2 + (PUNO, PUNO, PULGAR_CRUZADO, 0.0),
    "O": ((0.8, 1.0, 0.6),) * 4 + (PULGAR_OPUESTO, 0.0),
    "P": (ESTIRADO, (0.9, 0.2, 0.1), PUNO, PUNO, PULGAR_ARRIBA, 0.06),
    "R": (ESTIRADO, ESTIRADO, PUNO, PUNO, PULGAR_CRUZADO, -0.12),
    "S": (PUNO, PUNO, PUNO, PUNO, PULGAR_CRUZADO, 0.0),
    "T": ((1.2, 1.3, 0.7), PUNO, PUNO, PUNO, PULGAR_ARRIBA, 0.0),
    "U": (ESTIRADO, ESTIRADO, PUNO, PUNO, PULGAR_CRUZADO, 0.0),
    "V": (ESTIRADO, ESTIRADO, PUNO, PUNO, PULGAR_CRUZADO, 0.22),
    "W": (ESTIRADO, ESTIRADO, ESTIRADO, PUNO, PULGAR_CRUZADO, 0.2),
    "Y": (PUNO, PUNO, PUNO, ESTIRADO, PULGAR_AFUERA, 0.15),
}
RELAJADA = ((0.3, 0.4, 0.2),) * 4 + (PULGAR_AL_LADO, 0.0)


def _unitario(v):
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


def _falanges(base, direccion, doblez, angulos, largos):
    """Puntos tras la base; cada falange gira más hacia `doblez` (ángulos acumulados)."""
    doblez = _unitario(doblez - np.dot(doblez, direccion) * direccion)
    puntos, actual, theta = [], np.asarray(base, dtype=float), 0.0
    for angulo, largo in zip(angulos, largos):
        theta += angulo
        actual = actual + largo * (np.cos(theta) * direccion + np.sin(theta) * doblez)
        puntos.append(actual)
    return puntos


def forma_mano(indice, medio, anular, menique, pulgar, abertura):
    """21 puntos (orden de MediaPipe) de una mano derecha en su marco local."""
    mano = np.zeros((21, 3))
    direccion, doblez, angulos = pulgar
    mano[1] = PULGAR_BASE
    mano[2:5] = _falanges(PULGAR_BASE, _unitario(direccion), np.asarray(doblez, float), angulos, PULGAR_LARGOS)
    flexiones = {"indice": indice, "medio": medio, "anular": anular, "menique": menique}
    for nombre, (i, nudillo, largos, separacion) in NUDILLOS.items():
        s = separacion + abertura * ABRE[nombre]
        mano[i] = nudillo
        arriba = np.array([np.sin(s), np.cos(s), 0.0])
        mano[i + 1:i + 4] = _falanges(nudillo, arriba, np.array([0.0, 0.0, 1.0]), flexiones[nombre], largos)
    return mano


def orientar(forma, dedos_hacia, palma_hacia):
    """Gira la mano local: +y → `dedos_hacia`, +z → `palma_hacia` (ortogonalizado)."""
    y = _unitario(dedos_hacia)
    z = _unitario(np.asarray(palma_hacia, float) - np.dot(palma_hacia, y) * y)
    x = np.cross(y, z)
    return forma @ np.stack([x, y, z])


def espejo(forma):
    """La mano izquierda es la derecha reflejada en x."""
    return forma * np.array([-1.0, 1.0, 1.0])


# ---------- El cuerpo ----------

def codo_ik(hombro, muneca, polo):
    """IK de dos huesos: el codo que deja la muñeca en su lugar, doblado hacia `polo`."""
    hacia = muneca - hombro
    d = np.clip(np.linalg.norm(hacia), abs(BRAZO - ANTEBRAZO) + 1e-3, BRAZO + ANTEBRAZO - 1e-3)
    eje = _unitario(hacia)
    coseno = (BRAZO**2 + d**2 - ANTEBRAZO**2) / (2 * BRAZO * d)
    lado = _unitario(polo - np.dot(polo, eje) * eje)
    return hombro + BRAZO * (coseno * eje + np.sqrt(1 - coseno**2) * lado)


def mano_reposo(lado):
    """Mano relajada colgando, palma hacia el muslo."""
    if lado == "der":
        return orientar(forma_mano(*RELAJADA), (0, -1, 0), (1, 0, 0))
    return orientar(espejo(forma_mano(*RELAJADA)), (0, -1, 0), (-1, 0, 0))


def cuadro(muneca_der=None, mano_der=None):
    """Un cuadro del formato; el brazo izquierdo queda en reposo."""
    munecas = {"izq": MUNECA_REPOSO["izq"], "der": MUNECA_REPOSO["der"] if muneca_der is None else muneca_der}
    manos = {"izq": mano_reposo("izq"), "der": mano_reposo("der") if mano_der is None else mano_der}
    polos = {"izq": np.array([0.4, -1.0, -0.2]), "der": np.array([-0.4, -1.0, -0.2])}
    codos = {lado: codo_ik(HOMBROS[lado], munecas[lado], polos[lado]) for lado in ("izq", "der")}
    puntos = {**CABEZA, **CADERAS, 11: HOMBROS["izq"], 12: HOMBROS["der"], 13: codos["izq"],
              14: codos["der"], 15: munecas["izq"], 16: munecas["der"]}
    return {
        "cuerpo": [_redondear(puntos[i]) for i in CUERPO_INDICES],
        "manos": [_redondear(manos[lado] + munecas[lado]) for lado in ("izq", "der")],
    }


def _redondear(arreglo):
    return np.round(np.asarray(arreglo, dtype=float), 4).tolist()


def _suave(u):
    u = np.clip(u, 0.0, 1.0)
    return u * u * (3 - 2 * u)


def _mezcla(a, b, u):
    return np.asarray(a, float) + (np.asarray(b, float) - np.asarray(a, float)) * _suave(u)


# ---------- Las señas: muñeca derecha y orientación de la mano en cada instante u ∈ [0, 1] ----------

def _adios(u):
    muneca = _mezcla([-0.20, -0.25, 0.15], [-0.26, 0.10, 0.12], u / 0.25)
    giro = 0.4 * np.sin(2 * np.pi * 2.5 * max(0.0, u - 0.25) / 0.75)
    return muneca, orientar(forma_mano(*LETRAS["B"]), (np.sin(giro), np.cos(giro), 0.0), (0, 0, 1))


def _yo(u):
    toque = 0.03 * abs(np.sin(2 * np.pi * max(0.0, u - 0.3) / 0.7 * 1.0))
    muneca = _mezcla([-0.15, -0.38, 0.20], [-0.07, -0.20, 0.17], u / 0.3) + [0, 0, toque]
    return muneca, orientar(forma_mano(*LETRAS["D"]), (0.3, 0.1, -1.0), (0, -1, 0))


def _comer(u):
    toque = 0.035 * abs(np.sin(2 * np.pi * max(0.0, u - 0.3) / 0.7))
    muneca = _mezcla([-0.12, -0.30, 0.22], [-0.06, 0.02, 0.20], u / 0.3) + [0, -toque, toque]
    return muneca, orientar(forma_mano(*LETRAS["O"]), (0.35, 0.6, -0.6), (1.0, 0.0, 0.3))


def _gracias(u):
    avance = (u - 0.3) / 0.7
    muneca = _mezcla(_mezcla([-0.12, -0.30, 0.18], [-0.04, 0.02, 0.15], u / 0.3), [-0.10, -0.14, 0.40], avance)
    dedos = _mezcla([0.1, 1.0, 0.2], [0.0, 0.2, 1.0], avance)
    palma = _mezcla([0.0, 0.0, -1.0], [0.0, 1.0, 0.0], avance)
    return muneca, orientar(forma_mano(*LETRAS["B"]), dedos, palma)


SENAS = {"ADIOS": (_adios, 24), "YO": (_yo, 18), "COMER": (_comer, 20), "GRACIAS": (_gracias, 20)}


def animaciones():
    glosas = {}
    for glosa, (sena, cuadros) in SENAS.items():
        glosas[glosa] = {
            "origen": "sintetico",
            "cuadros": [cuadro(*sena(n / (cuadros - 1))) for n in range(cuadros)],
        }
    return {
        "sintetico": True,
        "nota": "Datos inventados para probar la página; no son señas grabadas.",
        "fps": FPS,
        "cuerpo_indices": CUERPO_INDICES,
        "glosas": glosas,
        "letras": {letra: {"mano": _redondear(forma_mano(*forma))} for letra, forma in LETRAS.items()},
        "letras_con_movimiento": LETRAS_CON_MOVIMIENTO,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("salida", type=Path)
    args = parser.parse_args()
    args.salida.parent.mkdir(parents=True, exist_ok=True)
    args.salida.write_text(json.dumps(animaciones(), ensure_ascii=False, separators=(",", ":")) + "\n",
                           encoding="utf-8")
    print(f"animaciones sintéticas → {args.salida}")


if __name__ == "__main__":
    main()
