"""Ningún participante puede estar en dos conjuntos (prueba 7 del plan)."""

import pytest

from lsm.datos import (
    PARTICIPANTES_PRUEBA,
    leer_nombre,
    pliegues_por_participante,
    verificar_sin_fugas,
)

ENTRENAMIENTO_OFICIAL = [p for p in range(1, 21) if p not in PARTICIPANTES_PRUEBA]


def test_leer_nombre():
    foto = leer_nombre("MSL-ABC/lsm-abc-B/train/N/S16-N-1-32.jpg")
    assert (foto.grupo, foto.particion, foto.letra) == ("B", "train", "N")
    assert (foto.participante, foto.codigo, foto.n) == (16, 1, 32)


@pytest.mark.parametrize("ruta", [
    "MSL-ABC/lsm-abc-A/train/N/S4-M-4-1.jpg",  # la letra de la carpeta no coincide
    "MSL-ABC/lsm-abc-A/train/N/foto.jpg",
    "MSL-ABC/lsm-abc-D/train/N/S4-N-4-1.jpg",
])
def test_rutas_invalidas(ruta):
    with pytest.raises(ValueError):
        leer_nombre(ruta)


def test_pliegues_cubren_a_todos_sin_repetir():
    pliegues = pliegues_por_participante(ENTRENAMIENTO_OFICIAL, k=6, semilla=1)
    assert len(pliegues) == 6
    assert set().union(*pliegues) == set(ENTRENAMIENTO_OFICIAL)
    assert sum(len(p) for p in pliegues) == len(ENTRENAMIENTO_OFICIAL)
    verificar_sin_fugas({f"pliegue {i}": p for i, p in enumerate(pliegues)})


def test_cada_validacion_queda_separada_de_su_entrenamiento_y_de_la_prueba():
    pliegues = pliegues_por_participante(ENTRENAMIENTO_OFICIAL, k=6)
    for i, validacion in enumerate(pliegues):
        entrenamiento = set().union(*(p for j, p in enumerate(pliegues) if j != i))
        verificar_sin_fugas({
            "entrenamiento": entrenamiento,
            "validacion": validacion,
            "prueba": PARTICIPANTES_PRUEBA,
        })


def test_detecta_una_fuga():
    with pytest.raises(ValueError, match="participante 3"):
        verificar_sin_fugas({"entrenamiento": [1, 2, 3], "validacion": [3, 4]})
