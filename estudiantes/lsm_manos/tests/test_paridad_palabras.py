"""Las features de secuencias y la GRU dan lo mismo en Python y en JavaScript."""

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from lsm.palabras.exportar import generar_paridad
from lsm.palabras.secuencias import D, T, secuencia

CARPETA = Path(__file__).resolve().parent.parent
PARIDAD = CARPETA / "web" / "fixtures" / "paridad_palabras.json"


def test_archivo_de_paridad_al_dia():
    """Si cambian secuencias.py o la GRU, hay que regenerarlo:
    python -m lsm.palabras.exportar paridad web/fixtures/paridad_palabras.json"""
    assert json.loads(PARIDAD.read_text()) == json.loads(json.dumps(generar_paridad()))


def test_zurdo_y_diestro_dan_la_misma_entrada():
    """Espejar toda la escena (x → 1 − x) e intercambiar los brazos no cambia la entrada."""
    rng = np.random.default_rng(0)
    F = 10
    cuerpo = np.zeros((F, 33, 4))
    cuerpo[:, :, :2] = rng.uniform(0.3, 0.7, (33, 2))
    cuerpo[:, 11, :2], cuerpo[:, 12, :2] = (0.6, 0.4), (0.4, 0.4)
    manos = rng.uniform(0.2, 0.8, (F, 2, 21, 3)) * np.array([1, 1, 0.1])
    manos[:, 1] += np.linspace(0, 0.3, F)[:, None, None]  # el brazo derecho es el que se mueve
    espejo_cuerpo = cuerpo.copy()
    espejo_cuerpo[..., 0] = 1 - cuerpo[..., 0]
    espejo_cuerpo[:, [11, 12]] = espejo_cuerpo[:, [12, 11]]
    espejo_manos = manos[:, ::-1].copy()
    espejo_manos[..., 0] = 1 - espejo_manos[..., 0]
    a = secuencia(cuerpo, manos, 1.0)
    b = secuencia(espejo_cuerpo, espejo_manos, 1.0)
    assert a.shape == (T, D)
    np.testing.assert_allclose(a, b, atol=1e-9)


@pytest.mark.skipif(shutil.which("node") is None, reason="node no está instalado")
def test_javascript_reproduce_a_python():
    resultado = subprocess.run(["node", "--test", "tests/web/secuencias.test.mjs"], cwd=CARPETA,
                               capture_output=True, text=True)
    assert resultado.returncode == 0, resultado.stdout + resultado.stderr
