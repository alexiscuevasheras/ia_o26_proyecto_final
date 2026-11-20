"""El archivo de paridad está al día y JavaScript lo reproduce (prueba 6)."""

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from lsm.exportar import generar_paridad, red_a_json, red_desde_json
from redneuronal.red import red_inicial

CARPETA = Path(__file__).resolve().parent.parent
PARIDAD = CARPETA / "web" / "fixtures" / "paridad.json"


def test_archivo_de_paridad_al_dia():
    """Si cambian features.py o la red, hay que regenerarlo:
    python -m lsm.exportar paridad web/fixtures/paridad.json"""
    guardado = json.loads(PARIDAD.read_text())
    assert guardado == json.loads(json.dumps(generar_paridad()))


def test_ida_y_vuelta_por_json():
    red = red_inicial((60, 16, 21), ("relu", "softmax"), semilla=3)
    vuelta = red_desde_json(red_a_json(red, "ABCDEFGHILMNOPRSTUVWY"))
    assert vuelta.anchos == red.anchos and vuelta.g == red.g
    np.testing.assert_allclose(vuelta.W[1], red.W[1], rtol=1e-6)  # float32


@pytest.mark.skipif(shutil.which("node") is None, reason="node no está instalado")
def test_javascript_reproduce_a_python():
    resultado = subprocess.run(["node", "--test", "tests/web/paridad.test.mjs"], cwd=CARPETA,
                               capture_output=True, text=True)
    assert resultado.returncode == 0, resultado.stdout + resultado.stderr
