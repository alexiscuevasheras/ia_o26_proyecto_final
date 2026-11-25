"""Exporta la GRU de palabras para la página y genera la prueba de paridad.

Uso:
    python -m lsm.palabras.exportar modelo modelos/palabras.json \
        --uaemex datos/secuencias_uaemex.npz [--iteso datos/secuencias_iteso.npz]
    python -m lsm.palabras.exportar paridad web/fixtures/paridad_palabras.json

El modelo final se entrena con todas las personas menos las de prueba.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from redneuronal.recurrente import gru_inicial, predecir

from .conjunto import cargar, personas_de_prueba
from .modelos import Estandar, ModeloGRU
from .secuencias import D, T, secuencia


def _f32(a):
    return np.asarray(a, dtype=np.float32).astype(float).round(6).tolist()


def modelo_a_json(red, estandar, glosas):
    return {
        "tipo": "gru", "T": T, "D": D, "glosas": list(glosas),
        "params": {k: _f32(v) for k, v in red.p.items()},
        "estandar": {"media": _f32(estandar.media), "desv": _f32(estandar.desv)},
        "features": {"version": 1},
    }


def calibracion(ruta_p5):
    """Temperatura y umbral del «no sé» elegidos en P5, o (1, None) sin ese archivo."""
    if ruta_p5 is None or not Path(ruta_p5).exists():
        return 1.0, None
    p5 = json.loads(Path(ruta_p5).read_text())
    return p5["temperatura_final"], p5["umbral_elegido"]["umbral"]


def entrenar_final(rutas):
    datos = cargar(rutas)
    datos = datos.filtrar(~np.isin(datos.persona, list(personas_de_prueba(datos.persona))))
    glosas = sorted(set(datos.glosa))
    indice = {g: k for k, g in enumerate(glosas)}
    modelo = ModeloGRU().ajustar(datos.X, np.array([indice[g] for g in datos.glosa]), len(glosas))
    return modelo_a_json(modelo.red, modelo.estandar, glosas)


def cuadros_de_prueba(rng, F):
    """Una seña sintética: cuerpo quieto, mano derecha que se mueve, mano
    izquierda con huecos (cortos y largos) y un cuadro sin cuerpo."""
    cuerpo = np.full((F, 33, 4), np.nan)
    base = rng.uniform(0.3, 0.7, (33, 2))
    cuerpo[:, :, :2] = base + rng.normal(0, 0.005, (F, 33, 2))
    cuerpo[:, :, 2:] = rng.uniform(0, 1, (F, 33, 2))
    if F > 2:
        cuerpo[F // 2] = np.nan
    manos = rng.uniform(0.2, 0.8, (F, 2, 21, 3)) * np.array([1, 1, 0.1])
    manos[:, 1] += np.linspace(0, 0.2, F)[:, None, None]
    if F > 3:
        manos[[1, 2], 0] = np.nan  # hueco corto: se interpola
    if F > 9:
        manos[6:, 0] = np.nan  # hueco largo: no se interpola
    return cuerpo, manos


def a_cuadros_js(cuerpo, manos):
    def lista(a):
        return None if np.isnan(a).any() else a.round(8).tolist()
    return [{"cuerpo": lista(c), "manos": [lista(m[0]), lista(m[1])]} for c, m in zip(cuerpo, manos)]


def caso_de_paridad(cuerpo, manos, aspecto, dominante, respaldo):
    x = secuencia(cuerpo, manos, aspecto, dominante, T, None if respaldo is None else np.array(respaldo))
    return {"cuadros": a_cuadros_js(cuerpo, manos), "aspecto": aspecto, "dominante": dominante,
            "respaldo": respaldo, "x": x.round(8).tolist()}


def generar_paridad(semilla=0):
    rng = np.random.default_rng(semilla)
    casos = []
    for F, aspecto, dominante in [(12, 4 / 3, None), (7, 16 / 9, 0), (20, 1.0, 1), (1, 4 / 3, None)]:
        cuerpo, manos = cuadros_de_prueba(rng, F)
        casos.append(caso_de_paridad(cuerpo, manos, aspecto, dominante, None))
    sin_cuerpo, manos = cuadros_de_prueba(rng, 9)
    sin_cuerpo[:] = np.nan
    casos.append(caso_de_paridad(sin_cuerpo, manos, 1.0, None, None))
    casos.append(caso_de_paridad(sin_cuerpo, manos, 1.0, None, [0.45, 0.35]))  # con centro de respaldo
    cuerpo, manos = cuadros_de_prueba(rng, 10)
    cuerpo[:, 11, :2], cuerpo[:, 12, :2] = (0.62, 0.4), (0.38, 0.4)  # hombros plausibles
    manos[:, 1, :, :2] *= 0.3  # palmas chicas: escala comparable a los hombros
    casos.append(caso_de_paridad(cuerpo, manos, 4 / 3, None, None))

    red = gru_inicial(D, 8, 5, semilla=semilla)
    for v in red.p.values():
        v += rng.normal(0, 0.1, v.shape)
    estandar = Estandar(rng.normal(0, 0.1, D), rng.uniform(0.5, 2, D))
    modelo = modelo_a_json(red, estandar, [f"G{k}" for k in range(5)])
    red_f32 = red.copia()
    for k in red_f32.p:
        red_f32.p[k] = np.array(modelo["params"][k])
    est = Estandar(np.array(modelo["estandar"]["media"]), np.array(modelo["estandar"]["desv"]))
    for caso in casos:
        X = est(np.array(caso["x"]))[:, :, None]
        caso["probabilidades"] = predecir(red_f32, X)[:, 0].tolist()
    return {"modelo": modelo, "casos": casos}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="que", required=True)
    m = sub.add_parser("modelo")
    m.add_argument("salida", type=Path)
    m.add_argument("--uaemex", type=Path, required=True)
    m.add_argument("--iteso", type=Path)
    m.add_argument("--propios", type=Path)
    m.add_argument("--no-se", type=Path, default=Path("resultados/p5_no_se.json"),
                   help="resultados de P5: temperatura y umbral del «no sé»")
    p = sub.add_parser("paridad")
    p.add_argument("salida", type=Path)
    args = ap.parse_args()

    if args.que == "modelo":
        datos = entrenar_final([args.uaemex] + [r for r in (args.iteso, args.propios) if r])
        datos["temperatura"], datos["umbral"] = calibracion(args.no_se)
    else:
        datos = generar_paridad()
    args.salida.parent.mkdir(parents=True, exist_ok=True)
    args.salida.write_text(json.dumps(datos, ensure_ascii=False))
    print(f"escrito {args.salida} ({args.salida.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
