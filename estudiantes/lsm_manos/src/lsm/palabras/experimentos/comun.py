"""Piezas compartidas por los experimentos de palabras (P1–P5)."""

from concurrent.futures import ProcessPoolExecutor

import numpy as np

from ...evaluar import matriz_confusion
from ...experimentos.comun import resumir
from ..conjunto import cargar, personas_de_prueba
from ..secuencias import T


def preparar(rutas, t=T):
    """Conjunto sin las personas de prueba, etiquetas y lista de glosas."""
    datos = cargar(rutas, t)
    prueba = personas_de_prueba(datos.persona)
    datos = datos.filtrar(~np.isin(datos.persona, list(prueba)))
    glosas = sorted(set(datos.glosa))
    indice = {g: k for k, g in enumerate(glosas)}
    return datos, np.array([indice[g] for g in datos.glosa]), glosas, sorted(prueba)


def personas_de_validacion(datos):
    """Las de UAEMex (sin las de prueba) y las del dataset propio. Las de ITESO
    solo entrenan: no se sabe con certeza quién es quién."""
    return sorted({p for p in datos.persona if not p.startswith("iteso-")})


def top_k(P, y, k):
    mejores = np.argsort(-P, axis=1)[:, :k]
    return float(np.mean((mejores == y[:, None]).any(axis=1)))


def evaluar(P, y, K):
    """Top-1, top-5 y F1 macro promediado solo sobre las glosas que la persona
    tiene (las demás valdrían 0 y lo bajarían sin decir nada)."""
    C = matriz_confusion(y, P.argmax(axis=1), K)
    verdaderos = np.diag(C).astype(float)
    denominador = C.sum(axis=0) + C.sum(axis=1)
    f1 = np.divide(2 * verdaderos, denominador, out=np.zeros_like(verdaderos), where=denominador > 0)
    return {"top1": float(np.mean(P.argmax(axis=1) == y)), "top5": top_k(P, y, 5),
            "f1_macro": float(f1[np.unique(y)].mean())}


def resumen_con_peor(valores):
    return {**resumir(valores), "peor": float(np.min(valores))}


def resumir_metricas(por_persona):
    return {m: resumen_con_peor([r[m] for r in por_persona.values()]) for m in ("top1", "top5", "f1_macro")}


def _pliegue(tarea):
    fabrica, X, y, es_val, K = tarea
    modelo = fabrica().ajustar(X[~es_val], y[~es_val], K)
    return modelo.probabilidades(X[es_val])


def dejando_una_persona_fuera(fabrica, datos, X, y, K, procesos=6):
    """Entrena sin cada persona de validación y evalúa en sus señas.
    Devuelve ({persona: métricas}, {persona: (P, y)})."""
    validacion = personas_de_validacion(datos)
    mascaras = [datos.persona == p for p in validacion]
    with ProcessPoolExecutor(procesos) as ejecutor:
        probabilidades = list(ejecutor.map(_pliegue, [(fabrica, X, y, m, K) for m in mascaras]))
    salidas = {p: (P, y[m]) for p, P, m in zip(validacion, probabilidades, mascaras)}
    return {p: evaluar(P, yv, K) for p, (P, yv) in salidas.items()}, salidas


def linea(nombre, resumen):
    r = resumen
    return (f"{nombre}: top1 {r['top1']['media']:.3f} ± {r['top1']['desviacion']:.3f} "
            f"(peor {r['top1']['peor']:.3f}), top5 {r['top5']['media']:.3f}, F1 {r['f1_macro']['media']:.3f}")
