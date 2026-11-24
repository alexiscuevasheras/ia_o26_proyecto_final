// Pase hacia adelante de la GRU de src/redneuronal/recurrente.py, con los pesos
// que exporta lsm.palabras.exportar. Mismas ecuaciones (ver el docstring en Python).

function matVec(M, v) {
  return M.map((fila) => fila.reduce((s, w, j) => s + w * v[j], 0));
}

const sigma = (x) => 0.5 * (1 + Math.tanh(0.5 * x));

function softmax(z) {
  const max = Math.max(...z);
  const e = z.map((v) => Math.exp(v - max));
  const suma = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / suma);
}

/**
 * modelo: { params, estandar: { media, desv }, glosas }.
 * x: T filas de D números (features crudas, sin estandarizar).
 * Devuelve { probabilidades, estados, resumen }: estados h_1..h_T (para dibujar
 * la red) y resumen = [h_T; h̄], la entrada de la capa de salida (para adaptarla).
 */
export function adelanteGRU(modelo, x) {
  const p = modelo.params;
  const { media, desv } = modelo.estandar;
  const n = p.U_z.length;
  let h = new Array(n).fill(0);
  const estados = [];
  for (const fila of x) {
    const xt = fila.map((v, j) => (v - media[j]) / desv[j]);
    const pre = (g) => matVec(p[`W_${g}`], xt).map((v, i) => v + p[`b_${g}`][i]);
    const zPre = pre("z");
    const rPre = pre("r");
    const uz = matVec(p.U_z, h);
    const ur = matVec(p.U_r, h);
    const z = zPre.map((v, i) => sigma(v + uz[i]));
    const r = rPre.map((v, i) => sigma(v + ur[i]));
    const uh = matVec(p.U_h, h.map((v, i) => r[i] * v));
    const c = pre("h").map((v, i) => Math.tanh(v + uh[i]));
    h = h.map((v, i) => (1 - z[i]) * v + z[i] * c[i]);
    estados.push(h);
  }
  const promedio = h.map((_, i) => estados.reduce((s, e) => s + e[i], 0) / estados.length);
  const resumen = [...h, ...promedio];
  return { probabilidades: salida(p.V, p.c, resumen), estados, resumen };
}

/** ŷ = softmax(V resumen + c). */
export function salida(V, c, resumen) {
  return softmax(matVec(V, resumen).map((v, k) => v + c[k]));
}

/** Las k glosas más probables: [{ glosa, probabilidad }]. */
export function mejoresGlosas(modelo, probabilidades, k = 3) {
  return probabilidades
    .map((p, i) => ({ glosa: modelo.glosas[i], probabilidad: p }))
    .sort((a, b) => b.probabilidad - a.probabilidad)
    .slice(0, k);
}
