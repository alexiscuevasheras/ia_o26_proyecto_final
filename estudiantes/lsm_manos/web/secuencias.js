// De los puntos crudos de una seña a la entrada de la GRU: T cuadros × D números.
// Hace exactamente lo mismo que src/lsm/palabras/secuencias.py (ver su docstring);
// tests/web/secuencias.test.mjs compara los dos.
//
// Un cuadro es { cuerpo: 33 [x, y, z, vis] o null, manos: [mano0, mano1] }, donde
// mano0 es la del brazo izquierdo de la persona y cada mano es 21 [x, y, z] o null.

import { normalizar } from "./features.js";

export const T = 16;
const HOMBRO_IZQ = 11;
const HOMBRO_DER = 12;
export const POR_MANO = 2 + 60 + 1;
export const D = 2 * POR_MANO;
const HUECO_MAXIMO = 3;

function mediana(valores) {
  const v = [...valores].sort((a, b) => a - b);
  const m = v.length >> 1;
  return v.length % 2 ? v[m] : (v[m - 1] + v[m]) / 2;
}

function corregirAspecto(cuadros, aspecto) {
  return cuadros.map(({ cuerpo, manos }) => ({
    cuerpo: cuerpo && cuerpo.map(([x, y, z, v]) => [x * aspecto, y, z, v]),
    manos: manos.map((m) => m && m.map(([x, y, z]) => [x * aspecto, y, z * aspecto])),
  }));
}

export function marcoDelCuerpo(cuadros) {
  const validos = cuadros.filter((c) => c.cuerpo && Number.isFinite(c.cuerpo[HOMBRO_IZQ][0]) && Number.isFinite(c.cuerpo[HOMBRO_DER][0]));
  if (!validos.length) return marcoSinCuerpo(cuadros);
  const h = [HOMBRO_IZQ, HOMBRO_DER].map((i) => [0, 1].map((e) => mediana(validos.map((c) => c.cuerpo[i][e]))));
  const centro = [(h[0][0] + h[1][0]) / 2, (h[0][1] + h[1][1]) / 2];
  return { centro, escala: Math.max(Math.hypot(h[0][0] - h[1][0], h[0][1] - h[1][1]), 1e-3) };
}

// Supuesto burdo, igual que en Python: palma ≈ 0.4 anchos de hombro y el pecho
// ~1.5 anchos debajo de la mano más alta.
function marcoSinCuerpo(cuadros) {
  const palmas = [];
  const munecas = [];
  for (const { manos } of cuadros) {
    for (const m of manos) {
      if (!m) continue;
      palmas.push(Math.hypot(m[9][0] - m[0][0], m[9][1] - m[0][1]));
      munecas.push(m[0]);
    }
  }
  const escala = palmas.length ? mediana(palmas) / 0.4 : 1;
  if (!munecas.length) return { centro: [0.5, 0.5], escala };
  const yMin = Math.min(...munecas.map((p) => p[1]));
  return { centro: [mediana(munecas.map((p) => p[0])), yMin + 1.5 * escala], escala };
}

function movimiento(cuadros, i) {
  const m = cuadros.map((c) => c.manos[i]).filter(Boolean).map((p) => p[0]);
  if (m.length < 2) return m.length * 1e-3;
  let recorrido = 0;
  for (let k = 1; k < m.length; k++) recorrido += Math.hypot(m[k][0] - m[k - 1][0], m[k][1] - m[k - 1][1]);
  return recorrido + (m.length / cuadros.length) * 1e-3;
}

/** 0 (brazo izquierdo) o 1 (derecho): la mano que más se mueve en la seña. */
export function manoQueMasSeMueve(cuadros) {
  return movimiento(cuadros, 1) >= movimiento(cuadros, 0) ? 1 : 0;
}

function rellenarHuecos(filas, presente) {
  const idx = presente.flatMap((p, i) => (p ? [i] : []));
  for (let k = 0; k + 1 < idx.length; k++) {
    const a = idx[k];
    const b = idx[k + 1];
    if (b - a > 1 && b - a <= HUECO_MAXIMO + 1) {
      for (let i = a + 1; i < b; i++) {
        const t = (i - a) / (b - a);
        filas[i] = filas[a].map((v, j) => v + t * (filas[b][j] - v));
        presente[i] = true;
      }
    }
  }
}

/** { x: F filas de D números, dominante }, sin remuestrear. */
export function cuadrosAEntrada(cuadrosCrudos, aspecto, dominante = null) {
  const cuadros = corregirAspecto(cuadrosCrudos, aspecto);
  const { centro, escala } = marcoDelCuerpo(cuadros);
  const dom = dominante ?? manoQueMasSeMueve(cuadros);
  const espejo = dom === 0;
  const F = cuadros.length;
  const x = Array.from({ length: F }, () => new Array(D).fill(0));
  [dom, 1 - dom].forEach((brazo, posicion) => {
    const presente = cuadros.map((c) => Boolean(c.manos[brazo]));
    const filas = cuadros.map((c) => {
      const mano = c.manos[brazo];
      if (!mano) return new Array(POR_MANO).fill(0);
      const muneca = [(mano[0][0] - centro[0]) / escala, (mano[0][1] - centro[1]) / escala];
      if (espejo) muneca[0] *= -1;
      // Misma entrada que la red de letras (ver el comentario en secuencias.py).
      const forma = normalizar(mano, brazo === 1).slice(1).flat();
      return [...muneca, ...forma, 1];
    });
    rellenarHuecos(filas, presente);
    filas.forEach((fila, i) => {
      if (presente[i]) x[i].splice(posicion * POR_MANO, POR_MANO, ...fila);
    });
  });
  return { x, dominante: dom };
}

/** F filas → t filas por interpolación lineal en el tiempo (como np.interp). */
export function remuestrear(x, t = T) {
  if (x.length === 1) return Array.from({ length: t }, () => x[0].slice());
  const F = x.length;
  return Array.from({ length: t }, (_, k) => {
    const pos = (k / (t - 1)) * (F - 1);
    const i = Math.min(Math.floor(pos), F - 2);
    const f = pos - i;
    return x[i].map((v, j) => v + f * (x[i + 1][j] - v));
  });
}

export function secuencia(cuadros, aspecto, dominante = null, t = T) {
  return remuestrear(cuadrosAEntrada(cuadros, aspecto, dominante).x, t);
}
