// De los 21 puntos de MediaPipe al vector de entrada de la red.
// Hace exactamente lo mismo que src/lsm/features.py (ver su docstring);
// tests/web/paridad.test.mjs compara los dos con landmarks fijos.

const MUNECA = 0;
const BASE_MEDIO = 9;
const NUM_PUNTOS = 21;

/**
 * puntos: 21 arreglos [x, y, z] tal como los da MediaPipe.
 * esIzquierda: lateralidad de la mano en la imagen sin espejo.
 * Devuelve los 21 puntos normalizados (la muñeca queda en el origen).
 */
export function normalizar(puntos, esIzquierda, { aspecto = 1, rotar = false } = {}) {
  if (puntos.length !== NUM_PUNTOS) {
    throw new Error(`se esperaban 21 puntos, llegaron ${puntos.length}`);
  }
  const espejo = esIzquierda ? -1 : 1;
  // z de MediaPipe viene en la escala de x: las dos se corrigen por la proporción.
  let P = puntos.map(([x, y, z]) => [x * aspecto * espejo, y, z * aspecto]);

  const [x0, y0, z0] = P[MUNECA];
  P = P.map(([x, y, z]) => [x - x0, y - y0, z - z0]);

  const s = Math.hypot(P[BASE_MEDIO][0], P[BASE_MEDIO][1]);
  if (s < 1e-9) throw new Error("palma de tamaño cero");
  P = P.map(([x, y, z]) => [x / s, y / s, z / s]);

  return rotar ? alinearPalmaHaciaArriba(P) : P;
}

// Gira x-y para que p9 − p0 apunte a (0, −1): en la imagen, «arriba» es −y.
function alinearPalmaHaciaArriba(P) {
  const anguloPalma = Math.atan2(P[BASE_MEDIO][1], P[BASE_MEDIO][0]);
  const giro = -Math.PI / 2 - anguloPalma;
  const c = Math.cos(giro);
  const s = Math.sin(giro);
  return P.map(([x, y, z]) => [c * x - s * y, s * x + c * y, z]);
}

/** El vector de entrada: 60 números (x, y, z de los puntos 1 a 20) o 40 sin z. */
export function vectorEntrada(puntos, esIzquierda, { aspecto = 1, rotar = false, usarZ = true } = {}) {
  const P = normalizar(puntos, esIzquierda, { aspecto, rotar });
  const vector = [];
  for (let j = 0; j < NUM_PUNTOS; j++) {
    if (j === MUNECA) continue;
    vector.push(P[j][0], P[j][1]);
    if (usarZ) vector.push(P[j][2]);
  }
  return vector;
}
