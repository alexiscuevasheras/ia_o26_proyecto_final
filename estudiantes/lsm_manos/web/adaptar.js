// Adapta la capa de salida de la GRU a quien está usando la página.
//
// Igual que en las letras (src/lsm/adaptar.py, experimento E8): las capas
// recurrentes quedan fijas y solo se reentrenan V y c con descenso de gradiente
// sobre los ejemplos que la persona confirmó. Con la representación fija el
// problema es convexo (regresión logística multinomial), así que 100 pasos con
// lote completo bastan y no hay riesgo de arruinar la red.

import { salida } from "./gru.js";

export const ETA = 0.05;
export const PASOS = 100;

/**
 * ejemplos: [{ resumen, k }] (k = índice de la glosa correcta).
 * Devuelve { V, c } nuevos; el modelo original no cambia.
 */
export function adaptarSalida(modelo, ejemplos, { eta = ETA, pasos = PASOS } = {}) {
  let V = modelo.params.V.map((fila) => fila.slice());
  let c = modelo.params.c.slice();
  const m = ejemplos.length;
  if (!m) return { V, c };
  for (let paso = 0; paso < pasos; paso++) {
    const gV = V.map((fila) => fila.map(() => 0));
    const gc = c.map(() => 0);
    for (const { resumen, k } of ejemplos) {
      const yGorro = salida(V, c, resumen);
      yGorro.forEach((p, i) => {
        const delta = (p - (i === k ? 1 : 0)) / m; // softmax con entropía cruzada: ŷ − y
        gc[i] += delta;
        for (let j = 0; j < resumen.length; j++) gV[i][j] += delta * resumen[j];
      });
    }
    V = V.map((fila, i) => fila.map((v, j) => v - eta * gV[i][j]));
    c = c.map((v, i) => v - eta * gc[i]);
  }
  return { V, c };
}
