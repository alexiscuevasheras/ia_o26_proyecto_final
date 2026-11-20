// Pase hacia adelante de la red, con los pesos que exporta src/lsm/exportar.py.
// z(l) = W(l) a(l−1) + b(l),  a(l) = g(l)(z(l)),  ŷ = a(L).
// Devuelve todas las activaciones para poder dibujar la red por dentro.

const ACTIVACIONES = {
  relu: (z) => z.map((v) => Math.max(0, v)),
  sigmoide: (z) => z.map((v) => 1 / (1 + Math.exp(-v))),
  identidad: (z) => z.slice(),
  softmax: (z) => {
    const max = Math.max(...z); // restar el máximo evita que exp se desborde
    const e = z.map((v) => Math.exp(v - max));
    const suma = e.reduce((a, b) => a + b, 0);
    return e.map((v) => v / suma);
  },
};

/**
 * red: { anchos, activaciones, letras, W, b } donde W[l-1] es W(l) como
 * arreglo de filas (filas = destino i, columnas = origen j).
 * Devuelve { z, a }: z[l] y a[l] para l = 1..L, con a[0] = x.
 */
export function adelante(red, x) {
  if (x.length !== red.anchos[0]) {
    throw new Error(`la entrada tiene ${x.length} números; la red espera ${red.anchos[0]}`);
  }
  const z = [null];
  const a = [x];
  for (let l = 1; l < red.anchos.length; l++) {
    const W = red.W[l - 1];
    const b = red.b[l - 1];
    const zl = W.map((fila, i) => fila.reduce((suma, w, j) => suma + w * a[l - 1][j], b[i]));
    z.push(zl);
    a.push(ACTIVACIONES[red.activaciones[l - 1]](zl));
  }
  return { z, a };
}

/** Las k letras más probables: [{ letra, probabilidad }], de mayor a menor. */
export function mejores(red, probabilidades, k = 3) {
  return probabilidades
    .map((p, i) => ({ letra: red.letras[i], probabilidad: p }))
    .sort((u, v) => v.probabilidad - u.probabilidad)
    .slice(0, k);
}
