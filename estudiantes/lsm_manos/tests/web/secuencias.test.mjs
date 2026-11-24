// Las features de secuencias y la GRU en JavaScript deben dar lo mismo que Python.
// Correr: node --test tests/web/   (también lo lanza pytest)

import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

import { secuencia } from "../../web/secuencias.js";
import { adelanteGRU } from "../../web/gru.js";

const TOL = 1e-5;
const paridad = JSON.parse(readFileSync(new URL("../../web/fixtures/paridad_palabras.json", import.meta.url)));

function cerca(obtenido, esperado, que) {
  assert.equal(obtenido.length, esperado.length, `${que}: longitud`);
  obtenido.forEach((v, i) => assert.ok(Math.abs(v - esperado[i]) < TOL, `${que}[${i}]: ${v} contra ${esperado[i]}`));
}

test("secuencias: misma entrada que Python", () => {
  paridad.casos.forEach((caso, n) => {
    const x = secuencia(caso.cuadros, caso.aspecto, caso.dominante);
    assert.equal(x.length, caso.x.length);
    x.forEach((fila, t) => cerca(fila, caso.x[t], `caso ${n}, t=${t}`));
  });
});

test("gru: mismas probabilidades que Python", () => {
  paridad.casos.forEach((caso, n) => {
    const { probabilidades } = adelanteGRU(paridad.modelo, caso.x);
    cerca(probabilidades, caso.probabilidades, `caso ${n}`);
  });
});
