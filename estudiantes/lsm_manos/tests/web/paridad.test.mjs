// JavaScript debe dar lo mismo que Python (prueba 6 del plan), dentro de 1e-5.
// Correr: node --test tests/web/paridad.test.mjs   (también lo lanza pytest)

import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

import { vectorEntrada } from "../../web/features.js";
import { adelante, mejores } from "../../web/red.js";

const TOL = 1e-5;
const paridad = JSON.parse(
  readFileSync(new URL("../../web/fixtures/paridad.json", import.meta.url)),
);

function cerca(obtenido, esperado, que) {
  assert.equal(obtenido.length, esperado.length, `${que}: longitud`);
  obtenido.forEach((v, i) => {
    assert.ok(Math.abs(v - esperado[i]) < TOL, `${que}[${i}]: ${v} contra ${esperado[i]}`);
  });
}

test("features: mismo vector que Python", () => {
  paridad.features.forEach((caso, n) => {
    const vector = vectorEntrada(caso.puntos, caso.es_izquierda, {
      aspecto: caso.aspecto,
      rotar: caso.rotar,
      usarZ: caso.usar_z,
    });
    cerca(vector, caso.vector, `caso ${n}`);
  });
});

test("red: mismas activaciones en cada capa que Python", () => {
  paridad.casos_red.forEach((caso, n) => {
    const { a } = adelante(paridad.red, caso.entrada);
    caso.activaciones.forEach((esperada, l) => cerca(a[l + 1], esperada, `caso ${n}, capa ${l + 1}`));
  });
});

test("red: las probabilidades suman 1 y las mejores vienen ordenadas", () => {
  const { a } = adelante(paridad.red, paridad.casos_red[0].entrada);
  const probabilidades = a[a.length - 1];
  assert.ok(Math.abs(probabilidades.reduce((s, p) => s + p, 0) - 1) < 1e-12);
  const top = mejores(paridad.red, probabilidades, 3);
  assert.equal(top.length, 3);
  assert.ok(top[0].probabilidad >= top[1].probabilidad && top[1].probabilidad >= top[2].probabilidad);
});
