import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

import { asignarManos, TraductorEnVivo } from "../../web/en_vivo.js";

const paridad = JSON.parse(readFileSync(new URL("../../web/fixtures/paridad_palabras.json", import.meta.url)));
const mano = (x, y) => Array.from({ length: 21 }, (_, i) => [x + 0.001 * i, y - 0.004 * i, 0]);

function cuerpo(munecaIzq, munecaDer) {
  const c = Array.from({ length: 33 }, () => [0.5, 0.5, 0, 1]);
  c[11] = [0.6, 0.3, 0, 1]; // hombro izquierdo de la persona: a la derecha de la imagen
  c[12] = [0.4, 0.3, 0, 1];
  c[15] = [...munecaIzq, 0, 1];
  c[16] = [...munecaDer, 0, 1];
  return c;
}

test("cada mano va al brazo con la muñeca más cercana", () => {
  const c = cuerpo([0.65, 0.6], [0.35, 0.6]);
  const [izq, der] = asignarManos([mano(0.36, 0.6), mano(0.64, 0.6)], c);
  assert.equal(der[0][0], 0.36);
  assert.equal(izq[0][0], 0.64);
});

test("sin cuerpo, una mano a la izquierda de la imagen es del brazo derecho", () => {
  const [izq, der] = asignarManos([mano(0.3, 0.5)], null);
  assert.equal(izq, null);
  assert.equal(der[0][0], 0.3);
});

test("una seña completa produce una glosa o un «no sé» con 3 opciones", () => {
  const traductor = new TraductorEnVivo(paridad.modelo, null, { umbral: 0.99 });
  const alturas = [1.5, 1.5, 1.5, 0.8, 0.4, 0.1, 0.0, 0.2, 0.7, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5];
  const eventos = alturas
    .map((y) => {
      const m = mano(0.4, 0.3 + y * 0.2);
      return traductor.agregar({ cuerpo: cuerpo([0.6, 0.6], [0.4, 0.3 + y * 0.2]), manos: [null, m] }, 1);
    })
    .filter(Boolean);
  assert.equal(eventos.length, 1);
  assert.equal(eventos[0].tipo, "no_se"); // con umbral 0.99 y una red al azar
  assert.equal(eventos[0].opciones.length, 3);
  assert.equal(eventos[0].resumen.length, 2 * paridad.modelo.params.U_z.length);
});
