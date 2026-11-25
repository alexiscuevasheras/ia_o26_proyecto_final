import { test } from "node:test";
import assert from "node:assert/strict";

import { Segmentador } from "../../web/segmentador.js";

const marco = { centro: [0.5, 0.3], escala: 0.2 };
// Una mano con la muñeca a la altura y (en anchos de hombro bajo los hombros).
const cuadro = (y) => ({ manos: [null, Array.from({ length: 21 }, () => [0.5, 0.3 + y * 0.2, 0])] });

test("corta una seña de reposo a reposo, con margen", () => {
  const seg = new Segmentador();
  const alturas = [1.5, 1.5, 1.5, 1.5, 0.9, 0.5, 0.2, 0.0, 0.3, 0.8, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5];
  const senas = alturas.map((y) => seg.agregar(cuadro(y), marco)).filter(Boolean);
  assert.equal(senas.length, 1);
  const ys = senas[0].map((c) => (c.manos[1][0][1] - 0.3) / 0.2);
  assert.ok(ys.includes(0), "incluye el punto más alto");
  assert.ok(Math.abs(ys[0] - 1.5) < 1e-9 && Math.abs(ys.at(-1) - 1.5) < 1e-9, "empieza y termina en reposo");
});

test("un temblor cerca del umbral no corta la seña", () => {
  const seg = new Segmentador();
  const alturas = [1.5, 1.5, 0.5, 0.4, 1.15, 1.2, 1.15, 0.3, 1.5, 1.5, 1.5, 1.5];
  const senas = alturas.map((y) => seg.agregar(cuadro(y), marco)).filter(Boolean);
  assert.equal(senas.length, 1);
});

test("sin manos a la vista no hay seña", () => {
  const seg = new Segmentador();
  for (let i = 0; i < 20; i++) assert.equal(seg.agregar({ manos: [null, null] }, marco), null);
  assert.equal(seg.enSena, false);
});
