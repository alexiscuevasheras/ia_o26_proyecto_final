// Texto en español → glosas, con el vocabulario real de la página.
// Correr: node --test tests/web/

import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

import {
  crearDiccionario, deletrearSinAnimacion, glosasALinea, letrasDeletreo, textoAGlosas,
} from "../../web/texto_a_glosas.js";

const vocabulario = JSON.parse(readFileSync(new URL("../../web/vocabulario.json", import.meta.url)));
const diccionario = crearDiccionario(vocabulario);
const glosas = (texto) => textoAGlosas(texto, diccionario).map((t) => t.glosa ?? `${t.texto}?`);

test("frase de varias palabras gana a sus partes", () => {
  assert.deepEqual(glosas("¡Buenos días!"), ["BUENOS-DIAS"]);
  assert.deepEqual(glosas("lengua de señas"), ["LENGUA-DE-SEÑAS"]);
  assert.deepEqual(glosas("buenos"), ["BUENOS"]);
});

test("verbo conjugado y la ambigüedad como / cómo", () => {
  assert.deepEqual(glosas("Yo como"), ["YO", "COMER"]);
  assert.deepEqual(glosas("¿Cómo estás?"), ["COMO-PREGUNTA", "ESTAR"]);
});

test("se quitan artículos, pero «él» es pronombre", () => {
  assert.deepEqual(glosas("el hermano de la maestra"), ["HERMANO", "MAESTRO"]);
  assert.deepEqual(glosas("él come"), ["EL", "COMER"]);
});

test("palabra desconocida se deletrea", () => {
  const [yo, comer, manzana] = textoAGlosas("yo como una manzana", diccionario);
  assert.equal(yo.glosa, "YO");
  assert.equal(comer.glosa, "COMER");
  assert.deepEqual(manzana, { tipo: "deletreo", texto: "manzana", letras: [..."MANZANA"] });
  assert.equal(glosasALinea([yo, comer, manzana]), "YO COMER MANZANA(deletreo)");
});

test("sin acentos también encuentra la glosa", () => {
  assert.deepEqual(glosas("adios papa"), ["ADIOS", "PAPA"]);
  assert.deepEqual(glosas("buenos dias"), ["BUENOS-DIAS"]);
});

test("deletreo: LL, RR y CH son una letra; la ñ se conserva", () => {
  assert.deepEqual(letrasDeletreo("Chihuahua"), ["CH", "I", "H", "U", "A", "H", "U", "A"]);
  assert.deepEqual(letrasDeletreo("perro"), ["P", "E", "RR", "O"]);
  assert.deepEqual(letrasDeletreo("llave"), ["LL", "A", "V", "E"]);
  assert.deepEqual(letrasDeletreo("Ñandú"), ["Ñ", "A", "N", "D", "U"]);
  assert.deepEqual(letrasDeletreo("42"), ["4", "2"]);
});

test("una glosa sin animación se deletrea y recuerda su glosa", () => {
  const tokens = deletrearSinAnimacion(textoAGlosas("yo cocino", diccionario), new Set(["YO"]));
  assert.equal(tokens[0].tipo, "glosa");
  assert.deepEqual(tokens[1], { tipo: "deletreo", texto: "cocino", letras: [..."COCINO"], glosa: "COCINAR" });
});
