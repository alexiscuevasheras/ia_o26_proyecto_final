import { test } from "node:test";
import assert from "node:assert/strict";

import { conjugar, glosasATexto } from "../../web/glosas_a_texto.js";
import { adaptarSalida } from "../../web/adaptar.js";
import { salida } from "../../web/gru.js";

const vocabulario = {
  glosas: {
    YO: { tema: "pronombres", sinonimos: ["yo"] },
    ELLA: { tema: "pronombres", sinonimos: ["ella"] },
    COMER: { tema: "verbos", sinonimos: ["comer", "como"] },
    DORMIR: { tema: "verbos", sinonimos: ["dormir"] },
    ESCUELA: { tema: "escuela", sinonimos: ["escuela"] },
  },
};
const g = (glosa) => ({ tipo: "glosa", glosa });
const l = (letra) => ({ tipo: "letra", letra });

test("conjuga en presente regulares e irregulares", () => {
  assert.equal(conjugar("comer", 0), "como");
  assert.equal(conjugar("vivir", 3), "vivimos");
  assert.equal(conjugar("dormir", 2), "duerme");
});

test("pronombre + verbo se conjuga; verbo suelto queda en infinitivo", () => {
  assert.equal(glosasATexto([g("YO"), g("COMER")], vocabulario), "Yo como.");
  assert.equal(glosasATexto([g("ELLA"), g("DORMIR")], vocabulario), "Ella duerme.");
  assert.equal(glosasATexto([g("COMER"), g("ESCUELA")], vocabulario), "Comer escuela.");
});

test("las letras seguidas forman una palabra", () => {
  assert.equal(glosasATexto([g("YO"), l("A"), l("N"), l("A")], vocabulario), "Yo Ana.");
});

test("adaptar la salida sube la probabilidad de la glosa confirmada", () => {
  const modelo = { params: { V: [[0.5, -0.2], [-0.1, 0.3], [0, 0]], c: [0, 0, 0] } };
  const resumen = [1, 1];
  const antes = salida(modelo.params.V, modelo.params.c, resumen)[2];
  const { V, c } = adaptarSalida(modelo, [{ resumen, k: 2 }]);
  assert.ok(salida(V, c, resumen)[2] > antes + 0.2);
  assert.deepEqual(modelo.params.c, [0, 0, 0]); // el original no cambia
});
