// La lógica del traductor en vivo, sin DOM ni cámara (se prueba con cuadros falsos).
//
// Por cuadro: resultados de MediaPipe → cuadro { cuerpo, manos } con cada mano en
// su brazo (igual que extraer.py) → segmentador. Al cerrar una seña:
// - si durante la seña la mano dominante se quedó quieta con una letra clara,
//   era deletreo y salen esas letras;
// - si no, la GRU clasifica la seña completa; con probabilidad baja dice «no sé»
//   y ofrece sus 3 mejores opciones para que la persona elija (y la red se adapte).

import { vectorEntrada } from "./features.js";
import { adelante } from "./red.js";
import { adelanteGRU, mejoresGlosas } from "./gru.js";
import { cuadrosAEntrada, marcoEnVivo, secuencia } from "./secuencias.js";
import { Segmentador } from "./segmentador.js";

const MUNECA_IZQ = 15;
const MUNECA_DER = 16;
export const UMBRAL_NO_SE = 0.5; // se recalibra con el experimento del «no sé»
const LETRA_QUIETA = 0.03; // anchos de hombro por cuadro
const LETRA_CUADROS = 8;
const LETRA_PROBABILIDAD = 0.8;
const CUADROS_DEL_MARCO = 90; // ~3 s de cuerpo para la mediana de los hombros
const SENAS_PARA_DOMINANCIA = 3;

/** Mano 0 = brazo izquierdo de la persona; igual que lado_de_mano en extraer.py. */
export function asignarManos(manosDetectadas, cuerpo) {
  const manos = [null, null];
  for (const mano of manosDetectadas) {
    const libres = [0, 1].filter((i) => !manos[i]);
    let orden;
    if (cuerpo) {
      const d = [MUNECA_IZQ, MUNECA_DER].map((k) => Math.hypot(mano[0][0] - cuerpo[k][0], mano[0][1] - cuerpo[k][1]));
      orden = d[0] <= d[1] ? [0, 1] : [1, 0];
    } else {
      orden = mano[0][0] > 0.5 ? [0, 1] : [1, 0];
    }
    const lado = orden.find((i) => libres.includes(i)) ?? orden[0];
    manos[lado] = mano;
  }
  return manos;
}

export class TraductorEnVivo {
  /** modeloPalabras: palabras.json; redLetras: pesos.json (o null para solo palabras). */
  constructor(modeloPalabras, redLetras, { umbral = UMBRAL_NO_SE } = {}) {
    this.modelo = modeloPalabras;
    this.letras = redLetras;
    this.umbral = umbral;
    this.segmentador = new Segmentador();
    this.recientes = [];
    this.centro = null;
    this.votosDominancia = [0, 0];
    this.reiniciarLetra();
  }

  reiniciarLetra() {
    this.letraActual = null;
    this.letraCuadros = 0;
    this.letrasDeLaSena = [];
    this.ultimaMuneca = null;
  }

  /** La mano dominante de quien usa la página: mayoría de sus señas, derecha al inicio. */
  get dominante() {
    const total = this.votosDominancia[0] + this.votosDominancia[1];
    if (total < SENAS_PARA_DOMINANCIA) return 1;
    return this.votosDominancia[1] >= this.votosDominancia[0] ? 1 : 0;
  }

  /**
   * cuadro: { cuerpo: 33 [x, y, z, vis] o null, manos: [m0, m1] } ya asignado.
   * Devuelve un evento o null:
   *   { tipo: "letra", letra, probabilidad }
   *   { tipo: "glosa", glosa, probabilidad, opciones, resumen, x, estados }
   *   { tipo: "no_se", opciones, resumen, x, estados }
   */
  agregar(cuadro, aspecto) {
    this.recientes = [...this.recientes, cuadro].slice(-CUADROS_DEL_MARCO);
    // Si en la ventana no hay hombros plausibles, se queda el último centro bueno.
    const marco = marcoEnVivo(this.recientes, aspecto, this.centro);
    this.centro = marco.centro;
    const enSenaAntes = this.segmentador.enSena;
    const sena = this.segmentador.agregar(cuadro, marco);
    let evento = null;
    if (this.segmentador.enSena && enSenaAntes) evento = this.revisarLetra(cuadro, aspecto, marco);
    if (sena) {
      evento = this.cerrarSena(sena, aspecto) ?? evento;
      this.reiniciarLetra();
    }
    return evento;
  }

  revisarLetra(cuadro, aspecto, marco) {
    const mano = cuadro.manos[this.dominante];
    if (!this.letras || !mano) return null;
    const muneca = [(mano[0][0] * aspecto) / marco.escala, mano[0][1] / marco.escala];
    const velocidad = this.ultimaMuneca ? Math.hypot(muneca[0] - this.ultimaMuneca[0], muneca[1] - this.ultimaMuneca[1]) : Infinity;
    this.ultimaMuneca = muneca;
    const x = vectorEntrada(mano, this.dominante === 1, { aspecto, rotar: this.letras.features.rotar, usarZ: this.letras.features.usar_z });
    const { a } = adelante(this.letras, x);
    const p = a[a.length - 1];
    const k = p.indexOf(Math.max(...p));
    const letra = this.letras.letras[k];
    const quieta = velocidad < LETRA_QUIETA && p[k] >= LETRA_PROBABILIDAD;
    if (!quieta) {
      this.letraActual = null;
      this.letraCuadros = 0;
      return null;
    }
    if (letra !== this.letraActual) {
      this.letraActual = letra;
      this.letraCuadros = 0;
    }
    this.letraCuadros += 1;
    if (this.letraCuadros !== LETRA_CUADROS) return null;
    this.letrasDeLaSena.push(letra);
    return { tipo: "letra", letra, probabilidad: p[k] };
  }

  cerrarSena(sena, aspecto) {
    this.votosDominancia[cuadrosAEntrada(sena, aspecto, null, this.centro).dominante] += 1;
    if (this.letrasDeLaSena.length) return null; // fue deletreo: las letras ya salieron
    const x = secuencia(sena, aspecto, this.dominante, undefined, this.centro);
    const { probabilidades, resumen, estados } = adelanteGRU(this.modelo, x);
    const opciones = mejoresGlosas(this.modelo, probabilidades, 3);
    const comun = { opciones, resumen, x, estados };
    if (opciones[0].probabilidad < this.umbral) return { tipo: "no_se", ...comun };
    return { tipo: "glosa", glosa: opciones[0].glosa, probabilidad: opciones[0].probabilidad, ...comun };
  }
}
