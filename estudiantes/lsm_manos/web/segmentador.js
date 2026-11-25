// Corta el video en vivo en señas: de «manos abajo» a «manos abajo».
//
// En UAEMex cada seña empieza y termina con las manos en reposo. En el marco de
// los hombros (y hacia abajo, en anchos de hombro), la muñeca dominante en reposo
// está en y ≈ 1.5 (mediana del primer y último cuadro), y durante la seña sube
// hasta y ≈ 0 (mediana del punto más alto). Por eso:
// - empieza una seña cuando alguna muñeca sube de SUBIR durante 2 cuadros;
// - termina cuando las dos están abajo (o fuera de cuadro) durante 4 cuadros.
// Los umbrales distintos (histéresis) evitan cortes por temblor cerca de la línea.
// Se guardan unos cuadros de reposo antes y después, como en los datos.

export const SUBIR = 1.1;
export const BAJAR = 1.25;
const CUADROS_PARA_EMPEZAR = 2;
const CUADROS_PARA_TERMINAR = 4;
const MARGEN = 3; // cuadros de reposo que se guardan alrededor de la seña
const MAXIMO = 150; // una seña de más de ~5 s a 30 fps se corta

/** Altura de cada muñeca en el marco de los hombros; Infinity si no se ve. */
export function alturas(cuadro, marco) {
  return cuadro.manos.map((m) => (m ? (m[0][1] - marco.centro[1]) / marco.escala : Infinity));
}

export class Segmentador {
  constructor() {
    this.previos = [];
    this.actual = null;
    this.arriba = 0;
    this.abajo = 0;
  }

  /**
   * Agrega un cuadro. Devuelve los cuadros de una seña completa cuando termina,
   * o null. `marco` es { centro, escala } del cuerpo (en coordenadas de imagen
   * ya corregidas por la proporción, igual que las manos del cuadro).
   */
  agregar(cuadro, marco) {
    const ys = alturas(cuadro, marco);
    const algunaArriba = ys.some((y) => y < SUBIR);
    const todasAbajo = ys.every((y) => y > BAJAR);

    if (!this.actual) {
      this.previos = [...this.previos, cuadro].slice(-MARGEN - CUADROS_PARA_EMPEZAR);
      this.arriba = algunaArriba ? this.arriba + 1 : 0;
      if (this.arriba >= CUADROS_PARA_EMPEZAR) {
        this.actual = this.previos;
        this.abajo = 0;
      }
      return null;
    }
    this.actual.push(cuadro);
    this.abajo = todasAbajo ? this.abajo + 1 : 0;
    if (this.abajo >= CUADROS_PARA_TERMINAR || this.actual.length >= MAXIMO) {
      const sena = this.actual.slice(0, this.actual.length - Math.max(0, this.abajo - MARGEN));
      this.actual = null;
      this.previos = [];
      this.arriba = 0;
      return sena;
    }
    return null;
  }

  get enSena() {
    return this.actual !== null;
  }
}
