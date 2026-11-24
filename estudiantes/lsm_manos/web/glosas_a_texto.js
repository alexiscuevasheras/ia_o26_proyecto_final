// De una sucesión de glosas reconocidas a una frase en español legible.
//
// La LSM tiene su propia gramática (orden, espacio, expresión facial), así que esto
// no es una traducción completa: es una lectura palabra por palabra con tres reglas
// que la hacen más natural, y la página lo dice.
// 1. Un pronombre sujeto seguido de un verbo conjuga el verbo en presente
//    (YO COMER → «yo como»).
// 2. Las letras deletreadas seguidas se juntan en una palabra (A N A → «Ana»).
// 3. Mayúscula al inicio y punto al final.

const PERSONA = {
  YO: 0, TU: 1, EL: 2, ELLA: 2, NOSOTROS: 3, NOSOTRAS: 3, USTEDES: 4, ELLOS: 4, ELLAS: 4,
};

// Presentes irregulares de los verbos del vocabulario: yo, tú, él, nosotros, ellos.
const IRREGULARES = {
  hacer: ["hago", "haces", "hace", "hacemos", "hacen"],
  dormir: ["duermo", "duermes", "duerme", "dormimos", "duermen"],
  jugar: ["juego", "juegas", "juega", "jugamos", "juegan"],
  recoger: ["recojo", "recoges", "recoge", "recogemos", "recogen"],
  reír: ["río", "ríes", "ríe", "reímos", "ríen"],
  cerrar: ["cierro", "cierras", "cierra", "cerramos", "cierran"],
  mentir: ["miento", "mientes", "miente", "mentimos", "mienten"],
  estar: ["estoy", "estás", "está", "estamos", "están"],
  dar: ["doy", "das", "da", "damos", "dan"],
  encender: ["enciendo", "enciendes", "enciende", "encendemos", "encienden"],
  contar: ["cuento", "cuentas", "cuenta", "contamos", "cuentan"],
  construir: ["construyo", "construyes", "construye", "construimos", "construyen"],
};

const TERMINACIONES = {
  ar: ["o", "as", "a", "amos", "an"],
  er: ["o", "es", "e", "emos", "en"],
  ir: ["o", "es", "e", "imos", "en"],
};

export function conjugar(infinitivo, persona) {
  if (IRREGULARES[infinitivo]) return IRREGULARES[infinitivo][persona];
  const tipo = infinitivo.slice(-2);
  if (!TERMINACIONES[tipo]) return infinitivo;
  return infinitivo.slice(0, -2) + TERMINACIONES[tipo][persona];
}

/** La forma en español de una glosa: su primer sinónimo, o la glosa en minúsculas. */
function forma(vocabulario, glosa) {
  const entrada = vocabulario.glosas[glosa];
  return entrada?.sinonimos?.[0] ?? glosa.toLowerCase().replaceAll("-", " ");
}

function esVerbo(vocabulario, glosa) {
  return vocabulario.glosas[glosa]?.tema === "verbos";
}

/**
 * elementos: [{ tipo: "glosa", glosa } | { tipo: "letra", letra }] en orden.
 * Devuelve la frase en español.
 */
export function glosasATexto(elementos, vocabulario) {
  const palabras = [];
  let persona = null;
  let deletreo = "";
  const cerrarDeletreo = () => {
    if (deletreo) palabras.push(deletreo[0] + deletreo.slice(1).toLowerCase());
    deletreo = "";
  };
  for (const e of elementos) {
    if (e.tipo === "letra") {
      deletreo += e.letra;
      continue;
    }
    cerrarDeletreo();
    const texto = forma(vocabulario, e.glosa);
    if (persona !== null && esVerbo(vocabulario, e.glosa)) {
      palabras.push(conjugar(texto, persona));
      persona = null;
      continue;
    }
    persona = PERSONA[e.glosa] ?? null;
    palabras.push(texto);
  }
  cerrarDeletreo();
  if (!palabras.length) return "";
  const frase = palabras.join(" ");
  return frase[0].toUpperCase() + frase.slice(1) + ".";
}
