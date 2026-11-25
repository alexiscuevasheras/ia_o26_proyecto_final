// Texto en español → secuencia de glosas de la LSM (palabra por palabra, sin gramática).
//
// Módulo puro (sin DOM) para poder probarlo con node. Recibe web/vocabulario.json.
// Cada token es {tipo: "glosa", glosa, texto} o {tipo: "deletreo", texto, letras}.

const MAX_PALABRAS = 4; // el sinónimo más largo del vocabulario tiene 4 palabras

// Palabras funcionales que la LSM no seña. Solo se quitan si no están en el
// vocabulario: «él» (pronombre) es la glosa EL, «el» (artículo) se quita.
const PALABRAS_VACIAS = new Set([
  "el", "la", "los", "las", "lo", "un", "una", "unos", "unas",
  "de", "del", "a", "al", "que", "y", "e", "o", "u", "en", "es", "son", "se",
]);

const DIGRAFOS = ["ll", "rr", "ch"]; // en la LSM son letras propias

/** Quita acentos y diéresis pero no la tilde de la ñ (año ≠ ano). */
export function sinAcentos(texto) {
  return texto.normalize("NFD").replace(/[́̈]/g, "").normalize("NFC");
}

/** Minúsculas y palabras sueltas; la puntuación se descarta. */
export function palabras(texto) {
  return texto.toLowerCase().normalize("NFC").match(/[\p{L}\p{N}]+/gu) ?? [];
}

function agregarSinPisar(mapa, clave, glosa) {
  if (!mapa.has(clave)) mapa.set(clave, glosa);
}

/**
 * Dos búsquedas: exacta (con acentos) y sin acentos. Si un sinónimo está en
 * varias glosas gana la primera del vocabulario.
 */
export function crearDiccionario(vocabulario) {
  const exacto = new Map();
  const relajado = new Map();
  for (const [glosa, { sinonimos }] of Object.entries(vocabulario.glosas)) {
    for (const sinonimo of sinonimos) {
      const clave = palabras(sinonimo).join(" ");
      agregarSinPisar(exacto, clave, glosa);
      agregarSinPisar(relajado, sinAcentos(clave), glosa);
    }
  }
  return { exacto, relajado };
}

/** Letras del alfabeto manual: «llorar» → LL, O, R, A, R; «niño» → N, I, Ñ, O. */
export function letrasDeletreo(palabra) {
  const limpia = sinAcentos(palabra.toLowerCase()).replace(/[^a-zñ0-9]/g, "");
  const letras = [];
  for (let i = 0; i < limpia.length; i++) {
    const par = limpia.slice(i, i + 2);
    if (DIGRAFOS.includes(par)) {
      letras.push(par.toUpperCase());
      i++;
    } else {
      letras.push(limpia[i].toUpperCase());
    }
  }
  return letras;
}

function deletreo(texto) {
  return { tipo: "deletreo", texto, letras: letrasDeletreo(texto) };
}

// Primero la frase más larga que exista; a igual largo, la exacta antes que la sin acentos.
function frasePrefijo(lista, i, diccionario) {
  for (let n = Math.min(MAX_PALABRAS, lista.length - i); n >= 2; n--) {
    const frase = lista.slice(i, i + n).join(" ");
    const glosa = diccionario.exacto.get(frase) ?? diccionario.relajado.get(sinAcentos(frase));
    if (glosa) return { glosa, texto: frase, n };
  }
  return null;
}

/** Español → tokens de glosa o de deletreo. */
export function textoAGlosas(texto, diccionario) {
  const lista = palabras(texto);
  const tokens = [];
  let i = 0;
  while (i < lista.length) {
    const frase = frasePrefijo(lista, i, diccionario);
    if (frase) {
      tokens.push({ tipo: "glosa", glosa: frase.glosa, texto: frase.texto });
      i += frase.n;
      continue;
    }
    const palabra = lista[i++];
    const exacta = diccionario.exacto.get(palabra);
    if (exacta) tokens.push({ tipo: "glosa", glosa: exacta, texto: palabra });
    else if (PALABRAS_VACIAS.has(palabra)) continue;
    else {
      const relajada = diccionario.relajado.get(sinAcentos(palabra));
      tokens.push(relajada ? { tipo: "glosa", glosa: relajada, texto: palabra } : deletreo(palabra));
    }
  }
  return tokens;
}

/** Las glosas sin animación se deletrean; se conserva la glosa para mostrarla. */
export function deletrearSinAnimacion(tokens, glosasDisponibles) {
  return tokens.map((token) =>
    token.tipo === "glosa" && !glosasDisponibles.has(token.glosa)
      ? { ...deletreo(token.texto), glosa: token.glosa }
      : token,
  );
}

/** Una línea legible: «YO COMER MANZANA(deletreo)». */
export function glosasALinea(tokens) {
  return tokens
    .map((t) => (t.tipo === "glosa" ? t.glosa : `${t.letras.join("")}(deletreo)`))
    .join(" ");
}
