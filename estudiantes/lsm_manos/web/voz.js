// Voz (o texto) → texto_a_glosas.js → avatar.js.
//
// Las animaciones reales salen de Python (modelos/animaciones.json, no se sube a
// git); si faltan, la página usa las sintéticas de fixtures/ y lo avisa.

import { crearAvatar } from "./avatar.js";
import { crearDiccionario, deletrearSinAnimacion, glosasALinea, textoAGlosas } from "./texto_a_glosas.js";

const ANIMACIONES = "../modelos/animaciones.json";
const ANIMACIONES_PRUEBA = "fixtures/animaciones_prueba.json";
const VOCABULARIO = "vocabulario.json";

const $ = (id) => document.getElementById(id);
const estado = $("estado");
const listaGlosas = $("glosas");
const botonMicrofono = $("microfono");
const botonRepetir = $("repetir");
const parcial = $("parcial");

async function leerJSON(ruta) {
  const respuesta = await fetch(ruta);
  if (!respuesta.ok) throw new Error(`no encontré ${ruta} (${respuesta.status})`);
  return respuesta.json();
}

async function cargarAnimaciones() {
  try {
    return await leerJSON(ANIMACIONES);
  } catch {
    $("aviso-sintetico").hidden = false;
    return leerJSON(ANIMACIONES_PRUEBA);
  }
}

function chip(token) {
  const li = document.createElement("li");
  li.className = token.tipo;
  if (token.tipo === "glosa") {
    li.textContent = token.glosa;
    return li;
  }
  li.textContent = token.letras.join("");
  const detalle = document.createElement("small");
  detalle.textContent = token.glosa ? ` (${token.glosa}, sin animación)` : " (deletreo)";
  li.append(detalle);
  return li;
}

function mostrarGlosas(tokens) {
  listaGlosas.replaceChildren(...tokens.map(chip));
  $("linea").textContent = glosasALinea(tokens);
}

function resaltar(indice) {
  [...listaGlosas.children].forEach((li, i) => li.classList.toggle("actual", i === indice));
}

function mostrarLetra(letra, nota) {
  $("letra").textContent = letra ?? "";
  $("nota-letra").textContent = nota ?? "";
}

/** Frases pendientes: la voz continua puede dictar otra mientras el avatar seña. */
function crearCola(avatar, diccionario) {
  const pendientes = [];
  let ocupado = false;
  let ultimos = [];

  function reproducir(tokens) {
    ultimos = tokens;
    ocupado = true;
    mostrarGlosas(tokens);
    botonRepetir.disabled = false;
    estado.textContent = `Señando ${tokens.length} ${tokens.length === 1 ? "glosa" : "glosas"}.`;
    avatar.reproducir(tokens);
  }

  return {
    agregar(texto) {
      const tokens = deletrearSinAnimacion(textoAGlosas(texto, diccionario), avatar.glosasDisponibles);
      if (tokens.length === 0) {
        estado.textContent = "No encontré nada que señar en ese texto.";
        return;
      }
      if (ocupado) pendientes.push(tokens);
      else reproducir(tokens);
    },
    repetir: () => reproducir(ultimos),
    alTerminar() {
      ocupado = false;
      resaltar(null);
      if (pendientes.length) reproducir(pendientes.shift());
      else estado.textContent = "Listo. Habla o escribe otra frase.";
    },
  };
}

function prepararMicrofono(alDecir) {
  const Reconocimiento = window.SpeechRecognition ?? window.webkitSpeechRecognition;
  if (!Reconocimiento) {
    botonMicrofono.disabled = true;
    botonMicrofono.textContent = "Voz no disponible";
    $("nota-voz").textContent = "Este navegador no tiene reconocimiento de voz: escribe el texto abajo.";
    return;
  }
  const reconocimiento = Object.assign(new Reconocimiento(), { lang: "es-MX", interimResults: true, continuous: true });
  let escuchando = false;

  const cambiar = (activo) => {
    escuchando = activo;
    botonMicrofono.setAttribute("aria-pressed", String(activo));
    botonMicrofono.textContent = activo ? "Dejar de escuchar" : "Hablar";
  };

  reconocimiento.addEventListener("result", (evento) => {
    let provisional = "";
    for (let i = evento.resultIndex; i < evento.results.length; i++) {
      const texto = evento.results[i][0].transcript;
      if (evento.results[i].isFinal) alDecir(texto);
      else provisional += texto;
    }
    parcial.textContent = provisional;
  });
  reconocimiento.addEventListener("end", () => cambiar(false));
  reconocimiento.addEventListener("error", (evento) => {
    estado.textContent = `El reconocimiento de voz falló: ${evento.error}. Puedes escribir el texto.`;
  });

  botonMicrofono.addEventListener("click", () => {
    if (escuchando) {
      reconocimiento.stop();
      return;
    }
    reconocimiento.start();
    cambiar(true);
    estado.textContent = "Escuchando…";
  });
}

async function main() {
  const [vocabulario, animaciones] = await Promise.all([leerJSON(VOCABULARIO), cargarAnimaciones()]);
  const diccionario = crearDiccionario(vocabulario);
  let cola;
  const avatar = crearAvatar($("lienzo"), animaciones, {
    alCambiarToken: resaltar,
    alCambiarLetra: mostrarLetra,
    alTerminar: () => cola.alTerminar(),
  });
  cola = crearCola(avatar, diccionario);

  $("formulario").addEventListener("submit", (evento) => {
    evento.preventDefault();
    cola.agregar($("texto").value);
  });
  botonRepetir.addEventListener("click", () => cola.repetir());
  $("velocidad").addEventListener("input", (evento) => {
    avatar.velocidad = Number(evento.target.value);
    $("valor-velocidad").textContent = `${Number(evento.target.value).toFixed(1)}×`;
  });
  prepararMicrofono((texto) => {
    $("texto").value = texto.trim();
    cola.agregar(texto);
  });

  const conSena = avatar.glosasDisponibles.size;
  estado.textContent = `Listo: ${Object.keys(vocabulario.glosas).length} glosas en el vocabulario, ${conSena} con animación.`;
}

main().catch((error) => {
  estado.textContent = `Error al cargar: ${error.message}`;
  console.error(error);
});
