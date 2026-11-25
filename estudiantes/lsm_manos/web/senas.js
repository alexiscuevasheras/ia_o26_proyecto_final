// Página señas → texto: cámara → MediaPipe (cuerpo y dos manos) → en_vivo.js → pantalla.
//
// La página se sirve desde la carpeta del proyecto (make demo); los modelos se leen de
// modelos/, que no se sube a git (el README dice cómo generarlos).

import { adaptarSalida } from "./adaptar.js";
import { cargarDetectores, cicloDeVideo, encenderCamara, HandLandmarker } from "./camara.js";
import { TraductorEnVivo } from "./en_vivo.js";
import { dibujarEscena, pintarMatriz } from "./dibujo.js";
import { glosasATexto } from "./glosas_a_texto.js";

const RUTAS = {
  palabras: "../modelos/palabras.json",
  letras: "../modelos/pesos.json",
  vocabulario: "vocabulario.json",
};

const $ = (id) => document.getElementById(id);
const video = $("video");
const lienzo = $("puntos");
const estado = $("estado");

const reconocidos = []; // [{ tipo: "glosa", glosa, opciones, resumen } | { tipo: "letra", letra }]
const confirmados = []; // [{ resumen, k }] para adaptar la capa de salida
let traductor;
let vocabulario;
let paramsOriginales;

async function cargarJSON(ruta) {
  const r = await fetch(ruta);
  if (!r.ok) throw new Error(`no encontré ${ruta} (ver «Cómo correrlo» en el README)`);
  return r.json();
}

// ---- Pantalla ----

function frase() {
  return glosasATexto(reconocidos.filter((r) => r.tipo !== "no_se"), vocabulario);
}

function mostrarFrase() {
  const texto = frase();
  $("frase").textContent = texto || "—";
  $("subtitulo").textContent = texto;
}

function mostrarReconocidos() {
  $("reconocidas").replaceChildren(...reconocidos.map((r, i) => {
    const li = document.createElement("li");
    const boton = document.createElement("button");
    boton.className = `ficha ${r.tipo}`;
    boton.textContent = r.tipo === "letra" ? r.letra : r.tipo === "no_se" ? "¿?" : r.glosa;
    boton.title = r.tipo === "letra" ? "Letra deletreada" : "Toca para corregir";
    boton.disabled = r.tipo === "letra";
    boton.addEventListener("click", () => mostrarOpciones(i));
    li.append(boton);
    return li;
  }));
  mostrarFrase();
}

function mostrarOpciones(i) {
  const r = reconocidos[i];
  const lista = $("mejores");
  lista.replaceChildren(...r.opciones.map(({ glosa, probabilidad }) => {
    const li = document.createElement("li");
    const boton = document.createElement("button");
    boton.className = "opcion";
    boton.innerHTML = `<span>${glosa}</span><span class="barra"><span style="width:${(100 * probabilidad).toFixed(0)}%"></span></span>
      <span class="porcentaje">${(100 * probabilidad).toFixed(0)} %</span>`;
    boton.addEventListener("click", () => confirmar(i, glosa));
    li.append(boton);
    return li;
  }));
}

/** La persona eligió la glosa correcta: se corrige y la red se adapta a ella. */
function confirmar(i, glosa) {
  const r = reconocidos[i];
  reconocidos[i] = { ...r, tipo: "glosa", glosa };
  confirmados.push({ resumen: r.resumen, k: traductor.modelo.glosas.indexOf(glosa) });
  const { V, c } = adaptarSalida({ params: paramsOriginales }, confirmados);
  traductor.modelo.params = { ...traductor.modelo.params, V, c };
  $("adaptacion").textContent = `La red se adaptó con ${confirmados.length} seña${confirmados.length > 1 ? "s" : ""} tuya${confirmados.length > 1 ? "s" : ""}.`;
  $("olvidar").disabled = false;
  $("mejores").replaceChildren();
  mostrarReconocidos();
}

function olvidar() {
  confirmados.length = 0;
  traductor.modelo.params = { ...paramsOriginales };
  $("adaptacion").textContent = "La red todavía no ha aprendido de ti.";
  $("olvidar").disabled = true;
}

function atenderEvento(evento) {
  reconocidos.push(evento);
  if (evento.tipo !== "letra") {
    pintarMatriz($("entrada"), evento.x, 2);
    pintarMatriz($("estados"), evento.estados, 2);
    mostrarOpciones(reconocidos.length - 1);
  }
  mostrarReconocidos();
}

function mostrarFase() {
  const fase = traductor.letraActual ? `deletreando (${traductor.letraActual})`
    : traductor.segmentador.enSena ? "señando" : "en reposo";
  $("fase").textContent = fase;
  $("fase").dataset.fase = traductor.segmentador.enSena ? "activa" : "reposo";
}

// ---- Ciclo de video ----

function alCuadro(cuadro) {
  dibujarEscena(lienzo, cuadro, HandLandmarker.HAND_CONNECTIONS, traductor.dominante);
  const evento = traductor.agregar(cuadro, video.videoWidth / video.videoHeight);
  if (evento) atenderEvento(evento);
  mostrarFase();
}

function alError(error) {
  estado.textContent = `Error en este cuadro: ${error.message}`;
  console.error(error);
}

async function main() {
  const [detectores, modelo, letras, vocab] = await Promise.all([
    cargarDetectores(), cargarJSON(RUTAS.palabras), cargarJSON(RUTAS.letras), cargarJSON(RUTAS.vocabulario),
  ]);
  vocabulario = vocab;
  paramsOriginales = { ...modelo.params };
  traductor = new TraductorEnVivo(modelo, letras);
  estado.textContent = `Listo: ${modelo.glosas.length} palabras y ${letras.letras.length} letras. Enciende la cámara.`;

  const boton = $("iniciar");
  boton.disabled = false;
  boton.addEventListener("click", async () => {
    boton.disabled = true;
    try {
      await encenderCamara(video, lienzo);
    } catch (error) {
      estado.textContent = `No pude abrir la cámara: ${error.message}`;
      boton.disabled = false;
      return;
    }
    estado.textContent = "Manos abajo para empezar; súbelas para señar.";
    cicloDeVideo(video, detectores, alCuadro, alError);
  });
  $("leer").addEventListener("click", () => {
    const texto = frase();
    if (texto && "speechSynthesis" in window) {
      const voz = new SpeechSynthesisUtterance(texto);
      voz.lang = "es-MX";
      speechSynthesis.speak(voz);
    }
  });
  $("borrar").addEventListener("click", () => {
    reconocidos.length = 0;
    $("mejores").replaceChildren();
    mostrarReconocidos();
  });
  $("olvidar").addEventListener("click", olvidar);
}

$("iniciar").disabled = true;
main().catch((error) => {
  estado.textContent = `Error al cargar: ${error.message}`;
});
