// Página para grabar el dataset propio: corta cada repetición con el mismo segmentador de
// la página en vivo y guarda solo los puntos (nada de video) en un .json que la persona
// descarga. lsm.palabras.importar_propios lo convierte al formato de extraer.py.

import { cargarDetectores, cicloDeVideo, encenderCamara, HandLandmarker } from "./camara.js";
import { dibujarEscena } from "./dibujo.js";
import { marcoEnVivo } from "./secuencias.js";
import { Segmentador } from "./segmentador.js";

const PREDETERMINADAS = `BUENOS-DIAS BUENAS-TARDES BUENAS-NOCHES GRACIAS POR-FAVOR NOS-VEMOS ADIOS
YO TU EL ELLA NOSOTROS USTEDES ELLOS
COMER DORMIR AYUDAR BUSCAR JUGAR ESCUCHAR OLVIDAR HACER LIMPIAR CERRAR ABRIR APRENDER
ESCRIBIR LEER ESTAR DAR AMAR
FAMILIA MAMA PAPA HERMANO HERMANA ESCUELA CASA BAÑO COMIDA AGUA DOCTOR MAESTRO HOSPITAL
DIA SEMANA AYER BIEN SORDO LENGUA-DE-SEÑAS`;
const MARCO_VENTANA = 90;

const $ = (id) => document.getElementById(id);
const video = $("video");
const lienzo = $("puntos");
const estado = $("estado");

const sesion = { version: 1, alias: "", inicio: new Date().toISOString(), aspecto: 4 / 3, muestras: [] };
let cola = []; // glosas pendientes, con repetición
let segmentador = new Segmentador();
let recientes = [];
let centro = null;
let grabando = false;
let guardada = true; // ¿ya se descargó todo lo grabado?

const redondear = (filas) => filas && filas.map((p) => p.map((v) => Math.round(v * 1e5) / 1e5));

function leerLista(glosasValidas) {
  const pedidas = $("lista").value.toUpperCase().split(/\s+/).filter(Boolean);
  const desconocidas = pedidas.filter((g) => !glosasValidas.has(g));
  if (desconocidas.length) estado.textContent = `No están en el vocabulario (se ignoran): ${desconocidas.join(", ")}`;
  return pedidas.filter((g) => glosasValidas.has(g));
}

function mostrarProgreso() {
  const total = sesion.muestras.length;
  if (!cola.length) {
    $("indicacion").textContent = grabando ? "¡Listo! Descarga la sesión." : "";
    $("progreso").textContent = `${total} repeticiones grabadas.`;
    grabando = false;
    return;
  }
  const glosa = cola[0];
  const hechas = sesion.muestras.filter((m) => m.glosa === glosa).length;
  const meta = Number($("repeticiones").value);
  $("indicacion").textContent = `${glosa}  (${hechas + 1}/${meta})`;
  $("progreso").textContent = `${total} grabadas · faltan ${cola.length}`;
}

function alCuadro(cuadro) {
  const aspecto = video.videoWidth / video.videoHeight;
  sesion.aspecto = aspecto;
  dibujarEscena(lienzo, cuadro, HandLandmarker.HAND_CONNECTIONS, 1);
  if (!grabando) return;
  recientes = [...recientes, cuadro].slice(-MARCO_VENTANA);
  const marco = marcoEnVivo(recientes, aspecto, centro);
  centro = marco.centro;
  const sena = segmentador.agregar(cuadro, marco);
  $("indicacion").dataset.fase = segmentador.enSena ? "activa" : "reposo";
  if (!sena) return;
  sesion.muestras.push({
    glosa: cola.shift(),
    cuadros: sena.map((c) => ({
      cuerpo: redondear(c.cuerpo), manos: c.manos.map(redondear),
      cuerpo3d: redondear(c.cuerpo3d), manos3d: c.manos3d.map(redondear),
    })),
  });
  guardada = false;
  $("descargar").disabled = false;
  $("descartar").disabled = false;
  mostrarProgreso();
}

function descartarUltima() {
  const ultima = sesion.muestras.pop();
  if (ultima) cola.unshift(ultima.glosa);
  $("descartar").disabled = !sesion.muestras.length;
  mostrarProgreso();
}

function saltar() {
  const glosa = cola[0];
  cola = cola.filter((g) => g !== glosa);
  mostrarProgreso();
}

function descargar() {
  sesion.alias = $("alias").value.toLowerCase().replace(/[^a-z0-9]/g, "") || "anonimo";
  const archivo = new Blob([JSON.stringify(sesion)], { type: "application/json" });
  const enlace = document.createElement("a");
  enlace.href = URL.createObjectURL(archivo);
  enlace.download = `sesion_${sesion.alias}_${sesion.inicio.slice(0, 16).replace(/[:T]/g, "-")}.json`;
  enlace.click();
  guardada = true;
  URL.revokeObjectURL(enlace.href);
}

async function main() {
  const [detectores, vocabulario] = await Promise.all([
    cargarDetectores(), fetch("vocabulario.json").then((r) => r.json()),
  ]);
  const glosasValidas = new Set(Object.keys(vocabulario.glosas));
  $("lista").value = PREDETERMINADAS;
  estado.textContent = "Listo. Elige alias y palabras, y empieza.";
  $("empezar").disabled = false;

  $("empezar").addEventListener("click", async () => {
    const glosas = leerLista(glosasValidas);
    const meta = Number($("repeticiones").value);
    // Intercaladas (todas las palabras una vez, luego otra vuelta…) para que el cansancio
    // y la luz no se concentren en una sola palabra.
    cola = Array.from({ length: meta }, () => glosas).flat();
    $("empezar").disabled = true;
    try {
      await encenderCamara(video, lienzo);
    } catch (error) {
      estado.textContent = `No pude abrir la cámara: ${error.message}`;
      $("empezar").disabled = false;
      return;
    }
    grabando = true;
    $("saltar").disabled = false;
    mostrarProgreso();
    cicloDeVideo(video, detectores, alCuadro, (error) => { estado.textContent = `Error: ${error.message}`; });
  });
  $("descartar").addEventListener("click", descartarUltima);
  $("saltar").addEventListener("click", saltar);
  $("descargar").addEventListener("click", descargar);
  window.addEventListener("beforeunload", (e) => {
    if (!guardada) e.preventDefault(); // avisar antes de perder lo grabado
  });
}

main().catch((error) => { estado.textContent = `Error al cargar: ${error.message}`; });
