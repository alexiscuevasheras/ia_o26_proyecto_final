// Cámara → MediaPipe (21 puntos) → features.js → red.js → letras y activaciones.
//
// La página se sirve desde la carpeta del proyecto (make demo), así que los
// pesos y el modelo de MediaPipe se leen de modelos/, que no se sube a git.

import { FilesetResolver, HandLandmarker } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.2.3/vision_bundle.mjs";
import { normalizar, vectorEntrada } from "./features.js";
import { adelante, mejores } from "./red.js";
import { crearCapas, dibujarMano, dibujarNormalizada } from "./dibujo.js";

const WASM = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.2.3/wasm";
const MODELO_MANOS = "../modelos/hand_landmarker.task";
const PESOS = "../modelos/pesos.json";

const $ = (id) => document.getElementById(id);
const video = $("video");
const lienzoPuntos = $("puntos");
const lienzoNormalizada = $("normalizada");
const estado = $("estado");
const lista = $("mejores");
const boton = $("iniciar");

async function cargarDetector() {
  const archivos = await FilesetResolver.forVisionTasks(WASM);
  const opciones = (delegate) => ({
    baseOptions: { modelAssetPath: MODELO_MANOS, delegate },
    runningMode: "VIDEO",
    numHands: 1,
  });
  try {
    return await HandLandmarker.createFromOptions(archivos, opciones("GPU"));
  } catch {
    return HandLandmarker.createFromOptions(archivos, opciones("CPU"));
  }
}

async function cargarRed() {
  const respuesta = await fetch(PESOS);
  if (!respuesta.ok) throw new Error(`no encontré ${PESOS}: corre «python -m lsm.exportar pesos»`);
  return respuesta.json();
}

function mostrarMejores(red, probabilidades) {
  lista.replaceChildren(...mejores(red, probabilidades, 3).map(({ letra, probabilidad }) => {
    const li = document.createElement("li");
    const porcentaje = (100 * probabilidad).toFixed(0);
    li.innerHTML = `<span class="letra">${letra}</span>
      <span class="barra"><span style="width:${porcentaje}%"></span></span>
      <span class="porcentaje">${porcentaje} %</span>`;
    return li;
  }));
}

/** Lo que pasa en cada cuadro del video. */
function procesarCuadro(detector, red, pintarCapas) {
  const resultado = detector.detectForVideo(video, performance.now());
  const mano = resultado.landmarks[0];
  if (!mano) {
    estado.textContent = "No veo ninguna mano.";
    dibujarMano(lienzoPuntos, null);
    dibujarNormalizada(lienzoNormalizada, null);
    return;
  }

  // La etiqueta Left/Right se interpreta igual que en Python porque el cuadro llega sin espejo.
  const esIzquierda = resultado.handedness[0][0].categoryName === "Left";
  const puntos = mano.map(({ x, y, z }) => [x, y, z]);
  dibujarMano(lienzoPuntos, puntos, HandLandmarker.HAND_CONNECTIONS);
  const opciones = { aspecto: video.videoWidth / video.videoHeight, rotar: red.features.rotar, usarZ: red.features.usar_z };
  const x = vectorEntrada(puntos, esIzquierda, opciones);
  const { a } = adelante(red, x);

  estado.textContent = `Mano ${esIzquierda ? "izquierda" : "derecha"} según MediaPipe.`;
  mostrarMejores(red, a[a.length - 1]);
  dibujarNormalizada(lienzoNormalizada, normalizar(puntos, esIzquierda, opciones), HandLandmarker.HAND_CONNECTIONS);
  pintarCapas(a);
}

async function encenderCamara() {
  video.srcObject = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
  await video.play();
  lienzoPuntos.width = video.videoWidth;
  lienzoPuntos.height = video.videoHeight;
}

async function main() {
  const [detector, red] = await Promise.all([cargarDetector(), cargarRed()]);
  const pintarCapas = crearCapas($("capas"), red);
  pintarCapas(red.anchos.map((ancho) => new Array(ancho).fill(0)));
  estado.textContent = `Listo: red ${red.anchos.join("–")}. Enciende la cámara.`;
  boton.disabled = false;

  boton.addEventListener("click", async () => {
    boton.disabled = true;
    try {
      await encenderCamara();
    } catch (error) {
      estado.textContent = `No pude abrir la cámara: ${error.message}`;
      boton.disabled = false;
      return;
    }
    let ultimoTiempo = -1;
    const cuadro = () => {
      if (video.currentTime !== ultimoTiempo) { // no repetir el mismo cuadro del video
        ultimoTiempo = video.currentTime;
        try {
          procesarCuadro(detector, red, pintarCapas);
        } catch (error) { // un cuadro con error no debe detener la demo
          estado.textContent = `Error en este cuadro: ${error.message}`;
          console.error(error);
        }
      }
      requestAnimationFrame(cuadro);
    };
    requestAnimationFrame(cuadro);
  });
}

boton.disabled = true;
main().catch((error) => {
  estado.textContent = `Error al cargar: ${error.message}`;
});
