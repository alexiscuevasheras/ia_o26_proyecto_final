// Cámara y detectores de MediaPipe compartidos por la página de señas y la de grabación,
// para que las dos produzcan exactamente los mismos cuadros.

import { FilesetResolver, HandLandmarker, PoseLandmarker } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.2.3/vision_bundle.mjs";
import { asignarManos } from "./en_vivo.js";

export { HandLandmarker };

const WASM = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.2.3/wasm";
const MODELO_MANOS = "../modelos/hand_landmarker.task";
const MODELO_CUERPO = "../modelos/pose_landmarker_full.task";

/** [detector de manos, detector de cuerpo], con GPU si se puede. */
export async function cargarDetectores() {
  const archivos = await FilesetResolver.forVisionTasks(WASM);
  const crear = async (Clase, opciones) => {
    try {
      return await Clase.createFromOptions(archivos, { ...opciones, baseOptions: { ...opciones.baseOptions, delegate: "GPU" } });
    } catch {
      return Clase.createFromOptions(archivos, { ...opciones, baseOptions: { ...opciones.baseOptions, delegate: "CPU" } });
    }
  };
  return Promise.all([
    crear(HandLandmarker, { baseOptions: { modelAssetPath: MODELO_MANOS }, runningMode: "VIDEO", numHands: 2,
      minHandDetectionConfidence: 0.1, minHandPresenceConfidence: 0.1 }), // igual que extraer.py (P7)
    crear(PoseLandmarker, { baseOptions: { modelAssetPath: MODELO_CUERPO }, runningMode: "VIDEO",
      minPoseDetectionConfidence: 0.3, minPosePresenceConfidence: 0.3 }),
  ]);
}

const aArreglos = (lm) => lm.map(({ x, y, z }) => [x, y, z]);

/**
 * Resultados de MediaPipe → cuadro { cuerpo, manos, cuerpo3d, manos3d }, con cada mano
 * en su brazo como en extraer.py (las «3d» son las coordenadas world, en metros).
 */
export function cuadroDe(rManos, rCuerpo) {
  const cuerpo = rCuerpo.landmarks[0]?.map(({ x, y, z, visibility }) => [x, y, z, visibility ?? 1]) ?? null;
  const cuerpo3d = rCuerpo.worldLandmarks?.[0] ? aArreglos(rCuerpo.worldLandmarks[0]) : null;
  const manos = rManos.landmarks.map(aArreglos);
  const asignadas = asignarManos(manos, cuerpo);
  const manos3d = asignadas.map((m) => {
    const i = manos.indexOf(m);
    return i >= 0 && rManos.worldLandmarks?.[i] ? aArreglos(rManos.worldLandmarks[i]) : null;
  });
  return { cuerpo, manos: asignadas, cuerpo3d, manos3d };
}

export async function encenderCamara(video, lienzo) {
  video.srcObject = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
  await video.play();
  lienzo.width = video.videoWidth;
  lienzo.height = video.videoHeight;
}

/**
 * Llama a alCuadro(cuadro) con cada cuadro nuevo del video. El cuadro va sin espejo,
 * igual que las fotos y videos del entrenamiento; el espejo es solo CSS.
 */
export function cicloDeVideo(video, [detectorManos, detectorCuerpo], alCuadro, alError) {
  let ultimoTiempo = -1;
  const paso = () => {
    if (video.currentTime !== ultimoTiempo) { // no repetir el mismo cuadro del video
      ultimoTiempo = video.currentTime;
      try {
        const t = performance.now();
        alCuadro(cuadroDe(detectorManos.detectForVideo(video, t), detectorCuerpo.detectForVideo(video, t)));
      } catch (error) { // un cuadro con error no debe detener la página
        alError(error);
      }
    }
    requestAnimationFrame(paso);
  };
  requestAnimationFrame(paso);
}
