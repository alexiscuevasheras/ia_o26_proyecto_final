// Avatar 3D que seña: recibe tokens (glosas y deletreo) y anima un cuerpo estilizado.
//
// Los datos siguen docs/FORMATOS.md: metros, origen entre los hombros, x a la
// izquierda de la persona, y arriba, z hacia la cámara. Es el mismo sistema
// que el de three.js con la cámara de frente, así que no hay que reflejar nada.

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const CONEXIONES_MANO = [
  [0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [5, 9], [9, 10], [10, 11],
  [11, 12], [9, 13], [13, 14], [14, 15], [15, 16], [13, 17], [0, 17], [17, 18], [18, 19], [19, 20],
];
const PALMA = [[0, 1, 5], [0, 5, 9], [0, 9, 13], [0, 13, 17]]; // triángulos sobre los puntos de la mano
const PUNTAS = new Set([4, 8, 12, 16, 20]);

// Tiempos en segundos (a velocidad 1×).
const TRANSICION = 0.25;
const CAMBIO_LETRA = 0.12;
const LETRA = 0.35;
const LETRA_CON_TRAZO = 0.8;
const A_REPOSO = 0.5;

// Las letras con movimiento no están en D1: se hacen con una forma estática parecida más un trazo.
// K y P comparten configuración en la LSM; Q y X son aproximaciones nuestras, sin validar con intérpretes.
const BASE_DE_MOVIMIENTO = { J: "I", Ñ: "N", Z: "D", K: "P", Q: "G", X: "D" };
// Trazos de la muñeca (metros) desde la vista de quien seña; +x es su izquierda.
const TRAZOS = {
  J: [[0, 0, 0], [0, -0.05, 0], [0.015, -0.075, 0.01], [0.035, -0.06, 0.01]],
  Z: [[0.03, 0.03, 0], [-0.03, 0.03, 0], [0.03, -0.03, 0], [-0.03, -0.03, 0]],
  Ñ: [[0, 0, 0], [0.015, 0, 0], [-0.015, 0, 0], [0.015, 0, 0], [0, 0, 0]],
  K: [[0, -0.02, 0], [0, 0.04, 0]],
  Q: [[0, 0, 0], [0, -0.04, 0.02]],
  X: [[0, 0, 0], [0, -0.02, 0.04]],
};
const DIGRAFOS = new Set(["LL", "RR", "CH"]);

// Dónde se deletrea: muñeca derecha frente al hombro derecho, relativa a ese hombro.
const ESPACIO_DELETREO = [0.04, -0.06, 0.28];
const POLO_CODO = { izq: [0.4, -1, -0.2], der: [-0.4, -1, -0.2] }; // el codo cae hacia abajo y afuera

const COLORES = { piel: 0xe2b08c, ropa: 0x4a6fa5, base: 0x34465f, rasgos: 0x2a2a30 };

// ---------- Vectores como arreglos [x, y, z] ----------

const sumar = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const restar = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const escalar = (a, k) => [a[0] * k, a[1] * k, a[2] * k];
const punto = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const norma = (a) => Math.hypot(a[0], a[1], a[2]);
const unitario = (a) => escalar(a, 1 / (norma(a) || 1));
const cruz = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const mezclar = (a, b, t) => sumar(a, escalar(restar(b, a), t));
const medio = (a, b) => mezclar(a, b, 0.5);
const espejo = (a) => [-a[0], a[1], a[2]];
const suave = (t) => t * t * (3 - 2 * t);

/** La componente de `v` perpendicular a `eje` (unitario); `otro` si casi no queda nada. */
function perpendicular(v, eje, otro) {
  const resto = restar(v, escalar(eje, punto(v, eje)));
  return norma(resto) > 1e-3 ? unitario(resto) : perpendicular(otro, eje, [0, 0, 1]);
}

function mediana(valores) {
  const orden = [...valores].sort((a, b) => a - b);
  return orden[Math.floor(orden.length / 2)];
}

// ---------- Poses: {cuerpo: [[x,y,z] × n], manos: [izquierda, derecha]} ----------

function mezclarManos(a, b, t) {
  if (!a || !b) return a ?? b;
  return a.map((p, i) => mezclar(p, b[i], t));
}

function mezclarPoses(a, b, t) {
  return {
    cuerpo: a.cuerpo.map((p, i) => mezclar(p, b.cuerpo[i], t)),
    manos: [0, 1].map((m) => mezclarManos(a.manos[m], b.manos[m], t)),
  };
}

function copiarPose(pose) {
  return { cuerpo: pose.cuerpo.map((p) => [...p]), manos: pose.manos.map((m) => m && m.map((p) => [...p])) };
}

/** Marco propio de una mano derecha: x hacia el índice, y hacia el dedo medio, z la palma. */
function marcoMano(forma) {
  const y = unitario(restar(forma[9], forma[0]));
  const x = perpendicular(restar(forma[5], forma[17]), y, [1, 0, 0]);
  return [x, y, cruz(x, y)];
}

/**
 * Coloca una forma de mano (derecha, cualquier orientación) en la muñeca, con los
 * dedos siguiendo al antebrazo y la palma hacia el cuerpo. La izquierda es su espejo.
 */
function orientarMano(forma, codo, muneca, esIzquierda) {
  if (!forma) return null;
  if (esIzquierda) return orientarMano(forma, espejo(codo), espejo(muneca), false).map(espejo);
  const [fx, fy, fz] = marcoMano(forma);
  const y = unitario(restar(muneca, codo));
  const z = perpendicular([1, 0, 0], y, [0, 0, 1]);
  const x = cruz(y, z);
  return forma.map((p) => {
    const q = restar(p, forma[0]);
    const local = [punto(q, fx), punto(q, fy), punto(q, fz)];
    return sumar(muneca, sumar(sumar(escalar(x, local[0]), escalar(y, local[1])), escalar(z, local[2])));
  });
}

/** IK de dos huesos: el codo que deja la muñeca en `objetivo`, doblado hacia `polo`. */
function codoIK(hombro, objetivo, brazo, antebrazo, polo) {
  const eje = unitario(restar(objetivo, hombro));
  const d = Math.min(Math.max(norma(restar(objetivo, hombro)), Math.abs(brazo - antebrazo) + 1e-3), brazo + antebrazo - 1e-3);
  const coseno = (brazo ** 2 + d ** 2 - antebrazo ** 2) / (2 * brazo * d);
  const lado = perpendicular(polo, eje, [0, -1, 0]);
  const codo = sumar(hombro, escalar(sumar(escalar(eje, coseno), escalar(lado, Math.sqrt(1 - coseno ** 2))), brazo));
  return { codo, muneca: sumar(hombro, escalar(eje, d)) };
}

/** Punto en una poligonal de trazo, con cada tramo suavizado; u ∈ [0, 1]. */
function puntoDeTrazo(trazo, u) {
  const tramos = trazo.length - 1;
  const i = Math.min(Math.floor(u * tramos), tramos - 1);
  return mezclar(trazo[i], trazo[i + 1], suave(u * tramos - i));
}

// ---------- De los datos a clips listos para reproducir ----------

function crearEsqueleto(datos) {
  const lugar = new Map(datos.cuerpo_indices.map((mp, i) => [mp, i]));
  const brazos = {
    izq: { hombro: lugar.get(11), codo: lugar.get(13), muneca: lugar.get(15), mano: 0 },
    der: { hombro: lugar.get(12), codo: lugar.get(14), muneca: lugar.get(16), mano: 1 },
  };
  return { lugar, brazos, p: (pose, mp) => pose.cuerpo[lugar.get(mp)] };
}

/** Pose de descanso: cabeza y torso promedio de todos los cuadros, brazos colgando. */
function crearReposo(datos, esqueleto, relajada) {
  const cuadros = Object.values(datos.glosas).flatMap((g) => g.cuadros);
  const n = datos.cuerpo_indices.length;
  const cuerpo = Array.from({ length: n }, (_, i) =>
    escalar(cuadros.reduce((s, c) => sumar(s, c.cuerpo[i]), [0, 0, 0]), 1 / cuadros.length));
  const largo = (a, b) => mediana(cuadros.flatMap((c) =>
    Object.values(esqueleto.brazos).map((br) => norma(restar(c.cuerpo[br[a]], c.cuerpo[br[b]])))));
  const brazo = largo("hombro", "codo");
  const antebrazo = largo("codo", "muneca");

  const manos = [null, null];
  for (const [lado, br] of Object.entries(esqueleto.brazos)) {
    const afuera = lado === "izq" ? 1 : -1;
    cuerpo[br.codo] = sumar(cuerpo[br.hombro], [0.02 * afuera, -brazo, 0]);
    cuerpo[br.muneca] = sumar(cuerpo[br.codo], [0.01 * afuera, -antebrazo, 0.04]);
    manos[br.mano] = orientarMano(relajada, cuerpo[br.codo], cuerpo[br.muneca], lado === "izq");
  }
  return { pose: { cuerpo, manos }, brazo, antebrazo };
}

/** Donde no se vio una mano: la última vista (o la siguiente) movida con la muñeca; si nunca, la relajada. */
function rellenarManos(cuadros, br, lado, relajada) {
  const vistos = cuadros.flatMap((c, i) => (c.manos[br.mano] ? [i] : []));
  cuadros.forEach((c, i) => {
    if (c.manos[br.mano]) return;
    const muneca = c.cuerpo[br.muneca];
    if (vistos.length === 0) {
      c.manos[br.mano] = orientarMano(relajada, c.cuerpo[br.codo], muneca, lado === "izq");
      return;
    }
    const fuente = cuadros[vistos.findLast((v) => v < i) ?? vistos[0]];
    const mover = restar(muneca, fuente.cuerpo[br.muneca]);
    c.manos[br.mano] = fuente.manos[br.mano].map((p) => sumar(p, mover));
  });
}

/** Promedio móvil de 3 cuadros: quita el temblor de MediaPipe sin borrar el movimiento. */
function suavizar(cuadros) {
  return cuadros.map((c, i) => {
    const vecinos = cuadros.slice(Math.max(0, i - 1), i + 2);
    const promedio = (elegir) => elegir(c).map((_, k) =>
      escalar(vecinos.reduce((s, v) => sumar(s, elegir(v)[k]), [0, 0, 0]), 1 / vecinos.length));
    return {
      cuerpo: promedio((v) => v.cuerpo),
      manos: [0, 1].map((m) => c.manos[m] && promedio((v) => v.manos[m])),
    };
  });
}

function prepararClip(glosa, esqueleto, relajada) {
  const cuadros = glosa.cuadros.map(copiarPose);
  for (const [lado, br] of Object.entries(esqueleto.brazos)) rellenarManos(cuadros, br, lado, relajada);
  return suavizar(cuadros);
}

// ---------- Escena de three.js ----------

function color(nombre) {
  return getComputedStyle(document.documentElement).getPropertyValue(nombre).trim();
}

function crearMateriales() {
  const material = (hex, extra = {}) => new THREE.MeshStandardMaterial({ color: hex, roughness: 0.65, ...extra });
  return {
    piel: material(COLORES.piel),
    ropa: material(COLORES.ropa, { roughness: 0.85 }),
    base: material(COLORES.base, { roughness: 0.9 }),
    rasgos: material(COLORES.rasgos),
    palma: material(COLORES.piel, { side: THREE.DoubleSide }),
  };
}

const GEOMETRIA = {
  esfera: new THREE.SphereGeometry(1, 24, 16),
  cilindro: new THREE.CylinderGeometry(1, 1, 1, 16),
  torso: new THREE.CylinderGeometry(1, 0.78, 1, 32),
  base: new THREE.CylinderGeometry(0.8, 1, 1, 32),
};

function malla(escena, geometria, material) {
  const m = new THREE.Mesh(geometria, material);
  m.castShadow = true;
  escena.add(m);
  return m;
}

const tmpA = new THREE.Vector3();
const tmpB = new THREE.Vector3();
const EJE_Y = new THREE.Vector3(0, 1, 0);

function colocarSegmento(m, a, b, radio) {
  tmpA.set(...a);
  tmpB.set(...b).sub(tmpA);
  const largo = tmpB.length();
  m.position.set(...medio(a, b));
  m.quaternion.setFromUnitVectors(EJE_Y, tmpB.divideScalar(largo || 1));
  m.scale.set(radio, largo, radio);
}

function colocarEsfera(m, p, radio) {
  m.position.set(...p);
  m.scale.setScalar(radio);
}

/** Orienta un objeto con la base (x, y, z) dada en coordenadas del mundo. */
function orientarObjeto(objeto, x, y, z) {
  objeto.quaternion.setFromRotationMatrix(new THREE.Matrix4().makeBasis(
    new THREE.Vector3(...x), new THREE.Vector3(...y), new THREE.Vector3(...z)));
}

function crearCabeza(escena, materiales) {
  const cabeza = new THREE.Group();
  const craneo = malla(cabeza, GEOMETRIA.esfera, materiales.piel);
  craneo.scale.set(1, 1.2, 1.05);
  for (const lado of [-1, 1]) {
    const ojo = malla(cabeza, GEOMETRIA.esfera, materiales.rasgos);
    ojo.position.set(0.36 * lado, 0.15, 0.93);
    ojo.scale.setScalar(0.09);
  }
  const nariz = malla(cabeza, GEOMETRIA.esfera, materiales.piel);
  nariz.position.set(0, -0.08, 1.02);
  nariz.scale.setScalar(0.13);
  escena.add(cabeza);
  return cabeza;
}

function crearMano(escena, materiales) {
  const geometria = new THREE.BufferGeometry();
  geometria.setAttribute("position", new THREE.BufferAttribute(new Float32Array(PALMA.length * 9), 3));
  const palma = malla(escena, geometria, materiales.palma);
  return {
    palma,
    huesos: CONEXIONES_MANO.map(() => malla(escena, GEOMETRIA.cilindro, materiales.piel)),
    nudos: Array.from({ length: 21 }, () => malla(escena, GEOMETRIA.esfera, materiales.piel)),
  };
}

function actualizarMano(mano, puntos) {
  const visible = Boolean(puntos);
  for (const m of [mano.palma, ...mano.huesos, ...mano.nudos]) m.visible = visible;
  if (!visible) return;
  // El grosor sigue al tamaño de la mano: los datos reales no miden 0.09 m de palma exactos.
  const palma = norma(restar(puntos[9], puntos[0]));
  CONEXIONES_MANO.forEach(([a, b], i) => colocarSegmento(mano.huesos[i], puntos[a], puntos[b], palma * 0.085));
  puntos.forEach((p, i) => colocarEsfera(mano.nudos[i], p, palma * (PUNTAS.has(i) ? 0.085 : 0.1)));
  const posiciones = mano.palma.geometry.attributes.position;
  PALMA.flat().forEach((indice, k) => posiciones.setXYZ(k, ...puntos[indice]));
  posiciones.needsUpdate = true;
  mano.palma.geometry.computeVertexNormals();
  mano.palma.geometry.computeBoundingSphere();
}

function crearCuerpo(escena) {
  const materiales = crearMateriales();
  const segmento = (material) => malla(escena, GEOMETRIA.cilindro, material);
  const esfera = (material) => malla(escena, GEOMETRIA.esfera, material);
  return {
    cabeza: crearCabeza(escena, materiales),
    cuello: segmento(materiales.piel),
    torso: malla(escena, GEOMETRIA.torso, materiales.ropa),
    hombros: segmento(materiales.ropa),
    base: malla(escena, GEOMETRIA.base, materiales.base),
    brazos: ["izq", "der"].map(() => ({
      hombro: esfera(materiales.ropa), brazo: segmento(materiales.ropa), codo: esfera(materiales.ropa),
      antebrazo: segmento(materiales.piel), muneca: esfera(materiales.piel),
    })),
    manos: [crearMano(escena, materiales), crearMano(escena, materiales)],
  };
}

function actualizarCabeza(cabeza, p) {
  const orejas = medio(p(7), p(8));
  const x = unitario(restar(p(7), p(8)));
  const arriba = perpendicular(restar(medio(p(2), p(5)), medio(p(9), p(10))), x, [0, 1, 0]);
  const radio = 0.62 * norma(restar(p(7), p(8)));
  cabeza.position.set(...sumar(orejas, escalar(arriba, 0.25 * radio)));
  cabeza.scale.setScalar(radio);
  orientarObjeto(cabeza, x, arriba, cruz(x, arriba));
  return { centro: cabeza.position.toArray(), arriba, radio };
}

function actualizarCuerpo(partes, pose, esqueleto, piso) {
  const p = (mp) => esqueleto.p(pose, mp);
  const hombros = medio(p(11), p(12));
  const caderas = medio(p(23), p(24));
  const ancho = norma(restar(p(11), p(12)));

  const cabeza = actualizarCabeza(partes.cabeza, p);
  colocarSegmento(partes.cuello, hombros, sumar(cabeza.centro, escalar(cabeza.arriba, -0.9 * cabeza.radio)), ancho * 0.17);

  const x = unitario(restar(p(11), p(12)));
  const y = perpendicular(restar(hombros, caderas), x, [0, 1, 0]);
  const alto = norma(restar(hombros, caderas)) + 0.04;
  partes.torso.position.set(...sumar(medio(hombros, caderas), escalar(y, -0.02)));
  orientarObjeto(partes.torso, x, y, cruz(x, y));
  partes.torso.scale.set(ancho * 0.5, alto, ancho * 0.3);
  colocarSegmento(partes.hombros, p(12), p(11), ancho * 0.16);

  const anchoCaderas = norma(restar(p(23), p(24)));
  const alturaBase = caderas[1] - piso;
  partes.base.position.set(caderas[0], piso + alturaBase / 2, caderas[2]);
  partes.base.scale.set(anchoCaderas * 0.62, alturaBase, anchoCaderas * 0.45);

  Object.values(esqueleto.brazos).forEach((br, i) => {
    const partesBrazo = partes.brazos[i];
    const [hombro, codo, muneca] = [br.hombro, br.codo, br.muneca].map((k) => pose.cuerpo[k]);
    colocarEsfera(partesBrazo.hombro, hombro, ancho * 0.16);
    colocarSegmento(partesBrazo.brazo, hombro, codo, ancho * 0.13);
    colocarEsfera(partesBrazo.codo, codo, ancho * 0.12);
    colocarSegmento(partesBrazo.antebrazo, codo, muneca, ancho * 0.1);
    colocarEsfera(partesBrazo.muneca, muneca, ancho * 0.09);
  });
  partes.manos.forEach((mano, m) => actualizarMano(mano, pose.manos[m]));
}

function crearPiso(escena, y) {
  const piso = new THREE.Mesh(new THREE.CircleGeometry(0.9, 64), new THREE.MeshStandardMaterial({ roughness: 1 }));
  piso.rotation.x = -Math.PI / 2;
  piso.position.y = y;
  piso.receiveShadow = true;
  const pintar = () => piso.material.color.set(color("--borde") || "#dddcd7");
  pintar();
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", pintar);
  escena.add(piso);
}

function crearLuces(escena, centro) {
  escena.add(new THREE.HemisphereLight(0xffffff, 0x8a8a90, 1.6));
  const sol = new THREE.DirectionalLight(0xffffff, 2.2);
  sol.position.set(1.2, 2.6, 2.2);
  sol.target.position.set(...centro);
  sol.castShadow = true;
  sol.shadow.mapSize.set(1024, 1024);
  Object.assign(sol.shadow.camera, { left: -1.2, right: 1.2, top: 1.2, bottom: -1.2, near: 0.5, far: 8 });
  sol.shadow.bias = -0.0005;
  escena.add(sol, sol.target);
}

// ---------- El avatar ----------

/**
 * Crea el avatar en `canvas`. `eventos`: alCambiarToken(i | null),
 * alCambiarLetra(letra | null, nota | null), alTerminar().
 */
export function crearAvatar(canvas, datos, eventos) {
  const esqueleto = crearEsqueleto(datos);
  const letras = datos.letras ?? {};
  const conMovimiento = new Set(datos.letras_con_movimiento ?? []);
  const manoGenerica = (letras.B ?? Object.values(letras)[0])?.mano ?? null;
  const reposo = crearReposo(datos, esqueleto, manoGenerica);
  const clips = new Map(Object.entries(datos.glosas).map(([g, glosa]) => [g, prepararClip(glosa, esqueleto, manoGenerica)]));
  const fps = datos.fps;

  const piso = Math.min(...reposo.pose.manos.flatMap((m) => (m ? m.map((q) => q[1]) : [])),
    esqueleto.p(reposo.pose, 16)[1] - 0.2) - 0.03;
  const cabezaY = esqueleto.p(reposo.pose, 0)[1] + 0.12;
  const centro = [0, (cabezaY + piso) / 2, 0];

  const escena = new THREE.Scene();
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  const camara = new THREE.PerspectiveCamera(32, 1, 0.05, 20);
  const distancia = 1.15 * (cabezaY - piso) / 2 / Math.tan(THREE.MathUtils.degToRad(16));
  camara.position.set(0, centro[1] + 0.05, distancia);
  const controles = new OrbitControls(camara, canvas);
  controles.target.set(...centro);
  Object.assign(controles, { enableDamping: true, enablePan: false, minDistance: 0.8, maxDistance: 6 });

  crearLuces(escena, centro);
  crearPiso(escena, piso);
  const partes = crearCuerpo(escena);

  new ResizeObserver(() => {
    const { clientWidth: ancho, clientHeight: alto } = canvas;
    renderer.setSize(ancho, alto, false);
    camara.aspect = ancho / alto;
    camara.updateProjectionMatrix();
  }).observe(canvas);

  // ----- Deletreo -----

  function poseDeLetra(forma, desplazamiento) {
    const pose = copiarPose(reposo.pose);
    const br = esqueleto.brazos.der;
    const hombro = pose.cuerpo[br.hombro];
    const objetivo = sumar(sumar(hombro, ESPACIO_DELETREO), desplazamiento);
    const { codo, muneca } = codoIK(hombro, objetivo, reposo.brazo, reposo.antebrazo, POLO_CODO.der);
    pose.cuerpo[br.codo] = codo;
    pose.cuerpo[br.muneca] = muneca;
    pose.manos[br.mano] = forma && forma.map((q) => sumar(muneca, restar(q, forma[0])));
    return pose;
  }

  function piezasDeLetra(letra) {
    if (conMovimiento.has(letra)) {
      const forma = letras[BASE_DE_MOVIMIENTO[letra]]?.mano ?? manoGenerica;
      return [{ forma, trazo: TRAZOS[letra], letra, nota: "trazo aproximado" }];
    }
    if (letras[letra]) return [{ forma: letras[letra].mano, letra }];
    if (DIGRAFOS.has(letra)) {
      const nota = `${letra} no tiene forma en los datos: se deletrea ${[...letra].join(" + ")}`;
      return [...letra].flatMap(piezasDeLetra).map((pieza) => ({ ...pieza, letra, nota }));
    }
    return [{ forma: manoGenerica, letra, nota: `sin forma de mano para «${letra}»` }];
  }

  // ----- Línea de tiempo: segmentos {dur, pose(t), token, letra, nota} -----

  let poseVisible = reposo.pose;

  function construirLinea(tokens) {
    const segmentos = [];
    let ultima = poseVisible;
    const irA = (destino, dur, info) => {
      const desde = ultima;
      segmentos.push({ dur, pose: (t) => mezclarPoses(desde, destino, suave(t / dur)), ...info });
      ultima = destino;
    };
    const sostener = (dur, pose, info) => {
      segmentos.push({ dur, pose, ...info });
      ultima = pose(dur);
    };

    tokens.forEach((token, i) => {
      if (token.tipo === "glosa") {
        const clip = clips.get(token.glosa);
        const duracion = Math.max(clip.length - 1, 1) / fps;
        irA(clip[0], TRANSICION, { token: i });
        sostener(duracion, (t) => poseDeClip(clip, t), { token: i });
        return;
      }
      token.letras.flatMap(piezasDeLetra).forEach((pieza, k) => {
        const info = { token: i, letra: pieza.letra, nota: pieza.nota ?? null };
        const enTrazo = (u) => poseDeLetra(pieza.forma, pieza.trazo ? puntoDeTrazo(pieza.trazo, u) : [0, 0, 0]);
        irA(enTrazo(0), k === 0 ? TRANSICION : CAMBIO_LETRA, info);
        const dur = pieza.trazo ? LETRA_CON_TRAZO : LETRA;
        sostener(dur, (t) => enTrazo(t / dur), info);
      });
    });
    irA(reposo.pose, A_REPOSO, { token: null });
    return segmentos;
  }

  function poseDeClip(clip, t) {
    const f = Math.min(t * fps, clip.length - 1);
    const i = Math.floor(f);
    return mezclarPoses(clip[i], clip[Math.min(i + 1, clip.length - 1)], f - i);
  }

  // ----- Reproducción -----

  let linea = null;
  let tiempo = 0;
  let velocidad = 1;
  let actual = { token: null, letra: null, nota: null };

  function avisarCambios(segmento) {
    const siguiente = { token: segmento?.token ?? null, letra: segmento?.letra ?? null, nota: segmento?.nota ?? null };
    if (siguiente.token !== actual.token) eventos.alCambiarToken?.(siguiente.token);
    if (siguiente.letra !== actual.letra || siguiente.nota !== actual.nota) {
      eventos.alCambiarLetra?.(siguiente.letra, siguiente.nota);
    }
    actual = siguiente;
  }

  function poseEnTiempo() {
    let t = tiempo;
    for (const segmento of linea) {
      if (t <= segmento.dur) {
        avisarCambios(segmento);
        return segmento.pose(t);
      }
      t -= segmento.dur;
    }
    linea = null;
    avisarCambios(null);
    eventos.alTerminar?.();
    return reposo.pose;
  }

  let antes = performance.now();
  function cuadro(ahora) {
    const dt = Math.min((ahora - antes) / 1000, 0.1); // una pestaña dormida no salta toda la seña
    antes = ahora;
    if (linea) {
      tiempo += dt * velocidad;
      poseVisible = poseEnTiempo();
    }
    actualizarCuerpo(partes, poseVisible, esqueleto, piso);
    controles.update();
    renderer.render(escena, camara);
    requestAnimationFrame(cuadro);
  }
  requestAnimationFrame(cuadro);

  return {
    glosasDisponibles: new Set(clips.keys()),
    reproducir(tokens) {
      linea = construirLinea(tokens);
      tiempo = 0;
    },
    set velocidad(valor) {
      velocidad = valor;
    },
  };
}
