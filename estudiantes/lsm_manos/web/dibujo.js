// Dibujos de la demo: la mano sobre el video, la mano normalizada y las activaciones.

const CELDA = 14; // píxeles por neurona en los mapas de activación

function color(nombre) {
  return getComputedStyle(document.documentElement).getPropertyValue(nombre).trim();
}

function dibujarHueso(ctx, puntos, conexiones, aPixeles) {
  ctx.beginPath();
  for (const { start, end } of conexiones) {
    const [x0, y0] = aPixeles(puntos[start]);
    const [x1, y1] = aPixeles(puntos[end]);
    ctx.moveTo(x0, y0);
    ctx.lineTo(x1, y1);
  }
  ctx.stroke();
  for (const p of puntos) {
    const [x, y] = aPixeles(p);
    ctx.beginPath();
    ctx.arc(x, y, 3, 0, 2 * Math.PI);
    ctx.fill();
  }
}

/** Los 21 puntos de MediaPipe (x, y en [0, 1]) sobre el video. */
export function dibujarMano(canvas, puntos, conexiones) {
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!puntos) return;
  ctx.strokeStyle = ctx.fillStyle = color("--acento");
  ctx.lineWidth = 3;
  dibujarHueso(ctx, puntos, conexiones, ([x, y]) => [x * canvas.width, y * canvas.height]);
}

/** La mano ya normalizada: muñeca en el centro, palma de largo 1. */
export function dibujarNormalizada(canvas, puntos, conexiones) {
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!puntos) return;
  const escala = canvas.width / 5; // la mano normalizada cabe en unas ±2.5 palmas
  const centro = [canvas.width / 2, canvas.height * 0.7];
  ctx.strokeStyle = ctx.fillStyle = color("--tinta");
  ctx.lineWidth = 2;
  dibujarHueso(ctx, puntos, conexiones, ([x, y]) => [centro[0] + x * escala, centro[1] + y * escala]);
}

/** Crea un canvas por capa; devuelve una función que los repinta. */
export function crearCapas(contenedor, red) {
  const capas = red.anchos.map((ancho, l) => {
    const columnas = Math.min(ancho, l === red.anchos.length - 1 ? ancho : 64);
    const filas = Math.ceil(ancho / columnas);
    const figura = document.createElement("div");
    figura.className = "capa";
    const titulo = document.createElement("h3");
    titulo.textContent = tituloCapa(red, l);
    const canvas = document.createElement("canvas");
    canvas.width = columnas * CELDA;
    canvas.height = filas * CELDA;
    canvas.style.maxWidth = `${canvas.width}px`;
    figura.append(titulo, canvas);
    contenedor.append(figura);
    return { canvas, columnas };
  });

  return (activaciones) => {
    activaciones.forEach((a, l) => {
      const esSalida = l === red.anchos.length - 1;
      pintarCapa(capas[l], a, esSalida ? red.letras : null, l === 0);
    });
  };
}

function tituloCapa(red, l) {
  const L = red.anchos.length - 1;
  if (l === 0) return `a(0): entrada, ${red.anchos[0]} números (x, y, z de los puntos 1 a 20)`;
  const g = red.activaciones[l - 1];
  return l === L
    ? `a(${l}) = ŷ: ${red.anchos[l]} probabilidades (${g})`
    : `a(${l}): ${red.anchos[l]} neuronas ocultas (${g})`;
}

// La entrada tiene signo: se pinta en dos colores. Las demás capas van de 0 a su máximo.
function pintarCapa({ canvas, columnas }, a, letras, conSigno) {
  const ctx = canvas.getContext("2d");
  ctx.fillStyle = color("--borde");
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  const maximo = Math.max(1e-9, ...a.map(Math.abs));
  a.forEach((v, i) => {
    const x = (i % columnas) * CELDA;
    const y = Math.floor(i / columnas) * CELDA;
    const t = Math.abs(v) / maximo;
    const tono = conSigno && v < 0 ? 25 : 220;
    ctx.fillStyle = `hsl(${tono} 80% ${15 + 60 * t}%)`;
    ctx.fillRect(x + 1, y + 1, CELDA - 2, CELDA - 2);
    if (letras) {
      ctx.fillStyle = t > 0.5 ? "#000" : "#fff";
      ctx.font = `${CELDA - 4}px system-ui`;
      ctx.textAlign = "center";
      ctx.fillText(letras[i], x + CELDA / 2, y + CELDA - 3);
    }
  });
}
