// Dibujos de la página de señas: el esqueleto sobre el video y las matrices de la red.

// Hombros, brazos y caderas de PoseLandmarker (la cara no se dibuja).
const HUESOS_CUERPO = [[11, 12], [11, 13], [13, 15], [12, 14], [14, 16], [11, 23], [12, 24], [23, 24]];

function color(nombre) {
  return getComputedStyle(document.documentElement).getPropertyValue(nombre).trim();
}

function lineas(ctx, puntos, pares, aPixeles) {
  ctx.beginPath();
  for (const [a, b] of pares) {
    if (!puntos[a] || !puntos[b]) continue;
    const [x0, y0] = aPixeles(puntos[a]);
    const [x1, y1] = aPixeles(puntos[b]);
    ctx.moveTo(x0, y0);
    ctx.lineTo(x1, y1);
  }
  ctx.stroke();
}

/**
 * cuadro: { cuerpo, manos: [brazo izquierdo, brazo derecho] } con x, y en [0, 1].
 * dominante: índice de la mano dominante, que se pinta con el color de acento.
 */
export function dibujarEscena(canvas, cuadro, conexionesMano, dominante) {
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!cuadro) return;
  const aPixeles = ([x, y]) => [x * canvas.width, y * canvas.height];
  if (cuadro.cuerpo) {
    ctx.strokeStyle = "rgba(255, 255, 255, 0.7)";
    ctx.lineWidth = 4;
    lineas(ctx, cuadro.cuerpo, HUESOS_CUERPO, aPixeles);
  }
  const pares = conexionesMano.map(({ start, end }) => [start, end]);
  cuadro.manos.forEach((mano, i) => {
    if (!mano) return;
    ctx.strokeStyle = ctx.fillStyle = i === dominante ? color("--acento") : "#f2a33a";
    ctx.lineWidth = 3;
    lineas(ctx, mano, pares, aPixeles);
    for (const p of mano) {
      const [x, y] = aPixeles(p);
      ctx.beginPath();
      ctx.arc(x, y, 3, 0, 2 * Math.PI);
      ctx.fill();
    }
  });
}

/**
 * Pinta una matriz como mapa de color: una columna por momento t, una fila por
 * número. Cada fila se escala a su propio rango para que se vea el cambio en el tiempo.
 */
export function pintarMatriz(canvas, columnas, alto = 2) {
  const T = columnas.length;
  const filas = columnas[0].length;
  canvas.width = T * 8;
  canvas.height = filas * alto;
  const ctx = canvas.getContext("2d");
  for (let i = 0; i < filas; i++) {
    const fila = columnas.map((c) => c[i]);
    const min = Math.min(...fila);
    const rango = Math.max(...fila) - min || 1;
    fila.forEach((v, t) => {
      ctx.fillStyle = `hsl(220 80% ${10 + 70 * ((v - min) / rango)}%)`;
      ctx.fillRect(t * 8, i * alto, 8, alto);
    });
  }
}
