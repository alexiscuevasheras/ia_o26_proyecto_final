# Reparto del trabajo (propuesta, a ajustar entre los tres)

**Todo lo que hay en el repo es preliminar.** Es un primer avance para presentarle el tema al
profesor el lunes 23 de noviembre; la arquitectura, las features, el reparto y los experimentos
pueden cambiar según lo que él diga y lo que decidamos entre los tres.

## Cómo empezar

1. Mándale a Alexis tu usuario de GitHub para que te agregue como colaborador del fork.
2. Clona el repo y entra a la carpeta del proyecto:

   ```bash
   git clone https://github.com/alexiscuevasheras/ia_o26_proyecto_final.git
   cd ia_o26_proyecto_final/estudiantes/lsm_manos
   ```

3. Crea el entorno y corre las pruebas (necesitas Python 3.12 o más nuevo y Node para la prueba
   de paridad con JavaScript). `make entorno` instala PyTorch y MediaPipe, que pesan bastante;
   solo se usan para comprobar gradientes y extraer puntos, la red no los usa.

   ```bash
   make entorno
   make pruebas   # deben pasar todas
   ```

4. Los datos no van en git (el repo es público y pesan mucho). Pídele a Alexis por Drive:
   - `landmarks_d1.csv` (367 MB, solo los puntos de la mano, sin fotos) → ponlo en `datos/`.
   - `pesos.json` (1 MB, la red preliminar) y `hand_landmarker.task` (el modelo de MediaPipe)
     → ponlos en `modelos/`.
5. Lee, en este orden: `README.md`, `docs/DATOS.md`, `docs/NOTAS_ESTUDIO.md` y este archivo.
6. Tu primer commit: agrega tu login a `equipo` en `proyecto.toml` y escribe tus respuestas a
   las preguntas de `NOTAS_ESTUDIO.md` (hitos 1 a 3). Son lo que nos pueden preguntar en el examen.

Forma de trabajar: una rama por tarea (`git checkout -b nadia/demo-camara`), commits pequeños en
español, pull request a `main` del fork, y lo revisa alguien distinto de quien lo hizo. Solo se
toca `estudiantes/lsm_manos/`. Nunca se suben `datos/`, `modelos/`, fotos ni caras.

## Áreas

| área | responsable | suplente | piezas |
|---|---|---|---|
| A · Red y experimentos | por definir | por definir | `src/redneuronal/`, `src/lsm/entrenar.py`, experimentos E1–E5 y E7 |
| B · Datos y evaluación | por definir | por definir | `src/lsm/landmarks.py`, `features.py`, `datos.py`, `evaluar.py`, E6 |
| C · Demo y entrega | por definir | por definir | `web/` (cámara, pantallas, activaciones), Docker, hosting, guion de la presentación |

Rotación obligatoria: cada quien toca al menos una pieza fuera de su área.

## Interfaces entre piezas

- **B → A:** `datos/landmarks_d1.csv`, una fila por foto: `ruta, grupo, particion, letra,
  participante, codigo, n, detectada, lateralidad, confianza, ancho, alto` y los 63 números
  `x0, y0, z0, …, x20, y20, z20`. `entrenar.cargar_landmarks` lo lee y
  `features.vector_entrada` lo convierte en `X` (60 columnas).
- **A → C:** `modelos/pesos.json` con `anchos`, `activaciones`, `letras`, `W`, `b` (float32) y
  `features` (las opciones con que se entrenó). Se genera con
  `PYTHONPATH=src .venv/bin/python -m lsm.exportar pesos modelos/pesos.json`.
- **A y B → C:** `web/fixtures/paridad.json`: entradas y salidas de Python para que `features.js`
  y `red.js` den lo mismo (lo comprueba `make pruebas`).

## Qué ya está (preliminar) y qué falta

Hecho: la red en NumPy con sus pruebas (gradientes contra diferencias finitas y PyTorch), la
extracción de puntos de D1, las features en Python y JavaScript, y los experimentos E1 a E3
(resultados en `resultados/` y resumen en `NOTAS_ESTUDIO.md`).

Falta, para repartir:

- **Demo (C):** página con la cámara, MediaPipe en el navegador, `features.js` + `red.js` con
  `pesos.json`, las 3 letras más probables y la visualización de activaciones.
- **Análisis (A o B):** matriz de confusión de la red; qué letras se confunden y por qué.
- **Experimentos:** E4 (lasso en W(1)), E5 (curva de datos), E6 (otro dataset y clips propios),
  E7 (calibración y «no sé»).
- **Entrega (C):** Docker, hosting con HTTPS, `BITACORA.md` al día y el guion de la presentación.

Revisar cada viernes cuántos commits y filas de bitácora tiene cada quien; si el trabajo se
concentra en una persona, se reparte de nuevo. Hoy casi todo lo ha hecho Alexis: lo que sigue
debería ser sobre todo de Nadia y Bernardo.
