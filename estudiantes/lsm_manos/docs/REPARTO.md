# Reparto del trabajo (propuesta, a ajustar entre los tres)

Cada pieza tiene una interfaz clara para que cada quien trabaje en su rama y haga sus
propios commits. Cada hito lo revisa **otra persona** distinta de quien lo hizo.

| área | responsable | suplente | piezas |
|---|---|---|---|
| A · Red y experimentos | por definir | por definir | `src/redneuronal/`, `src/lsm/entrenar.py`, experimentos E1–E5 y E7 |
| B · Datos y evaluación | por definir | por definir | `src/lsm/landmarks.py`, `features.py`, `datos.py`, `evaluar.py`, baselines, E6 |
| C · Demo y entrega | por definir | por definir | `web/` (`red.js`, `features.js`, pantallas), Docker, hosting, guion de la presentación |

Rotación obligatoria: cada quien toca al menos una pieza fuera de su área (por ejemplo,
quien hace la demo escribe un experimento, y quien hace la red escribe `features.js`).

## Interfaces entre piezas

- **B → A:** `datos/landmarks_d1.csv` con una fila por foto: `ruta, grupo, particion, letra,
  participante, es_izquierda, ancho, alto` y los 63 números `x0, y0, z0, …, x20, y20, z20`.
  `features.vector_entrada` lo convierte en `X` (60 columnas).
- **A → C:** `modelos/pesos.json` con `anchos`, `activaciones`, `letras`, `W` y `b` (float32),
  más un archivo de prueba con entradas y salidas para que `red.js` dé lo mismo que Python.
- **B → C:** el mismo archivo de prueba para `features.js` contra `features.py`.

## Primeras tareas sugeridas (semana del 23 de noviembre)

- Quien no sea Alexis: correr `make pruebas`, contestar las preguntas de `NOTAS_ESTUDIO.md` y
  agregar su login a `proyecto.toml` (su primer commit).
- B: `landmarks.py` sobre D1 (Alexis ya descargó el archivo; se comparte el CSV, no las fotos).
- C: `web/features.js` y `web/red.js` con sus pruebas en Node contra los archivos de Python.
- A: baselines y primer entrenamiento con validación agrupada por participante.

Revisar cada viernes cuántos commits y filas de bitácora tiene cada quien; si el trabajo se
concentra en una persona, se reparte de nuevo.
