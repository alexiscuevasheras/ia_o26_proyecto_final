# Formatos entre Python y la página

Python genera estos archivos y la página web los lee. Los que van en `modelos/` no se suben a git: se generan con los comandos del README.

## `modelos/animaciones.json`: de las glosas a las manos del avatar

Python (`lsm.palabras.animaciones`) elige, para cada glosa, la repetición más representativa: el medoide, es decir, la grabación más parecida en promedio a las demás de esa glosa. No se promedian grabaciones porque el promedio emborrona la seña.

```json
{
  "fps": 15,
  "cuerpo_indices": [0, 2, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 23, 24],
  "glosas": {
    "COMER": {
      "origen": "uaemex-03/190",
      "cuadros": [
        {
          "cuerpo": [[x, y, z], ...],
          "manos": [ [[x, y, z] × 21] o null,  [[x, y, z] × 21] o null ]
        }
      ]
    }
  },
  "letras": {
    "A": { "mano": [[x, y, z] × 21] }
  },
  "letras_con_movimiento": ["J", "K", "Ñ", "Q", "X", "Z"]
}
```

**Ejes y unidades:** metros, en el sistema del avatar, no en el de la imagen.
- Origen en el punto medio de los hombros.
- **x** apunta a la izquierda de la persona, que es la derecha de quien la mira de frente. **y** apunta hacia arriba. **z** apunta hacia la cámara, hacia quien mira.
- Las manos se guardan ya escaladas y colocadas: el punto 0 de cada mano (la muñeca) coincide con la muñeca del brazo correspondiente (índices 15 y 16 del cuerpo).

**`cuerpo`:** los puntos de PoseLandmarker que indica `cuerpo_indices`, en ese orden.
- 0 nariz; 2 y 5 ojos; 7 y 8 orejas; 9 y 10 boca.
- 11 y 12 hombros; 13 y 14 codos; 15 y 16 muñecas; 23 y 24 caderas.
- En MediaPipe, los índices impares son el lado izquierdo de la persona.

**`manos`:**
- `manos[0]` es la mano del brazo izquierdo y `manos[1]` la del derecho. Vale `null` en los cuadros donde no se vio.
- Los 21 puntos siguen el orden de MediaPipe Hands.

**`letras`:** forma estática de la mano dominante para el deletreo.
- Se calcula con el medoide de D1, centrado en la muñeca, con la palma (‖p9 − p0‖) escalada a 0.09 m y los mismos ejes de arriba, como mano derecha.
- El avatar la coloca en la muñeca derecha, frente al pecho.
- Las letras de `letras_con_movimiento` no están en D1: la página las hace con la forma base más un trazo programado, y lo indica en pantalla.

## `modelos/palabras.json`: la red que reconoce palabras

```json
{
  "tipo": "gru",
  "T": 16, "D": 126,
  "glosas": ["ABRAZAR", ...],
  "params": {"W_z": [[...]], "U_z": [[...]], "b_z": [...], ..., "V": [[...]], "c": [...]},
  "estandar": {"media": [...D], "desv": [...D]},
  "temperatura": 1.7,
  "umbral": 0.35,
  "features": {"version": 1}
}
```

`estandar` normaliza cada una de las D columnas con la media y la desviación del entrenamiento. `temperatura` (τ) y `umbral` salen de `resultados/p5_no_se.json`: ŷ = softmax(z/τ), y la página dice «no sé» si max ŷ < umbral. Los números del ejemplo son ilustrativos.

La entrada es la de `lsm/palabras/secuencias.py`. `web/secuencias.js` la reproduce y una prueba de paridad compara las dos.

## `web/vocabulario.json`: del texto a las glosas

Se genera de `src/lsm/palabras/vocabulario.csv`:

```json
{ "glosas": { "COMER": { "tema": "verbos", "sinonimos": ["comer", "como", "come"] } } }
```
