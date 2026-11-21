# Notas de estudio

Cada hito agrega una sección: qué hicimos, por qué, y tres preguntas que cada quien debe
poder contestar sin ver el código. **Otra persona distinta del autor** corre el hito y contesta.

---

## Hito 1 · El núcleo de la red (2026-11-20)

### Qué hicimos

- `src/redneuronal/red.py`: la red como listas `W[l]`, `b[l]`, `g[l]` (la posición 0 vacía porque la
  capa 0 es la entrada). Pase hacia adelante por lotes: `Z(l) = W(l) A(l−1) + b(l)`, `A(l) = g(l)(Z(l))`,
  con una **columna por observación** `k`.
- Retropropagación: `δ(L)` en la salida y `δ(l) = (W(l+1)ᵀ δ(l+1)) ⊙ g(l)'(z(l))` hacia atrás;
  `∇W(l) = δ(l) A(l−1)ᵀ / m` (error del destino por activación del origen, promediado en el lote).
- `perdida.py`: pérdida cuadrática del curso (`Σ(ŷ−y)²`, sin 1/2), entropía cruzada (extensión) y
  castigos ridge (`2μW`) y lasso (`μ sign(W)`), que pueden tocar solo algunas capas (para Q1: solo `W(1)`).
- `optimizador.py`: paso de descenso y entrenamiento por minibatch barajado.
- `gradcheck.py`: diferencias finitas centrales con `h = 1e-6` y error relativo por bloque.

### Pruebas (`make pruebas`: 24 pasan, 1 se salta)

- El hilo 2–3–2–1 reproduce todas las cifras de la página: `ŷ = 0.546`, `ℓ = 0.206`, `Ω = 2.24`,
  `δ(3) = −0.908`, `δ(1) = (−0.163, 0, −0.363)`, un paso con `η = 0.1` da `ŷ = 0.879` y `w(3)_11 = 0.744`
  (0.742 con `μ = 0.01`).
- Diferencias finitas en el hilo: peor bloque `8.9e-11` (el curso dice menor que `1e-10`). Con lasso
  queda en `1.0e-10`, por redondeo; ahí la prueba usa `1e-9`.
- Sin el factor 2 en `δ(L)`, todos los bloques dan exactamente 1/3 (ejercicio 10.1.4.5).
- Red del tamaño real `60–128–64–21` con softmax: error menor que `1e-6`; tiene 17,429 parámetros.
- Sin activaciones, la red se colapsa en `W′x + b′` (observación 10.1.2.3).
- La prueba contra PyTorch está escrita pero se salta: torch no está instalado (falta decidir si se instala).

### Por qué softmax con entropía cruzada da `δ(L) = ŷ − y` (extensión)

Con `ŷ_i = e^{z_i} / Σ_j e^{z_j}` y `ℓ = −Σ_i y_i log ŷ_i`, con `y` one-hot (`Σ_i y_i = 1`):

1. `log ŷ_i = z_i − log Σ_j e^{z_j}`, así que `ℓ = −Σ_i y_i z_i + log Σ_j e^{z_j}`.
2. `∂ℓ/∂z_m = −y_m + e^{z_m} / Σ_j e^{z_j} = ŷ_m − y_m`.

Entonces `δ(L) = ∇_{z(L)} ℓ = ŷ − y`. No se usa la fórmula `∇_a ℓ ⊙ g'(z)` del curso porque softmax
no actúa componente a componente (cada `ŷ_i` depende de todas las `z`). La prueba de diferencias
finitas sobre la red grande confirma la derivación.

### Tres preguntas (contestar sin ver el código)

1. En `W(2)` de la red 2–3–2–1, ¿qué significan las filas y las columnas, y por qué la fila 2 de
   `∇W(1) ℓ` es cero en el hilo?
2. Si olvidas el 2 en `δ(L) = 2(ŷ − y)`, ¿por qué la comprobación por diferencias finitas da 1/3 en
   **todos** los bloques, y no solo en la última capa?
3. Con `η = 0.1` y `μ = 0.01`, ¿cuánto acercan a cero un peso de 0.7 el ridge y el lasso en un paso?
   ¿Por qué el lasso deja pesos exactamente en cero y el ridge no? (Esto es lo que usaremos en Q1.)

---

## Hito 2 · Datos, features y primer experimento (2026-11-20)

### Qué hicimos

- Descargamos D1 (4.8 GB, md5 verificado) y extrajimos los 21 puntos de cada foto con MediaPipe
  (`landmarks.py`): 279,716 fotos, mano detectada en el 98.7 %.
- `features.py`: corrige la proporción de la imagen (x y z vienen en la escala del ancho), espeja las
  manos izquierdas, centra en la muñeca, escala por la palma y, opcionalmente, rota. `features.js` hace
  lo mismo en el navegador y una prueba lo compara con Python.
- `evaluar.py`: F1 macro (métrica principal), MCC y exactitud, probadas contra scikit-learn.
- E1 (`resultados/e1_baselines.json`): 6 pliegues agrupados por participante, solo con la partición
  oficial de entrenamiento. La prueba oficial (participantes 18 y 20) sigue cerrada.

### Resultados de E1 (media ± desviación en 6 pliegues)

| modelo | F1 macro |
|---|---|
| clase mayoritaria | 0.004 ± 0.000 |
| regresión logística (red sin capas ocultas) | 0.870 ± 0.019 |
| kNN (k = 5) | 0.887 ± 0.023 |
| SVM RBF (15 mil fotos) | 0.931 ± 0.016 |
| red 60–128–64–21 (sin ajustar) | 0.932 ± 0.021 |

La red y el SVM empatan: la diferencia es mucho menor que la variación entre pliegues. Las capas
ocultas sí ayudan: la red supera por unos 6 puntos a la misma red sin capas ocultas. La brecha
entrenamiento–validación de la red es de unos 3 puntos (0.965 contra 0.932), y el peor pliegue es el
de los participantes 6, 8 y 17 (0.890): cambiar de persona es lo difícil.

### Tres preguntas (contestar sin ver el código)

1. ¿Por qué separamos por participante y no al azar por foto? ¿Qué número saldría inflado si no lo hiciéramos?
2. La regresión logística es «la misma red con L = 1». ¿Qué le falta para llegar a 0.93, y qué dice eso
   sobre si las letras se separan con una frontera lineal en los 60 números?
3. ¿Por qué la clase mayoritaria da F1 macro 0.004 pero exactitud 0.049? ¿Qué ventaja tiene el F1 macro
   como métrica principal?
