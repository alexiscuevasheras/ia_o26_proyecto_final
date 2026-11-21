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

---

## Hito 3 · Ajuste, castigo y ablaciones (2026-11-21)

### Qué hicimos

- `experimentos/comun.py`: E1, E2 y E3 usan exactamente los mismos 6 pliegues y las mismas muestras.
- E2 (`resultados/e2_ajuste.json`): rejilla de 27 configuraciones (anchos 64 / 128–64 / 256–128,
  η ∈ {0.02, 0.05, 0.1}, μ ∈ {0, 1e-4, 1e-3}), 40 épocas, con curvas medias por época.
- E3 (`resultados/e3_ablaciones.json`): con la configuración de E2, cuatro variantes del vector de entrada.
- `python -m lsm.exportar pesos modelos/pesos.json`: red preliminar entrenada con toda la partición
  oficial de entrenamiento (1 MB; `modelos/` no se sube) para probar la demo.

### E2: lo que salió

- Mejor configuración: 60–256–128–21, η = 0.1, μ = 0, época 39: F1 macro 0.942 ± 0.022. Ese número
  está **sesgado hacia arriba** (se eligió mirando estos mismos pliegues); la estimación honesta será la
  prueba oficial, que se abre una vez al final.
- Las mejores configuraciones (0.938–0.942) difieren mucho menos que la variación entre pliegues:
  es una meseta, no un ganador claro. Contra la red sin ajustar de E1 (0.932) tampoco hay una mejora
  que podamos afirmar.
- Con η = 0.1 fijo, las curvas oscilan de época a época; la «mejor época» tiene algo de suerte.
- **Castigo (Q4).** En la 60–256–128–21 con η = 0.1, a la época 40:

  | μ | pérdida entrenamiento | pérdida validación | brecha | F1 validación |
  |---|---|---|---|---|
  | 0 | 0.038 | 0.224 | 0.186 | 0.940 |
  | 1e-4 | 0.068 | 0.210 | 0.142 | 0.934 |
  | 1e-3 | 0.208 | 0.283 | 0.075 | 0.909 |

  El ridge sí achica la brecha, pero con 1e-3 la red ya no ajusta bien ni el entrenamiento y el F1 baja.
  Con estos datos, lo que separa entrenamiento de validación es el cambio de persona, no el sobreajuste
  que el ridge corrige.

### E3: ablaciones (media ± desviación, 6 pliegues)

| entrada | F1 macro | pliegues en que pierde contra la base |
|---|---|---|
| x, y, z sin rotar (60), la base | 0.942 ± 0.022 | — |
| x, y, z rotado (60) | 0.931 ± 0.035 | 5 de 6 |
| x, y sin rotar (40) | 0.941 ± 0.018 | 4 de 6, por muy poco |
| x, y rotado (40) | 0.930 ± 0.033 | 5 de 6 |

Rotar la palma hacia arriba empeora casi siempre, y en el pliegue difícil (participantes 6, 8 y 17)
cae 0.046: la orientación de la mano lleva información. z no aporta ni estorba de forma clara.
Nos quedamos con x, y, z sin rotar.

### Tres preguntas (contestar sin ver el código)

1. ¿Por qué el F1 de la configuración elegida en E2 no sirve como estimación final, y qué número sí?
2. Con μ = 1e-3 la brecha baja a menos de la mitad, pero el F1 empeora. ¿Cómo se explica con
   «sesgo contra varianza»?
3. ¿Qué letras de la LSM esperarías que se confundan si se quita la orientación al rotar? Comprobarlo
   en la matriz de confusión es una tarea pendiente.
