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

---

## Hito 4 · Adaptación a la persona (2026-11-22)

### Qué hicimos

- `adaptar.py`: reentrena solo W(L) y b(L) con pocos ejemplos de una persona; las capas ocultas
  quedan fijas. Hiperparámetros fijados antes de ver resultados: lote completo, η = 0.05, 100 pasos.
- Revisamos los códigos de grabación de D1: para una misma persona y letra, los 5 códigos tienen
  distinto número de cuadros y posición, y la distancia entre códigos es de 3 a 40 veces la de dentro
  de un código. Los tratamos como grabaciones distintas (no está confirmado por el autor).
- E8 (`resultados/e8_adaptacion.json`): para cada participante de validación, se adapta con k fotos
  por letra de **una** grabación y se evalúa con sus **otras 4**, para no evaluar con cuadros vecinos.

### Resultados de E8 (F1 macro, media ± desviación sobre los 18 participantes)

| k por letra | F1 macro | participantes que mejoran | peor participante |
|---|---|---|---|
| 0 (red base) | 0.939 ± 0.037 | — | 0.824 |
| 1 | 0.939 ± 0.032 | 6 de 18 | 0.843 |
| 2 | 0.940 ± 0.031 | 5 de 18 | 0.854 |
| 5 | 0.942 ± 0.031 | 8 de 18 | 0.857 |
| 10 | 0.943 ± 0.030 | 10 de 18 | 0.862 |
| 20 | 0.944 ± 0.030 | 10 de 18 | 0.863 |

En promedio la mejora es mínima (+0.005 con k = 20). Pero ayuda justo donde la red falla: el
participante 8 pasa de 0.824 a 0.863 y el 6 de 0.939 a 0.967; los que ya estaban arriba de 0.95
quedan igual o bajan un poco. En D1 todas las fotos vienen del mismo montaje de cámara, así que el
cambio entre personas es chico; con una webcam distinta el cambio debería ser mayor, y eso se medirá
con clips propios (E6). Probar más pasos o adaptar más capas exige elegirlos sin ver estos mismos
participantes.

### Tres preguntas (contestar sin ver el código)

1. ¿Por qué evaluar con fotos de la misma grabación que se usó para adaptar inflaría el resultado?
2. Si las capas ocultas quedan fijas, ¿por qué adaptar la última capa es un problema convexo y
   reentrenar toda la red no?
3. ¿Por qué la adaptación puede empeorar a alguien que ya estaba en 0.97?

## Hito 5 · De letras a palabras: la GRU (2026-11-24)

### Por qué una red recurrente

Una letra estática es una foto: un vector x de 60 números alcanza. Una palabra es un
movimiento: una secuencia x_1, …, x_T con T = 16 momentos de 126 números cada uno. Hay dos
formas de pasársela a una red:
- **Aplanarla** (16 · 126 = 2016 entradas) y usar la red densa de la unidad 10. Es la línea
  base `mlp` de P1.
- **Una celda recurrente** que lee un momento a la vez y lleva un estado h_t que resume lo visto
  hasta t. Comparte los mismos pesos en todos los momentos, así que tiene muchos menos
  parámetros que la red aplanada.

### La celda (src/redneuronal/recurrente.py)

    z_t = σ(W_z x_t + U_z h_{t−1} + b_z)            ¿cuánto actualizo? (0 = conservo h_{t−1})
    r_t = σ(W_r x_t + U_r h_{t−1} + b_r)            ¿cuánto del pasado uso para el candidato?
    h̃_t = tanh(W_h x_t + U_h (r_t ⊙ h_{t−1}) + b_h)
    h_t = (1 − z_t) ⊙ h_{t−1} + z_t ⊙ h̃_t
    ŷ  = softmax(V [h_T; h̄] + c),   h̄ = (1/T) Σ h_t

### Retropropagación en el tiempo (BPTT)

Es la misma regla de la cadena de la unidad 10. Se «desenrolla» la red en T capas que comparten
pesos y se suman los gradientes de todas.
1. **Salida:** δ = ŷ − y (softmax con entropía cruzada, como en las letras). Entonces
   ∇V = δ [h_T; h̄]ᵀ y ∇c = δ.
2. **Lo que llega a los estados:** a h_T le llega Vᵀδ por la parte de h_T. A cada h_t le llega
   además (1/T) de la parte de h̄.
3. **Hacia atrás**, con dh = ∂L/∂h_t, para t = T, …, 1:
   - dz = dh ⊙ (h̃_t − h_{t−1})
   - dh̃ = dh ⊙ z_t
   - da_h = dh̃ ⊙ (1 − h̃_t²), porque tanh' = 1 − tanh²
   - dr = (U_hᵀ da_h) ⊙ h_{t−1}
   - da_z = dz ⊙ z_t(1 − z_t) y da_r = dr ⊙ r_t(1 − r_t), porque σ' = σ(1 − σ)
   - ∇W_g += da_g x_tᵀ, ∇U_g += da_g (entrada de U_g)ᵀ, ∇b_g += da_g, con g ∈ {z, r, h}
   - ∂L/∂h_{t−1} = dh ⊙ (1 − z_t) + (U_hᵀ da_h) ⊙ r_t + U_zᵀ da_z + U_rᵀ da_r, más su parte de h̄
4. **Comprobación:** `tests/test_recurrente.py` compara cada parámetro contra diferencias finitas
   (error < 1e-6).

### Por qué Adam y recorte

Con SGD simple la GRU aprende muy lento. Adam escala cada paso por la magnitud reciente del
gradiente de cada peso. El recorte (norma ≤ 5) evita que un gradiente que «explota» en la BPTT
arruine los pesos de un solo paso.

### Tres preguntas (contestar sin ver el código)

1. Si z_t = 0 en todos los momentos, ¿qué vale h_T? ¿Qué aprende entonces la red?
2. ¿Por qué la salida usa h̄ además de h_T? Piensa en cuántos productos de matrices recorre el
   gradiente hasta llegar a x_1 en cada caso.
3. ¿Por qué adaptar solo V y c a una persona nueva es un problema convexo, y adaptar toda la GRU
   no?

## Hito 6 · Qué tan bien reconoce palabras, y por qué falla (2026-11-25)

### Qué hicimos

- P1: cinco modelos con validación dejando una persona fuera en UAEMex (249 glosas).
- P2: ablaciones de la GRU (quitar forma, posición, mano no dominante, aumento; cambiar T).
- P3: entrenar en un dataset y evaluar en el otro, en las 11 glosas compartidas.
- P5: calibración por temperatura y umbral de «no sé».
- P6: diagnóstico con división al azar (personas conocidas) para ubicar el error.
- P7: bajar el umbral de confianza de manos de MediaPipe y medir si mejora el reconocimiento.

Todos los números están en el README y en `resultados/p*.json`.

### La lección de P6: separar las causas del error

Con personas nuevas (P1, extracción original) la GRU acertaba 0.107 en UAEMex. Hay al menos tres explicaciones
posibles y cada una pide un remedio distinto:

| Si el problema fuera… | …con personas conocidas esperaríamos | Remedio |
|---|---|---|
| generalizar a personas | acierto alto | adaptar a quien usa la página (P4) |
| pocos ejemplos por glosa | acierto alto en ITESO completo, bajo con pocos | más datos |
| la calidad de los datos | acierto bajo aun con personas conocidas | otros datos o mejor extracción |

Resultado (extracción original): UAEMex 0.090 con personas conocidas (igual que con nuevas);
ITESO 0.984, y 0.876 con solo 7 ejemplos por glosa. Lo que más pesa en UAEMex es la calidad de los datos (MediaPipe no
ve mano en el 27 % de los cuadros), aunque P6 no separa del todo el número de glosas (249 contra
42) ni que en ITESO la misma persona repite cada seña.

Moraleja: antes de cambiar la red, mide dónde está el error. Un 10 % podía leerse como «la GRU es
mala»; P6 muestra que con buenos datos la misma GRU llega a 98 %.

### P7: atacar la causa que P6 señaló

Si el problema es que MediaPipe pierde la mano, el remedio está en la extracción, no en la red.
Bajamos el umbral de confianza de manos de 0.3 a 0.1, elegido viendo solo la detección (400 fotos
al azar y una revisión a ojo de las manos nuevas), no la exactitud. Cuadros sin mano: de 26.9 % a
15.1 %. La GRU sube de 0.107 a 0.146 con personas nuevas y mejoran 8 de 9 personas (Wilcoxon
p = 0.008). Con la extracción nueva se volvieron a correr P1–P6; los números del README son ésos.

Dos detalles de método que conviene saber explicar:
- La comparación es pareada: la misma persona evaluada con las dos extracciones. Con 9 personas
  la prueba de signos solo puede dar p ≥ 0.004 (9 de 9); por eso también se reporta Wilcoxon.
- Al rehacer P5 su criterio eligió un umbral que respondía al 0.7 % de las señas (14 de 15 bien).
  Un criterio de «exactitud ≥ 90 %» sin cobertura mínima premia responder casi nada; ahora exige
  responder al menos al 20 %.

### Tres preguntas (contestar sin ver el código)

1. ¿Por qué P6 no sirve como resultado de generalización, y aun así sirve para algo?
2. En P2, quitar la forma de la mano baja de 0.146 a 0.052 y quitar la posición solo a 0.126.
   ¿Qué te dice eso sobre qué información usa la GRU?
3. P5 da AUROC 0.566 entre señas conocidas y desconocidas. ¿Qué significa un AUROC de 0.5 y por
   qué la temperatura no lo mejora?
4. ¿Por qué el umbral de P7 se eligió con la tasa de detección y no con la exactitud de P1?
