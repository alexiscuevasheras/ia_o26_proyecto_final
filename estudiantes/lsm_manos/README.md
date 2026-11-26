# Manos que hablan: traductor LSM ⇄ español con redes hechas desde cero

Un traductor bidireccional entre la Lengua de Señas Mexicana (LSM) y el español que corre
completo en el navegador:

- **Señas → texto (y voz):** la cámara ve a la persona señar. MediaPipe saca los puntos del
  cuerpo y de las dos manos, y una **red recurrente GRU escrita desde cero en NumPy** reconoce
  palabras de un vocabulario de 285 glosas. Si la persona deletrea, la red densa de la unidad 10
  reconoce las letras. La frase sale como subtítulo y se puede leer en voz alta.
- **Voz → señas:** la persona habla (o escribe) en español, el texto se convierte en glosas y
  un **avatar 3D** las seña con movimientos tomados de grabaciones reales. Las palabras que no
  conoce las deletrea.

**Qué no es:** la LSM tiene su propia gramática (orden, uso del espacio, expresión facial).
Este sistema trabaja palabra por palabra (glosas) con un vocabulario limitado; no es una
traducción gramatical completa, y la página lo dice.

Además de reconocer, el sistema ataca el problema de fondo que medimos: **la exactitud cae con
personas que el modelo no vio.**
1. **Se adapta a quien lo usa.** Cuando la persona corrige una palabra, la página reentrena en
   el navegador la capa de salida de la GRU con descenso de gradiente (experimentos E8 y P4).
2. **Dice «no sé».** En lugar de inventar una palabra, ofrece sus 3 mejores opciones. El umbral
   se elige con validación por persona (P5).

> Estado: en desarrollo. Los números de este README salen de `resultados/*.json`. P1–P3, P5 y
> P6 están corridos; P4 espera el dataset propio.

## Cómo correrlo

Pruebas (Python 3.12+ y Node 20+):

```bash
make entorno     # crea .venv con numpy, pytest y las herramientas de comprobación
make pruebas     # gradientes contra diferencias finitas, paridad Python–JavaScript, etc.
```

Datos y modelos (no se suben al repo; `datos/` y `modelos/` están ignorados). Detalle en
[`docs/DATOS.md`](docs/DATOS.md):

```bash
export PYTHONPATH=src
# 1. Palabras: UAEMex (1.9 GB, CC BY 4.0) e ITESO-MX-100 (10.7 GB, CC BY 4.0)
.venv/bin/python -m lsm.palabras.descargar uaemex datos/crudos/uaemex --zip
.venv/bin/python -m lsm.palabras.descargar iteso datos/crudos/iteso
# 2. Puntos de cada seña (solo puntos: las fotos y videos se quedan locales)
.venv/bin/python -m lsm.palabras.extraer uaemex datos/crudos/uaemex datos/secuencias_uaemex.npz
.venv/bin/python -m lsm.palabras.extraer iteso datos/crudos/iteso datos/secuencias_iteso.npz
# 3. Modelos para la página
.venv/bin/python -m lsm.palabras.exportar modelo modelos/palabras.json \
    --uaemex datos/secuencias_uaemex.npz --iteso datos/secuencias_iteso.npz
.venv/bin/python -m lsm.palabras.animaciones modelos/animaciones.json \
    --secuencias datos/secuencias_uaemex.npz datos/secuencias_iteso.npz --letras datos/landmarks_d1.csv
.venv/bin/python -m lsm.exportar pesos modelos/pesos.json   # letras (ver docs/DATOS.md para D1)
```

Los detectores de MediaPipe (`hand_landmarker.task` y `pose_landmarker_full.task`) van en
`modelos/` y se bajan del almacenamiento oficial de MediaPipe (ver `docs/DATOS.md`).

Demo local:

```bash
make demo        # abrir http://localhost:8000/web/ en Chrome (pide la cámara)
```

Con Docker (probado con Docker 29.5; la imagen pesa ~200 MB y no lleva modelos, que se montan
de `modelos/` en solo lectura):

```bash
docker compose up --build   # abrir http://localhost:8000
```

## Cómo funciona

| Pieza | Dónde | Qué hace |
|---|---|---|
| Red densa (unidad 10) | `src/redneuronal/red.py` | `z(l) = W(l) a(l−1) + b(l)`, retropropagación, ridge/lasso |
| GRU desde cero | `src/redneuronal/recurrente.py` | compuertas z, r; BPTT comprobada con diferencias finitas; Adam |
| Features de letras | `src/lsm/features.py` | mano centrada en la muñeca, escalada por la palma, espejo canónico |
| Features de señas | `src/lsm/palabras/secuencias.py` | 16 momentos × 126 números: muñecas en el marco de los hombros + forma de cada mano |
| En el navegador | `web/secuencias.js`, `web/gru.js`, `web/red.js` | lo mismo que Python; pruebas de paridad a 1e-5 |
| En vivo | `web/segmentador.js`, `web/en_vivo.js` | corta de reposo a reposo, distingue deletreo de palabras, «no sé» |
| Voz → señas | `web/texto_a_glosas.js`, `web/avatar.js` | texto → glosas → avatar three.js con cinemática inversa para deletrear |

**Extensiones fuera de las páginas de la unidad 10**, cada una derivada y probada:
- salida softmax con entropía cruzada (`δ(L) = ŷ − y`, en `docs/NOTAS_ESTUDIO.md`);
- la celda GRU con retropropagación en el tiempo;
- Adam con recorte del gradiente.

## Evaluación

Toda la evaluación separa por persona: el modelo nunca ve a la persona con la que se evalúa.
En cada dataset hay personas de prueba fijadas antes de ver resultados, que no se usan para
elegir nada.

- **Letras (D1):** E1–E3 y E8, en `resultados/e*.json`.
- **Palabras** (`resultados/p*.json`; azar con 249 glosas: 0.004):

  | Experimento | Qué mide | Resultado (top-1) |
  |---|---|---|
  | P1, UAEMex | persona nueva, 249 glosas | GRU 0.146 ± 0.048 (top-5 0.339); logística 0.114, SVM 0.105, red densa 0.105, kNN 0.069 |
  | P1, + ITESO al entrenar | persona nueva de UAEMex, 280 glosas | GRU 0.110; no ayuda |
  | P2 | qué parte importa (GRU) | sin la forma de la mano 0.052; solo la mano dominante 0.108; sin la posición 0.126; sin aumento 0.130; T=8 0.133; T=32 0.135 |
  | P3 | de un dataset al otro, 11 glosas compartidas (azar 0.091) | UAEMex → ITESO 0.344; ITESO → UAEMex 0.232 |
  | P4 | adaptación con k ejemplos | pendiente: necesita el dataset propio (ver abajo) |
  | P5 | calibración y «no sé» | ECE de 0.367 a 0.029 con temperatura 2.47; ningún umbral llega a 90 % respondiendo al menos al 20 %; AUROC conocidas/desconocidas 0.566 |
  | P5, + ITESO al entrenar | calibración del modelo de la página | ECE de 0.433 a 0.017 con temperatura 2.78; AUROC 0.491 |
  | P6 | diagnóstico con personas conocidas (división al azar) | UAEMex 0.121; ITESO 0.974; ITESO con 7 ejemplos por glosa 0.876 |
  | P7 | umbral de manos de MediaPipe 0.3 → 0.1 en UAEMex | cuadros sin mano de 26.9 % a 15.1 %; GRU de 0.107 a 0.146 (8 de 9 personas mejoran, Wilcoxon p = 0.008) |

  **Qué dicen juntos.** P6 no mide generalización: las mismas personas están en entrenamiento
  y en evaluación. Sirve para ubicar el error. En ITESO la GRU acierta casi todo, incluso con
  tan pocos ejemplos por glosa como UAEMex, así que la red y la entrada funcionan. En UAEMex
  no mejora con personas conocidas (0.121 contra 0.146 con nuevas): ahí el límite está en los
  datos, no en generalizar a personas. Dos diferencias que P6 no separa: UAEMex tiene 249 glosas
  contra 42, y cada persona hace cada seña una sola vez, mientras que en ITESO la misma persona
  la repite. P7 ataca la calidad directamente: con fondo y ropa negros MediaPipe perdía la mano
  en el 27 % de los cuadros; bajando su umbral de confianza la pierde en el 15 % y todos los
  modelos mejoran. Todos los números de la tabla salen ya de la extracción con umbral 0.1.

  **Siguiente paso:** el dataset propio, grabado con `web/grabar.html` en las mismas
  condiciones que la demo (video continuo de cámara web, ~10 repeticiones por persona y
  palabra). Con él se corren P1 con `--propios` y P4.

## Créditos

Ver `docs/DATOS.md` para las citas de los datasets (D1, UAEMex, MX-ITESO-100) y de MediaPipe.
