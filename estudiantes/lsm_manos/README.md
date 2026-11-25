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

> Estado: en desarrollo. Los números de este README salen de `resultados/*.json`; los
> experimentos de palabras (P1–P5) se están corriendo.

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
- **Palabras:**
  - P1: modelos, dejando una persona fuera.
  - P2: ablaciones.
  - P3: de un dataset al otro.
  - P4: adaptación con k ejemplos.
  - P5: «no sé».

  Todos se guardan en `resultados/p*.json`.

## Créditos

Ver `docs/DATOS.md` para las citas de los datasets (D1, UAEMex, MX-ITESO-100) y de MediaPipe.
