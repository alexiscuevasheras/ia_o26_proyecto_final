# Manos que hablan: alfabeto de la LSM con una red hecha desde cero

Reconocimiento en tiempo real, en el navegador, del alfabeto manual (dactilología) de la
Lengua de Señas Mexicana con una red neuronal escrita desde cero en NumPy, con la notación
de la unidad 10 del curso. **El alfabeto manual no es la LSM completa**: este sistema no
traduce la LSM.

La red pierde exactitud con personas que no vio al entrenar: ese es el problema que ataca el
proyecto. Tres ideas, todas con la maquinaria de la unidad 10 y medidas con validación por
participante:

1. **Se adapta a quien la usa:** con unos cuantos ejemplos de cada letra, la página reentrena la
   última capa con descenso de gradiente, en el navegador (experimento E8).
2. **Sabe decir «no sé»:** probabilidades calibradas y un umbral elegido con validación (E7).
3. **Explica su decisión:** ∂ŷ/∂x por retropropagación hasta la entrada, para ver qué puntos de
   la mano pesaron más.

> Estado: preliminar, para presentar la propuesta. Ya existen la red con sus pruebas, la
> extracción de puntos, los experimentos E1–E3 y una primera demo con la cámara.

## Cómo correrlo

Pruebas del núcleo de la red (Python 3.12 o más nuevo):

```bash
make entorno     # crea .venv con numpy, pytest y las herramientas de comprobación
make pruebas     # reproduce el hilo 2–3–2–1 del curso, compara gradientes y la paridad Python–JS
```

Demo local (preliminar), con `modelos/pesos.json` y `modelos/hand_landmarker.task` en su lugar:

```bash
make demo        # y abrir http://localhost:8000/web/ en Chrome; pide permiso para la cámara
```

La red corre en el navegador con `web/red.js`; el detector de manos (MediaPipe Tasks Vision
1.2.3) se baja de jsDelivr. La versión con Docker todavía sirve la página de la plantilla.

## Datos y modelos

Ver [`docs/DATOS.md`](docs/DATOS.md): fuentes, licencias, tamaños y estructura verificados.
Los datos y los modelos no se suben al repo (`datos/` y `modelos/` están ignorados).

## Cómo funciona

- `src/redneuronal/`: la red `d_0–d_1–…–d_L` por lotes (observaciones en columnas),
  ReLU/sigmoide/identidad, pérdida cuadrática del curso, retropropagación, descenso por
  minibatch con castigo ridge o lasso, y comprobación por diferencias finitas.
- **Extensión fuera de las páginas de la unidad 10:** salida softmax con entropía cruzada,
  donde `δ(L) = ŷ − y` (derivación en `docs/NOTAS_ESTUDIO.md`).

## Créditos

Ver `docs/DATOS.md` para las citas de los datasets y de MediaPipe.
