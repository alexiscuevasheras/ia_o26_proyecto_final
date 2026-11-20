# Manos que hablan: alfabeto de la LSM con una red hecha desde cero

Reconocimiento en tiempo real, en el navegador, del alfabeto manual (dactilología) de la
Lengua de Señas Mexicana con una red neuronal escrita desde cero en NumPy, con la notación
de la unidad 10 del curso. **El alfabeto manual no es la LSM completa**: este sistema no
traduce la LSM.

> Estado: en construcción. Por ahora existe el núcleo de la red (`src/redneuronal/`) con sus
> pruebas. La demo, los datos y los experimentos vienen en los siguientes hitos.

## Cómo correrlo

Pruebas del núcleo de la red (Python 3.12 o más nuevo):

```bash
make entorno     # crea .venv e instala numpy y pytest
make pruebas     # reproduce el hilo 2–3–2–1 del curso y compara gradientes
```

La demo con Docker (`docker compose up --build` y abrir <http://localhost:8000>) llega en
la semana 2; por ahora el contenedor sirve la página de ejemplo de la plantilla.

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
