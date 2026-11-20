# Decisiones

Una entrada por decisión: qué, por qué y qué alternativas se descartaron.

## 2026-11-20 · Estructura de la red

Listas `W[l]`, `b[l]`, `g[l]` con la posición 0 vacía, como el notebook del curso, para que el índice
del código sea el `l` de las páginas. Observaciones en columnas. float64 para entrenar.

## 2026-11-20 · Softmax solo con entropía cruzada

Softmax no actúa componente a componente, así que `∇_a ℓ ⊙ g'(z)` no aplica. El código exige
entropía cruzada cuando la salida es softmax y usa `δ(L) = ŷ − y`. Para la comparación «del curso»
se usa salida sigmoide (o identidad) con pérdida cuadrática sobre `y` one-hot.

## 2026-11-20 · Docker y D1 (pendiente de decidir con el equipo)

D1 es un solo `MSL-ABC.7z` de 4.8 GB. Descargarlo en cada `docker build` no es razonable (tarda,
y un build del check del PR podría fallar). Opciones a discutir:

1. Commitear un `pesos.json` pequeño (menos de 300 KB, fuera de `modelos/`): hay que preguntarle al profesor.
2. Que la imagen baje los pesos ya entrenados de un hosting externo (release de GitHub del fork u otro) al construirse.
3. Que el contenedor entrene con un archivo pequeño de **landmarks ya extraídos** (CSV de unos cientos de KB,
   derivado de datos CC BY 4.0, con atribución) en lugar de con imágenes.
