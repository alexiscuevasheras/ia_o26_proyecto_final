# Datos: qué verificamos y cómo

Verificado el 2026-11-20 contra la API de Zenodo, PyPI, npm y las páginas oficiales.
Lo que no se pudo confirmar dice «sin verificar».

## D1. Alfabeto estático LSM (21 letras)

| campo | valor |
|---|---|
| Registro | https://zenodo.org/records/10067509 (el DOI 10.5281/zenodo.10067508 es el concepto y resuelve a esta misma versión; es la única) |
| Título y autoría en Zenodo | «Mexican Sign Language Alphabet (static signs only)», creador: Ricardo Morfín |
| Licencia | CC BY 4.0 (hay que atribuir) |
| Archivo | `MSL-ABC.7z`, **4,810.9 MB**, md5 `d263efa295467f90a8a8ae93bb643f26` |
| Contenido | 279,716 JPG (~18 KB cada uno, ~5.0 GB descomprimido) |
| Clases | A B C D E F G H I L M N O P R S T U V W Y |

Listamos el contenido **sin descargar el archivo**: Zenodo acepta peticiones por rangos, así que
bajamos solo el encabezado (32 bytes) y el final (16 MB) donde 7z guarda el índice, y lo leímos con
`bsdtar -tvf`.

Estructura real (difiere un poco de la descripción: la carpeta es `lsm-abc-A`, no `lss-abc-A`):

```
MSL-ABC/<grupo>/<train|test>/<Letra>/S<participante>-<Letra>-<código>-<n>.jpg
```

- **Sí hay id de participante** en el nombre: `S1` a `S20`. Eso permite validación agrupada por participante.
- Prueba oficial: participantes **18 y 20** en los tres grupos. Entrenamiento: los otros 18.
- `<código>`: grupo A usa `4`, grupo C usa `5`, el grupo B usa `1`, `2` y `3` (probablemente un eje de rotación cada uno; sin verificar).
- Imágenes por grupo: A 83,264 train / 9,439 test; B 84,019 / 8,841; C 84,814 / 9,339. ~4,600 imágenes por participante en el grupo A.

Descarga verificada el 2026-11-20: el md5 coincide con el publicado. Todas las fotos son de 360 × 360.
En `lsm-abc-A/test` (9,439 fotos), MediaPipe detectó la mano en el 99.8 % (la letra D es la peor, 97.1 %)
y marcó todas como «Left».

Zenodo limita cada descarga a unos 0.6 MB/s (4.8 GB tardan más de 2 horas). Si la computadora se
duerme, se cortan las conexiones: conviene correrla con `caffeinate` y conectada a la corriente.

**Problema práctico:** es un solo 7z de 4.8 GB, así que no se puede bajar «un subconjunto» directamente.
Eso afecta el plan de Docker (sección 11 del plan); ver DECISIONES.md.

## D2. Alfabeto dinámico LSM (J, K, Ñ, Q, X, Z)

| campo | valor |
|---|---|
| Registro | https://zenodo.org/records/19154645 (v2; la v1 14689869 pesa 3.6 GB) |
| Autoría | Jesús Antonio Navarrete-López, Irvin Hussein López-Nava |
| Licencia | CC BY 4.0 |
| Archivos | `MSL-dynamic-signs-frontal.7z` 291.9 MB (md5 `f21cf359fe6297debdc686feff4650cc`), `MSL-dynamic-signs-perfil.7z` 289.4 MB (md5 `3a4a2152eaa3cf3b790143dae8247193`) |
| Nombres | `S<IdSujeto>-<Letra>-<Vista>-<Repetición>`; `train` = primeros 18 participantes, `test` = 2 restantes |
| Complementa | artículo DOI 10.1007/978-3-031-96255-4_11 (título y congreso sin verificar: Springer pide login) |

## D3. Dataset externo (solo prueba cruzada)

Mendeley Data, «Mexican Sign Language's Dactylology and Ten First Numbers», v1 (2023-05-30):
personas 1–5 (DOI 10.17632/5s4mt7xrd9.1) y 6–11 (DOI 10.17632/67htnzmwbb.1). Autoría: Mario Rodriguez,
Outmane Oubram (UAEM), Bassam Ali (UADY), Noureddine Lakouari (INAOE).

- Licencia: **CC BY-NC 3.0** (no comercial). Sirve para un proyecto escolar, pero hay que atribuir y
  no redistribuir los datos (de todas formas no se suben).
- Tamaño total y cómo identifican persona y letra en los nombres: sin verificar. La API da un enlace
  de descarga por archivo, sin login aparente.

## P1. Palabras: UAEMex «Mexican sign language dataset»

- Cita: Espejel, J.; Domínguez, L. Y.; Cervantes, J.; Cervantes, J. (2023). *Mexican sign language
  dataset*. Mendeley Data, V1. doi:10.17632/6rj76z6y3n.1 (Universidad Autónoma del Estado de México).
- Licencia: CC BY 4.0, verificada en la API pública de Mendeley (`data_licence`).
- Contenido verificado con la API: 31,417 archivos y 2.03 GB, de los cuales 31,415 son JPG, más
  `classes.xlsx` y un `.mov`. El zip en caché pesa 1,885,633,282 bytes.
- Estructura: `MSLwords1/<clase>/<persona><clase>/<cuadro>.jpg`. Son 249 clases y 11 personas
  (01–11); hay 2,447 carpetas de persona y clase, no 2,739, porque faltan algunas. Las personas 10
  y 11 tienen menos señas (163 y 86 carpetas).
- Unos 12 cuadros por seña. Fondo y ropa negros. **Las caras vienen pixeladas en el dataset.**
- `classes.xlsx` da el nombre en inglés de cada clase (número = carpeta). **Supuesto sin
  confirmar:** que el número del xlsx corresponde a la carpeta; se revisará con señas obvias
  (YO apunta al pecho, TÚ al frente).
- Detección: con el umbral por omisión (0.5), PoseLandmarker encuentra el cuerpo en 47 % de los
  cuadros y con 0.2 en 74 % (150 cuadros de prueba). El extractor usa 0.2 y toma la mediana de
  los hombros de cada seña.
- **Prueba final:** las personas 10 y 11, fijadas por regla (las dos con número más alto) antes
  de ver resultados.

## P2. Palabras: MX-ITESO-100

- Cita: Martínez Sánchez, V. H. (2026). *ITESO-MX-100*. Mendeley Data, V1. doi:10.17632/g45jkzys4f.1.
  Artículo: «Exploring a Novel Mexican Sign Language Lexicon Video Dataset», *Multimodal
  Technologies and Interaction* 7(8):83 (2023), doi:10.3390/mti7080083.
- Licencia: CC BY 4.0, verificada en la API.
- El artículo describe 100 señas, 5,000 videos y 3 personas. **Lo publicado en Mendeley son 49
  carpetas (de «Abrir» a «Él»):** 2,159 videos (2,107 MP4 y 52 MOV) que pesan 10.73 GB.
- No dice quién es quién. Teníamos la hipótesis de que el sufijo del archivo (`12`, `12_1`,
  `12_2`) era la persona. **Es falsa:** 1,854 de los 2,159 videos no tienen sufijo,
  y van numerados del 1 al ~52 por palabra. Como no sabemos quién es quién, ITESO se usa solo
  para entrenar, para el avatar y para P3; no se puede evaluar por persona con él.
- Bajamos los 2,159 videos (36 fallaron la primera vez y bajaron al reintentar). Cada uno tiene 75 cuadros a 15 fps
  (5 s), y la seña ocupa poco más de un segundo. MediaPipe ve el cuerpo en el 100 % de los
  cuadros y las dos manos en el 48 %, mucho mejor que en UAEMex.
- Videos de teléfono en condiciones no controladas. El extractor los baja a 15 cuadros por
  segundo y a 960 px de lado mayor.

## Vocabulario

`src/lsm/palabras/vocabulario.csv` asigna a cada clase una glosa en español y sinónimos (para la
voz). Lo tradujimos nosotros desde los nombres en inglés de UAEMex y los nombres en español de
ITESO, así que puede tener errores. Las glosas que aparecen en los dos datasets se unen; la
misma glosa puede tener variantes regionales distintas.

## Herramientas

- MediaPipe Python `mediapipe==1.2.3` (Apache 2.0, publicado el 2026-11-11), con ruedas `py3` para
  macOS arm64, Linux x86_64/aarch64 y Windows.
- Web: `@mediapipe/tasks-vision@1.2.3` (Apache 2.0).
- Modelo `hand_landmarker.task`: 7.8 MB (más de 5 MB, **nunca va al repo**). La página oficial lo
  enlaza en https://huggingface.co/litert-community/MediaPipe-HandLandmarker-Full ; también responde
  `storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task`.
  Licencia del modelo: sin verificar (falta leer la tarjeta del modelo).
- Modelo `pose_landmarker_full.task`: 9,398,198 bytes, bajado de
  `storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task`.
  También va en `modelos/` y nunca al repo.
- Lateralidad: según la documentación, izquierda y derecha son las del sujeto, y una imagen reflejada
  invierte la etiqueta. Hay que probarlo con la cámara en modo espejo.

## Trabajo previo (pendiente de leer)

- Gortarez-Pelayo, Navarrete-López, López-Nava, «Enhancing Keypoint Selection for Hand Model in
  Recognition of Mexican Sign Language Alphabet», *Computación y Sistemas*: **no lo encontramos todavía**
  (ni URL ni DOI); no citaremos su F1 hasta leerlo.
- daktilos.cicese.mx: el certificado TLS del sitio falló al consultarlo; sin verificar.
