# Evaluación con personas (protocolo)

Los experimentos P1–P5 miden el sistema contra datasets. Falta lo que más importa: si sirve con
personas reales, con cámaras reales y con señantes de verdad. Este protocolo lo define **antes**
de correrlo, para que los números no se ajusten después.

## Participantes

- **Señantes de LSM** (idealmente personas sordas o intérpretes certificados): al menos 3.
  Prueban señas → texto y califican el avatar.
- **Oyentes sin LSM:** al menos 3. Prueban voz → señas y el flujo de conversación.
- Consentimiento informado por escrito. **No se graba video:** el sistema solo guarda los puntos
  y los eventos de la página. Nada personal sale de la computadora.

## Tarea A: señas → texto (persona nueva)

1. Se usa una lista fija de 30 glosas del vocabulario, elegidas al azar con semilla 0 antes de
   la sesión, y 5 frases cortas armadas con ellas (por ejemplo YO COMER ESCUELA).
2. La persona seña cada glosa una vez **sin adaptación**. Se registra:
   - top-1 y top-3;
   - si la red dijo «no sé»;
   - la latencia, del fin de la seña a la palabra en pantalla.
3. La persona corrige los errores con los botones (eso adapta la red) y repite las 30 glosas.
   Se registra lo mismo.
4. Las 5 frases: se mide la tasa de error por palabra entre las glosas reconocidas y las
   intentadas.

**Métrica principal:** top-1 antes y después de adaptar, por persona. Con 30 glosas se reporta
el intervalo de confianza de Wilson al 95 %.

## Tarea B: voz → señas (inteligibilidad del avatar)

1. Se eligen 20 glosas con animación y 5 palabras deletreadas, en orden aleatorio.
2. El avatar seña cada una sin texto en pantalla. El señante escribe qué entendió y califica la
   naturalidad de 1 a 5.
3. **Métricas:**
   - fracción entendida correctamente, separando glosas grabadas de deletreo;
   - naturalidad media;
   - comentarios sobre variantes regionales (la seña del dataset puede no ser la que usa la
     persona).

## Tarea C: conversación

Un señante y un oyente intentan 3 intercambios guiados (saludo, preguntar por la familia,
despedida) usando solo el sistema. Se anota cuántos intercambios se completan y dónde falla.

## Qué reportamos

Todo, aunque salga mal. Los resultados van a `resultados/humana_*.json`, con los registros de la
página exportados en JSON, sin video. Los comentarios cualitativos de los señantes van en el
README tal como los dijeron.

## Pendiente

- Contactar a la comunidad sorda o a intérpretes. Nos lo pueden orientar el profesor, el área de
  inclusión del ITAM o una asociación de personas sordas.
- Agregar a la página un botón «exportar sesión» que guarde los eventos (sin video).
