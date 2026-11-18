# Proyecto final — Inteligencia Artificial O26

Aquí se entrega el proyecto final de la clase de IA. **Cada equipo tiene una carpeta**,
`estudiantes/<tu_equipo>/`, y ahí va **todo su proyecto**: el código, el Docker, el README,
la bitácora. Se entrega con dos *pull requests*: uno para la **propuesta** y otro para la
**entrega**.

En el sitio del curso:

- [El proyecto final](https://rayalucaria.org/ia_o26/proyectos/proyecto-final/): qué puede ser,
  los requisitos y cómo se presenta (8 a 12 minutos, con reloj exacto).
- [Cómo se entrega](https://rayalucaria.org/ia_o26/proyectos/como-entregar/): este mismo paso a paso.

Este README es lo oficial para entregar. Si algo difiere, manda este README.

> [!IMPORTANT]
> - **Equipo de 1 o 2.** Si quieres otro tamaño, lo hablas con el profesor.
> - **Antes de empezar, habla con el profesor** para que autorice tu tema. Luego abre el PR de la
>   propuesta. **Cuando el profesor hace merge de tu propuesta, tu tema queda autorizado.**
> - **La entrega es el día del examen final de IA.** La hora la anuncia el profesor. A esa hora se
>   deja de hacer merge: lo que esté aceptado es lo que se presenta.
> - **Solo tocas tu carpeta.** Un PR que cambia algo fuera de `estudiantes/<tu_equipo>/` no pasa.

## Cómo funciona

```text
  habla con el profe ──► PR «Propuesta» ──► merge = tema autorizado
                                                   │
                         trabajan en su fork, con commits de los dos
                                                   │
  día del examen final ◄── merge ◄── PR «Entrega» ◄┘
```

Cada PR lo revisa un check automático (**revisar-entrega**). Si sale ❌, abre el check, lee el
resumen, arregla lo que diga y vuelve a hacer `push`: el check corre solo otra vez.

## Paso a paso

### 1. Una persona del equipo hace el fork

1. Arriba a la derecha de esta página: **Fork → Create fork**. Queda una copia en tu cuenta:
   `github.com/<tu-login>/ia_o26_proyecto_final`.
2. Si son dos: en **tu fork**, ve a **Settings → Collaborators → Add people** y agrega a tu
   compañero. Él acepta la invitación desde su correo. Así los dos pueden hacer `push` al mismo fork.

### 2. Los dos clonan el fork

```bash
git clone https://github.com/<login-del-fork>/ia_o26_proyecto_final.git
cd ia_o26_proyecto_final
```

Cada quien trabaja con **su propia cuenta de GitHub**: así sus commits cuentan como su trabajo.
Revisa que git sepa quién eres:

```bash
git config user.name      # tu nombre
git config user.email     # el correo de tu cuenta de GitHub
```

Si no sale nada o sale otro correo: `git config --global user.email "tu-correo-de-github"`.
Si el correo no es el de tu cuenta, **tus commits no se cuentan como tuyos**.

### 3. Copia la plantilla a tu carpeta

El nombre de tu carpeta es el nombre corto del proyecto: de 3 a 30 caracteres, solo minúsculas
sin acentos, dígitos y `_`. Por ejemplo, `agente_snake`.

```bash
cp -r plantilla estudiantes/<tu_equipo>
```

Y llena `estudiantes/<tu_equipo>/proyecto.toml`:

```toml
estado = "propuesta"
titulo = "Un agente que aprende a jugar Snake"
tema = "Q-learning sobre un ambiente propio; usa la unidad de Markov y la de agentes"
equipo = ["usuario-a", "usuario-b"]   # sus logins de GitHub; si vas solo, uno
demo = ""
```

### 4. Revisa antes de subir

```bash
python3 herramientas/revisar.py estudiantes/<tu_equipo>
```

Es el mismo revisor del check. Si sale ✅ aquí, el check también pasa (salvo Docker en la entrega).

### 5. Sube y abre el PR de la propuesta

```bash
git add estudiantes/<tu_equipo>
git commit -m "Propuesta: <tu_equipo>"
git push
```

En GitHub, en tu fork: **Contribute → Open pull request**. Título: `Propuesta: <tu_equipo>`.
Llena las casillas del PR y espera el check y el merge.

### 6. Trabajen

Sigan haciendo commits en su fork, cada quien con su cuenta. Lleven la `BITACORA.md` al día.
No hace falta abrir PR hasta la entrega.

### 7. El PR de la entrega

Cambia en `proyecto.toml`:

```toml
estado = "entrega"
demo = "https://…"     # o "no aplica", si así lo acordaron con el profesor
```

Corre otra vez `python3 herramientas/revisar.py estudiantes/<tu_equipo>`, prueba
`docker compose up --build` en tu carpeta, haz `commit` y `push`, y abre otro PR:
`Entrega: <tu_equipo>`.

## Tu carpeta

La plantilla ya trae todo esto. Cambia lo que haga falta, pero **no quites ninguno**:

```text
estudiantes/<tu_equipo>/
├── proyecto.toml      estado, título, tema, equipo, demo
├── README.md          qué es y la sección «## Cómo correrlo» con los comandos exactos
├── BITACORA.md        tu evidencia de trabajo (ver abajo)
├── LICENSE            MIT; cambia el año o los nombres si quieres
├── Dockerfile         y/o compose.yaml: `docker compose up --build` lo levanta
├── compose.yaml
├── .gitignore         datos/, modelos/ y .env ya están ignorados
├── .env.example       las variables que necesitas, sin valores
└── …                  tu código, como lo organices
```

El `main.py`, `requirements.txt` y el `Dockerfile` de la plantilla son un ejemplo que funciona
(sirve una página en `http://localhost:8000`). Cámbialos por los tuyos.

## Datos, modelos y llaves: se quedan en tu computadora

El repo es **público**. Por eso:

- **Datos y modelos** van en `datos/` y `modelos/` dentro de tu carpeta. Están en `.gitignore`:
  no se suben. En el README pon de dónde se bajan (Kaggle, Hugging Face, un script) y el comando
  para conseguirlos.
- **Ningún archivo de más de 5 MB**, y la carpeta completa, máximo 50 MB.
- **Llaves (OpenAI, Anthropic, Hugging Face, …)** van en `.env`, que no se sube. Sube
  `.env.example` sin valores. El check busca llaves en tus archivos; si sube una por error,
  **revócala** en el sitio que la dio, aunque luego la borres: ya quedó en el historial.
- Nada de caras ni datos personales.

Evita los datos simulados, salvo que sean necesarios.

## La bitácora

Lo que GitHub ya sabe (commits, PRs, issues) **no se copia**: se toma de GitHub. La bitácora es
para lo que no deja commits: pruebas, diseño, entrenamientos largos, un robot. Una fila por entrada:

```markdown
| fecha | login | qué hice | evidencia |
|---|---|---|---|
| 2026-11-20 | usuario-a | Entrené el agente 2 h; la recompensa se estancó en 40 | evidencia/2026-11-20-curva.png |
| 2026-11-22 | usuario-b | Probé la demo en tres navegadores | https://youtu.be/… |
```

- **fecha** como `AAAA-MM-DD`; **login** uno de `equipo`.
- **evidencia**: un link `https://…` o un archivo de tu carpeta (por ejemplo, en `evidencia/`).
  Sin evidencia, la fila no cuenta.

## Qué revisa el check

En cada PR:

- Que solo cambie **una** carpeta `estudiantes/<tu_equipo>/`, con nombre válido.
- Que quien abre el PR esté en `equipo`. Si la carpeta ya existe, solo su equipo puede cambiarla.
- Que estén `proyecto.toml` (bien llenado), `README.md` con «Cómo correrlo», `LICENSE` y `BITACORA.md`.
- Que la bitácora tenga el formato de arriba, con evidencia en cada fila.
- Que no haya archivos de más de 5 MB, `datos/`, `modelos/`, `node_modules/`, `.venv/`, `.env` ni llaves.

Además, en la entrega:

- Que haya `Dockerfile` o `compose.yaml`, y que **la imagen se construya**.
- Que `demo` tenga una URL (o `"no aplica"`, si así se acordó).

## Qué se revisa para calificar

- Que la carpeta pase el check: licencia, README, Docker.
- Que la demo responda, si declaras una.
- Que cada integrante tenga evidencia de trabajo: sus commits y su bitácora.

Los reportes dicen hechos verificables, por ejemplo cuántos commits tiene cada quien. Nunca
acusan a nadie.

## ¿Algo falla?

- **«El PR toca archivos fuera de…»**: cambiaste algo fuera de tu carpeta. El resumen del
  check dice qué archivo es. Regrésalo a como estaba en este repo, haz `commit` y `push`, y el
  check vuelve a correr. Si no sabes cómo, pregunta.
- **«Quien abre el PR no está en `equipo`»**: agrega tu login en `proyecto.toml`.
- **El check dice que hay una llave**: bórrala del archivo, ponla en `.env` y revócala.
- **El check no corre**: si es tu primer PR, puede que el profesor tenga que aprobarlo primero.

Este repo (la plantilla y las herramientas) es MIT. Cada proyecto lleva su propia `LICENSE`
en su carpeta.
