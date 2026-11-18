"""Revisa la carpeta de un proyecto final: `estudiantes/<tu_equipo>/`.

Úsalo en tu computadora antes de abrir el pull request:

    python3 herramientas/revisar.py estudiantes/<tu_equipo>

El check del PR («revisar-entrega») corre este mismo script, con `--pr`, y además
revisa que el PR solo toque tu carpeta:

    python3 herramientas/revisar.py --pr --base <sha> --head <sha> --autor <login>

Revisa solo lo que se subiría a GitHub (lo que `.gitignore` ignora no cuenta).
Sale con 0 si no hay errores y con 1 si hay alguno. Los avisos no bloquean.
Solo biblioteca estándar, Python 3.10 o más nuevo.
"""

from __future__ import annotations

import argparse
import ast
import datetime as dt
import os
import re
import subprocess
import sys
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

RAIZ = "estudiantes"
NOMBRE = re.compile(r"^[a-z0-9_]{3,30}$")
RESERVADOS = {"plantilla", "ejemplo"}
LOGIN = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9-]{0,38})$")
ESTADOS = ("propuesta", "entrega")

TOPE_ARCHIVO = 5 * 1024 * 1024  # 5 MB por archivo
TOPE_CARPETA = 50 * 1024 * 1024  # 50 MB toda la carpeta

LICENCIAS = ("LICENSE", "LICENSE.md", "LICENSE.txt")
COMPOSE = ("compose.yaml", "compose.yml", "docker-compose.yaml", "docker-compose.yml")
# `datos/` y `modelos/` son la convención de la plantilla para lo pesado que se queda en
# tu computadora (están en .gitignore); las demás son dependencias o bitácoras de runs.
CARPETAS_PROHIBIDAS = {
    "datos", "modelos", "node_modules", ".venv", "venv", "__pycache__",
    ".ipynb_checkpoints", "wandb", "mlruns",
}
SECRETOS = [
    ("una llave de OpenAI o Anthropic", re.compile(r"sk-(?:ant-)?[A-Za-z0-9_-]{20,}")),
    ("una llave de AWS", re.compile(r"AKIA[0-9A-Z]{16}")),
    ("un token de GitHub", re.compile(r"(?:ghp_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,})")),
    ("una llave de Google", re.compile(r"AIza[0-9A-Za-z_-]{35}")),
    ("un token de Hugging Face", re.compile(r"hf_[A-Za-z0-9]{30,}")),
    ("una llave privada", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
]
DEFAULTS_PLANTILLA = {
    "titulo": "Título del proyecto",
    "tema": "Una línea: qué hace y qué parte del curso usa",
}


@dataclass
class Informe:
    errores: list[str] = field(default_factory=list)
    avisos: list[str] = field(default_factory=list)

    def error(self, texto: str) -> None:
        self.errores.append(texto)

    def aviso(self, texto: str) -> None:
        self.avisos.append(texto)


# --------------------------------------------------------------------------- utilidades


def sh(*args: str, cwd: Path | None = None) -> str:
    return subprocess.run(args, capture_output=True, check=True, cwd=cwd).stdout.decode()


def sin_acentos(texto: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", texto) if unicodedata.category(c) != "Mn"
    ).lower()


def leer_toml(texto: str) -> dict:
    """TOML de `proyecto.toml`. Usa tomllib si existe (3.11+); si no, un lector mínimo
    que entiende `clave = "texto"` y `clave = ["a", "b"]` en una línea, como la plantilla."""
    try:
        import tomllib  # type: ignore[import-not-found]

        return tomllib.loads(texto)
    except ModuleNotFoundError:
        pass
    datos: dict = {}
    for n, linea in enumerate(texto.splitlines(), 1):
        limpia = _quitar_comentario(linea).strip()
        if not limpia:
            continue
        if "=" not in limpia:
            raise ValueError(f"línea {n}: se esperaba `clave = valor`")
        clave, valor = (p.strip() for p in limpia.split("=", 1))
        try:
            datos[clave] = ast.literal_eval(valor)
        except (ValueError, SyntaxError) as e:
            raise ValueError(f"línea {n}: valor no válido ({valor})") from e
    return datos


def _quitar_comentario(linea: str) -> str:
    dentro = False
    for i, c in enumerate(linea):
        if c == '"':
            dentro = not dentro
        elif c == "#" and not dentro:
            return linea[:i]
    return linea


def archivos_que_se_suben(carpeta: Path, repo: Path) -> list[Path]:
    """Lo versionado más lo nuevo que `.gitignore` no ignora: lo que llegaría al PR."""
    rel = carpeta.resolve().relative_to(repo.resolve()).as_posix()
    salida = sh("git", "ls-files", "-z", "-c", "-o", "--exclude-standard", "--", rel, cwd=repo)
    vistos = sorted({p for p in salida.split("\0") if p})
    return [repo / p for p in vistos if (repo / p).is_file()]


# --------------------------------------------------------------------------- rutas del PR


def revisar_rutas(base: str, head: str, informe: Informe) -> str | None:
    """El PR solo toca UNA carpeta `estudiantes/<tu_equipo>/`. Devuelve su nombre."""
    crudo = sh("git", "diff", "--raw", "--no-renames", "--no-abbrev", "-z", f"{base}...{head}")
    partes = crudo.split("\0")
    rutas: list[tuple[str, tuple[str, str]]] = []
    i = 0
    while i + 1 < len(partes) and partes[i]:
        meta = partes[i].lstrip(":").split()
        rutas.append((partes[i + 1], (meta[0], meta[1])))
        i += 2
    if not rutas:
        informe.error("El PR no cambia ningún archivo.")
        return None
    equipos: set[str] = set()
    fuera = []
    for ruta, modos in rutas:
        if "160000" in modos:
            informe.error(f"`{ruta}` es un submódulo de git. Copia los archivos dentro de tu carpeta.")
        p = ruta.split("/")
        if len(p) >= 3 and p[0] == RAIZ:
            equipos.add(p[1])
        else:
            fuera.append(ruta)
    if fuera:
        informe.error(
            "El PR toca archivos fuera de `estudiantes/<tu_equipo>/`. Quítalos del PR:\n"
            + "\n".join(f"  - {r}" for r in fuera)
        )
    if len(equipos) > 1:
        informe.error(
            "El PR toca varias carpetas de equipo y solo puede tocar la tuya: "
            + ", ".join(f"`{RAIZ}/{e}/`" for e in sorted(equipos))
        )
    if informe.errores or not equipos:
        return None
    (equipo,) = equipos
    return equipo


def equipo_en_base(base: str, equipo: str) -> list[str] | None:
    """El `equipo = [...]` de esa carpeta en `main`, o None si la carpeta es nueva."""
    try:
        texto = sh("git", "show", f"{base}:{RAIZ}/{equipo}/proyecto.toml")
    except subprocess.CalledProcessError:
        return None
    try:
        return [str(x).lower() for x in leer_toml(texto).get("equipo", [])]
    except ValueError:
        return []


# --------------------------------------------------------------------------- la carpeta


def revisar_carpeta(carpeta: Path, repo: Path, informe: Informe, autor: str | None = None) -> dict:
    nombre = carpeta.name
    if carpeta.parent.name != RAIZ:
        informe.error(f"La carpeta tiene que estar dentro de `{RAIZ}/`: `{RAIZ}/{nombre}/`.")
    if not NOMBRE.fullmatch(nombre):
        informe.error(
            f"El nombre de carpeta «{nombre}» no es válido. Usa de 3 a 30 caracteres: "
            "minúsculas sin acentos, dígitos y guion bajo (ej.: `agente_snake`)."
        )
    elif nombre in RESERVADOS:
        informe.error(f"`{RAIZ}/{nombre}/` está reservada. Copia la plantilla a una carpeta con el nombre de tu proyecto.")
    if not carpeta.is_dir():
        informe.error(f"No existe la carpeta `{carpeta}`.")
        return {}

    archivos = archivos_que_se_suben(carpeta, repo)
    nombres = {p.relative_to(carpeta).as_posix() for p in archivos}

    datos = revisar_proyecto_toml(carpeta, informe, autor)
    estado = datos.get("estado")
    equipo = [str(x).lower() for x in datos.get("equipo", []) if isinstance(x, str)]

    # Archivos que siempre van.
    if "README.md" not in nombres:
        informe.error("Falta `README.md` (cópialo de la plantilla).")
    else:
        texto = (carpeta / "README.md").read_text(encoding="utf-8", errors="replace")
        encabezados = [sin_acentos(l) for l in texto.splitlines() if l.startswith("#")]
        if not any("como correrlo" in e for e in encabezados):
            informe.error("El `README.md` necesita una sección `## Cómo correrlo` con los comandos exactos.")
    if not any(l in nombres for l in LICENCIAS):
        informe.error("Falta `LICENSE` (la plantilla trae MIT; cambia el año y los nombres si quieres).")
    if "BITACORA.md" not in nombres:
        informe.error("Falta `BITACORA.md` (cópiala de la plantilla).")
    else:
        revisar_bitacora(carpeta, set(nombres), equipo, estado, informe)

    tiene_docker = "Dockerfile" in nombres or any(c in nombres for c in COMPOSE)
    if not tiene_docker:
        texto = "Falta `Dockerfile` o `compose.yaml` en la raíz de tu carpeta."
        (informe.error if estado == "entrega" else informe.aviso)(
            texto + (" En la entrega es obligatorio." if estado != "entrega" else "")
        )

    if estado == "entrega":
        demo = str(datos.get("demo", "")).strip()
        if not demo:
            informe.error('En la entrega, `demo` no puede ir vacío: pon la URL, o `"no aplica"` si así lo acordamos.')
        elif demo != "no aplica" and not demo.startswith(("https://", "http://")):
            informe.error(f'`demo` tiene que ser una URL (https://…) o `"no aplica"`, no «{demo}».')

    revisar_archivos(carpeta, archivos, informe)
    return {"nombre": nombre, "estado": estado, "equipo": equipo, "datos": datos, "archivos": nombres}


def revisar_proyecto_toml(carpeta: Path, informe: Informe, autor: str | None) -> dict:
    ruta = carpeta / "proyecto.toml"
    if not ruta.is_file():
        informe.error("Falta `proyecto.toml` (cópialo de la plantilla).")
        return {}
    try:
        datos = leer_toml(ruta.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001 — cualquier error de lectura se explica igual
        informe.error(f"`proyecto.toml` no se puede leer: {e}. Compáralo con el de la plantilla.")
        return {}

    estado = datos.get("estado")
    if estado not in ESTADOS:
        informe.error(f'`estado` tiene que ser "propuesta" o "entrega", no «{estado}».')
    for clave in ("titulo", "tema"):
        valor = datos.get(clave)
        if not isinstance(valor, str) or not valor.strip():
            informe.error(f"`{clave}` falta o está vacío en `proyecto.toml`.")
        elif valor.strip() == DEFAULTS_PLANTILLA[clave]:
            informe.error(f"`{clave}` todavía dice lo de la plantilla. Escribe el tuyo.")
    equipo = datos.get("equipo")
    if not isinstance(equipo, list) or not equipo or not all(isinstance(x, str) for x in equipo):
        informe.error('`equipo` tiene que ser una lista de logins de GitHub: `equipo = ["usuario-a", "usuario-b"]`.')
        equipo = []
    else:
        for login in equipo:
            if not LOGIN.fullmatch(login) or login in ("usuario-a", "usuario-b"):
                informe.error(f"«{login}» no es un login de GitHub válido (o es el de la plantilla).")
        if len({x.lower() for x in equipo}) != len(equipo):
            informe.error("Hay un login repetido en `equipo`.")
        if len(equipo) > 2:
            informe.aviso(f"El equipo tiene {len(equipo)} personas. Solo vale si lo hablaste con el profesor.")
    if "demo" in datos and not isinstance(datos["demo"], str):
        informe.error('`demo` tiene que ir entre comillas: `demo = "https://…"`.')
    if autor and equipo and autor.lower() not in {x.lower() for x in equipo}:
        informe.error(
            f"Quien abre el PR (`{autor}`) no está en `equipo`. Agrega tu login a `proyecto.toml`."
        )
    return datos


FILA = re.compile(r"^\s*\|(.+)\|\s*$")


def revisar_bitacora(carpeta: Path, nombres: set[str], equipo: list[str], estado: str | None, informe: Informe) -> dict[str, int]:
    """Cada fila: | fecha | login | qué hice | evidencia |. Devuelve filas válidas por login."""
    lineas = (carpeta / "BITACORA.md").read_text(encoding="utf-8", errors="replace").splitlines()
    por_login: dict[str, int] = {x: 0 for x in equipo}
    encabezado_visto = False
    for n, linea in enumerate(lineas, 1):
        m = FILA.match(linea)
        if not m:
            continue
        celdas = [c.strip() for c in m.group(1).split("|")]
        if all(set(c) <= set("-: ") for c in celdas):
            continue  # separador |---|---|
        if not encabezado_visto:
            encabezado_visto = True
            if sin_acentos(celdas[0]) != "fecha":
                informe.error(f"`BITACORA.md:{n}`: la tabla tiene que empezar con `| fecha | login | qué hice | evidencia |`.")
            continue
        if len(celdas) != 4:
            informe.error(f"`BITACORA.md:{n}`: la fila tiene {len(celdas)} columnas; van 4 (fecha, login, qué hice, evidencia).")
            continue
        fecha, login, que, evidencia = celdas
        try:
            dt.date.fromisoformat(fecha)
        except ValueError:
            informe.error(f"`BITACORA.md:{n}`: la fecha «{fecha}» va como AAAA-MM-DD (ej.: 2026-11-20).")
        if equipo and login.lower() not in por_login:
            informe.error(f"`BITACORA.md:{n}`: «{login}» no está en `equipo` de `proyecto.toml`.")
            continue
        if not que:
            informe.error(f"`BITACORA.md:{n}`: falta qué hiciste.")
        if not evidencia:
            informe.error(f"`BITACORA.md:{n}`: falta la evidencia (un link o un archivo de tu carpeta). Sin evidencia la línea no cuenta.")
            continue
        enlace = re.search(r"\]\(([^)]+)\)", evidencia)
        destino = enlace.group(1) if enlace else evidencia
        if not destino.startswith(("https://", "http://")):
            destino = destino.removeprefix("./")
            if destino not in nombres:
                informe.error(f"`BITACORA.md:{n}`: la evidencia «{destino}» no es un link ni un archivo de tu carpeta.")
                continue
        if login.lower() in por_login:
            por_login[login.lower()] += 1
    if not encabezado_visto:
        informe.error("`BITACORA.md` no tiene la tabla de la plantilla.")
    if estado == "entrega":
        for login, cuenta in por_login.items():
            if cuenta == 0:
                informe.aviso(f"`{login}` no tiene ninguna línea en la bitácora. Sus commits también cuentan, pero revisa que esté su trabajo.")
    return por_login


def revisar_archivos(carpeta: Path, archivos: list[Path], informe: Informe) -> None:
    total = 0
    for p in archivos:
        rel = p.relative_to(carpeta).as_posix()
        partes = rel.split("/")
        tam = p.stat().st_size
        total += tam
        prohibida = next((d for d in partes[:-1] if d in CARPETAS_PROHIBIDAS), None)
        if prohibida:
            informe.error(f"`{rel}`: la carpeta `{prohibida}/` no se sube (datos, modelos o dependencias se quedan en tu computadora).")
            continue
        base = partes[-1]
        if base == ".env" or (base.startswith(".env.") and base != ".env.example"):
            informe.error(f"`{rel}`: los archivos `.env` no se suben nunca. Sube un `.env.example` sin valores.")
            continue
        if tam > TOPE_ARCHIVO:
            informe.error(f"`{rel}` pesa {tam / 2**20:.1f} MB; el máximo es 5 MB. Déjalo en `datos/` o `modelos/` y explica en el README cómo conseguirlo.")
            continue
        if tam <= 2 * 1024 * 1024:
            try:
                texto = p.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            for n, linea in enumerate(texto.splitlines(), 1):
                for que, patron in SECRETOS:
                    if patron.search(linea):
                        informe.error(f"`{rel}:{n}` parece tener {que}. Bórrala del archivo, ponla en `.env` y **revócala**: el repo es público.")
    if total > TOPE_CARPETA:
        informe.error(f"Tu carpeta pesa {total / 2**20:.1f} MB; el máximo es 50 MB.")


# --------------------------------------------------------------------------- salida


def imprimir(informe: Informe, titulo: str, resumen: Path | None) -> None:
    lineas = [f"## {titulo}", ""]
    if informe.errores:
        lineas += ["**Hay que arreglar esto antes de aceptar el PR:**", ""]
        lineas += [f"- ❌ {e}" for e in informe.errores]
    else:
        lineas += ["✅ Todo en orden."]
    if informe.avisos:
        lineas += ["", "**Avisos (no bloquean):**", ""] + [f"- ⚠️ {a}" for a in informe.avisos]
    texto = "\n".join(lineas) + "\n"
    print(texto)
    if resumen:
        with resumen.open("a", encoding="utf-8") as f:
            f.write(texto)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="Revisa la carpeta de un proyecto final.")
    ap.add_argument("carpeta", nargs="?", help="estudiantes/<tu_equipo>")
    ap.add_argument("--pr", action="store_true", help="modo del check del PR")
    ap.add_argument("--base")
    ap.add_argument("--head")
    ap.add_argument("--autor")
    a = ap.parse_args(argv)

    repo = Path(sh("git", "rev-parse", "--show-toplevel").strip())
    informe = Informe()
    resumen = Path(os.environ["GITHUB_STEP_SUMMARY"]) if os.environ.get("GITHUB_STEP_SUMMARY") else None

    if a.pr:
        if not (a.base and a.head and a.autor):
            ap.error("--pr necesita --base, --head y --autor")
        equipo = revisar_rutas(a.base, a.head, informe)
        if equipo is None:
            imprimir(informe, "Revisión del PR", resumen)
            return 1
        anteriores = equipo_en_base(a.base, equipo)
        if anteriores is not None and a.autor.lower() not in anteriores:
            informe.error(
                f"`{RAIZ}/{equipo}/` ya existe y es de {', '.join(anteriores) or 'otro equipo'}. "
                "Solo su equipo puede cambiarla; si tu proyecto es otro, usa otro nombre de carpeta."
            )
        carpeta = repo / RAIZ / equipo
    else:
        if not a.carpeta:
            ap.error("di qué carpeta revisar: python3 herramientas/revisar.py estudiantes/<tu_equipo>")
        carpeta = Path(a.carpeta).resolve()

    info = revisar_carpeta(carpeta, repo, informe, autor=a.autor if a.pr else None)
    estado = info.get("estado") or "?"
    imprimir(informe, f"`{RAIZ}/{carpeta.name}/` · {estado}", resumen)

    salida = os.environ.get("GITHUB_OUTPUT")
    if salida and info:
        with open(salida, "a", encoding="utf-8") as f:
            f.write(f"equipo={carpeta.name}\nestado={estado}\n")
    return 1 if informe.errores else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
