"""Tabla para calificar: un renglón por proyecto de `estudiantes/`.

    python3 herramientas/reporte.py > reporte.md          # con commits de GitHub y demos
    python3 herramientas/reporte.py --sin-red > reporte.md

Por proyecto dice: estado, equipo, si pasa `revisar.py`, commits en su carpeta por login
(de la API de GitHub, con `gh`), filas de bitácora con evidencia por login, y si la demo
responde. Son hechos verificables; ninguno acusa a nadie. Se corre sobre `main` ya
actualizado (`git pull`).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import urllib.request
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from revisar import RAIZ, Informe, revisar_bitacora, revisar_carpeta  # noqa: E402

REPO = "raya-lucaria/ia_o26_proyecto_final"


def commits_por_login(equipo: str) -> Counter:
    """Commits (sin merges) que tocan la carpeta, por login de GitHub."""
    salida = subprocess.run(
        [
            "gh", "api", "-X", "GET", f"repos/{REPO}/commits",
            "-f", f"path={RAIZ}/{equipo}", "-f", "per_page=100", "--paginate",
            "--jq", '.[] | select((.parents | length) == 1)'
                    ' | (.author.login // ("sin cuenta: " + .commit.author.name))',
        ],
        capture_output=True, text=True, check=True,
    ).stdout
    return Counter(l for l in salida.splitlines() if l)


def demo_responde(url: str) -> str:
    if not url.startswith(("http://", "https://")):
        return url or "—"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "revision-proyecto-final"})
        with urllib.request.urlopen(req, timeout=15) as r:
            return f"✅ {r.status}"
    except Exception as e:  # noqa: BLE001
        return f"❌ {type(e).__name__}"


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sin-red", action="store_true", help="no consulta GitHub ni las demos")
    a = ap.parse_args(argv)

    repo = Path(__file__).resolve().parent.parent
    carpetas = sorted(p for p in (repo / RAIZ).iterdir() if p.is_dir())
    print("| Proyecto | Estado | Equipo | Revisión | Commits por login | Bitácora con evidencia | Demo |")
    print("|---|---|---|---|---|---|---|")
    for carpeta in carpetas:
        informe = Informe()
        info = revisar_carpeta(carpeta, repo, informe)
        equipo = info.get("equipo", [])
        bitacora = (
            revisar_bitacora(carpeta, info.get("archivos", set()), equipo, None, Informe())
            if (carpeta / "BITACORA.md").is_file() else {}
        )
        if a.sin_red:
            commits, demo = "—", "—"
        else:
            c = commits_por_login(carpeta.name)
            commits = ", ".join(f"{k}: {v}" for k, v in sorted(c.items())) or "0"
            demo = demo_responde(str(info.get("datos", {}).get("demo", "")))
        revision = "✅" if not informe.errores else f"❌ {len(informe.errores)} errores"
        print(
            f"| `{carpeta.name}` | {info.get('estado') or '?'} | {', '.join(equipo)} | {revision} "
            f"| {commits} | {', '.join(f'{k}: {v}' for k, v in bitacora.items()) or '0'} | {demo} |"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
