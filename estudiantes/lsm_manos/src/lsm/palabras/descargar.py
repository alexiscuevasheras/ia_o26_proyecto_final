"""Descarga un dataset público de Mendeley Data archivo por archivo.

Dos modos:
- `--zip`: el zip completo que Mendeley guarda en caché (rápido; UAEMex pesa
  1.9 GB). Se descomprime en el destino y se borra el zip.
- archivo por archivo con la API pública (lista de archivos con sha256 y de
  carpetas para reconstruir las rutas). Se puede reanudar, pero Mendeley
  limita la velocidad con miles de archivos chicos: con UAEMex bajó a ~25
  fotos por minuto, por eso existe el modo zip.

Uso:
    python -m lsm.palabras.descargar uaemex datos/crudos/uaemex --zip
    python -m lsm.palabras.descargar iteso datos/crudos/iteso

Los dos tienen licencia CC BY 4.0. Las fotos y videos muestran la cara de
las personas: se quedan en datos/ (ignorado por git) y nunca se suben.
"""

import argparse
import hashlib
import json
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

API = "https://data.mendeley.com/public-api/datasets"
# Mendeley responde 403 al User-Agent por defecto de urllib.
AGENTE = {"User-Agent": "lsm-manos-descarga/1.0"}
ENCABEZADO = {**AGENTE, "Accept": "application/vnd.mendeley-public-dataset.1+json"}
DATASETS = {
    "uaemex": ("6rj76z6y3n", 1),  # Espejel et al. 2023, 249 palabras, 11 personas
    "iteso": ("g45jkzys4f", 1),  # MX-ITESO-100 (49 de las 100 señas publicadas)
}


def pedir_json(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers=ENCABEZADO), timeout=600) as r:
        return json.load(r)


def rutas_de_carpetas(carpetas):
    """{id de carpeta: 'a/b/c'} siguiendo parent_id hasta la raíz."""
    por_id = {c["id"]: c for c in carpetas}

    def ruta(cid):
        partes = []
        while cid in por_id:
            partes.append(por_id[cid]["name"].strip())
            cid = por_id[cid].get("parent_id")
        return "/".join(reversed(partes))

    return {cid: ruta(cid) for cid in por_id}


def pedir_json_guardado(url, copia):
    """La lista de UAEMex pesa 24 MB y a veces tarda minutos: se guarda una copia."""
    if copia.exists():
        return json.loads(copia.read_text())
    datos = pedir_json(url)
    copia.write_text(json.dumps(datos))
    return datos


def inventario(clave, destino):
    """[(ruta relativa, url, sha256 o None, tamaño)] de todos los archivos."""
    ident, version = DATASETS[clave]
    meta = pedir_json_guardado(f"{API}/{ident}", destino / "_mendeley.json")
    carpetas = rutas_de_carpetas(pedir_json_guardado(f"{API}/{ident}/folders/{version}", destino / "_carpetas.json"))
    archivos = []
    for f in meta["files"]:
        detalle = f.get("content_details") or {}
        url = detalle.get("download_url") or f"{API.replace('public-api', 'public-files')}/{ident}/files/{f['id']}/file_downloaded"
        ruta = "/".join(p for p in (carpetas.get(f["folder_id"], ""), f["filename"].strip()) if p)
        archivos.append((ruta, url, detalle.get("sha256_hash"), f.get("size") or detalle.get("size", 0)))
    return archivos


def sha256(ruta):
    h = hashlib.sha256()
    with ruta.open("rb") as f:
        for bloque in iter(lambda: f.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


def bajar(destino, ruta, url, esperado, intentos=5):
    """Baja un archivo; reintenta ante errores de red o sha256 distinto. Devuelve
    (ruta, estado) y nunca lanza: un archivo fallido no detiene a los demás."""
    archivo = destino / ruta
    if archivo.exists() and (esperado is None or sha256(archivo) == esperado):
        return ruta, "ya estaba"
    archivo.parent.mkdir(parents=True, exist_ok=True)
    temporal = archivo.with_name(archivo.name + ".parcial")
    for intento in range(intentos):
        try:
            # Timeout corto: Mendeley a veces deja la conexión abierta sin mandar nada.
            with urllib.request.urlopen(urllib.request.Request(url, headers=AGENTE), timeout=60) as r, \
                    temporal.open("wb") as f:
                while bloque := r.read(1 << 20):
                    f.write(bloque)
            if esperado is None or sha256(temporal) == esperado:
                temporal.rename(archivo)
                return ruta, "ok"
        except OSError:
            time.sleep(2 ** intento)
    temporal.unlink(missing_ok=True)
    return ruta, "falló"


def bajar_zip(clave, destino):
    """El zip en caché de Mendeley: se baja, se descomprime en `destino` y se borra."""
    ident, version = DATASETS[clave]
    zip_local = destino / f"{clave}.zip"
    url = f"{API.replace('datasets', 'zip')}/{ident}/download/{version}"
    with urllib.request.urlopen(urllib.request.Request(url, headers=AGENTE), timeout=120) as r, zip_local.open("wb") as f:
        while bloque := r.read(1 << 22):
            f.write(bloque)
    with zipfile.ZipFile(zip_local) as z:
        z.extractall(destino)
    zip_local.unlink()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("dataset", choices=DATASETS)
    ap.add_argument("destino", type=Path)
    ap.add_argument("--hilos", type=int, default=16)
    ap.add_argument("--zip", action="store_true", help="bajar el zip completo en caché")
    args = ap.parse_args()

    args.destino.mkdir(parents=True, exist_ok=True)
    if args.zip:
        bajar_zip(args.dataset, args.destino)
        print(f"{args.dataset}: descomprimido en {args.destino}")
        return
    archivos = inventario(args.dataset, args.destino)
    total = sum(a[3] for a in archivos)
    print(f"{args.dataset}: {len(archivos)} archivos, {total / 1e9:.2f} GB", flush=True)
    with ThreadPoolExecutor(args.hilos) as ejecutor:
        futuros = [ejecutor.submit(bajar, args.destino, r, u, s) for r, u, s, _ in archivos]
        fallidos = []
        for i, futuro in enumerate(as_completed(futuros), 1):
            ruta, estado = futuro.result()
            if estado == "falló":
                fallidos.append(ruta)
            if i % 500 == 0 or i == len(futuros):
                print(f"  {i} de {len(futuros)} ({len(fallidos)} fallidos)", flush=True)
    if fallidos:
        print("fallaron (vuelve a correr el comando para reintentar):", *fallidos[:20], sep="\n  ")


if __name__ == "__main__":
    main()
