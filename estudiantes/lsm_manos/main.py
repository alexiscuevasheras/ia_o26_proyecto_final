"""Sirve la demo en http://localhost:8000: la página (web/) y los modelos (modelos/).

Los modelos no están en la imagen: compose.yaml monta ./modelos (ver README,
«Cómo correrlo»). La raíz redirige a la página de señas.
"""

from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

CARPETA = Path(__file__).resolve().parent


class Demo(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self.send_response(302)
            self.send_header("Location", "/web/")
            self.end_headers()
            return
        super().do_GET()


if __name__ == "__main__":
    faltan = [m for m in ("palabras.json", "pesos.json", "hand_landmarker.task", "pose_landmarker_full.task",
                          "animaciones.json") if not (CARPETA / "modelos" / m).exists()]
    if faltan:
        print(f"Aviso: faltan en modelos/: {', '.join(faltan)} (ver README)", flush=True)
    print("Sirviendo en http://localhost:8000", flush=True)
    ThreadingHTTPServer(("0.0.0.0", 8000), partial(Demo, directory=str(CARPETA))).serve_forever()
