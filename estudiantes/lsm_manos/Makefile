# Uso: make entorno && make pruebas; make demo abre la demo en http://localhost:8000/web/
PY = .venv/bin/python

entorno:
	python3 -m venv .venv && $(PY) -m pip install -r requirements-dev.txt

pruebas:
	$(PY) -m pytest -q

demo:
	@echo "Abre http://localhost:8000/web/ (necesita modelos/pesos.json y modelos/hand_landmarker.task)"
	python3 -m http.server 8000 --bind 127.0.0.1

.PHONY: entorno pruebas demo
