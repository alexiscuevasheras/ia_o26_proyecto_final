# Uso: make entorno && make pruebas
PY = .venv/bin/python

entorno:
	python3 -m venv .venv && $(PY) -m pip install -r requirements-dev.txt

pruebas:
	$(PY) -m pytest -q

.PHONY: entorno pruebas
