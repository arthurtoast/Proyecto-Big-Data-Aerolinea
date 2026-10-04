# Atajos para el pipeline. Todo usa el Python del entorno virtual .venv.
PY := .venv/bin/python

.PHONY: help venv all organize bronze weather silver gold mysql train report predict evaluate export figures summary deliverables mlflow-ui clean-derived

help:
	@echo "make venv        crea .venv (Python 3.12) e instala requirements.txt"
	@echo "make all         corre el pipeline completo (python -m src.run_pipeline)"
	@echo "make organize | bronze | weather | silver | gold | mysql | train | report | predict"
	@echo "make deliverables  indicadores + CSV Power BI + figuras + zip de entrega"
	@echo "make mlflow-ui   abre la UI de MLflow en http://127.0.0.1:5000"
	@echo "make clean-derived  borra bronze/silver/gold/models (NO toca data/raw)"

venv:
	/opt/homebrew/bin/python3.12 -m venv .venv
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements.txt

all:
	$(PY) -m src.run_pipeline

organize:
	$(PY) -m src.organize_data

bronze:
	$(PY) -m src.bronze

weather:
	$(PY) -m src.weather

silver:
	$(PY) -m src.silver

gold:
	$(PY) -m src.gold

mysql:
	$(PY) -m src.mysql_loader

train:
	$(PY) -m src.train

report:
	$(PY) -m src.report

predict:
	$(PY) -m src.predict

evaluate:
	$(PY) -m src.evaluate

export:
	$(PY) -m src.export_powerbi

figures:
	$(PY) -m src.figures

summary:
	$(PY) -m src.summary

deliverables: evaluate export figures summary
	rm -f deliverables_fase3.zip
	zip -r -X deliverables_fase3.zip deliverables -x "*.DS_Store"

mlflow-ui:
	.venv/bin/mlflow ui --backend-store-uri sqlite:///mlflow.db

# Solo capas derivadas: data/raw (CSV, clima cacheado, aeropuertos) se conserva.
clean-derived:
	rm -rf data/bronze/* data/silver/* data/gold/* data/models
