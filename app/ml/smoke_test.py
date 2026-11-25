"""
Prueba de carga de los modelos sklearn en un proceso nuevo y SIN pyspark.

Antes de cualquier import se instala app/ml/sin_spark.py: cualquier intento de importar
pyspark (o src, que importa pyspark) falla con ModuleNotFoundError, como si pyspark no
estuviera instalado. Luego carga los modelos de PRODUCCION (app/models/*.joblib, ene-dic) y,
para comparar, los de EVALUACION (app/models/eval/*_ene_oct.joblib), y predice vuelos de
ejemplo de 2026.

Uso: .venv/bin/python app/ml/smoke_test.py
"""
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from app.ml import sin_spark  # noqa: E402  (primero: bloquea pyspark y src)

MODELS_DIR = os.path.join(ROOT, "app", "models")

# Clima promedio de DFW en la misma semana ISO de 2025, calculado desde
# data/silver/weather_daily (ver app/ml/verificar.py). Banderas con las reglas de Silver.
CLIMA_DFW_SEM11_2025 = dict(precip_mm=0.06, snow_cm=0.0, wind_max_kmh=18.9,
                            temp_max_c=25.19, temp_min_c=11.37)   # 10 a 16 de marzo de 2025
CLIMA_DFW_SEM51_2025 = dict(precip_mm=0.23, snow_cm=0.0, wind_max_kmh=16.29,
                            temp_max_c=17.27, temp_min_c=7.61)    # 15 a 21 de diciembre de 2025

# AA de DFW a ORD a las 18 h. 15/03/2026 y 20/12/2026 son domingo (DAY_OF_WEEK=7).
VUELOS = [
    ("2026-03-15", dict(MONTH=3, DAY_OF_WEEK=7), CLIMA_DFW_SEM11_2025),
    ("2026-12-20", dict(MONTH=12, DAY_OF_WEEK=7), CLIMA_DFW_SEM51_2025),
]
BASE = dict(aerolinea="AA", origen="DFW", destino="ORD", dep_hour=18, DISTANCE=802.0)


def main():
    try:
        import pyspark  # noqa: F401
        raise SystemExit("ERROR: pyspark se pudo importar")
    except ModuleNotFoundError as e:
        print(f"pyspark no disponible en este proceso: OK ({e})")

    t = time.perf_counter()
    import joblib
    import sklearn.compose, sklearn.ensemble, sklearn.linear_model  # noqa: F401,E401
    from app.ml.features import clima_con_banderas, flight_features
    from app.ml.modelos import cargar_modelos
    t_imports = time.perf_counter() - t
    t = time.perf_counter()
    cls, reg = cargar_modelos(MODELS_DIR)   # clasificador segun metadata "clasificador_activo"
    t_modelos = time.perf_counter() - t
    t_carga = t_imports + t_modelos
    import json
    activo = json.load(open(os.path.join(MODELS_DIR, "metadata.json")))["clasificador_activo"]
    print(f"Imports (pandas, sklearn, joblib): {t_imports:.3f} s | carga de los 2 modelos de "
          f"produccion (clasificador_activo={activo}): {t_modelos:.3f} s | carga total: {t_carga:.3f} s")
    ev = os.path.join(MODELS_DIR, "eval")
    modelos = {"produccion ene-dic": (joblib.load(os.path.join(MODELS_DIR, "clasificador.joblib")), reg),
               "variante mes ene-dic": (joblib.load(os.path.join(MODELS_DIR, "clasificador_mes.joblib")), reg),
               "evaluacion ene-oct": (joblib.load(os.path.join(ev, "clasificador_ene_oct.joblib")),
                                      joblib.load(os.path.join(ev, "regresor_ene_oct.joblib")))}

    def predecir(c, r, X):
        return float(c.predict_proba(X)[0, 1]), float(r.predict(X)[0])

    def variables(campos, clima):
        return flight_features(**BASE, **campos, clima=clima_con_banderas(**clima))

    # Tiempo por prediccion con produccion: construir las 18 variables + 2 modelos.
    _, campos, clima = VUELOS[0]
    t = time.perf_counter()
    predecir(cls, reg, variables(campos, clima))
    t_primera = time.perf_counter() - t
    n = 200
    t = time.perf_counter()
    for _ in range(n):
        predecir(cls, reg, variables(campos, clima))
    t_pred = (time.perf_counter() - t) / n
    print(f"Primera prediccion: {1000 * t_primera:.1f} ms | promedio de {n}: {1000 * t_pred:.1f} ms "
          f"(incluye construir las 18 variables; el clima ya viene dado)")
    print(f"Objetivo por prediccion < 1 s: {'OK' if t_primera < 1 else 'NO'}")

    print("\nAA DFW-ORD 18 h, con el clima promedio de DFW de esa semana en 2025:")
    print(f"{'fecha':<11} {'modelo':<21} {'prob. retraso':>13} {'minutos':>8}")
    for fecha, campos, clima in VUELOS:
        X = variables(campos, clima)
        for nombre, (c, r) in modelos.items():
            p, m = predecir(c, r, X)
            print(f"{fecha:<11} {nombre:<21} {p:>13.3f} {m:>8.1f}")
            assert 0.0 < p < 1.0 and r.p01 <= m <= r.p99

    assert not sin_spark.cargados(), sin_spark.cargados()
    print("\npyspark/src en sys.modules: ninguno. Smoke test OK")


if __name__ == "__main__":
    main()
