"""
Variables de entrada del modelo ligero (scikit-learn) para la app 2026.

Este modulo NO importa pyspark ni nada de src/ (src.features importa pyspark y la app no
puede depender de Spark). Las listas de variables estan COPIADAS de src/features.py
(CATEGORICAL, CYCLIC_SMALL, WEATHER_NUM, WEATHER_FLAGS, NUMERIC, LEAKAGE_COLS) y
add_model_columns() reproduce en pandas la funcion del mismo nombre de src/features.py.
Si cambian alla, hay que cambiarlas aqui.

REGLA DE FUGA DE DATOS: solo variables conocidas ANTES del vuelo.
"""
import numpy as np
import pandas as pd

# ---- Copiado de src/features.py ----
CATEGORICAL = ["OP_UNIQUE_CARRIER", "ORIGIN", "DEST"]
CYCLIC_SMALL = ["DAY_OF_WEEK", "dep_hour"]
WEATHER_NUM = ["precip_mm", "snow_cm", "wind_max_kmh", "temp_max_c", "temp_min_c"]
WEATHER_FLAGS = ["rain_flag", "snow_flag", "high_wind_flag", "has_weather"]
NUMERIC = ["month_sin", "month_cos", "is_weekend", "DISTANCE"] + WEATHER_NUM + WEATHER_FLAGS
LEAKAGE_COLS = ["DEP_DELAY", "DEP_DEL15", "ARR_DELAY", "ARR_DEL15", "CARRIER_DELAY",
                "WEATHER_DELAY", "NAS_DELAY", "SECURITY_DELAY", "LATE_AIRCRAFT_DELAY"]
INPUT_COLS = CATEGORICAL + CYCLIC_SMALL + NUMERIC
# ---- Fin de la copia ----


def check_no_leakage(cols=INPUT_COLS):
    bad = set(cols) & set(LEAKAGE_COLS)
    if bad:
        raise RuntimeError(f"Fuga de datos: columnas prohibidas como entrada: {bad}")


def add_model_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Equivalente pandas de src.features.add_model_columns: mes ciclico, fin de semana y
    relleno de clima. Espera MONTH, DAY_OF_WEEK, dep_hour, DISTANCE y las columnas de clima
    (posiblemente nulas o ausentes). Modifica y devuelve df."""
    angle = 2 * np.pi * (df["MONTH"].astype("float64") - 1) / 12
    df["month_sin"] = np.sin(angle)
    df["month_cos"] = np.cos(angle)
    df["is_weekend"] = df["DAY_OF_WEEK"].isin([6, 7]).astype("int8")
    if "has_weather" not in df:
        df["has_weather"] = 0
    # Sin clima -> 0 en variables y banderas; has_weather=0 indica "dato no disponible".
    df["has_weather"] = df["has_weather"].fillna(0).astype("int8")
    for c in WEATHER_NUM:
        df[c] = (df[c] if c in df else pd.Series(np.nan, index=df.index)).fillna(0.0).astype("float64")
    for c in WEATHER_FLAGS:
        df[c] = (df[c] if c in df else pd.Series(0, index=df.index)).fillna(0).astype("int8")
    return df


# Nombres de las entradas crudas (las que llena la app) -> nombre de la variable del modelo.
RAW_TO_MODEL = {"aerolinea": "OP_UNIQUE_CARRIER", "origen": "ORIGIN", "destino": "DEST"}
RAW_INPUTS = (["aerolinea", "origen", "destino", "MONTH", "DAY_OF_WEEK", "dep_hour", "DISTANCE"]
              + WEATHER_NUM + ["rain_flag", "snow_flag", "high_wind_flag", "has_weather"])


def make_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Entradas crudas -> las 18 variables del modelo (en el orden de INPUT_COLS).

    raw: una fila por vuelo con aerolinea, origen, destino (codigos IATA), MONTH (1-12),
    DAY_OF_WEEK (1=lunes ... 7=domingo, convencion BTS), dep_hour (0-23), DISTANCE (millas) y,
    opcionalmente, el clima del origen ese dia (precip_mm, snow_cm, wind_max_kmh, temp_max_c,
    temp_min_c, rain_flag, snow_flag, high_wind_flag, has_weather). Clima ausente o nulo se
    rellena con 0 y has_weather=0, igual que en el entrenamiento de Spark.
    La usan el entrenamiento (app/ml/datos.py) y la app, asi no hay diferencias."""
    df = raw.rename(columns=RAW_TO_MODEL).copy()
    for c in CATEGORICAL:
        if not isinstance(df[c].dtype, pd.CategoricalDtype):
            df[c] = df[c].astype(str)
    for c in ["MONTH", "DAY_OF_WEEK", "dep_hour"]:
        df[c] = df[c].astype("int8")
    df["DISTANCE"] = df["DISTANCE"].astype("float64")
    df = add_model_columns(df)
    check_no_leakage(df.columns.intersection(INPUT_COLS))
    return df[INPUT_COLS]


# Umbrales de banderas de clima (copiados de src/silver.py) y clima_con_banderas(): viven en
# app/ml/umbrales.py (solo biblioteca estandar, lo usa app/weather.py) y se reexportan aqui.
from app.ml.umbrales import HIGH_WIND_KMH, RAIN_MM, SNOW_CM, clima_con_banderas  # noqa: E402,F401


def flight_features(aerolinea, origen, destino, MONTH, DAY_OF_WEEK, dep_hour, DISTANCE,
                    clima=None) -> pd.DataFrame:
    """Atajo para un solo vuelo. clima: dict opcional con las variables de clima."""
    row = {"aerolinea": aerolinea, "origen": origen, "destino": destino, "MONTH": MONTH,
           "DAY_OF_WEEK": DAY_OF_WEEK, "dep_hour": dep_hour, "DISTANCE": DISTANCE,
           **(clima or {})}
    return make_features(pd.DataFrame([row]))
