"""
Carga de datos para entrenar el modelo scikit-learn, sin Spark (pandas + pyarrow).

Reproduce src.features.build_features() para los vuelos completados (data/silver/flights),
que es lo que uso src/train.py: se leen solo las columnas base (NO el clima que ya trae
Silver) y se une data/silver/weather_daily por ORIGIN y FL_DATE con un left join, igual que
weather_table() + build_features(). Luego make_features() (el mismo codigo que usa la app).
"""
import os

import pandas as pd
import pyarrow.dataset as ds

from app.ml.features import INPUT_COLS, RAW_TO_MODEL, WEATHER_NUM, make_features

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SILVER_FLIGHTS = os.path.join(ROOT, "data", "silver", "flights")
SILVER_WEATHER = os.path.join(ROOT, "data", "silver", "weather_daily")

# Copiado de src/silver.py (BUSINESS_KEY) y src/train.py (meses del split).
BUSINESS_KEY = ["FL_DATE", "OP_UNIQUE_CARRIER", "OP_CARRIER_FL_NUM", "ORIGIN", "DEST",
                "CRS_DEP_TIME"]
TRAIN_MONTHS = list(range(1, 11))
TEST_MONTHS = [11, 12]
BASE_COLS = list(dict.fromkeys(BUSINESS_KEY + ["MONTH", "DAY_OF_WEEK", "dep_hour", "DISTANCE",
                                               "ARR_DEL15", "ARR_DELAY"]))
STR_COLS = ["OP_UNIQUE_CARRIER", "ORIGIN", "DEST"]


def load_weather() -> pd.DataFrame:
    w = pd.read_parquet(SILVER_WEATHER)
    w = w.rename(columns={"airport": "ORIGIN", "date": "FL_DATE"})
    w = w[["ORIGIN", "FL_DATE"] + WEATHER_NUM + ["rain_flag", "snow_flag", "high_wind_flag"]]
    w["has_weather"] = 1
    return w


def load_completed() -> pd.DataFrame:
    """Vuelos completados con variables de entrada y etiquetas (ARR_DEL15, ARR_DELAY)."""
    table = ds.dataset(SILVER_FLIGHTS, format="parquet", partitioning="hive").to_table(
        columns=BASE_COLS)
    df = table.to_pandas()
    w = load_weather()
    # Categorias comunes para que el merge no convierta a object.
    for c in STR_COLS:
        df[c] = df[c].astype("category")
    w["ORIGIN"] = pd.Categorical(w["ORIGIN"], categories=df["ORIGIN"].cat.categories)
    df = df.merge(w, on=["ORIGIN", "FL_DATE"], how="left", validate="many_to_one")
    for c in ["MONTH", "DAY_OF_WEEK", "dep_hour"]:
        df[c] = df[c].astype("int8")
    df["ARR_DEL15"] = df["ARR_DEL15"].astype("int8")
    # Mismo camino que la app: entradas crudas -> make_features().
    raw = df.rename(columns={v: k for k, v in RAW_TO_MODEL.items()})
    feats = make_features(raw)
    extra = [c for c in df.columns if c not in INPUT_COLS]
    return pd.concat([df[extra], feats], axis=1)


def split(df: pd.DataFrame):
    return (df[df["MONTH"].isin(TRAIN_MONTHS)].reset_index(drop=True),
            df[df["MONTH"].isin(TEST_MONTHS)].reset_index(drop=True))
