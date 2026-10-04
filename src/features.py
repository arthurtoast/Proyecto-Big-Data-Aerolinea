"""
Variables de entrada del modelo (compartidas por entrenamiento, scoring y predict()).

REGLA DE FUGA DE DATOS: la app predice ANTES del vuelo. Solo se usan variables conocidas de
antemano: mes, dia de semana, hora programada, aerolinea, origen, destino, distancia y clima.
Nunca DEP_DELAY, DEP_DEL15, ARR_DELAY ni las columnas de causa (CARRIER_DELAY, etc.).
ARR_DEL15 y ARR_DELAY solo aparecen como etiquetas (label), nunca como entrada.

Nota: el clima usado es el observado del dia (Open-Meteo archive). En produccion la app usaria
el pronostico de ese dia, que es la version "conocida de antemano" de esta misma variable.
"""
import math

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from src.silver import BUSINESS_KEY, SILVER_DISRUPTED, SILVER_FLIGHTS, SILVER_WEATHER

CATEGORICAL = ["OP_UNIQUE_CARRIER", "ORIGIN", "DEST"]
# Dia de semana y hora: categoricos para la regresion logistica (efecto no lineal),
# numericos para los arboles (pueden cortar donde convenga).
CYCLIC_SMALL = ["DAY_OF_WEEK", "dep_hour"]
WEATHER_NUM = ["precip_mm", "snow_cm", "wind_max_kmh", "temp_max_c", "temp_min_c"]
WEATHER_FLAGS = ["rain_flag", "snow_flag", "high_wind_flag", "has_weather"]
NUMERIC = ["month_sin", "month_cos", "is_weekend", "DISTANCE"] + WEATHER_NUM + WEATHER_FLAGS

# Columnas prohibidas como entrada (se valida en el entrenamiento).
LEAKAGE_COLS = ["DEP_DELAY", "DEP_DEL15", "ARR_DELAY", "ARR_DEL15", "CARRIER_DELAY",
                "WEATHER_DELAY", "NAS_DELAY", "SECURITY_DELAY", "LATE_AIRCRAFT_DELAY"]
INPUT_COLS = CATEGORICAL + CYCLIC_SMALL + NUMERIC


def add_model_columns(df: DataFrame) -> DataFrame:
    """Mes ciclico y relleno de clima. Espera MONTH, DAY_OF_WEEK, dep_hour, DISTANCE y las
    columnas de clima (posiblemente nulas)."""
    angle = 2 * math.pi * (F.col("MONTH") - 1) / 12
    out = (df.withColumn("month_sin", F.sin(angle))
             .withColumn("month_cos", F.cos(angle))
             .withColumn("is_weekend", F.col("DAY_OF_WEEK").isin(6, 7).cast("int"))
             .withColumn("has_weather", F.coalesce(F.col("has_weather"), F.lit(0)).cast("int")))
    # Sin clima -> 0 en variables y banderas; has_weather=0 le dice al modelo que ese 0 no
    # es "dia despejado" sino "dato no disponible".
    return out.fillna(0.0, subset=WEATHER_NUM).fillna(0, subset=WEATHER_FLAGS)


def weather_table(spark) -> DataFrame:
    return (spark.read.parquet(SILVER_WEATHER)
            .select(F.col("airport").alias("ORIGIN"), F.col("date").alias("FL_DATE"),
                    *WEATHER_NUM, "rain_flag", "snow_flag", "high_wind_flag",
                    F.lit(1).alias("has_weather")))


def build_features(spark) -> DataFrame:
    """Todos los vuelos (completados y no completados) con vuelo_key, variables de entrada,
    etiquetas y bandera is_completed. vuelo_key se calcula igual que en Gold."""
    completed = spark.read.parquet(SILVER_FLIGHTS).withColumn("is_completed", F.lit(1))
    disrupted = spark.read.parquet(SILVER_DISRUPTED).withColumn("is_completed", F.lit(0))
    base_cols = list(dict.fromkeys(BUSINESS_KEY + ["MONTH", "DAY_OF_WEEK", "dep_hour", "DISTANCE",
                                                   "ARR_DEL15", "ARR_DELAY", "is_completed"]))
    flights = completed.select(*base_cols).unionByName(disrupted.select(*base_cols))
    df = (flights.withColumn("vuelo_key", F.xxhash64(*BUSINESS_KEY))
          .join(F.broadcast(weather_table(spark)), ["ORIGIN", "FL_DATE"], "left"))
    return add_model_columns(df)
