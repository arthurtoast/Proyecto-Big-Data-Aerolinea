"""
Paso 4: capa Silver (limpieza y enriquecimiento).

Reglas:
- Tipos: enteros para calendario y banderas, double para minutos y distancia, date para
  FL_DATE ('2/1/2025 12:00:00 AM' -> 2025-02-01). Se usa double (no decimal) porque
  decimal llega a pandas como object y es mas lento.
- Nulos, politica explicita:
    * columnas clave (fecha, aerolinea, origen, destino, hora programada, distancia,
      banderas CANCELLED/DIVERTED) nulas o invalidas -> se descarta la fila (se reporta).
    * vuelos completados sin ARR_DELAY o ARR_DEL15 -> se descarta (no tienen etiqueta).
    * columnas de causa (CARRIER_DELAY, etc.): el BTS solo las llena cuando el vuelo llego
      con 15+ min de retraso; en completados se rellenan con 0.0 (sin retraso atribuido).
    * DEP_DELAY/DEP_DEL15 nulos en completados se dejan nulos: no se usan para modelar.
    * clima: nulo cuando el aeropuerto de origen no esta en el top N; has_weather lo indica.
- Duplicados: por clave de negocio (FL_DATE, OP_UNIQUE_CARRIER, OP_CARRIER_FL_NUM,
  ORIGIN, DEST, CRS_DEP_TIME).
- Cancelados y desviados -> data/silver/flights_disrupted (analisis).
  Completados -> data/silver/flights (modelado), con clima de ORIGEN por (aeropuerto, fecha).
- Derivadas: dep_hour (de CRS_DEP_TIME; 2400 se trata como 0), is_weekend (BTS: 1=lunes,
  6=sabado, 7=domingo), route, y banderas de clima:
    rain = precipitation_sum >= 1 mm, snow = snowfall_sum > 0, high_wind = viento >= 30 km/h.

Uso: python -m src.silver
"""
import os
import sys

from pyspark import StorageLevel
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from src.bronze import BRONZE_FLIGHTS
from src.config import SILVER_DIR
from src.spark_session import get_spark
from src.utils import timed, update_stage_metrics
from src.weather import WEATHER_DAILY_CSV

SILVER_FLIGHTS = os.path.join(SILVER_DIR, "flights")
SILVER_DISRUPTED = os.path.join(SILVER_DIR, "flights_disrupted")
SILVER_WEATHER = os.path.join(SILVER_DIR, "weather_daily")

# Umbrales de banderas de clima (justificados con la distribucion real del Paso 3).
RAIN_MM = 1.0
SNOW_CM = 0.0
HIGH_WIND_KMH = 30.0

BUSINESS_KEY = ["FL_DATE", "OP_UNIQUE_CARRIER", "OP_CARRIER_FL_NUM", "ORIGIN", "DEST",
                "CRS_DEP_TIME"]
INT_COLS = ["YEAR", "MONTH", "DAY_OF_MONTH", "DAY_OF_WEEK", "OP_CARRIER_FL_NUM",
            "DISTANCE_GROUP"]
# Banderas vienen como '0.00'/'1.00': double primero y luego int.
FLAG_COLS = ["DEP_DEL15", "ARR_DEL15", "CANCELLED", "DIVERTED"]
DOUBLE_COLS = ["DEP_DELAY", "ARR_DELAY", "DISTANCE"]
CAUSE_COLS = ["CARRIER_DELAY", "WEATHER_DELAY", "NAS_DELAY", "SECURITY_DELAY",
              "LATE_AIRCRAFT_DELAY"]


def cast_types(bronze: DataFrame) -> DataFrame:
    df = bronze.drop("_ingested_at")
    for c in INT_COLS:
        df = df.withColumn(c, F.col(c).cast("int"))
    for c in FLAG_COLS:
        df = df.withColumn(c, F.col(c).cast("double").cast("int"))
    for c in DOUBLE_COLS + CAUSE_COLS:
        df = df.withColumn(c, F.col(c).cast("double"))
    # Solo la parte de fecha: '2/1/2025 12:00:00 AM' -> '2/1/2025'.
    df = df.withColumn("FL_DATE", F.to_date(F.split("FL_DATE", " ").getItem(0), "M/d/yyyy"))
    # Horas hhmm como int (0659 -> 659); strings vacios -> null.
    for c in ["CRS_DEP_TIME", "CRS_ARR_TIME"]:
        df = df.withColumn(c, F.when(F.trim(c) != "", F.col(c).cast("int")))
    for c in ["OP_UNIQUE_CARRIER", "ORIGIN", "DEST", "CANCELLATION_CODE"]:
        df = df.withColumn(c, F.when(F.trim(c) != "", F.trim(c)))
    return df


def invalid_key_condition():
    """Filas sin datos minimos para ser un vuelo util."""
    return (
        F.col("FL_DATE").isNull()
        | F.col("OP_UNIQUE_CARRIER").isNull()
        | F.col("ORIGIN").isNull() | F.col("DEST").isNull()
        | F.col("CRS_DEP_TIME").isNull() | ~F.col("CRS_DEP_TIME").between(0, 2400)
        | F.col("DISTANCE").isNull() | (F.col("DISTANCE") <= 0)
        | F.col("CANCELLED").isNull() | F.col("DIVERTED").isNull()
        # Coherencia: la fecha debe coincidir con YEAR/MONTH/DAY_OF_MONTH.
        | (F.year("FL_DATE") != F.col("YEAR"))
        | (F.month("FL_DATE") != F.col("MONTH"))
        | (F.dayofmonth("FL_DATE") != F.col("DAY_OF_MONTH"))
    )


def add_derived(df: DataFrame) -> DataFrame:
    return (df
            .withColumn("dep_hour", (F.floor(F.col("CRS_DEP_TIME") / 100) % 24).cast("int"))
            .withColumn("is_weekend", F.col("DAY_OF_WEEK").isin(6, 7).cast("int"))
            .withColumn("route", F.concat_ws("-", "ORIGIN", "DEST")))


def load_weather(spark) -> DataFrame:
    if not os.path.exists(WEATHER_DAILY_CSV):
        raise FileNotFoundError(f"No existe {WEATHER_DAILY_CSV}. Corre: python -m src.weather")
    w = (spark.read.option("header", "true").csv(WEATHER_DAILY_CSV)
         .select(F.col("airport").alias("w_airport"),
                 F.to_date("date").alias("w_date"),
                 F.col("precipitation_sum").cast("double").alias("precip_mm"),
                 F.col("snowfall_sum").cast("double").alias("snow_cm"),
                 F.col("wind_speed_10m_max").cast("double").alias("wind_max_kmh"),
                 F.col("temperature_2m_max").cast("double").alias("temp_max_c"),
                 F.col("temperature_2m_min").cast("double").alias("temp_min_c"))
         .dropDuplicates(["w_airport", "w_date"]))
    return (w
            .withColumn("rain_flag", (F.col("precip_mm") >= RAIN_MM).cast("int"))
            .withColumn("snow_flag", (F.col("snow_cm") > SNOW_CM).cast("int"))
            .withColumn("high_wind_flag", (F.col("wind_max_kmh") >= HIGH_WIND_KMH).cast("int")))


def build_silver(spark) -> dict:
    stats = {}
    bronze = spark.read.parquet(BRONZE_FLIGHTS)
    typed = cast_types(bronze).persist(StorageLevel.MEMORY_AND_DISK)
    stats["bronze_rows"] = typed.count()

    # Diagnostico: DAY_OF_WEEK del BTS (1=lunes) vs calculado desde la fecha.
    stats["dow_mismatch"] = typed.filter(
        ((F.dayofweek("FL_DATE") + 5) % 7 + 1) != F.col("DAY_OF_WEEK")).count()

    invalid = invalid_key_condition()
    stats["dropped_invalid_key"] = typed.filter(invalid).count()
    valid = typed.filter(~invalid)

    deduped = valid.dropDuplicates(BUSINESS_KEY)
    n_valid = stats["bronze_rows"] - stats["dropped_invalid_key"]
    n_dedup = deduped.count()
    stats["dropped_duplicates"] = n_valid - n_dedup

    deduped = add_derived(deduped)
    is_disrupted = (F.col("CANCELLED") == 1) | (F.col("DIVERTED") == 1)

    # Cancelados y desviados: tal cual, con sus derivadas, para analisis.
    disrupted = deduped.filter(is_disrupted)
    disrupted.write.mode("overwrite").partitionBy("MONTH").parquet(SILVER_DISRUPTED)
    disrupted = spark.read.parquet(SILVER_DISRUPTED)
    stats["disrupted_rows"] = disrupted.count()
    stats["cancelled_rows"] = disrupted.filter("CANCELLED = 1").count()
    stats["diverted_rows"] = disrupted.filter("DIVERTED = 1").count()

    completed = deduped.filter(~is_disrupted)
    no_label = F.col("ARR_DELAY").isNull() | F.col("ARR_DEL15").isNull()
    stats["dropped_completed_no_label"] = completed.filter(no_label).count()
    # CANCELLATION_CODE solo aplica a cancelados: en completados es 100% nula, se descarta.
    completed = (completed.filter(~no_label).fillna(0.0, subset=CAUSE_COLS)
                 .drop("CANCELLATION_CODE"))

    weather = load_weather(spark)
    weather.withColumnRenamed("w_airport", "airport").withColumnRenamed("w_date", "date") \
        .write.mode("overwrite").parquet(SILVER_WEATHER)

    # Left join: el vuelo se conserva aunque su origen no tenga clima.
    # broadcast: la tabla de clima (~18 mil filas) cabe en memoria, evita un shuffle de 5M filas.
    enriched = (completed
                .join(F.broadcast(weather),
                      (completed.ORIGIN == weather.w_airport) & (completed.FL_DATE == weather.w_date),
                      "left")
                .withColumn("has_weather", F.col("w_airport").isNotNull().cast("int"))
                .drop("w_airport", "w_date"))

    enriched.write.mode("overwrite").partitionBy("MONTH").parquet(SILVER_FLIGHTS)
    silver = spark.read.parquet(SILVER_FLIGHTS)
    stats["silver_rows"] = silver.count()
    stats["silver_with_weather"] = silver.filter("has_weather = 1").count()
    stats["weather_coverage_pct"] = round(100 * stats["silver_with_weather"] / stats["silver_rows"], 2)
    stats["delay_rate_pct"] = round(100 * silver.agg(F.avg("ARR_DEL15")).first()[0], 2)

    # Nulos restantes en Silver (todo nulo debe estar explicado por la politica).
    null_counts = silver.select([F.sum(F.col(c).isNull().cast("int")).alias(c)
                                 for c in silver.columns]).first().asDict()
    stats["remaining_nulls"] = {k: v for k, v in null_counts.items() if v}

    flags = silver.filter("has_weather = 1").agg(
        *[F.round(100 * F.avg(c), 2).alias(c) for c in ["rain_flag", "snow_flag", "high_wind_flag"]]
    ).first().asDict()
    stats["weather_flag_pct"] = flags
    typed.unpersist()
    return stats


def main():
    spark = get_spark("silver")
    try:
        with timed("silver") as t:
            s = build_silver(spark)
    except Exception as e:
        print(f"ERROR en Silver: {e}")
        sys.exit(1)

    print("\n=== Silver ===")
    print(f"Filas Bronze (antes):                 {s['bronze_rows']:>10,}")
    print(f"  - clave invalida o nula:            {s['dropped_invalid_key']:>10,}")
    print(f"  - duplicados por clave de negocio:  {s['dropped_duplicates']:>10,}")
    print(f"  - cancelados + desviados (aparte):  {s['disrupted_rows']:>10,} "
          f"(cancelados {s['cancelled_rows']:,}, desviados {s['diverted_rows']:,})")
    print(f"  - completados sin etiqueta:         {s['dropped_completed_no_label']:>10,}")
    print(f"Filas Silver completados (despues):   {s['silver_rows']:>10,}")
    print(f"Con clima de origen:                  {s['silver_with_weather']:>10,} "
          f"({s['weather_coverage_pct']}%)")
    print(f"Tasa de retraso ARR_DEL15:            {s['delay_rate_pct']}%")
    print(f"% dias-vuelo con lluvia/nieve/viento: {s['weather_flag_pct']}")
    print(f"DAY_OF_WEEK incoherente con fecha:    {s['dow_mismatch']:,}")
    print(f"Nulos restantes: {s['remaining_nulls']}")

    update_stage_metrics("silver", seconds=t["seconds"], **s)
    spark.stop()


if __name__ == "__main__":
    main()
