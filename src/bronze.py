"""
Paso 2: capa Bronze (ingesta cruda).

- Lee los 12 CSV con un schema explicito (sin inferSchema: inferir obliga a recorrer los
  datos dos veces y puede adivinar mal tipos).
- Todas las columnas como StringType: Bronze conserva el dato tal cual llego (por ejemplo
  '0659' en CRS_DEP_TIME o '2/1/2025 12:00:00 AM' en FL_DATE). Los tipos se corrigen en Silver.
- mode=FAILFAST: si una fila no trae las 31 columnas la ingesta falla, en vez de rellenar
  con nulos en silencio.
- Solo se agregan dos columnas de linaje (_source_file, _ingested_at); no se modifica ningun
  valor original.
- Escribe Parquet particionado por MONTH en data/bronze/flights.
- Descarga ademas el catalogo de aeropuertos de OpenFlights a data/raw/airports.

Uso: python -m src.bronze
"""
import os
import sys

from pyspark.sql import functions as F
from pyspark.sql.types import StringType, StructField, StructType

from src.config import BRONZE_DIR, EXPECTED_HEADER, RAW_AIRPORTS_DIR, RAW_FLIGHTS_DIR
from src.spark_session import get_spark
from src.utils import http_get, timed, update_stage_metrics

BRONZE_FLIGHTS = os.path.join(BRONZE_DIR, "flights")
AIRPORTS_URL = "https://raw.githubusercontent.com/jpatokal/openflights/master/data/airports.dat"
AIRPORTS_PATH = os.path.join(RAW_AIRPORTS_DIR, "airports.dat")

FLIGHTS_SCHEMA = StructType([StructField(c, StringType(), True) for c in EXPECTED_HEADER])


def download_airports(force: bool = False) -> str:
    """Descarga airports.dat de OpenFlights (se omite si ya existe)."""
    if os.path.exists(AIRPORTS_PATH) and not force:
        print(f"airports.dat ya existe ({os.path.getsize(AIRPORTS_PATH)/1024:.0f} KB), se reutiliza")
        return AIRPORTS_PATH
    os.makedirs(RAW_AIRPORTS_DIR, exist_ok=True)
    r = http_get(AIRPORTS_URL)
    tmp = AIRPORTS_PATH + ".part"
    with open(tmp, "wb") as f:
        f.write(r.content)
    os.replace(tmp, AIRPORTS_PATH)
    n = r.content.count(b"\n")
    print(f"airports.dat descargado: {len(r.content)/1024:.0f} KB, {n:,} aeropuertos")
    return AIRPORTS_PATH


def ingest_flights(spark) -> dict:
    files = sorted(os.path.join(RAW_FLIGHTS_DIR, f) for f in os.listdir(RAW_FLIGHTS_DIR)
                   if f.startswith("vuelos_") and f.endswith(".csv"))
    if len(files) != 12:
        raise RuntimeError(f"Se esperaban 12 CSV en {RAW_FLIGHTS_DIR} y hay {len(files)}. "
                           "Corre primero: python -m src.organize_data")

    df = (spark.read
          .option("header", "true")
          .option("mode", "FAILFAST")
          .option("quote", '"')
          .option("escape", '"')
          .schema(FLIGHTS_SCHEMA)
          .csv(files)
          .withColumn("_source_file", F.regexp_extract(F.input_file_name(), r"([^/]+)$", 1))
          .withColumn("_ingested_at", F.current_timestamp()))

    # overwrite: la ingesta es idempotente, re-correrla reemplaza Bronze completo.
    (df.write.mode("overwrite")
       .partitionBy("MONTH")
       .parquet(BRONZE_FLIGHTS))

    # Verificacion leyendo lo escrito (no el CSV): confirma que Bronze quedo completo.
    bronze = spark.read.parquet(BRONZE_FLIGHTS)
    per_month = (bronze.groupBy("MONTH").count()
                 .withColumn("m", F.col("MONTH").cast("int")).orderBy("m").collect())
    return {"per_month": {int(r["MONTH"]): r["count"] for r in per_month},
            "total": sum(r["count"] for r in per_month)}


def main():
    download_airports()
    spark = get_spark("bronze")
    try:
        with timed("ingesta bronze") as t:
            res = ingest_flights(spark)
    except Exception as e:
        print(f"ERROR en Bronze: {e}")
        sys.exit(1)

    print("\nFilas por mes en Bronze:")
    for m, c in res["per_month"].items():
        print(f"  {m:02d}: {c:>10,}")
    print(f"  TOTAL: {res['total']:>8,}")
    print(f"Tiempo de ingesta: {t['seconds']:.1f} s")

    update_stage_metrics("bronze", seconds=t["seconds"], rows=res["total"],
                         rows_per_month={f"{m:02d}": c for m, c in res["per_month"].items()})
    spark.stop()


if __name__ == "__main__":
    main()
