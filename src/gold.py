"""
Paso 5: capa Gold (esquema estrella).

Grano de FACT_VUELOS: un vuelo programado (completados, cancelados y desviados), para que
BI pueda analizar tambien cancelaciones. El modelo (Paso 6) entrena solo con completados.

Dimensiones:
- DIM_TIEMPO     (fecha_key = yyyymmdd): calendario completo 2025, con feriados federales.
- DIM_AEROLINEA  (aerolinea_key): codigo BTS, nombre y tipo (principal, bajo costo, regional).
- DIM_AEROPUERTO (aeropuerto_key): una sola dimension, usada como origen y destino
                 (role-playing): FACT_VUELOS tiene origen_key y destino_key.
- DIM_CLIMA      (clima_key): un registro por aeropuerto y dia, con miembro -1 "sin_datos"
                 para vuelos cuyo origen no tiene clima (evita llaves foraneas nulas).

vuelo_key es xxhash64 de la clave de negocio: estable entre corridas, asi las predicciones
del Paso 6 se pueden volver a unir al hecho sin depender del orden de las filas.

Uso: python -m src.gold [--skip-mysql]
"""
import argparse
import os
import sys

import pandas as pd
from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

from src.config import GOLD_DIR, YEAR
from src.silver import (BUSINESS_KEY, SILVER_DISRUPTED, SILVER_FLIGHTS, SILVER_WEATHER)
from src.spark_session import get_spark
from src.utils import timed, update_stage_metrics
from src.weather import load_openflights

GOLD_PATHS = {name: os.path.join(GOLD_DIR, name.lower()) for name in
              ["FACT_VUELOS", "DIM_TIEMPO", "DIM_AEROLINEA", "DIM_AEROPUERTO", "DIM_CLIMA"]}

# Nombres de las aerolineas presentes en los datos 2025 (catalogo de transportistas del BTS).
CARRIERS = {
    "WN": ("Southwest Airlines", "bajo_costo"),
    "DL": ("Delta Air Lines", "principal"),
    "AA": ("American Airlines", "principal"),
    "OO": ("SkyWest Airlines", "regional"),
    "UA": ("United Airlines", "principal"),
    "YX": ("Republic Airways", "regional"),
    "MQ": ("Envoy Air", "regional"),
    "OH": ("PSA Airlines", "regional"),
    "AS": ("Alaska Airlines", "principal"),
    "B6": ("JetBlue Airways", "bajo_costo"),
    "F9": ("Frontier Airlines", "bajo_costo"),
    "NK": ("Spirit Airlines", "bajo_costo"),
    "G4": ("Allegiant Air", "bajo_costo"),
    "HA": ("Hawaiian Airlines", "principal"),
}

# Feriados federales de EE. UU. en 2025 (se conocen de antemano: utiles para BI y modelo).
US_HOLIDAYS_2025 = ["2025-01-01", "2025-01-20", "2025-02-17", "2025-05-26", "2025-06-19",
                    "2025-07-04", "2025-09-01", "2025-10-13", "2025-11-11", "2025-11-27",
                    "2025-12-25"]
MESES = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto",
         "septiembre", "octubre", "noviembre", "diciembre"]
DIAS = ["lunes", "martes", "miercoles", "jueves", "viernes", "sabado", "domingo"]

FACT_MEASURES = ["DEP_DELAY", "DEP_DEL15", "ARR_DELAY", "ARR_DEL15", "CARRIER_DELAY",
                 "WEATHER_DELAY", "NAS_DELAY", "SECURITY_DELAY", "LATE_AIRCRAFT_DELAY"]


def date_key(col):
    return F.date_format(col, "yyyyMMdd").cast("int")


def build_dim_tiempo(spark) -> DataFrame:
    df = spark.sql(f"SELECT explode(sequence(DATE'{YEAR}-01-01', DATE'{YEAR}-12-31')) AS fecha")
    meses = F.array(*[F.lit(m) for m in MESES])
    dias = F.array(*[F.lit(d) for d in DIAS])
    dow = ((F.dayofweek("fecha") + 5) % 7 + 1)  # 1=lunes ... 7=domingo, igual que el BTS
    return (df.select(
        date_key("fecha").alias("fecha_key"),
        "fecha",
        F.year("fecha").alias("anio"),
        F.quarter("fecha").alias("trimestre"),
        F.month("fecha").alias("mes"),
        meses[F.month("fecha") - 1].alias("nombre_mes"),
        F.dayofmonth("fecha").alias("dia"),
        dow.alias("dia_semana"),
        dias[dow - 1].alias("nombre_dia"),
        F.weekofyear("fecha").alias("semana_anio"),
        dow.isin(6, 7).cast("int").alias("es_fin_semana"),
        F.col("fecha").isin(*[F.lit(d).cast("date") for d in US_HOLIDAYS_2025])
         .cast("int").alias("es_feriado"),
    ))


def build_dim_aerolinea(spark, flights: DataFrame) -> DataFrame:
    codes = sorted(r[0] for r in flights.select("OP_UNIQUE_CARRIER").distinct().collect())
    rows = [(i + 1, c, *CARRIERS.get(c, (c, "desconocido"))) for i, c in enumerate(codes)]
    unknown = [c for c in codes if c not in CARRIERS]
    if unknown:
        print(f"AVISO: aerolineas sin nombre en el catalogo: {unknown}")
    return spark.createDataFrame(rows, "aerolinea_key int, codigo string, nombre string, tipo string")


def build_dim_aeropuerto(spark, flights: DataFrame, weather_airports: set) -> DataFrame:
    # Ciudad y estado desde el propio BTS (la grafia mas frecuente por codigo).
    side = lambda p: flights.select(F.col(p).alias("iata"),
                                    F.col(f"{p}_CITY_NAME").alias("ciudad"),
                                    F.col(f"{p}_STATE_ABR").alias("estado"))
    bts = (side("ORIGIN").unionByName(side("DEST"))
           .groupBy("iata", "ciudad", "estado").count()
           .withColumn("rn", F.row_number().over(Window.partitionBy("iata").orderBy(F.desc("count"))))
           .filter("rn = 1").drop("rn", "count").toPandas())

    cat = load_openflights()[["name", "lat", "lon", "tz"]].rename(
        columns={"name": "nombre", "lat": "latitud", "lon": "longitud", "tz": "zona_horaria"})
    dim = bts.merge(cat, left_on="iata", right_index=True, how="left").sort_values("iata")
    missing = dim[dim["nombre"].isna()]["iata"].tolist()
    if missing:
        print(f"AVISO: aeropuertos sin datos en OpenFlights (sin coordenadas): {missing}")
    dim["nombre"] = dim["nombre"].fillna(dim["ciudad"])
    dim["tiene_clima"] = dim["iata"].isin(weather_airports).astype("int32")
    dim.insert(0, "aeropuerto_key", range(1, len(dim) + 1))
    dim["aeropuerto_key"] = dim["aeropuerto_key"].astype("int32")
    dim = dim.astype({"latitud": "float64", "longitud": "float64"})
    # Pandas usa NaN; Spark necesita None para nulos en string.
    dim = dim.astype(object).where(dim.notna(), None)
    return spark.createDataFrame(
        dim[["aeropuerto_key", "iata", "nombre", "ciudad", "estado", "latitud", "longitud",
             "zona_horaria", "tiene_clima"]],
        "aeropuerto_key int, iata string, nombre string, ciudad string, estado string, "
        "latitud double, longitud double, zona_horaria string, tiene_clima int")


def build_dim_clima(spark, dim_aeropuerto: DataFrame) -> DataFrame:
    w = spark.read.parquet(SILVER_WEATHER)
    condicion = (F.when(F.col("snow_flag") == 1, "nieve")
                 .when(F.col("rain_flag") == 1, "lluvia")
                 .when(F.col("high_wind_flag") == 1, "viento_fuerte")
                 .otherwise("despejado"))
    dim = (w.join(dim_aeropuerto.select("aeropuerto_key", F.col("iata").alias("airport")), "airport")
           .select(
               F.row_number().over(Window.orderBy("airport", "date")).alias("clima_key"),
               "aeropuerto_key",
               F.col("airport").alias("iata"),
               date_key("date").alias("fecha_key"),
               "precip_mm", "snow_cm", "wind_max_kmh", "temp_max_c", "temp_min_c",
               F.col("rain_flag").alias("lluvia"), F.col("snow_flag").alias("nieve"),
               F.col("high_wind_flag").alias("viento_fuerte"),
               condicion.alias("condicion")))
    # Miembro desconocido: los vuelos sin clima apuntan a -1 en vez de a NULL.
    unknown = spark.createDataFrame(
        [(-1, None, None, None, None, None, None, None, None, None, None, None, "sin_datos")],
        dim.schema)
    return unknown.unionByName(dim)


def build_fact(silver_all: DataFrame, dims: dict) -> DataFrame:
    aer = dims["DIM_AEROLINEA"].select("aerolinea_key", F.col("codigo").alias("OP_UNIQUE_CARRIER"))
    ori = dims["DIM_AEROPUERTO"].select(F.col("aeropuerto_key").alias("origen_key"),
                                        F.col("iata").alias("ORIGIN"))
    des = dims["DIM_AEROPUERTO"].select(F.col("aeropuerto_key").alias("destino_key"),
                                        F.col("iata").alias("DEST"))
    cli = dims["DIM_CLIMA"].filter("clima_key > 0").select(
        "clima_key", F.col("iata").alias("ORIGIN"), "fecha_key")

    f = (silver_all
         .withColumn("vuelo_key", F.xxhash64(*BUSINESS_KEY))
         .withColumn("fecha_key", date_key("FL_DATE"))
         .join(F.broadcast(aer), "OP_UNIQUE_CARRIER")
         .join(F.broadcast(ori), "ORIGIN")
         .join(F.broadcast(des), "DEST")
         .join(F.broadcast(cli), ["ORIGIN", "fecha_key"], "left")
         .fillna({"clima_key": -1}))
    return f.select(
        "vuelo_key", "fecha_key", "aerolinea_key", "origen_key", "destino_key", "clima_key",
        "OP_CARRIER_FL_NUM", "CRS_DEP_TIME", "dep_hour", "CRS_ARR_TIME", "DISTANCE",
        "DISTANCE_GROUP", "CANCELLED", "CANCELLATION_CODE", "DIVERTED", *FACT_MEASURES,
        F.lit(None).cast("double").alias("pred_prob"),   # se llenan en el Paso 6
        F.lit(None).cast("double").alias("pred_min"),
        "MONTH")


def build_gold(spark) -> dict:
    completed = spark.read.parquet(SILVER_FLIGHTS)
    disrupted = spark.read.parquet(SILVER_DISRUPTED)
    # Completados no traen CANCELLATION_CODE (en ellos es siempre nula): se agrega como nula.
    # Las columnas de clima de completados no se usan aqui; el hecho apunta a DIM_CLIMA.
    completed = completed.withColumn("CANCELLATION_CODE", F.lit(None).cast("string"))
    cols = [c for c in disrupted.columns if c in completed.columns]
    silver_all = completed.select(*cols).unionByName(disrupted.select(*cols))

    weather_airports = {r[0] for r in spark.read.parquet(SILVER_WEATHER)
                        .select("airport").distinct().collect()}
    dims = {"DIM_TIEMPO": build_dim_tiempo(spark),
            "DIM_AEROLINEA": build_dim_aerolinea(spark, silver_all),
            "DIM_AEROPUERTO": build_dim_aeropuerto(spark, silver_all, weather_airports)}
    dims["DIM_CLIMA"] = build_dim_clima(spark, dims["DIM_AEROPUERTO"])

    counts = {}
    for name, df in dims.items():
        df.write.mode("overwrite").parquet(GOLD_PATHS[name])
        counts[name] = spark.read.parquet(GOLD_PATHS[name]).count()
    # Se releen las dimensiones ya escritas para que el hecho use exactamente esas llaves.
    dims = {n: spark.read.parquet(GOLD_PATHS[n]) for n in dims}

    fact = build_fact(silver_all, dims)
    fact.write.mode("overwrite").partitionBy("MONTH").parquet(GOLD_PATHS["FACT_VUELOS"])
    fact = spark.read.parquet(GOLD_PATHS["FACT_VUELOS"])
    counts["FACT_VUELOS"] = fact.count()

    # Integridad: el hecho no debe perder filas en los joins ni duplicar llaves.
    checks = {
        "fact_rows_expected": silver_all.count(),
        "fact_distinct_vuelo_key": fact.select("vuelo_key").distinct().count(),
        "fact_with_weather": fact.filter("clima_key > 0").count(),
        "fact_completed": fact.filter("CANCELLED = 0 AND DIVERTED = 0").count(),
    }
    return {"counts": counts, "checks": checks}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-mysql", action="store_true")
    args = ap.parse_args()

    spark = get_spark("gold")
    try:
        with timed("gold (parquet)") as t:
            res = build_gold(spark)
    except Exception as e:
        print(f"ERROR en Gold: {str(e).splitlines()[0][:300]}")
        sys.exit(1)

    c, k = res["counts"], res["checks"]
    print("\n=== Gold (data/gold) ===")
    for name, n in c.items():
        print(f"  {name:<15} {n:>10,} filas")
    print(f"  Filas esperadas en el hecho (Silver completados + no completados): {k['fact_rows_expected']:,}")
    print(f"  vuelo_key distintos: {k['fact_distinct_vuelo_key']:,} "
          f"({'OK, sin colisiones' if k['fact_distinct_vuelo_key'] == c['FACT_VUELOS'] else 'COLISION'})")
    print(f"  Vuelos con clima_key valido: {k['fact_with_weather']:,} "
          f"({100 * k['fact_with_weather'] / c['FACT_VUELOS']:.2f}%)")
    update_stage_metrics("gold", seconds=t["seconds"], table_rows=c, **k)

    if not args.skip_mysql:
        from src.mysql_loader import load_gold_to_mysql
        load_gold_to_mysql()
    spark.stop()


if __name__ == "__main__":
    main()
