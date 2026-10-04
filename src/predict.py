"""
predict(aerolinea, origen, destino, fecha, hora): probabilidad de retraso y minutos estimados
para un vuelo, usando los modelos entrenados en el Paso 6.

- aerolinea: codigo BTS ('AA') o nombre ('American Airlines').
- origen/destino: codigo IATA ('ORD').
- fecha: 'YYYY-MM-DD' o datetime.date.  hora: 17 o '17:45' (hora local programada de salida).
- Distancia: mediana historica de la ruta; si la ruta no existe, distancia en linea recta
  (haversine) con las coordenadas de DIM_AEROPUERTO.
- Clima: el del aeropuerto de origen ese dia si existe en Silver (en produccion seria el
  pronostico); si no, has_weather=0 y el modelo usa solo las demas variables.

Uso: python -m src.predict  (imprime ejemplos)
"""
import json
import math
from datetime import date, datetime

from pyspark.ml import PipelineModel
from pyspark.ml.functions import vector_to_array
from pyspark.sql import Row
from pyspark.sql import functions as F

from src.features import INPUT_COLS, WEATHER_NUM, add_model_columns, weather_table
from src.gold import GOLD_PATHS
from src.silver import SILVER_FLIGHTS
from src.spark_session import get_spark
from src.train import CLIP_PATH, MODEL_PATHS, prior_corrected
from src.utils import STAGE_METRICS_PATH

_cache = {}


def _load(spark):
    """Carga modelos y catalogos una sola vez (las siguientes llamadas son mas rapidas)."""
    if _cache:
        return _cache
    with open(CLIP_PATH) as f:
        cfg = json.load(f)
    with open(STAGE_METRICS_PATH) as f:
        stage = json.load(f)
    best = stage.get("models", {}).get("best_regressor", "gbt")
    aer = spark.read.parquet(GOLD_PATHS["DIM_AEROLINEA"]).toPandas()
    apt = spark.read.parquet(GOLD_PATHS["DIM_AEROPUERTO"]).toPandas().set_index("iata")
    routes = (spark.read.parquet(SILVER_FLIGHTS).groupBy("ORIGIN", "DEST")
              .agg(F.percentile_approx("DISTANCE", 0.5).alias("d")).toPandas())
    _cache.update(
        cfg=cfg, best=best,
        clf=PipelineModel.load(MODEL_PATHS["logreg"]),
        reg=PipelineModel.load(MODEL_PATHS[best]),
        carriers={**{c.upper(): c for c in aer["codigo"]},
                  **{n.upper(): c for n, c in zip(aer["nombre"], aer["codigo"])}},
        carrier_names=dict(zip(aer["codigo"], aer["nombre"])),
        airports=apt,
        routes={(o, d): float(x) for o, d, x in zip(routes["ORIGIN"], routes["DEST"], routes["d"])},
        weather=weather_table(spark).cache(),
    )
    return _cache


def _haversine_miles(a, b):
    lat1, lon1, lat2, lon2 = map(math.radians, [a["latitud"], a["longitud"], b["latitud"], b["longitud"]])
    h = math.sin((lat2 - lat1) / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    return 2 * 3958.8 * math.asin(math.sqrt(h))


def predict(aerolinea: str, origen: str, destino: str, fecha, hora) -> dict:
    spark = get_spark("predict")
    c = _load(spark)

    carrier = c["carriers"].get(str(aerolinea).strip().upper())
    if carrier is None:
        raise ValueError(f"Aerolinea desconocida: {aerolinea}. Opciones: {sorted(c['carrier_names'])}")
    origen, destino = origen.strip().upper(), destino.strip().upper()
    for code in (origen, destino):
        if code not in c["airports"].index:
            raise ValueError(f"Aeropuerto desconocido: {code}")
    if isinstance(fecha, str):
        fecha = datetime.strptime(fecha, "%Y-%m-%d").date()
    if isinstance(hora, str):
        hora = int(hora.split(":")[0])
    if not (isinstance(fecha, date) and 0 <= int(hora) <= 23):
        raise ValueError("fecha debe ser 'YYYY-MM-DD' y hora entre 0 y 23")

    distance = c["routes"].get((origen, destino))
    distance_src = "mediana historica de la ruta"
    if distance is None:
        a, b = c["airports"].loc[origen], c["airports"].loc[destino]
        if any(math.isnan(x) for x in [a["latitud"], b["latitud"]]):
            raise ValueError("Ruta sin historial y sin coordenadas para estimar la distancia")
        distance, distance_src = round(_haversine_miles(a, b)), "linea recta (ruta sin historial)"

    row = spark.createDataFrame([Row(
        OP_UNIQUE_CARRIER=carrier, ORIGIN=origen, DEST=destino, FL_DATE=fecha,
        MONTH=fecha.month, DAY_OF_WEEK=fecha.isoweekday(),  # 1=lunes, igual que el BTS
        dep_hour=int(hora), DISTANCE=float(distance))])
    row = row.join(c["weather"], ["ORIGIN", "FL_DATE"], "left")
    row = add_model_columns(row)

    p = c["clf"].transform(row).select(
        prior_corrected(vector_to_array("probability")[1], c["cfg"]["w_pos"], c["cfg"]["w_neg"])
        .alias("p"), *INPUT_COLS).first()
    minutes = c["reg"].transform(row.select(*INPUT_COLS)).select("prediction").first()[0]

    return {
        "aerolinea": f"{carrier} ({c['carrier_names'][carrier]})",
        "ruta": f"{origen} -> {destino}",
        "fecha": fecha.isoformat(), "hora": int(hora),
        "prob_retraso": round(float(p["p"]), 3),
        "minutos_estimados": round(float(minutes), 1),
        "distancia_millas": distance, "distancia_fuente": distance_src,
        "clima_usado": ({k: p[k] for k in WEATHER_NUM} if p["has_weather"] else
                        "sin datos (origen fuera del top 50 o fecha fuera de 2025)"),
        "modelo_minutos": c["best"],
    }


if __name__ == "__main__":
    ejemplos = [
        ("AA", "ORD", "LGA", "2025-12-22", "17:45"),       # tarde, invierno, hub congestionado
        ("Delta Air Lines", "ATL", "MCO", "2025-04-15", 7),  # manana, primavera
        ("AS", "MCO", "ANC", "2025-07-04", 20),            # ruta sin historial: distancia haversine
        ("ZZ", "ORD", "LGA", "2025-06-01", 9),             # aerolinea invalida: error controlado
    ]
    for args in ejemplos:
        try:
            print(f"predict{args}")
            print(json.dumps(predict(*args), indent=2, ensure_ascii=False, default=str))
        except ValueError as e:
            print(f"  -> error de entrada: {e}")
