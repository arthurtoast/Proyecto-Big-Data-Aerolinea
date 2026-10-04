"""
Paso B (cierre Fase 3): tablas AGREGADAS en CSV para Power BI en deliverables/powerbi/.

- No se exportan los 7M de vuelos: se agregan en Spark sobre el esquema estrella de Gold.
- CSV UTF-8, separador coma, punto decimal, fechas YYYY-MM-DD, encabezados en espanol sin
  acentos (evita problemas de codificacion al importar), sin indice de pandas.
- Tasas en porcentaje (0 a 100); probabilidades por vuelo en 0 a 1.
- Todos los numeros salen de Gold y de deliverables/kpis.json (Paso A).

Uso: python -m src.export_powerbi   (requiere haber corrido python -m src.evaluate)
"""
import json
import os

import pandas as pd
from pyspark.sql import functions as F

from src.config import ROOT
from src.evaluate import KPIS_OUT, TEST_PREDS
from src.gold import GOLD_PATHS, MESES
from src.spark_session import get_spark

OUT_DIR = os.path.join(ROOT, "deliverables", "powerbi")
MAX_MB = 25
SAMPLE_N = 50_000
SEED = 42
NOMBRE_MES = [m.capitalize() for m in MESES]
NOMBRE_DIA = ["Lunes", "Martes", "Miercoles", "Jueves", "Viernes", "Sabado", "Domingo"]
CAUSAS = [("CARRIER_DELAY", "Aerolinea"), ("WEATHER_DELAY", "Clima"),
          ("NAS_DELAY", "Sistema aereo nacional (NAS)"), ("SECURITY_DELAY", "Seguridad"),
          ("LATE_AIRCRAFT_DELAY", "Avion tardio (vuelo anterior)")]
CONDICION = {"despejado": "Despejado", "lluvia": "Lluvia", "nieve": "Nieve",
             "viento_fuerte": "Viento fuerte", "sin_datos": "Sin datos de clima"}


def metrics():
    """Metricas estandar por grupo, sobre el hecho completo (incluye cancelados/desviados)."""
    done = (F.col("CANCELLED") == 0) & (F.col("DIVERTED") == 0)
    return [
        F.count("*").alias("vuelos_programados"),
        F.sum(F.when(done, 1).otherwise(0)).alias("vuelos_completados"),
        F.sum("CANCELLED").alias("vuelos_cancelados"),
        F.sum("DIVERTED").alias("vuelos_desviados"),
        F.sum(F.when(done, F.col("ARR_DEL15"))).alias("vuelos_con_retraso"),
        F.round(100 * F.avg(F.when(done, F.col("ARR_DEL15"))), 2).alias("tasa_retraso_pct"),
        F.round(F.avg(F.when(done, F.col("ARR_DELAY"))), 2).alias("retraso_promedio_min"),
        F.round(F.avg(F.when(done & (F.col("ARR_DEL15") == 1), F.col("ARR_DELAY"))), 2)
         .alias("retraso_promedio_vuelos_retrasados_min"),
        F.round(100 * F.avg("CANCELLED"), 2).alias("tasa_cancelacion_pct"),
        F.round(100 * F.avg(F.when(done, F.col("pred_prob"))), 2).alias("prob_retraso_predicha_promedio_pct"),
        F.round(F.avg(F.when(done, F.col("pred_min"))), 2).alias("minutos_predichos_promedio"),
    ]


def write(df: pd.DataFrame, name: str) -> dict:
    path = os.path.join(OUT_DIR, name)
    df.to_csv(path, index=False, encoding="utf-8", sep=",", decimal=".", date_format="%Y-%m-%d")
    mb = os.path.getsize(path) / 1024 ** 2
    if mb > MAX_MB:
        raise RuntimeError(f"{name} pesa {mb:.1f} MB (> {MAX_MB} MB)")
    return {"archivo": name, "filas": len(df), "columnas": df.shape[1], "mb": mb}


def kpis_resumen(k: dict) -> pd.DataFrame:
    ev, vol, tim, q = k["evaluacion"], k["volumen"], k["tiempos"], k["consultas_dashboard"]
    clf, reg = ev["clasificacion"], ev["regresion"]
    m, bg, br = reg["modelo_gbt"], reg["baseline_media_global"], reg["baseline_promedio_ruta"]
    lr = clf["modelo_regresion_logistica"]
    rows = [
        ("Vuelos procesados (Bronze)", vol["filas"]["bronze"], "vuelos", "Registros BTS 2025 ingeridos sin transformar"),
        ("Vuelos completados (Silver)", vol["filas"]["silver_completados"], "vuelos", "Vuelos que llegaron a destino, usados para modelar"),
        ("Vuelos cancelados o desviados", vol["filas"]["silver_cancelados_desviados"], "vuelos", "Separados en su propia tabla para analisis"),
        ("Filas en FACT_VUELOS (Gold)", vol["filas"]["gold"]["FACT_VUELOS"], "filas", "Hechos del esquema estrella, con predicciones"),
        ("Filas cargadas en MySQL (FACT_VUELOS)", vol["disco_mb"]["mysql"]["FACT_VUELOS"]["filas_exactas"]
         if isinstance(vol["disco_mb"]["mysql"], dict) else None, "filas", "Conteo exacto verificado con COUNT(*)"),
        ("Tamano CSV crudos", vol["disco_mb"]["raw_flights_csv"], "MB", "12 CSV mensuales del BTS"),
        ("Tamano Bronze (Parquet)", vol["disco_mb"]["bronze"], "MB", "Parquet snappy particionado por mes"),
        ("Tamano Silver (Parquet)", vol["disco_mb"]["silver"], "MB", "Completados + cancelados/desviados + clima"),
        ("Tamano Gold (Parquet)", vol["disco_mb"]["gold"], "MB", "Hecho + 4 dimensiones"),
        ("Cobertura de clima (vuelos completados)", vol["cobertura_clima_pct"]["vuelos_completados"], "%", "Vuelos cuyo origen esta en el top 50 con clima"),
        ("Aeropuertos con clima", vol["cobertura_clima_pct"]["aeropuertos_con_clima"], "aeropuertos", "Top 50 aeropuertos de origen por volumen"),
        ("Tasa de retraso global", round(100 * k["tasa_retraso_global"], 2), "%", "Vuelos completados con llegada 15+ min tarde (ARR_DEL15)"),
        ("ROC AUC modelo", round(lr["roc_auc"], 4), "0-1", "Regresion logistica, prueba nov-dic"),
        ("ROC AUC baseline tasa por ruta", round(clf["baseline_tasa_historica_ruta"]["roc_auc"], 4), "0-1", "Tasa historica de la ruta (ene-oct)"),
        ("Precision modelo", round(lr["precision"], 4), "0-1", "De los vuelos marcados con retraso, cuantos lo tuvieron"),
        ("Recall modelo", round(lr["recall"], 4), "0-1", "De los vuelos con retraso, cuantos detecto"),
        ("F1 modelo", round(lr["f1"], 4), "0-1", "Media armonica de precision y recall"),
        ("Accuracy modelo", round(lr["accuracy"], 4), "0-1", "Aciertos totales con umbral = tasa base de entrenamiento"),
        ("MAE modelo GBT", round(m["mae"], 2), "minutos", "Error absoluto medio, ARR_DELAY recortado [-37, 222]"),
        ("RMSE modelo GBT", round(m["rmse"], 2), "minutos", "Raiz del error cuadratico medio, ARR_DELAY recortado"),
        ("MAE baseline media global", round(bg["mae"], 2), "minutos", "Predecir siempre la media de ene-oct"),
        ("MAE baseline promedio por ruta", round(br["mae"], 2), "minutos", "Promedio historico de la ruta (ene-oct)"),
        ("Reduccion MAE vs media global", round(bg["reduccion_mae_modelo_pct"], 2), "%", "Mejora del modelo frente al metodo tradicional"),
        ("Reduccion MAE vs promedio por ruta", round(br["reduccion_mae_modelo_pct"], 2), "%", "Mejora del modelo frente al promedio por ruta"),
        ("Reduccion RMSE vs media global", round(bg["reduccion_rmse_modelo_pct"], 2), "%", "Mejora del modelo frente al metodo tradicional"),
        ("Reduccion RMSE vs promedio por ruta", round(br["reduccion_rmse_modelo_pct"], 2), "%", "Mejora del modelo frente al promedio por ruta"),
        ("Tiempo pipeline completo", tim["pipeline_completo_s"], "segundos", "De zips a reporte, un comando, MacBook Air M4"),
        ("Throughput pipeline", tim["pipeline_filas_por_segundo"], "filas/s", "7,001,619 filas / tiempo total"),
        ("Tiempo ingesta Bronze", tim["ingesta_bronze"]["segundos"], "segundos", "12 CSV a Parquet"),
        ("Throughput ingesta Bronze", tim["ingesta_bronze"]["filas_por_segundo"], "filas/s", "Filas por segundo en la ingesta"),
        ("Tiempo descarga clima (inicial)", tim["clima_descarga_inicial_50_aeropuertos_s"], "segundos", "50 llamadas a Open-Meteo; luego se usa cache"),
        ("Tiempo Silver", tim["silver"]["segundos"], "segundos", "Limpieza, separacion y union con clima"),
        ("Tiempo Gold", tim["gold"]["segundos"], "segundos", "Esquema estrella en Parquet"),
        ("Tiempo entrenamiento (3 modelos)", tim["entrenamiento_3_modelos"]["segundos"], "segundos", "Logistica + lineal + GBT"),
        ("Tiempo scoring 7M vuelos", tim["scoring_y_escritura_gold"]["segundos"], "segundos", "pred_prob y pred_min escritos en Gold"),
        ("Tiempo carga MySQL", tim["carga_mysql"]["segundos"], "segundos", "7M filas + indices + llaves foraneas"),
        ("Tiempo promedio por consulta de dashboard", q["promedio_mediana_s"], "segundos", "Promedio de medianas de 5 consultas sobre Gold con Spark"),
        ("Costo por consulta", None, "USD", "No calculado: completar con la tarifa de capacidad elegida (tiempo x precio)"),
    ]
    df = pd.DataFrame(rows, columns=["nombre", "valor", "unidad", "descripcion"])
    # object: los enteros se escriben como enteros (no 7001619.0) y el vacio queda vacio.
    df["valor"] = pd.array([int(v) if isinstance(v, (int, float)) and v is not None and float(v).is_integer()
                            else v for v in df["valor"]], dtype=object)
    return df


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(KPIS_OUT) as f:
        k = json.load(f)
    spark = get_spark("export_powerbi")
    fact = spark.read.parquet(GOLD_PATHS["FACT_VUELOS"])
    dt = spark.read.parquet(GOLD_PATHS["DIM_TIEMPO"])
    aer = spark.read.parquet(GOLD_PATHS["DIM_AEROLINEA"])
    apt = spark.read.parquet(GOLD_PATHS["DIM_AEROPUERTO"])
    cli = spark.read.parquet(GOLD_PATHS["DIM_CLIMA"])
    done = "CANCELLED = 0 AND DIVERTED = 0"
    out = []

    out.append(write(kpis_resumen(k), "kpis_resumen.csv"))

    m = fact.join(dt, "fecha_key").groupBy("mes").agg(*metrics()).orderBy("mes").toPandas()
    m.insert(1, "nombre_mes", [NOMBRE_MES[i - 1] for i in m["mes"]])
    m.insert(2, "fecha_inicio_mes", pd.to_datetime([f"2025-{i:02d}-01" for i in m["mes"]]))
    out.append(write(m, "retraso_por_mes.csv"))

    d = fact.join(dt, "fecha_key").groupBy("dia_semana").agg(*metrics()).orderBy("dia_semana").toPandas()
    d.insert(1, "nombre_dia", [NOMBRE_DIA[i - 1] for i in d["dia_semana"]])
    out.append(write(d, "retraso_por_dia_semana.csv"))

    h = (fact.groupBy(F.col("dep_hour").alias("hora_salida_programada")).agg(*metrics())
         .orderBy("hora_salida_programada").toPandas())
    out.append(write(h, "retraso_por_hora.csv"))

    a = (fact.join(aer, "aerolinea_key")
         .groupBy(F.col("codigo").alias("codigo_aerolinea"), F.col("nombre").alias("nombre_aerolinea"),
                  F.col("tipo").alias("tipo_aerolinea"))
         .agg(*metrics()).orderBy(F.desc("vuelos_programados")).toPandas())
    out.append(write(a, "retraso_por_aerolinea.csv"))

    o = apt.select(F.col("aeropuerto_key").alias("origen_key"), F.col("iata").alias("origen"),
                   F.col("ciudad").alias("ciudad_origen"), F.col("estado").alias("estado_origen"),
                   F.col("latitud").alias("latitud_origen"), F.col("longitud").alias("longitud_origen"))
    de = apt.select(F.col("aeropuerto_key").alias("destino_key"), F.col("iata").alias("destino"),
                    F.col("ciudad").alias("ciudad_destino"), F.col("estado").alias("estado_destino"),
                    F.col("latitud").alias("latitud_destino"), F.col("longitud").alias("longitud_destino"))
    r = (fact.join(o, "origen_key").join(de, "destino_key")
         .groupBy("origen", "ciudad_origen", "estado_origen", "latitud_origen", "longitud_origen",
                  "destino", "ciudad_destino", "estado_destino", "latitud_destino", "longitud_destino")
         .agg(F.round(F.avg("DISTANCE"), 0).alias("distancia_millas"), *metrics())
         .orderBy(F.desc("vuelos_programados")).limit(50).toPandas())
    r.insert(0, "ruta", r["origen"] + " - " + r["destino"])
    r.insert(0, "posicion_por_volumen", range(1, len(r) + 1))
    out.append(write(r, "top_rutas.csv"))

    ap = (fact.join(apt.withColumnRenamed("aeropuerto_key", "origen_key"), "origen_key")
          .groupBy(F.col("iata").alias("codigo_aeropuerto"), F.col("nombre").alias("nombre_aeropuerto"),
                   "ciudad", "estado", "latitud", "longitud", "tiene_clima")
          .agg(*metrics()).orderBy(F.desc("vuelos_programados")).toPandas())
    ap = ap.rename(columns={"vuelos_programados": "salidas_programadas"})
    out.append(write(ap, "retraso_por_aeropuerto.csv"))

    c = (fact.join(cli.select("clima_key", "condicion"), "clima_key").groupBy("condicion")
         .agg(*metrics()).toPandas())
    base = c.loc[c["condicion"] == "despejado"].iloc[0]
    c["incluido_en_comparacion"] = (c["condicion"] != "sin_datos").astype(int)
    c["diferencia_vs_despejado_pp"] = (c["tasa_retraso_pct"] - base["tasa_retraso_pct"]).round(2)
    c["variacion_vs_despejado_pct"] = (100 * (c["tasa_retraso_pct"] / base["tasa_retraso_pct"] - 1)).round(2)
    c.loc[c["condicion"] == "sin_datos", ["diferencia_vs_despejado_pp", "variacion_vs_despejado_pct"]] = None
    order = list(CONDICION)
    c = c.set_index("condicion").loc[[x for x in order if x in set(c["condicion"])]].reset_index()
    c.insert(1, "nombre_condicion", c["condicion"].map(CONDICION))
    out.append(write(c, "efecto_clima.csv"))

    comp = fact.filter(done)
    agg = comp.agg(*[F.sum(col).alias(f"{col}_min") for col, _ in CAUSAS],
                   *[F.sum((F.col(col) > 0).cast("int")).alias(f"{col}_n") for col, _ in CAUSAS]).first()
    total = sum(agg[f"{col}_min"] for col, _ in CAUSAS)
    causas = pd.DataFrame([{
        "causa": name, "columna_bts": col,
        "minutos_totales": round(agg[f"{col}_min"], 0),
        "pct_minutos_de_retraso": round(100 * agg[f"{col}_min"] / total, 2),
        "vuelos_afectados": agg[f"{col}_n"],
        "minutos_promedio_por_vuelo_afectado": round(agg[f"{col}_min"] / agg[f"{col}_n"], 2) if agg[f"{col}_n"] else None,
    } for col, name in CAUSAS]).sort_values("minutos_totales", ascending=False)
    out.append(write(causas, "causas_retraso.csv"))

    ev = k["evaluacion"]
    mv = []
    for key, label in [("modelo_regresion_logistica", "Modelo: regresion logistica"),
                       ("baseline_siempre_a_tiempo", "Baseline: siempre a tiempo"),
                       ("baseline_tasa_historica_ruta", "Baseline: tasa historica por ruta")]:
        x = ev["clasificacion"][key]
        mv.append({"tarea": "Clasificacion ARR_DEL15", "opcion": label, "es_modelo": int(key.startswith("modelo")),
                   **{n: round(x[n], 4) for n in ["roc_auc", "accuracy", "precision", "recall", "f1"]}})
    for key, label in [("modelo_gbt", "Modelo: GBT"), ("baseline_media_global", "Baseline: media global"),
                       ("baseline_promedio_ruta", "Baseline: promedio por ruta")]:
        x = ev["regresion"][key]
        mv.append({"tarea": "Regresion minutos ARR_DELAY", "opcion": label, "es_modelo": int(key.startswith("modelo")),
                   "mae_min": round(x["mae"], 2), "rmse_min": round(x["rmse"], 2),
                   "mae_sin_recorte_min": round(x["mae_sin_recorte"], 2),
                   "rmse_sin_recorte_min": round(x["rmse_sin_recorte"], 2),
                   "reduccion_mae_del_modelo_pct": round(x["reduccion_mae_modelo_pct"], 2) if "reduccion_mae_modelo_pct" in x else None,
                   "reduccion_rmse_del_modelo_pct": round(x["reduccion_rmse_modelo_pct"], 2) if "reduccion_rmse_modelo_pct" in x else None})
    cols = ["tarea", "opcion", "es_modelo", "roc_auc", "accuracy", "precision", "recall", "f1", "mae_min",
            "rmse_min", "mae_sin_recorte_min", "rmse_sin_recorte_min", "reduccion_mae_del_modelo_pct",
            "reduccion_rmse_del_modelo_pct"]
    out.append(write(pd.DataFrame(mv)[cols], "modelo_vs_baseline.csv"))

    imp = k["importancia_variables"]
    iv = [{"modelo": "Regresion logistica (clasificacion)", "posicion": i + 1, "variable": x["variable"],
           "tipo": x["tipo"], "importancia": round(x["importancia_estandarizada"], 4),
           "coeficiente": round(x["coeficiente"], 4),
           "efecto": "aumenta el riesgo" if x["coeficiente"] > 0 else "reduce el riesgo",
           "frecuencia_entrenamiento_pct": round(100 * x["frecuencia_entrenamiento"], 2)
           if x["frecuencia_entrenamiento"] is not None else None}
          for i, x in enumerate(imp["regresion_logistica_top15"])]
    iv += [{"modelo": "GBT (regresion de minutos)", "posicion": i + 1, "variable": x["variable"],
            "tipo": "variable de entrada", "importancia": round(x["importancia"], 4), "coeficiente": None,
            "efecto": None, "frecuencia_entrenamiento_pct": None}
           for i, x in enumerate(imp["gbt_todas"])]
    out.append(write(pd.DataFrame(iv), "importancia_variables.csv"))

    # Muestra aleatoria del set de prueba (vuelos completados nov-dic), semilla fija.
    t = pd.read_parquet(TEST_PREDS).sample(n=SAMPLE_N, random_state=SEED)
    aer_pd = aer.toPandas().rename(columns={"codigo": "codigo_aerolinea", "nombre": "nombre_aerolinea"})
    cli_pd = cli.select("clima_key", "condicion").toPandas()
    dt_pd = dt.select("fecha_key", "fecha", "dia_semana").toPandas()
    t = (t.merge(aer_pd[["aerolinea_key", "codigo_aerolinea", "nombre_aerolinea"]], on="aerolinea_key")
          .merge(cli_pd, on="clima_key").merge(dt_pd, on="fecha_key"))
    thr = k["evaluacion"]["conjunto_prueba"]["umbral_clasificacion"]
    s = pd.DataFrame({
        "vuelo_id": t["vuelo_key"].astype("int64"),
        "fecha": pd.to_datetime(t["fecha"]),
        "mes": t["MONTH"], "dia_semana": t["dia_semana"],
        "hora_salida_programada": t["dep_hour"],
        "codigo_aerolinea": t["codigo_aerolinea"], "nombre_aerolinea": t["nombre_aerolinea"],
        "origen": t["ORIGIN"], "destino": t["DEST"], "ruta": t["ORIGIN"] + " - " + t["DEST"],
        "distancia_millas": t["DISTANCE"].round(0),
        "condicion_clima": t["condicion"].map(CONDICION),
        "prob_retraso_predicha": t["pred_prob"].round(4),
        "retraso_predicho": (t["pred_prob"] >= thr).astype(int),
        "retraso_real": t["ARR_DEL15"].astype(int),
        "minutos_predichos": t["pred_min"].round(1),
        "minutos_reales": t["ARR_DELAY"].round(1),
    }).sort_values(["fecha", "hora_salida_programada"])
    out.append(write(s, "predicciones_muestra.csv"))
    spark.stop()

    print(f"\n{'archivo':<32}{'filas':>8}{'columnas':>10}{'tamano':>12}")
    for r in out:
        size = f"{r['mb'] * 1024:.1f} KB" if r["mb"] < 1 else f"{r['mb']:.2f} MB"
        print(f"{r['archivo']:<32}{r['filas']:>8,}{r['columnas']:>10}{size:>12}")
    print(f"Total: {len(out)} archivos, {sum(r['mb'] for r in out):.2f} MB en {os.path.relpath(OUT_DIR)}")


if __name__ == "__main__":
    main()
