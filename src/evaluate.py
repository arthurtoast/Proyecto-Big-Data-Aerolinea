"""
Paso A (cierre Fase 3): indicadores tecnicos medidos.

1. Modelo vs baselines en el MISMO set de prueba (vuelos completados de nov y dic):
   - Clasificacion: siempre "a tiempo" y tasa historica por ruta (solo ene a oct).
   - Regresion: media global y promedio por ruta (solo ene a oct), con ARR_DELAY recortado
     igual que el modelo ([p1, p99] de entrenamiento).
2. Tiempos por etapa (de reports/stage_metrics.json), filas por segundo y specs de la maquina.
3. Costo por consulta: 5 consultas tipicas del dashboard sobre Gold, cronometradas.
4. Volumen y tamano en disco por capa, cobertura de clima.
5. Importancia de variables (regresion logistica: |coeficiente|; GBT: featureImportances).
6. Auditoria de limpieza: filas afectadas por cada regla de Silver.

Salidas: deliverables/kpis.json y data/eval/test_preds.parquet (insumo de figuras y CSV).
Uso: python -m src.evaluate
"""
import json
import os
import platform
import re
import statistics
import subprocess
import time

import numpy as np
import pandas as pd
from pyspark.ml import PipelineModel
from pyspark.sql import functions as F
from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error, mean_squared_error,
                             precision_score, recall_score, roc_auc_score)

from src.config import DATA_DIR, GOLD_DIR, RAW_DIR, ROOT, BRONZE_DIR, SILVER_DIR
from src.features import CATEGORICAL, CYCLIC_SMALL, NUMERIC
from src.gold import GOLD_PATHS
from src.spark_session import get_spark
from src.train import CLIP_PATH, MODEL_PATHS, TEST_MONTHS, TRAIN_MONTHS
from src.utils import STAGE_METRICS_PATH

DELIV_DIR = os.path.join(ROOT, "deliverables")
EVAL_DIR = os.path.join(DATA_DIR, "eval")
TEST_PREDS = os.path.join(EVAL_DIR, "test_preds.parquet")
KPIS_OUT = os.path.join(DELIV_DIR, "kpis.json")
QUERY_REPEATS = 5


# ---------------------------------------------------------------- 1. baselines
def load_eval_frames(spark, clip):
    """Prueba (nov-dic) en pandas con modelo y baselines; agregados de ruta solo de ene-oct."""
    fact = spark.read.parquet(GOLD_PATHS["FACT_VUELOS"]).filter("CANCELLED = 0 AND DIVERTED = 0")
    apt = spark.read.parquet(GOLD_PATHS["DIM_AEROPUERTO"])
    o = apt.select(F.col("aeropuerto_key").alias("origen_key"), F.col("iata").alias("ORIGIN"))
    d = apt.select(F.col("aeropuerto_key").alias("destino_key"), F.col("iata").alias("DEST"))
    clipped = F.least(F.greatest(F.col("ARR_DELAY"), F.lit(clip["p01"])), F.lit(clip["p99"]))
    base = (fact.join(F.broadcast(o), "origen_key").join(F.broadcast(d), "destino_key")
            .withColumn("label_clip", clipped))

    train = base.filter(F.col("MONTH").isin(TRAIN_MONTHS))
    g = train.agg(F.avg("ARR_DEL15").alias("rate"), F.avg("label_clip").alias("mean")).first()
    routes = (train.groupBy("ORIGIN", "DEST")
              .agg(F.avg("ARR_DEL15").alias("route_rate"), F.avg("label_clip").alias("route_mean"),
                   F.count("*").alias("route_n_train")))
    test = (base.filter(F.col("MONTH").isin(TEST_MONTHS))
            .join(routes, ["ORIGIN", "DEST"], "left")
            .select("vuelo_key", "fecha_key", "MONTH", "aerolinea_key", "ORIGIN", "DEST",
                    "dep_hour", "DISTANCE", "clima_key", "ARR_DEL15", "ARR_DELAY", "label_clip",
                    "pred_prob", "pred_min", "route_rate", "route_mean", "route_n_train")
            .toPandas())
    test["route_seen"] = test["route_rate"].notna()
    test["route_rate"] = test["route_rate"].fillna(g["rate"])
    test["route_mean"] = test["route_mean"].fillna(g["mean"])
    return test, g["rate"], g["mean"]


def clf_metrics(y, score, pred):
    return {"roc_auc": float(roc_auc_score(y, score)) if np.unique(score).size > 1 else 0.5,
            "accuracy": float(accuracy_score(y, pred)),
            "precision": float(precision_score(y, pred, zero_division=0)),
            "recall": float(recall_score(y, pred, zero_division=0)),
            "f1": float(f1_score(y, pred, zero_division=0))}


def reg_metrics(y_clip, y_raw, pred):
    return {"mae": float(mean_absolute_error(y_clip, pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_clip, pred))),
            "mae_sin_recorte": float(mean_absolute_error(y_raw, pred)),
            "rmse_sin_recorte": float(np.sqrt(mean_squared_error(y_raw, pred)))}


def baselines(test, train_rate, train_mean):
    y = test["ARR_DEL15"].astype(int).to_numpy()
    # Umbral del modelo: p=0.5 del modelo ponderado equivale a pred_prob (corregida) >= tasa
    # base de entrenamiento. Se usa el mismo umbral para la tasa por ruta.
    thr = train_rate
    clf = {
        "modelo_regresion_logistica": clf_metrics(y, test["pred_prob"], (test["pred_prob"] >= thr).astype(int)),
        "baseline_siempre_a_tiempo": clf_metrics(y, np.zeros(len(y)), np.zeros(len(y), dtype=int)),
        "baseline_tasa_historica_ruta": clf_metrics(y, test["route_rate"], (test["route_rate"] >= thr).astype(int)),
    }
    yc, yr = test["label_clip"].to_numpy(), test["ARR_DELAY"].to_numpy()
    reg = {
        "modelo_gbt": reg_metrics(yc, yr, test["pred_min"]),
        "baseline_media_global": reg_metrics(yc, yr, np.full(len(yc), train_mean)),
        "baseline_promedio_ruta": reg_metrics(yc, yr, test["route_mean"]),
    }
    m = reg["modelo_gbt"]
    for k in ["baseline_media_global", "baseline_promedio_ruta"]:
        b = reg[k]
        b["reduccion_mae_modelo_pct"] = 100 * (b["mae"] - m["mae"]) / b["mae"]
        b["reduccion_rmse_modelo_pct"] = 100 * (b["rmse"] - m["rmse"]) / b["rmse"]
    meta = {"filas_prueba": int(len(test)), "tasa_retraso_prueba": float(y.mean()),
            "tasa_retraso_entrenamiento": float(train_rate), "umbral_clasificacion": float(thr),
            "media_retraso_entrenamiento_recortado_min": float(train_mean),
            "vuelos_prueba_ruta_sin_historial": int((~test["route_seen"]).sum())}
    return clf, reg, meta


# ---------------------------------------------------------------- 2. specs y tiempos
def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=20).stdout.strip() \
            or subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=20).stderr.strip()
    except Exception as e:
        return f"no medible ({e})"


def machine_specs(spark):
    java_home = os.environ.get("JAVA_HOME", "")
    java = sh(f'"{java_home}/bin/java" -version 2>&1 | head -1') if java_home else "no medible"
    mysql_v = None
    try:
        from src.mysql_loader import connect
        c = connect(); cur = c.cursor(); cur.execute("SELECT VERSION(), @@innodb_buffer_pool_size")
        v, bp = cur.fetchone(); c.close()
        mysql_v = {"version": v, "innodb_buffer_pool_mb": int(bp) // 1024 ** 2}
    except Exception as e:
        mysql_v = f"no medible ({type(e).__name__})"
    return {
        "cpu": sh("sysctl -n machdep.cpu.brand_string"),
        "nucleos_totales": int(sh("sysctl -n hw.ncpu")),
        "nucleos_rendimiento": int(sh("sysctl -n hw.perflevel0.physicalcpu") or 0),
        "nucleos_eficiencia": int(sh("sysctl -n hw.perflevel1.physicalcpu") or 0),
        "ram_gb": round(int(sh("sysctl -n hw.memsize")) / 1024 ** 3, 1),
        "modelo_equipo": sh("sysctl -n hw.model"),
        "sistema": f"macOS {platform.mac_ver()[0]}",
        "python": platform.python_version(),
        "spark": spark.version,
        "java": java,
        "spark_master": spark.sparkContext.master,
        "spark_driver_memory": spark.conf.get("spark.driver.memory"),
        "spark_shuffle_partitions": spark.conf.get("spark.sql.shuffle.partitions"),
        "mysql": mysql_v,
    }


def timings(stage):
    """Tiempos de la ultima corrida completa, con filas/s. Clima: la ultima corrida uso cache;
    la descarga real de 50 aeropuertos (237.3 s) se midio en la primera ejecucion."""
    m = stage["models"]
    n_bronze = stage["bronze"]["rows"]
    rows_train = 5_752_264
    rows = {
        "ingesta_bronze": (stage["bronze"]["seconds"], n_bronze),
        "clima_con_cache": (stage["weather"]["seconds"], None),
        "silver": (stage["silver"]["seconds"], n_bronze),
        "gold": (stage["gold"]["seconds"], stage["gold"]["table_rows"]["FACT_VUELOS"]),
        "entrenamiento_3_modelos": (round(m["logreg_s"] + m["linreg_s"] + m["gbt_s"], 2), rows_train),
        "scoring_y_escritura_gold": (m["scoring_s"], stage["gold"]["table_rows"]["FACT_VUELOS"]),
        "carga_mysql": (stage["mysql"]["seconds"], stage["mysql"]["table_rows"]["FACT_VUELOS"]),
        "organizar_zips": (stage["organize"]["seconds"], stage["organize"]["rows"]),
        "reporte": (stage["report"]["seconds"], None),
    }
    out = {k: {"segundos": s, "filas": n, "filas_por_segundo": round(n / s) if n and s else None}
           for k, (s, n) in rows.items()}
    out["detalle_entrenamiento_s"] = {"regresion_logistica": m["logreg_s"], "regresion_lineal": m["linreg_s"],
                                      "gbt": m["gbt_s"]}
    out["clima_descarga_inicial_50_aeropuertos_s"] = 237.3
    pipe = stage.get("pipeline", {})
    out["pipeline_completo_s"] = pipe.get("seconds")
    out["pipeline_filas_por_segundo"] = round(n_bronze / pipe["seconds"]) if pipe.get("seconds") else None
    out["pipeline_por_etapa_reloj_s"] = pipe.get("stages_wall_s")
    return out


# ---------------------------------------------------------------- 3. consultas
def bench(fn, repeats=QUERY_REPEATS):
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter(); fn(); times.append(time.perf_counter() - t0)
    return {"primera_s": round(times[0], 3), "mediana_s": round(statistics.median(times), 3),
            "min_s": round(min(times), 3), "max_s": round(max(times), 3), "repeticiones": repeats}


def query_costs(spark):
    gp = GOLD_PATHS
    fact = lambda: spark.read.parquet(gp["FACT_VUELOS"]).filter("CANCELLED = 0 AND DIVERTED = 0")
    dim = lambda n: spark.read.parquet(gp[n])

    def q_mes():
        return (fact().join(dim("DIM_TIEMPO"), "fecha_key").groupBy("mes")
                .agg(F.avg("ARR_DEL15")).collect())

    def q_aerolinea():
        return (fact().join(dim("DIM_AEROLINEA"), "aerolinea_key").groupBy("nombre")
                .agg(F.avg("ARR_DEL15"), F.count("*")).collect())

    def q_rutas():
        a = dim("DIM_AEROPUERTO")
        o = a.select(F.col("aeropuerto_key").alias("origen_key"), F.col("iata").alias("o"))
        d = a.select(F.col("aeropuerto_key").alias("destino_key"), F.col("iata").alias("d"))
        return (fact().join(o, "origen_key").join(d, "destino_key").groupBy("o", "d")
                .agg(F.avg("ARR_DEL15").alias("t"), F.count("*").alias("n"))
                .filter("n >= 1000").orderBy(F.desc("t")).limit(10).collect())

    def q_clima():
        return (fact().join(dim("DIM_CLIMA"), "clima_key").filter("clima_key > 0")
                .groupBy("condicion").agg(F.avg("ARR_DEL15")).collect())

    from src.predict import predict
    t0 = time.perf_counter()
    predict("AA", "ORD", "LGA", "2025-12-22", "17:45")  # carga modelos y catalogos (arranque)
    predict_cold = time.perf_counter() - t0

    res = {"retraso_por_mes": bench(q_mes), "retraso_por_aerolinea": bench(q_aerolinea),
           "top_rutas": bench(q_rutas), "efecto_clima": bench(q_clima),
           "prediccion_un_vuelo": bench(lambda: predict("AA", "ORD", "LGA", "2025-12-22", "17:45"))}
    res["prediccion_un_vuelo"]["arranque_carga_modelos_s"] = round(predict_cold, 3)
    meds = [v["mediana_s"] for v in res.values()]
    return {"consultas": res,
            "promedio_mediana_s": round(sum(meds) / len(meds), 3),
            "metodo": f"Spark local sobre Parquet de Gold, {QUERY_REPEATS} repeticiones por consulta, "
                      "tiempo de reloj hasta collect(). La primera repeticion incluye lectura de "
                      "metadatos Parquet; la mediana representa el uso tipico.",
            "costo_por_consulta_usd": None,
            "costo_formula": "costo = mediana_s x precio_por_segundo_de_capacidad (no se inventan "
                             "precios: completar con la tarifa de la capacidad de Fabric elegida)"}


# ---------------------------------------------------------------- 4. volumen
def du_mb(path):
    if not os.path.exists(path):
        return None
    total = 0
    for root, _, files in os.walk(path):
        total += sum(os.path.getsize(os.path.join(root, f)) for f in files)
    return round(total / 1024 ** 2, 1)


def volume(spark, stage):
    rp = lambda p: spark.read.parquet(p).count()
    gold_rows = {n: rp(p) for n, p in GOLD_PATHS.items()}
    mysql_mb = None
    try:
        from src.mysql_loader import connect
        c = connect(); cur = c.cursor()
        cur.execute("SELECT table_name, table_rows, ROUND((data_length + index_length)/1024/1024, 1) "
                    "FROM information_schema.tables WHERE table_schema = DATABASE()")
        exact = stage.get("mysql", {}).get("table_rows", {})
        # table_rows de information_schema es una estimacion de InnoDB: se reporta el conteo
        # exacto verificado con COUNT(*) al cargar.
        mysql_mb = {r[0]: {"filas_exactas": exact.get(r[0]), "mb": float(r[2])} for r in cur.fetchall()}
        c.close()
    except Exception as e:
        mysql_mb = f"no medible ({type(e).__name__})"
    return {
        "filas": {
            "raw_csv": stage["organize"]["rows"],
            "bronze": rp(os.path.join(BRONZE_DIR, "flights")),
            "silver_completados": rp(os.path.join(SILVER_DIR, "flights")),
            "silver_cancelados_desviados": rp(os.path.join(SILVER_DIR, "flights_disrupted")),
            "silver_clima_diario": rp(os.path.join(SILVER_DIR, "weather_daily")),
            "gold": gold_rows,
        },
        "disco_mb": {
            "raw_flights_csv": du_mb(os.path.join(RAW_DIR, "flights")),
            "raw_weather_json": du_mb(os.path.join(RAW_DIR, "weather")),
            "raw_airports": du_mb(os.path.join(RAW_DIR, "airports")),
            "bronze": du_mb(BRONZE_DIR),
            "silver": du_mb(SILVER_DIR),
            "silver_detalle": {n: du_mb(os.path.join(SILVER_DIR, n))
                               for n in ["flights", "flights_disrupted", "weather_daily"]},
            "gold": du_mb(GOLD_DIR),
            "gold_detalle": {n: du_mb(p) for n, p in GOLD_PATHS.items()},
            "modelos": du_mb(os.path.join(DATA_DIR, "models")),
            "mysql": mysql_mb,
        },
        "cobertura_clima_pct": {
            "vuelos_completados": stage["silver"]["weather_coverage_pct"],
            "todos_los_vuelos_gold": round(100 * stage["gold"]["fact_with_weather"] /
                                           stage["gold"]["table_rows"]["FACT_VUELOS"], 2),
            "aeropuertos_con_clima": stage["weather"]["airports_ok"],
            "aeropuertos_totales": gold_rows["DIM_AEROPUERTO"],
        },
    }


# ---------------------------------------------------------------- 5. importancia
def importance(spark):
    from src.features import build_features
    lr_model = PipelineModel.load(MODEL_PATHS["logreg"])
    sample = lr_model.transform(build_features(spark).limit(1))
    attrs = sample.schema["features"].metadata["ml_attr"]["attrs"]
    names = {a["idx"]: a["name"] for g in attrs.values() for a in g}
    coefs = lr_model.stages[-1].coefficients.toArray()

    def pretty(n):
        m = re.match(r"num_scaled_(\d+)$", n)
        if m:
            return NUMERIC[int(m.group(1))], "numerica (escalada)"
        for c in CATEGORICAL + CYCLIC_SMALL:
            for pre in (f"{c}_idx_ohe_", f"{c}_ohe_"):
                if n.startswith(pre):
                    return f"{c} = {n[len(pre):]}", "categoria (one-hot)"
        return n, "otra"

    # Frecuencia de cada categoria en entrenamiento (ene-oct, completados) para estandarizar:
    # importancia = |coef| x desviacion estandar de la variable. Una categoria rara con
    # coeficiente grande afecta a pocos vuelos; sin esto domina el ranking.
    feats = build_features(spark).filter("is_completed = 1").filter(F.col("MONTH").isin(TRAIN_MONTHS))
    n_train = feats.count()
    freq = {}
    for c in CATEGORICAL + CYCLIC_SMALL:
        for r in feats.groupBy(c).count().collect():
            freq[f"{c} = {r[c]}"] = r["count"] / n_train

    rows = []
    for i, c in enumerate(coefs):
        var, tipo = pretty(names[i])
        if tipo.startswith("numerica"):
            sd, p = 1.0, None  # ya escaladas a desviacion estandar 1
        else:
            p = freq.get(var, 0.0)
            sd = (p * (1 - p)) ** 0.5
        rows.append({"variable": var, "tipo": tipo, "coeficiente": float(c),
                     "frecuencia_entrenamiento": p, "importancia_estandarizada": abs(float(c)) * sd})
    lr_top = sorted(rows, key=lambda r: -r["importancia_estandarizada"])[:15]
    raw_top = sorted(rows, key=lambda r: -abs(r["coeficiente"]))[:5]

    gbt = PipelineModel.load(MODEL_PATHS["gbt"]).stages[-1].featureImportances.toArray()
    gbt_names = CATEGORICAL + CYCLIC_SMALL + NUMERIC
    gbt_rows = sorted([{"variable": n, "importancia": float(v)} for n, v in zip(gbt_names, gbt)],
                      key=lambda r: -r["importancia"])
    return {"regresion_logistica_top15": lr_top, "gbt_todas": gbt_rows,
            "regresion_logistica_top5_coeficiente_crudo": raw_top,
            "nota": "Regresion logistica: importancia = |coeficiente| x desviacion estandar de la "
                    "variable (numericas ya escaladas: sd 1; categorias one-hot: sqrt(p(1-p)) con p su "
                    "frecuencia en entrenamiento). Signo positivo = mas riesgo de retraso frente a la "
                    "categoria de referencia. Sin estandarizar, el top lo dominan aeropuertos con muy "
                    "pocos vuelos (ver top5_coeficiente_crudo). GBT: importancia por ganancia, suma 1."}


# ---------------------------------------------------------------- 6. auditoria limpieza
def cleaning_audit(spark, stage):
    b = spark.read.parquet(os.path.join(BRONZE_DIR, "flights"))
    done = b.filter((F.col("CANCELLED").cast("double") == 0) & (F.col("DIVERTED").cast("double") == 0))
    nul = lambda c: F.sum((F.col(c).isNull() | (F.trim(F.col(c)) == "")).cast("int"))
    r = done.agg(nul("CARRIER_DELAY").alias("causas_nulas"), nul("DEP_DELAY").alias("dep_delay_nulo"),
                 F.sum((F.col("CRS_DEP_TIME") == "2400").cast("int")).alias("crs_2400"),
                 F.count("*").alias("completados_bronze")).first().asDict()
    s = stage["silver"]
    return {
        "filas_bronze": s["bronze_rows"],
        "descartadas_clave_invalida_o_nula": s["dropped_invalid_key"],
        "descartadas_duplicados_clave_negocio": s["dropped_duplicates"],
        "fecha_incoherente_con_year_month_day_o_day_of_week": s["dow_mismatch"],
        "separadas_cancelados": s["cancelled_rows"],
        "separadas_desviados": s["diverted_rows"],
        "descartadas_completados_sin_arr_delay": s["dropped_completed_no_label"],
        "completados_con_causas_nulas_rellenadas_con_0": r["causas_nulas"],
        "completados_con_dep_delay_nulo": r["dep_delay_nulo"],
        "crs_dep_time_2400_tratado_como_hora_0": r["crs_2400"],
        "completados_sin_clima_origen_has_weather_0": s["silver_rows"] - s["silver_with_weather"],
        "filas_silver_completados": s["silver_rows"],
    }


def main():
    os.makedirs(DELIV_DIR, exist_ok=True)
    os.makedirs(EVAL_DIR, exist_ok=True)
    spark = get_spark("evaluate")
    with open(CLIP_PATH) as f:
        clip = json.load(f)
    with open(STAGE_METRICS_PATH) as f:
        stage = json.load(f)
    with open(os.path.join(ROOT, "reports", "kpis.json")) as f:
        base_kpis = json.load(f)

    t0 = time.perf_counter()
    test, train_rate, train_mean = load_eval_frames(spark, clip)
    test.to_parquet(TEST_PREDS, index=False)
    clf, reg, meta = baselines(test, train_rate, train_mean)
    print(f"[A1] baselines en {time.perf_counter() - t0:.1f} s")

    specs = machine_specs(spark)
    tim = timings(stage)
    t0 = time.perf_counter(); q = query_costs(spark); print(f"[A3] consultas en {time.perf_counter() - t0:.1f} s")
    t0 = time.perf_counter(); vol = volume(spark, stage); print(f"[A4] volumen en {time.perf_counter() - t0:.1f} s")
    imp = importance(spark)
    audit = cleaning_audit(spark, stage)
    spark.stop()

    kpis = {
        "generado": time.strftime("%Y-%m-%d %H:%M:%S"),
        "maquina": specs,
        "volumen": vol,
        "limpieza": audit,
        "tiempos": tim,
        "consultas_dashboard": q,
        "evaluacion": {"conjunto_prueba": meta, "clasificacion": clf, "regresion": reg,
                       "matriz_confusion_modelo": base_kpis["matriz_confusion"],
                       "recorte_arr_delay_min": [clip["p01"], clip["p99"]],
                       "pesos_clase": {"retrasado": clip["w_pos"], "a_tiempo": clip["w_neg"]}},
        "importancia_variables": imp,
        "tasa_retraso_global": base_kpis["tasa_retraso_global"],
    }
    with open(KPIS_OUT, "w") as f:
        json.dump(kpis, f, indent=2, ensure_ascii=False)

    # ------------------------------------------------ salida
    pct = lambda x: f"{100 * x:.2f}%"
    print(f"\n=== A1. Clasificacion (prueba nov-dic, {meta['filas_prueba']:,} vuelos, umbral {meta['umbral_clasificacion']:.4f}) ===")
    print(f"{'opcion':<32}{'ROC AUC':>9}{'accuracy':>10}{'precision':>11}{'recall':>9}{'F1':>8}")
    for k, m in clf.items():
        print(f"{k:<32}{m['roc_auc']:>9.4f}{m['accuracy']:>10.4f}{m['precision']:>11.4f}{m['recall']:>9.4f}{m['f1']:>8.4f}")
    print(f"Vuelos de prueba en rutas sin historial ene-oct (usan tasa global): {meta['vuelos_prueba_ruta_sin_historial']:,}")
    print(f"\n=== A1. Regresion (ARR_DELAY recortado a [{clip['p01']:.0f}, {clip['p99']:.0f}]) ===")
    print(f"{'opcion':<26}{'MAE':>8}{'RMSE':>8}{'red. MAE':>10}{'red. RMSE':>11}{'MAE sin rec.':>14}{'RMSE sin rec.':>15}")
    for k, m in reg.items():
        rm = f"{m['reduccion_mae_modelo_pct']:.2f}%" if "reduccion_mae_modelo_pct" in m else "-"
        rr = f"{m['reduccion_rmse_modelo_pct']:.2f}%" if "reduccion_rmse_modelo_pct" in m else "-"
        print(f"{k:<26}{m['mae']:>8.2f}{m['rmse']:>8.2f}{rm:>10}{rr:>11}{m['mae_sin_recorte']:>14.2f}{m['rmse_sin_recorte']:>15.2f}")
    print("(red. = % de reduccion de error del modelo GBT frente a ese baseline)")

    print("\n=== A2. Tiempos (ultima corrida completa) ===")
    for k, v in tim.items():
        if isinstance(v, dict) and "segundos" in v:
            fps = f"{v['filas_por_segundo']:,} filas/s" if v["filas_por_segundo"] else ""
            print(f"  {k:<28}{v['segundos']:>8.1f} s   {fps}")
    print(f"  pipeline completo           {tim['pipeline_completo_s']:>8.1f} s   "
          f"{tim['pipeline_filas_por_segundo']:,} filas/s (7,001,619 filas de entrada)")
    print(f"  clima: descarga inicial real de 50 aeropuertos = {tim['clima_descarga_inicial_50_aeropuertos_s']} s")
    print("Maquina:", json.dumps(specs, ensure_ascii=False))

    print("\n=== A3. Consultas del dashboard sobre Gold (Spark local) ===")
    for k, v in q["consultas"].items():
        extra = f" | arranque {v['arranque_carga_modelos_s']} s" if "arranque_carga_modelos_s" in v else ""
        print(f"  {k:<24} primera {v['primera_s']:>6.3f} s | mediana {v['mediana_s']:>6.3f} s{extra}")
    print(f"  promedio de medianas: {q['promedio_mediana_s']} s | costo_usd: no calculado (campo vacio)")

    print("\n=== A4. Volumen ===")
    print(json.dumps(vol, indent=1, ensure_ascii=False))

    print("\n=== A5. Top 15 variables (regresion logistica, coeficiente estandarizado) ===")
    print(f"  {'variable':<30}{'importancia':>12}{'coef':>9}{'frec. entr.':>13}")
    for r in imp["regresion_logistica_top15"]:
        fr = f"{100 * r['frecuencia_entrenamiento']:.2f}%" if r["frecuencia_entrenamiento"] is not None else "numerica"
        print(f"  {r['variable']:<30}{r['importancia_estandarizada']:>12.4f}{r['coeficiente']:>+9.4f}{fr:>13}")
    print("GBT (regresion):", [(r["variable"], round(r["importancia"], 3)) for r in imp["gbt_todas"][:8]])
    print("\nAuditoria de limpieza:", json.dumps(audit, ensure_ascii=False))
    print(f"\nEscrito: {os.path.relpath(KPIS_OUT)} y {os.path.relpath(TEST_PREDS)}")


if __name__ == "__main__":
    main()
