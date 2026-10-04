"""
Paso 6: modelos de retraso (Spark MLlib) con registro en MLflow.

- Split temporal: entrena con enero a octubre, prueba con noviembre y diciembre (simula
  predecir el futuro; un split aleatorio mezclaria dias y sobreestimaria el desempeno).
- Solo vuelos completados para entrenar/evaluar (cancelados no tienen ARR_DELAY).
- a) Clasificacion ARR_DEL15: regresion logistica con pesos de clase balanceados.
     Metricas con umbral 0.5 del modelo ponderado. pred_prob se guarda corregida a la tasa
     real (correccion de prior), porque los pesos inflan la probabilidad.
- b) Regresion de minutos ARR_DELAY recortados a [p1, p99] de entrenamiento (winsorizacion):
     baseline lineal y GBT de Spark (sobre una muestra del entrenamiento, configurable).
- Escribe pred_prob y pred_min en data/gold/fact_vuelos (todos los vuelos, incluidos
  cancelados: la prediccion se hace antes del vuelo) y recarga FACT_VUELOS en MySQL.

Uso: python -m src.train [--gbt-sample 0.3] [--skip-mysql]
"""
import argparse
import json
import math
import os
import shutil
import sys

# Silencia el aviso informativo que MLflow 3 imprime al importarse.
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
import mlflow  # noqa: E402
from pyspark import StorageLevel
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator, RegressionEvaluator
from pyspark.ml.feature import OneHotEncoder, StandardScaler, StringIndexer, VectorAssembler
from pyspark.ml.functions import vector_to_array
from pyspark.ml.regression import GBTRegressor, LinearRegression
from pyspark.sql import functions as F

from src.config import DATA_DIR, ROOT
from src.features import (CATEGORICAL, CYCLIC_SMALL, INPUT_COLS, LEAKAGE_COLS, NUMERIC,
                          build_features)
from src.gold import GOLD_PATHS
from src.spark_session import get_spark
from src.utils import timed, update_stage_metrics

MODELS_DIR = os.path.join(DATA_DIR, "models")
MODEL_PATHS = {m: os.path.join(MODELS_DIR, m) for m in ["logreg", "linreg", "gbt"]}
CLIP_PATH = os.path.join(MODELS_DIR, "clip_and_prior.json")
MLFLOW_URI = f"sqlite:///{os.path.join(ROOT, 'mlflow.db')}"
EXPERIMENT = "retrasos_vuelos_2025"
TRAIN_MONTHS = list(range(1, 11))
TEST_MONTHS = [11, 12]
SEED = 42


def indexers():
    # handleInvalid='keep': una aerolinea o aeropuerto nuevo en prueba va a un indice extra
    # en vez de romper el pipeline.
    return [StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep")
            for c in CATEGORICAL]


def linear_stages(predictor):
    """Pipeline para modelos lineales: one-hot de categoricos y escalado de numericos."""
    ohe_in = [f"{c}_idx" for c in CATEGORICAL] + CYCLIC_SMALL
    ohe_out = [f"{c}_ohe" for c in ohe_in]
    return indexers() + [
        OneHotEncoder(inputCols=ohe_in, outputCols=ohe_out, handleInvalid="keep"),
        VectorAssembler(inputCols=NUMERIC, outputCol="num_raw"),
        # withMean=False: no densifica; la escala basta para que la regularizacion sea justa.
        StandardScaler(inputCol="num_raw", outputCol="num_scaled", withMean=False),
        VectorAssembler(inputCols=ohe_out + ["num_scaled"], outputCol="features"),
        predictor,
    ]


def tree_stages(predictor):
    """Pipeline para arboles: indices categoricos directos (sin one-hot). VectorAssembler
    conserva la metadata nominal de StringIndexer, asi GBT los trata como categoricos."""
    return indexers() + [
        VectorAssembler(inputCols=[f"{c}_idx" for c in CATEGORICAL] + CYCLIC_SMALL + NUMERIC,
                        outputCol="features"),
        predictor,
    ]


def check_no_leakage():
    bad = set(INPUT_COLS) & set(LEAKAGE_COLS)
    if bad:
        raise RuntimeError(f"Fuga de datos: columnas prohibidas como entrada: {bad}")


def classification_metrics(pred, prob_col="p1"):
    auc = BinaryClassificationEvaluator(labelCol="ARR_DEL15", rawPredictionCol="rawPrediction",
                                        metricName="areaUnderROC").evaluate(pred)
    cm = (pred.groupBy("ARR_DEL15", "prediction").count().collect())
    m = {(int(r["ARR_DEL15"]), int(r["prediction"])): r["count"] for r in cm}
    tp, fp = m.get((1, 1), 0), m.get((0, 1), 0)
    fn, tn = m.get((1, 0), 0), m.get((0, 0), 0)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"roc_auc": auc, "precision": precision, "recall": recall, "f1": f1,
            "accuracy": (tp + tn) / (tp + tn + fp + fn)}, {"tn": tn, "fp": fp, "fn": fn, "tp": tp}


def regression_metrics(pred, label="label_clip"):
    ev = lambda m: RegressionEvaluator(labelCol=label, predictionCol="prediction",
                                       metricName=m).evaluate(pred)
    out = {"mae": ev("mae"), "rmse": ev("rmse"), "r2": ev("r2")}
    # Referencia sin recorte: cuanto error queda frente a los minutos reales.
    out["rmse_sin_recorte"] = RegressionEvaluator(labelCol="ARR_DELAY", predictionCol="prediction",
                                                  metricName="rmse").evaluate(pred)
    return out


def prior_corrected(p_col, w_pos, w_neg):
    """Convierte la probabilidad del modelo ponderado a la escala de la tasa real.
    Ponderar positivos por w_pos y negativos por w_neg multiplica los odds por w_pos/w_neg;
    se deshace multiplicando por w_neg/w_pos."""
    odds = p_col / (1 - p_col) * (w_neg / w_pos)
    return odds / (1 + odds)


def swap_dir(tmp, final):
    """Reemplaza un directorio Parquet: Spark no puede sobrescribir lo que esta leyendo."""
    old = final + "_old"
    if os.path.exists(old):
        shutil.rmtree(old)
    os.rename(final, old)
    os.rename(tmp, final)
    shutil.rmtree(old)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gbt-sample", type=float, default=float(os.getenv("GBT_SAMPLE", "0.3")))
    ap.add_argument("--gbt-iter", type=int, default=int(os.getenv("GBT_ITER", "60")))
    ap.add_argument("--skip-mysql", action="store_true")
    args = ap.parse_args()
    check_no_leakage()
    os.makedirs(MODELS_DIR, exist_ok=True)
    spark = get_spark("train")
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT)
    timings, results = {}, {}

    try:
        feats = build_features(spark)
        labeled = feats.filter("is_completed = 1")
        train = labeled.filter(F.col("MONTH").isin(TRAIN_MONTHS))
        test = labeled.filter(F.col("MONTH").isin(TEST_MONTHS))

        # Recorte (winsorizacion) con percentiles SOLO de entrenamiento: usar los de prueba
        # seria filtrar informacion del futuro.
        p01, p99 = train.approxQuantile("ARR_DELAY", [0.01, 0.99], 0.0005)
        clip = lambda df: df.withColumn(
            "label_clip", F.least(F.greatest(F.col("ARR_DELAY"), F.lit(p01)), F.lit(p99)))
        n_pos = train.filter("ARR_DEL15 = 1").count()
        n_tot = train.count()
        n_neg = n_tot - n_pos
        # Pesos balanceados: cada clase aporta lo mismo a la funcion de perdida.
        w_pos, w_neg = n_tot / (2 * n_pos), n_tot / (2 * n_neg)
        train = (clip(train)
                 .withColumn("w", F.when(F.col("ARR_DEL15") == 1, w_pos).otherwise(w_neg))
                 .persist(StorageLevel.MEMORY_AND_DISK))
        test = clip(test).persist(StorageLevel.MEMORY_AND_DISK)
        n_test = test.count()
        print(f"Entrenamiento (ene-oct): {n_tot:,} vuelos | Prueba (nov-dic): {n_test:,} vuelos")
        print(f"Tasa de retraso en entrenamiento: {100 * n_pos / n_tot:.2f}% "
              f"-> pesos: retrasado {w_pos:.3f}, a tiempo {w_neg:.3f}")
        print(f"Recorte de ARR_DELAY a [p1, p99] de entrenamiento = [{p01:.0f}, {p99:.0f}] min")
        print(f"Variables de entrada ({len(INPUT_COLS)}): {INPUT_COLS}")
        with open(CLIP_PATH, "w") as f:
            json.dump({"p01": p01, "p99": p99, "w_pos": w_pos, "w_neg": w_neg}, f, indent=2)

        common = {"train_months": "1-10", "test_months": "11-12", "features": ",".join(INPUT_COLS),
                  "train_rows": n_tot, "test_rows": n_test}

        # ---------- a) Regresion logistica ponderada ----------
        with timed("logreg") as t, mlflow.start_run(run_name="logreg_arr_del15"):
            lr = LogisticRegression(featuresCol="features", labelCol="ARR_DEL15", weightCol="w",
                                    maxIter=50, regParam=0.001, elasticNetParam=0.0)
            model = Pipeline(stages=linear_stages(lr)).fit(train)
            pred = model.transform(test).withColumn("p1", vector_to_array("probability")[1])
            m, cm = classification_metrics(pred)
            # Calibracion: promedio de la probabilidad corregida vs tasa real en prueba.
            calib = pred.select(
                F.avg(prior_corrected(F.col("p1"), w_pos, w_neg)).alias("p_corr"),
                F.avg("p1").alias("p_raw"), F.avg("ARR_DEL15").alias("real")).first()
            m.update({"mean_prob_raw": calib["p_raw"], "mean_prob_corrected": calib["p_corr"],
                      "real_delay_rate_test": calib["real"]})
            mlflow.log_params({**common, "model": "LogisticRegression", "maxIter": 50,
                               "regParam": 0.001, "class_weight_pos": round(w_pos, 4),
                               "class_weight_neg": round(w_neg, 4), "threshold": 0.5})
            mlflow.log_metrics(m)
            mlflow.log_dict(cm, "confusion_matrix.json")
            model.write().overwrite().save(MODEL_PATHS["logreg"])
        timings["logreg_s"] = t["seconds"]
        results["logreg"] = {**m, "confusion_matrix": cm}

        # ---------- b1) Baseline lineal ----------
        with timed("linreg") as t, mlflow.start_run(run_name="linreg_arr_delay"):
            linr = LinearRegression(featuresCol="features", labelCol="label_clip",
                                    maxIter=50, regParam=0.001)
            model = Pipeline(stages=linear_stages(linr)).fit(train)
            m = regression_metrics(model.transform(test))
            mlflow.log_params({**common, "model": "LinearRegression", "regParam": 0.001,
                               "clip_p01": p01, "clip_p99": p99})
            mlflow.log_metrics(m)
            model.write().overwrite().save(MODEL_PATHS["linreg"])
        timings["linreg_s"] = t["seconds"]
        results["linreg"] = m

        # ---------- b2) GBT (Spark MLlib) sobre muestra ----------
        with timed("gbt") as t, mlflow.start_run(run_name="gbt_arr_delay"):
            gbt_train = train.sample(fraction=args.gbt_sample, seed=SEED)
            # maxBins >= numero de aeropuertos (352) para que cada uno sea su propia categoria.
            gbt = GBTRegressor(featuresCol="features", labelCol="label_clip", maxIter=args.gbt_iter,
                               maxDepth=6, stepSize=0.1, maxBins=400, subsamplingRate=0.8,
                               seed=SEED)
            model = Pipeline(stages=tree_stages(gbt)).fit(gbt_train)
            m = regression_metrics(model.transform(test))
            mlflow.log_params({**common, "model": "GBTRegressor", "maxIter": args.gbt_iter,
                               "maxDepth": 6, "stepSize": 0.1, "maxBins": 400,
                               "train_sample_fraction": args.gbt_sample,
                               "clip_p01": p01, "clip_p99": p99})
            mlflow.log_metrics(m)
            model.write().overwrite().save(MODEL_PATHS["gbt"])
        timings["gbt_s"] = t["seconds"]
        results["gbt"] = m
        train.unpersist(); test.unpersist()

        # ---------- Predicciones en Gold ----------
        with timed("scoring + escritura gold") as t:
            best_reg = "gbt" if results["gbt"]["rmse"] <= results["linreg"]["rmse"] else "linreg"
            lr_model = PipelineModel.load(MODEL_PATHS["logreg"])
            reg_model = PipelineModel.load(MODEL_PATHS[best_reg])
            scored = lr_model.transform(feats).select(
                "vuelo_key", *[c for c in feats.columns if c != "vuelo_key"],
                prior_corrected(vector_to_array("probability")[1], w_pos, w_neg).alias("pred_prob"))
            scored = reg_model.transform(scored.select("vuelo_key", *INPUT_COLS, "pred_prob")) \
                .select("vuelo_key", "pred_prob", F.round("prediction", 1).alias("pred_min"))

            fact = spark.read.parquet(GOLD_PATHS["FACT_VUELOS"]).drop("pred_prob", "pred_min")
            cols = [c for c in spark.read.parquet(GOLD_PATHS["FACT_VUELOS"]).columns]
            out = fact.join(scored, "vuelo_key", "left").select(*cols)
            tmp = GOLD_PATHS["FACT_VUELOS"] + "_tmp"
            out.write.mode("overwrite").partitionBy("MONTH").parquet(tmp)
            swap_dir(tmp, GOLD_PATHS["FACT_VUELOS"])
            fact = spark.read.parquet(GOLD_PATHS["FACT_VUELOS"])
            chk = fact.agg(F.count("*").alias("n"), F.count("pred_prob").alias("n_prob"),
                           F.count("pred_min").alias("n_min")).first()
        timings["scoring_s"] = t["seconds"]
    except Exception as e:
        print(f"ERROR en modelos: {str(e).splitlines()[0][:300]}")
        raise

    lr_m, cm = results["logreg"], results["logreg"]["confusion_matrix"]
    print("\n=== a) Clasificacion ARR_DEL15: regresion logistica ponderada (prueba nov-dic) ===")
    print(f"ROC AUC {lr_m['roc_auc']:.4f} | precision {lr_m['precision']:.4f} | "
          f"recall {lr_m['recall']:.4f} | F1 {lr_m['f1']:.4f} | accuracy {lr_m['accuracy']:.4f}")
    print("Matriz de confusion (filas = real, columnas = predicho):")
    print(f"               pred 0      pred 1")
    print(f"  real 0  {cm['tn']:>11,} {cm['fp']:>11,}")
    print(f"  real 1  {cm['fn']:>11,} {cm['tp']:>11,}")
    print(f"Calibracion: prob. media ponderada {lr_m['mean_prob_raw']:.3f} | corregida "
          f"{lr_m['mean_prob_corrected']:.3f} | tasa real {lr_m['real_delay_rate_test']:.3f}")
    print("\n=== b) Regresion de minutos (ARR_DELAY recortado), prueba nov-dic ===")
    for k in ["linreg", "gbt"]:
        r = results[k]
        print(f"{k:<7} MAE {r['mae']:.2f} | RMSE {r['rmse']:.2f} | R2 {r['r2']:.4f} | "
              f"RMSE vs minutos sin recortar {r['rmse_sin_recorte']:.2f}")
    print(f"Modelo usado para pred_min: {best_reg}")
    print(f"\nGold FACT_VUELOS: {chk['n']:,} filas | con pred_prob {chk['n_prob']:,} | "
          f"con pred_min {chk['n_min']:,}")

    update_stage_metrics("models", **timings, best_regressor=best_reg, clip=[p01, p99],
                         logreg=results["logreg"], linreg=results["linreg"], gbt=results["gbt"])
    spark.stop()

    if not args.skip_mysql:
        from src.mysql_loader import load_gold_to_mysql
        load_gold_to_mysql()


if __name__ == "__main__":
    main()
