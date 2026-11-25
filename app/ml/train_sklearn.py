"""
Fase 4: modelos ligeros en scikit-learn para la app 2026 (sin Spark).

Replica src/train.py (Spark MLlib) con las mismas variables, split, pesos y recorte:
- 1B Clasificacion ARR_DEL15: LogisticRegression L2 ponderada, probabilidad corregida a la
     tasa base dentro del artefacto (ClasificadorCorregido).
- 1C Regresion de minutos: HistGradientBoostingRegressor sobre ARR_DELAY recortado con los
     mismos p01/p99 de data/models/clip_and_prior.json, early stopping con octubre.
- 1D Comparacion contra las predicciones de Spark guardadas en data/gold/fact_vuelos.
- 1E Guardado con joblib, metadata.json y registro en MLflow.
- Produccion: reentrena ambos modelos con ene-dic 2025 y la misma configuracion (pesos
  recalculados con ene-dic, misma correccion, mismo regParam equivalente, mismos p01/p99 de
  ene-oct y el numero de arboles fijo que eligio el early stopping, sin early stopping).

Archivos:
- app/models/clasificador.joblib, regresor.joblib: PRODUCCION (ene-dic), los usa la app.
- app/models/eval/clasificador_ene_oct.joblib, regresor_ene_oct.joblib: EVALUACION (ene-oct),
  de donde salen todas las metricas de prueba (nov-dic) y la comparacion con Spark.

Uso: .venv/bin/python -m app.ml.train_sklearn
"""
import json
import os
import sys
import time

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
from app.ml import sin_spark  # noqa: E402  (primero: bloquea pyspark en este proceso)
import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sklearn  # noqa: E402
from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import brier_score_loss, roc_auc_score  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import (FunctionTransformer, OneHotEncoder,  # noqa: E402
                                   OrdinalEncoder, StandardScaler)

from app.ml import datos, metricas  # noqa: E402
from app.ml.features import (CATEGORICAL, CYCLIC_SMALL, INPUT_COLS, LEAKAGE_COLS,  # noqa: E402
                             NUMERIC, check_no_leakage)
from app.ml.modelos import (ClasificadorCorregido, RegresorRecortado,  # noqa: E402
                            mes_desde_ciclico)

ROOT = datos.ROOT
MODELS_DIR = os.path.join(ROOT, "app", "models")
EVAL_DIR = os.path.join(MODELS_DIR, "eval")
PROD_PATHS = {"cls": os.path.join(MODELS_DIR, "clasificador.joblib"),
              "reg": os.path.join(MODELS_DIR, "regresor.joblib"),
              "cls_mes": os.path.join(MODELS_DIR, "clasificador_mes.joblib")}
EVAL_PATHS = {"cls": os.path.join(EVAL_DIR, "clasificador_ene_oct.joblib"),
              "reg": os.path.join(EVAL_DIR, "regresor_ene_oct.joblib")}
CLIP_PATH = os.path.join(ROOT, "data", "models", "clip_and_prior.json")
SPARK_MODELS = os.path.join(ROOT, "data", "models")
GOLD = os.path.join(ROOT, "data", "gold")
MLFLOW_URI = f"sqlite:///{os.path.join(ROOT, 'mlflow.db')}"
EXPERIMENT = "retrasos_vuelos_sklearn"
SEED = 42
REG_PARAM = 0.001          # regParam de la LogisticRegression de Spark (src/train.py)
MAX_CAT_TREE = 255         # limite de categorias de HistGradientBoosting (max_bins=255)


def du_mb(path):
    if os.path.isfile(path):
        return os.path.getsize(path) / 2**20
    return sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(path) for f in fs) / 2**20


def print_cls(nombre, m, cm):
    print(f"[{nombre}] ROC AUC {m['roc_auc']:.4f} | precision {m['precision']:.4f} | "
          f"recall {m['recall']:.4f} | F1 {m['f1']:.4f} | accuracy {m['accuracy']:.4f}")
    print("  Matriz de confusion (filas = real, columnas = predicho):")
    print(f"               pred 0      pred 1\n  real 0  {cm['tn']:>11,} {cm['fp']:>11,}\n"
          f"  real 1  {cm['fn']:>11,} {cm['tp']:>11,}")
    print(f"  Calibracion: prob. media corregida {m['mean_prob_corrected']:.4f} | "
          f"tasa real {m['real_delay_rate_test']:.4f}")


def print_reg(nombre, m):
    print(f"[{nombre}] MAE {m['mae']:.3f} | RMSE {m['rmse']:.3f} | R2 {m['r2']:.4f} | "
          f"RMSE vs minutos sin recortar {m['rmse_sin_recorte']:.3f}")


# ---------------------------------------------------------------- 1B
def clasificador_pipeline(n_train, con_mes=False):
    # Spark: regParam * 1/2 ||b||^2 + (1/sum w) * sum w_i * logloss_i.
    # sklearn: 1/2 ||b||^2 + C * sum w_i * logloss_i. Dividiendo sklearn por C*sum(w) quedan
    # iguales si 1/(C*sum w) = regParam, o sea C = 1/(regParam*sum w). Con pesos balanceados
    # sum w = n_pos*n/(2n_pos) + n_neg*n/(2n_neg) = n, asi que C = 1/(regParam*n).
    C = 1.0 / (REG_PARAM * n_train)
    partes = [("ohe", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL + CYCLIC_SMALL)]
    if con_mes:
        # Variante: el mes tambien como categorico (one-hot), ademas de month_sin/month_cos.
        # El mes se reconstruye de month_sin/month_cos, asi la entrada sigue siendo la misma.
        partes.append(("mes", Pipeline([
            ("mes", FunctionTransformer(mes_desde_ciclico)),
            ("ohe", OneHotEncoder(handle_unknown="ignore"))]), ["month_sin", "month_cos"]))
    # Como StandardScaler(withMean=False) de Spark: divide por la desviacion, no centra.
    partes.append(("num", StandardScaler(with_mean=False), NUMERIC))
    pre = ColumnTransformer(partes, sparse_threshold=1.0)
    return Pipeline([
        ("pre", pre),
        # Spark LogisticRegression tiene standardization=True por defecto: internamente
        # escala TODAS las columnas (one-hot incluidas) por su desviacion ponderada y aplica la
        # penalizacion L2 sobre esos coeficientes. Este escalado sin centrar (con pesos) lo
        # reproduce, para que regParam signifique lo mismo en ambos.
        ("std", StandardScaler(with_mean=False)),
        ("lr", LogisticRegression(C=C, class_weight="balanced", solver="lbfgs", max_iter=300,
                                  tol=1e-6)),
    ]), C


def paso_1b(tr, te, clip):
    print("\n===== 1B Clasificacion ARR_DEL15 (LogisticRegression) =====")
    n = len(tr)
    n_pos = int(tr["ARR_DEL15"].sum())
    w_pos, w_neg = n / (2 * n_pos), n / (2 * (n - n_pos))
    assert abs(w_pos - clip["w_pos"]) < 1e-9 and abs(w_neg - clip["w_neg"]) < 1e-9
    pipe, C = clasificador_pipeline(n)
    print(f"Pesos balanceados n/(2*n_clase): w_pos {w_pos:.4f}, w_neg {w_neg:.4f} | "
          f"C = 1/(regParam*n) = 1/({REG_PARAM}*{n:,}) = {C:.6e}")
    w = np.where(tr["ARR_DEL15"].to_numpy() == 1, w_pos, w_neg)
    t = time.time()
    pipe.fit(tr[INPUT_COLS], tr["ARR_DEL15"], std__sample_weight=w)
    secs = time.time() - t
    lr = pipe.named_steps["lr"]
    print(f"Entrenamiento: {secs:.1f} s | iteraciones lbfgs {lr.n_iter_[0]} | "
          f"coeficientes {lr.coef_.shape[1]}")
    model = ClasificadorCorregido(pipe, w_pos, w_neg)
    p_corr = model.predict_proba(te[INPUT_COLS])[:, 1]
    y_pred = model.predict(te[INPUT_COLS])
    p_raw = model.predict_proba_ponderada(te[INPUT_COLS])
    m, cm = metricas.clasificacion(te["ARR_DEL15"], p_corr, y_pred)
    m["mean_prob_raw"] = float(p_raw.mean())
    print_cls("sklearn prueba nov-dic", m, cm)
    print(f"  (prob. media ponderada sin corregir {m['mean_prob_raw']:.4f}; umbral 0.5 ponderado "
          f"= {model.umbral_corregido:.4f} en escala corregida)")
    return model, m, cm, secs, p_corr, y_pred, {"C": C, "w_pos": w_pos, "w_neg": w_neg,
                                                "n_iter": int(lr.n_iter_[0])}


# ---------------------------------------------------------------- 1C
def arbol_pre():
    # ORIGIN y DEST tienen ~350 categorias y HistGradientBoosting admite hasta 255: se
    # conservan las 254 mas frecuentes del entrenamiento y el resto se agrupa en una sola
    # categoria "infrecuente" (OTRO). Un aeropuerto nunca visto queda como faltante (NaN).
    enc = OrdinalEncoder(max_categories=MAX_CAT_TREE, handle_unknown="use_encoded_value",
                         unknown_value=np.nan, encoded_missing_value=np.nan, dtype=np.float64)
    return ColumnTransformer(
        [("cat", enc, CATEGORICAL), ("rest", "passthrough", CYCLIC_SMALL + NUMERIC)],
        verbose_feature_names_out=False)


def hgb(loss="squared_error", **kw):
    # Spark GBT: maxDepth=6, stepSize=0.1, subsamplingRate=0.8. Aqui profundidad 6 y tasa 0.1;
    # el numero de arboles lo decide el early stopping con octubre.
    return HistGradientBoostingRegressor(
        loss=loss, learning_rate=0.1, max_depth=6, max_leaf_nodes=63,
        min_samples_leaf=100, categorical_features=[0, 1, 2], random_state=SEED, **kw)


def paso_1c(tr, te, clip):
    print("\n===== 1C Regresion de minutos (HistGradientBoostingRegressor) =====")
    p01, p99 = clip["p01"], clip["p99"]
    y_tr = tr["ARR_DELAY"].clip(p01, p99)
    y_te = te["ARR_DELAY"].clip(p01, p99)
    print(f"Etiqueta: ARR_DELAY recortado a [{p01:.0f}, {p99:.0f}] (clip_and_prior.json)")
    print("Muestra: entrenamiento COMPLETO (Spark GBT uso 30%; aqui cabe en memoria)")

    # a) Early stopping: entrena ene-sep, valida oct (nunca toca nov-dic).
    fit_m = tr["MONTH"] <= 9
    val_m = tr["MONTH"] == 10
    pre = arbol_pre().fit(tr.loc[fit_m, INPUT_COLS])
    cats = pre.named_transformers_["cat"]
    for col, ci in zip(CATEGORICAL, cats.categories_):
        infreq = cats.infrequent_categories_[CATEGORICAL.index(col)]
        n_inf = 0 if infreq is None else len(infreq)
        share = 0.0 if infreq is None else tr.loc[fit_m, col].isin(infreq).mean()
        print(f"  {col}: {len(ci)} categorias -> {len(ci) - n_inf} propias + "
              f"{'1 OTRO' if n_inf else '0 OTRO'} ({n_inf} agrupadas, {100 * share:.2f}% de vuelos)")
    t = time.time()
    m_es = hgb(max_iter=1000, early_stopping=True, n_iter_no_change=20, tol=1e-7,
               scoring="loss")
    m_es.fit(pre.transform(tr.loc[fit_m, INPUT_COLS]), y_tr[fit_m],
             X_val=pre.transform(tr.loc[val_m, INPUT_COLS]), y_val=y_tr[val_m])
    secs_es = time.time() - t
    # validation_score_ es -loss (mayor es mejor); [0] es antes del primer arbol. La perdida
    # squared_error de HGB es MSE/2, por eso RMSE = sqrt(2 * loss).
    best_iter = int(np.argmax(m_es.validation_score_))
    print(f"Early stopping (ene-sep entrena, oct valida): {m_es.n_iter_} arboles corridos, mejor "
          f"en {best_iter} (RMSE val {np.sqrt(-2 * m_es.validation_score_[best_iter]):.3f}) | "
          f"{secs_es:.1f} s")

    # b) Modelo final: ene-oct completo con el numero de arboles elegido.
    pipe = Pipeline([("pre", arbol_pre()), ("hgb", hgb(max_iter=best_iter, early_stopping=False))])
    t = time.time()
    pipe.fit(tr[INPUT_COLS], y_tr)
    secs = time.time() - t
    print(f"Modelo final ene-oct ({len(tr):,} vuelos, {best_iter} arboles): {secs:.1f} s")
    model = RegresorRecortado(pipe, p01, p99)
    pred = model.predict(te[INPUT_COLS])
    m = metricas.regresion(y_te, te["ARR_DELAY"], pred)
    print_reg("sklearn prueba nov-dic", m)
    return model, m, secs, secs_es, pred, {"best_iter": best_iter}


# ---------------------------------------------------------------- 1D
def spark_gold_test(te):
    """Predicciones de Spark (Gold) para los completados de nov-dic, alineadas con te.
    Gold tiene vuelo_key (xxhash64 de la clave de negocio en Spark) y llaves sustitutas; se
    reconstruye la clave de negocio con las dimensiones y se une por ella."""
    import pyarrow.dataset as ds
    cols = ["vuelo_key", "fecha_key", "aerolinea_key", "origen_key", "destino_key",
            "OP_CARRIER_FL_NUM", "CRS_DEP_TIME", "CANCELLED", "DIVERTED", "ARR_DEL15",
            "ARR_DELAY", "pred_prob", "pred_min", "MONTH"]
    f = ds.dataset(os.path.join(GOLD, "fact_vuelos"), format="parquet", partitioning="hive")
    g = f.to_table(columns=cols, filter=ds.field("MONTH").isin([11, 12])).to_pandas()
    g = g[(g.CANCELLED == 0) & (g.DIVERTED == 0)]
    aer = pd.read_parquet(os.path.join(GOLD, "dim_aerolinea")).set_index("aerolinea_key")["codigo"]
    apt = pd.read_parquet(os.path.join(GOLD, "dim_aeropuerto")).set_index("aeropuerto_key")["iata"]
    g["OP_UNIQUE_CARRIER"] = g["aerolinea_key"].map(aer)
    g["ORIGIN"] = g["origen_key"].map(apt)
    g["DEST"] = g["destino_key"].map(apt)
    g["FL_DATE"] = pd.to_datetime(g["fecha_key"].astype(str), format="%Y%m%d")
    key = datos.BUSINESS_KEY
    left = te[key + ["ARR_DEL15", "ARR_DELAY"]].copy()
    left["FL_DATE"] = pd.to_datetime(left["FL_DATE"])
    for c in ["OP_UNIQUE_CARRIER", "ORIGIN", "DEST"]:
        left[c] = left[c].astype(str)
    left["_i"] = np.arange(len(left))
    j = left.merge(g[key + ["vuelo_key", "pred_prob", "pred_min", "ARR_DEL15", "ARR_DELAY"]],
                   on=key, how="left", validate="one_to_one", suffixes=("", "_gold"))
    j = j.sort_values("_i").reset_index(drop=True)
    print(f"Gold nov-dic completados: {len(g):,} | prueba sklearn: {len(te):,} | unidos: "
          f"{j['vuelo_key'].notna().sum():,} | vuelo_key distintos: {j['vuelo_key'].nunique():,}")
    assert j["vuelo_key"].notna().all(), "hay vuelos de prueba sin prediccion de Spark"
    assert (j["ARR_DEL15_gold"].to_numpy() == te["ARR_DEL15"].to_numpy()).all()
    assert np.allclose(j["ARR_DELAY_gold"].to_numpy(), te["ARR_DELAY"].to_numpy())
    return j


def paso_1d(tr, te, clip, sk):
    print("\n===== 1D Comparacion contra Spark (mismo set de prueba) =====")
    j = spark_gold_test(te)
    w_pos, w_neg = clip["w_pos"], clip["w_neg"]
    # Gold guarda la prob. corregida. La decision de Spark (prob. ponderada > 0.5) equivale a
    # prob. corregida > w_neg/(w_pos+w_neg), porque la correccion es monotona.
    thr = w_neg / (w_pos + w_neg)
    sp_m, sp_cm = metricas.clasificacion(te["ARR_DEL15"], j["pred_prob"], (j["pred_prob"] > thr).astype(int))
    y_clip = te["ARR_DELAY"].clip(clip["p01"], clip["p99"])
    sp_r = metricas.regresion(y_clip, te["ARR_DELAY"], j["pred_min"])
    print_cls("Spark (Gold) prueba nov-dic", sp_m, sp_cm)
    print_reg("Spark GBT (Gold, pred_min redondeado a 0.1)", sp_r)

    sm = json.load(open(os.path.join(ROOT, "reports", "stage_metrics.json")))["models"]
    print(f"  Control vs stage_metrics.json de Spark: AUC {sm['logreg']['roc_auc']:.4f}, "
          f"F1 {sm['logreg']['f1']:.4f}, MAE GBT {sm['gbt']['mae']:.3f}, RMSE GBT {sm['gbt']['rmse']:.3f}")

    tabla = [
        {"modelo": "Spark LogisticRegression", "roc_auc": sp_m["roc_auc"], "f1": sp_m["f1"],
         "mae": None, "rmse": None, "train_s": sm["logreg_s"],
         "disk_mb": du_mb(os.path.join(SPARK_MODELS, "logreg"))},
        {"modelo": "sklearn LogisticRegression", "roc_auc": sk["cls_m"]["roc_auc"],
         "f1": sk["cls_m"]["f1"], "mae": None, "rmse": None, "train_s": sk["cls_s"],
         "disk_mb": sk.get("cls_mb")},
        {"modelo": "Spark GBTRegressor (30%)", "roc_auc": None, "f1": None, "mae": sp_r["mae"],
         "rmse": sp_r["rmse"], "train_s": sm["gbt_s"],
         "disk_mb": du_mb(os.path.join(SPARK_MODELS, "gbt"))},
        {"modelo": "sklearn HistGradientBoosting (100%)", "roc_auc": None, "f1": None,
         "mae": sk["reg_m"]["mae"], "rmse": sk["reg_m"]["rmse"], "train_s": sk["reg_s"],
         "disk_mb": sk.get("reg_mb")},
    ]
    d_auc = abs(sp_m["roc_auc"] - sk["cls_m"]["roc_auc"])
    d_mae = abs(sp_r["mae"] - sk["reg_m"]["mae"])
    print(f"Diferencias: |AUC| {d_auc:.4f} (limite 0.01) -> {'OK' if d_auc < 0.01 else 'NO'} | "
          f"|MAE| {d_mae:.3f} min (limite 1) -> {'OK' if d_mae < 1 else 'NO'}")

    rng = np.random.default_rng(SEED)
    idx = np.sort(rng.choice(len(te), 5, replace=False))
    ej = te.loc[idx, ["FL_DATE", "OP_UNIQUE_CARRIER", "ORIGIN", "DEST", "dep_hour", "ARR_DEL15",
                      "ARR_DELAY"]].copy()
    ej["vuelo_key"] = j.loc[idx, "vuelo_key"].astype("int64").to_numpy()
    ej["prob_spark"] = j.loc[idx, "pred_prob"].round(3).to_numpy()
    ej["prob_sklearn"] = sk["p_corr"][idx].round(3)
    ej["min_spark"] = j.loc[idx, "pred_min"].to_numpy()
    ej["min_sklearn"] = sk["pred_min"][idx].round(1)
    print("\n5 vuelos de prueba (seed 42):")
    print(ej.to_string(index=False))
    corr_p = np.corrcoef(j["pred_prob"], sk["p_corr"])[0, 1]
    corr_m = np.corrcoef(j["pred_min"], sk["pred_min"])[0, 1]
    print(f"Correlacion Spark vs sklearn en toda la prueba: prob {corr_p:.4f} | minutos {corr_m:.4f}")
    return {"spark_cls": sp_m, "spark_cm": sp_cm, "spark_reg": sp_r, "tabla": tabla,
            "diff_auc": d_auc, "diff_mae": d_mae, "corr_prob": corr_p, "corr_min": corr_m}


def print_tabla(tabla):
    f = lambda v, fmt: "-" if v is None else format(v, fmt)
    print(f"\n{'modelo':<38}| {'ROC AUC':>7} | {'F1':>6} | {'MAE':>6} | {'RMSE':>6} | "
          f"{'entren. s':>9} | {'disco MB':>8}")
    for r in tabla:
        print(f"{r['modelo']:<38}| {f(r['roc_auc'], '.4f'):>7} | {f(r['f1'], '.4f'):>6} | "
              f"{f(r['mae'], '.3f'):>6} | {f(r['rmse'], '.3f'):>6} | {f(r['train_s'], '.1f'):>9} | "
              f"{f(r['disk_mb'], '.2f'):>8}")


# ---------------------------------------------------------------- produccion
def paso_produccion(tr, te, clip, n_trees):
    """Modelos de produccion con ene-dic 2025. Misma configuracion que los de evaluacion; no
    hay set de prueba propio, solo se reporta el ajuste dentro del entrenamiento."""
    print("\n===== Produccion: reentrenamiento con ene-dic 2025 =====")
    full = pd.concat([tr, te], ignore_index=True)
    n = len(full)
    n_pos = int(full["ARR_DEL15"].sum())
    w_pos, w_neg = n / (2 * n_pos), n / (2 * (n - n_pos))
    pipe, C = clasificador_pipeline(n)
    print(f"Vuelos ene-dic: {n:,} | tasa real {n_pos / n:.4f} | pesos w_pos {w_pos:.4f}, "
          f"w_neg {w_neg:.4f} | C = 1/({REG_PARAM}*{n:,}) = {C:.6e}")
    t = time.time()
    pipe.fit(full[INPUT_COLS], full["ARR_DEL15"],
             std__sample_weight=np.where(full["ARR_DEL15"].to_numpy() == 1, w_pos, w_neg))
    cls_s = time.time() - t
    n_iter = int(pipe.named_steps["lr"].n_iter_[0])
    cls = ClasificadorCorregido(pipe, w_pos, w_neg)
    print(f"Clasificador: {cls_s:.1f} s | iteraciones lbfgs {n_iter}")

    # Variante: mes categorico ademas de month_sin/month_cos; todo lo demas igual.
    pipe_m, _ = clasificador_pipeline(n, con_mes=True)
    t = time.time()
    pipe_m.fit(full[INPUT_COLS], full["ARR_DEL15"],
               std__sample_weight=np.where(full["ARR_DEL15"].to_numpy() == 1, w_pos, w_neg))
    cls_mes_s = time.time() - t
    n_iter_mes = int(pipe_m.named_steps["lr"].n_iter_[0])
    cls_mes = ClasificadorCorregido(pipe_m, w_pos, w_neg)
    print(f"Clasificador variante con mes categorico: {cls_mes_s:.1f} s | iteraciones lbfgs "
          f"{n_iter_mes} | coeficientes {pipe_m.named_steps['lr'].coef_.shape[1]}")

    y = full["ARR_DELAY"].clip(clip["p01"], clip["p99"])
    rpipe = Pipeline([("pre", arbol_pre()), ("hgb", hgb(max_iter=n_trees, early_stopping=False))])
    t = time.time()
    rpipe.fit(full[INPUT_COLS], y)
    reg_s = time.time() - t
    reg = RegresorRecortado(rpipe, clip["p01"], clip["p99"])
    print(f"Regresor: {reg_s:.1f} s | {rpipe.named_steps['hgb'].n_iter_} arboles fijos "
          f"(recorte [{clip['p01']:.0f}, {clip['p99']:.0f}] de ene-oct)")

    # Ajuste DENTRO del entrenamiento (no es una metrica de prueba).
    p = cls.predict_proba(full[INPUT_COLS])[:, 1]
    p_mes = cls_mes.predict_proba(full[INPUT_COLS])[:, 1]
    m = reg.predict(full[INPUT_COLS])
    por_mes = (pd.DataFrame({"MONTH": full["MONTH"].to_numpy(), "y": full["ARR_DEL15"].to_numpy(),
                             "p": p, "p_mes": p_mes, "min_real": y.to_numpy(), "min_pred": m})
               .groupby("MONTH").agg(vuelos=("y", "size"), tasa_real=("y", "mean"),
                                     prob_media_corregida=("p", "mean"),
                                     prob_media_variante_mes=("p_mes", "mean"),
                                     minutos_reales_media=("min_real", "mean"),
                                     minutos_predichos_media=("min_pred", "mean")))
    print(f"Ajuste en ene-dic (dentro del entrenamiento): prob media corregida {p.mean():.4f} | "
          f"tasa real {n_pos / n:.4f} | minutos predichos {m.mean():.2f} vs reales "
          f"(recortados) {y.mean():.2f}")
    print(por_mes.round(4).to_string())
    info = {"filas": n, "fechas": [str(pd.Timestamp(full["FL_DATE"].min()).date()),
                                   str(pd.Timestamp(full["FL_DATE"].max()).date())],
            "w_pos": w_pos, "w_neg": w_neg, "C": C, "iteraciones_lbfgs": n_iter,
            "arboles": int(rpipe.named_steps["hgb"].n_iter_),
            "umbral_en_prob_corregida": cls.umbral_corregido,
            "tiempos_s": {"clasificador": cls_s, "clasificador_mes": cls_mes_s, "regresor": reg_s},
            "variante_mes": {"iteraciones_lbfgs": n_iter_mes,
                             "coeficientes": int(pipe_m.named_steps["lr"].coef_.shape[1]),
                             "roc_auc_en_entrenamiento": float(roc_auc_score(full["ARR_DEL15"], p_mes)),
                             "brier_en_entrenamiento": float(brier_score_loss(full["ARR_DEL15"], p_mes))},
            "actual_en_entrenamiento": {
                "roc_auc": float(roc_auc_score(full["ARR_DEL15"], p)),
                "brier": float(brier_score_loss(full["ARR_DEL15"], p))},
            "ajuste_en_entrenamiento": {
                "nota": "calculado sobre los mismos datos de entrenamiento (ene-dic); NO es una "
                        "metrica de prueba",
                "tasa_real": n_pos / n, "prob_media_corregida": float(p.mean()),
                "prob_media_variante_mes": float(p_mes.mean()),
                "minutos_reales_media_recortada": float(y.mean()),
                "minutos_predichos_media": float(m.mean()),
                "por_mes": por_mes.round(4).reset_index().to_dict("records")}}
    return cls, cls_mes, reg, info


# ---------------------------------------------------------------- main
def main():
    check_no_leakage()
    assert not set(INPUT_COLS) & set(LEAKAGE_COLS)
    clip = json.load(open(CLIP_PATH))
    t0 = time.time()
    df = datos.load_completed()
    tr, te = datos.split(df)
    del df
    print(f"Datos: entrenamiento {len(tr):,} | prueba {len(te):,} | {time.time() - t0:.1f} s")

    cls, cls_m, cls_cm, cls_s, p_corr, _, cls_info = paso_1b(tr, te, clip)
    reg, reg_m, reg_s, reg_es_s, pred_min, reg_info = paso_1c(tr, te, clip)

    os.makedirs(EVAL_DIR, exist_ok=True)
    cls_path, reg_path = EVAL_PATHS["cls"], EVAL_PATHS["reg"]
    joblib.dump(cls, cls_path, compress=3)
    joblib.dump(reg, reg_path, compress=3)
    sk = {"cls_m": cls_m, "cls_s": cls_s, "reg_m": reg_m, "reg_s": reg_s, "p_corr": p_corr,
          "pred_min": pred_min, "cls_mb": du_mb(cls_path), "reg_mb": du_mb(reg_path)}
    cmp = paso_1d(tr, te, clip, sk)
    print_tabla(cmp["tabla"])

    prod_cls, prod_cls_mes, prod_reg, prod = paso_produccion(tr, te, clip, reg_info["best_iter"])
    joblib.dump(prod_cls, PROD_PATHS["cls"], compress=3)
    joblib.dump(prod_cls_mes, PROD_PATHS["cls_mes"], compress=3)
    joblib.dump(prod_reg, PROD_PATHS["reg"], compress=3)

    print("\n===== 1E Guardado =====")
    for p in [PROD_PATHS["cls"], PROD_PATHS["cls_mes"], PROD_PATHS["reg"], cls_path, reg_path]:
        mb = du_mb(p)
        print(f"{os.path.relpath(p, ROOT)}: {mb:.2f} MB {'(OK < 50 MB)' if mb < 50 else '(EXCEDE 50 MB)'}")
        assert mb < 50
    fechas = lambda d: [str(pd.Timestamp(d["FL_DATE"].min()).date()),
                        str(pd.Timestamp(d["FL_DATE"].max()).date())]
    meta = {
        "sklearn_version": sklearn.__version__, "pandas_version": pd.__version__,
        "python_version": sys.version.split()[0],
        "entrenado_en": time.strftime("%Y-%m-%d %H:%M:%S"),
        "variables": {"input_cols": INPUT_COLS, "categorical": CATEGORICAL,
                      "cyclic_small": CYCLIC_SMALL, "numeric": NUMERIC,
                      "leakage_cols_prohibidas": LEAKAGE_COLS},
        "clasificador_activo": "clasificador.joblib",
        "clasificador_activo_opciones": {
            "clasificador.joblib": "produccion ene-dic, mismas 18 variables que Spark (mes solo "
                                   "como month_sin/month_cos)",
            "clasificador_mes.joblib": "variante produccion ene-dic: ademas el mes como "
                                       "categorico (one-hot); misma configuracion en lo demas",
            "nota": "pendiente de decidir con la validacion 2026 (Paso 5). La app debe cargar el "
                    "clasificador con app.ml.modelos.cargar_modelos(), que lee esta clave."},
        "recorte_arr_delay": {"p01": clip["p01"], "p99": clip["p99"],
                              "fuente": "data/models/clip_and_prior.json (Spark, p01/p99 de ene-oct); "
                                        "se usan en evaluacion y en produccion"},
        "modelo_produccion": {
            "nota": "Es el que usa la app. Entrenado con ene-dic 2025 con la misma configuracion "
                    "que el de evaluacion. NO tiene metrica de prueba propia: no quedan datos de "
                    "2025 fuera del entrenamiento. Las metricas de referencia son las del modelo "
                    "de evaluacion (seccion metricas_prueba).",
            "archivos": {"clasificador": "clasificador.joblib",
                         "clasificador_variante_mes": "clasificador_mes.joblib",
                         "regresor": "regresor.joblib"},
            "entrenamiento": {"meses": list(range(1, 13)), "fechas": prod["fechas"],
                              "filas": prod["filas"]},
            "pesos_clase": {"w_pos": prod["w_pos"], "w_neg": prod["w_neg"],
                            "formula": "n/(2*n_clase), recalculados con ene-dic"},
            "clasificador": {"modelo": "LogisticRegression L2", "C": prod["C"],
                             "equivale_regParam_spark": REG_PARAM,
                             "iteraciones": prod["iteraciones_lbfgs"], "umbral_ponderado": 0.5,
                             "umbral_en_prob_corregida": prod["umbral_en_prob_corregida"],
                             "predict_proba": "corregida a la tasa base de ene-dic",
                             **{f"{k}_en_entrenamiento": v
                                for k, v in prod["actual_en_entrenamiento"].items()}},
            "clasificador_variante_mes": {
                "archivo": "clasificador_mes.joblib",
                "cambio": "one-hot de MONTH (reconstruido de month_sin/month_cos dentro del "
                          "pipeline) ademas de month_sin/month_cos; mismos pesos, correccion, C "
                          "y escalado",
                **prod["variante_mes"]},
            "regresor": {"modelo": "HistGradientBoostingRegressor", "arboles": prod["arboles"],
                         "arboles_origen": "fijo: el que eligio el early stopping del modelo de "
                                           "evaluacion (ene-sep entrena, oct valida)",
                         "early_stopping": False, "max_depth": 6, "learning_rate": 0.1,
                         "categoricas_arbol": f"top {MAX_CAT_TREE - 1} por frecuencia + OTRO"},
            "metricas_prueba": None,
            "ajuste_en_entrenamiento": prod["ajuste_en_entrenamiento"],
            "tiempos_s": prod["tiempos_s"],
        },
        "modelo_evaluacion": {
            "nota": "Entrenado con ene-oct y evaluado con nov-dic. De aqui salen TODAS las "
                    "metricas reportadas (metricas_prueba, tabla_comparacion, verificaciones).",
            "archivos": {"clasificador": "eval/clasificador_ene_oct.joblib",
                         "regresor": "eval/regresor_ene_oct.joblib"},
            "split": {"entrenamiento": {"meses": datos.TRAIN_MONTHS, "fechas": fechas(tr), "filas": len(tr)},
                      "prueba": {"meses": datos.TEST_MONTHS, "fechas": fechas(te), "filas": len(te)},
                      "validacion_early_stopping": "octubre 2025 (dentro de entrenamiento)"},
            "pesos_clase": {"w_pos": clip["w_pos"], "w_neg": clip["w_neg"],
                            "formula": "n/(2*n_clase) con ene-oct (iguales a Spark)"},
            "clasificador": {"modelo": "LogisticRegression L2", "C": cls_info["C"],
                             "equivale_regParam_spark": REG_PARAM,
                             "iteraciones": cls_info["n_iter"], "umbral_ponderado": 0.5,
                             "umbral_en_prob_corregida": cls.umbral_corregido},
            "regresor": {"modelo": "HistGradientBoostingRegressor",
                         "arboles": reg_info["best_iter"], "max_depth": 6, "learning_rate": 0.1,
                         "muestra_entrenamiento": 1.0},
            "tiempos_s": {"clasificador": cls_s, "regresor_final": reg_s,
                          "regresor_early_stopping": reg_es_s},
        },
        "metricas_prueba": {"nota": "modelo de evaluacion (ene-oct) sobre la prueba nov-dic",
                            "sklearn_clasificador": cls_m, "sklearn_confusion": cls_cm,
                            "sklearn_regresor": reg_m, "spark_clasificador": cmp["spark_cls"],
                            "spark_confusion": cmp["spark_cm"], "spark_regresor": cmp["spark_reg"],
                            "diff_auc": cmp["diff_auc"], "diff_mae": cmp["diff_mae"],
                            "corr_prob": cmp["corr_prob"], "corr_min": cmp["corr_min"]},
        "tabla_comparacion": cmp["tabla"],
    }
    meta_path = os.path.join(MODELS_DIR, "metadata.json")
    with open(meta_path, "w") as fh:
        json.dump(meta, fh, indent=2, ensure_ascii=False)
    print(f"{os.path.relpath(meta_path, ROOT)} escrito")

    import mlflow
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name="sklearn_logreg_hgb") as run:
        mlflow.log_params({
            "train_months": "1-10", "test_months": "11-12", "train_rows": len(tr),
            "test_rows": len(te), "features": ",".join(INPUT_COLS),
            "sklearn_version": sklearn.__version__, "lr_C": cls_info["C"],
            "lr_equiv_regParam": REG_PARAM, "class_weight": "balanced", "threshold": 0.5,
            "w_pos": round(clip["w_pos"], 4), "w_neg": round(clip["w_neg"], 4),
            "hgb_trees": reg_info["best_iter"], "hgb_max_depth": 6, "hgb_learning_rate": 0.1,
            "hgb_sample_fraction": 1.0, "hgb_val": "octubre", "clip_p01": clip["p01"],
            "clip_p99": clip["p99"], "prod_train_months": "1-12", "prod_train_rows": prod["filas"],
            "prod_w_pos": round(prod["w_pos"], 4), "prod_w_neg": round(prod["w_neg"], 4),
            "prod_lr_C": prod["C"], "prod_hgb_trees": prod["arboles"]})
        mlflow.log_metrics({**{f"cls_{k}": v for k, v in cls_m.items()},
                            **{f"reg_{k}": v for k, v in reg_m.items()},
                            **{f"spark_cls_{k}": v for k, v in cmp["spark_cls"].items()},
                            **{f"spark_reg_{k}": v for k, v in cmp["spark_reg"].items()},
                            "diff_auc": cmp["diff_auc"], "diff_mae": cmp["diff_mae"],
                            "cls_train_s": cls_s, "reg_train_s": reg_s,
                            "cls_disk_mb": sk["cls_mb"], "reg_disk_mb": sk["reg_mb"],
                            "prod_insample_tasa_real": prod["ajuste_en_entrenamiento"]["tasa_real"],
                            "prod_insample_prob_media": prod["ajuste_en_entrenamiento"]["prob_media_corregida"]})
        mlflow.log_artifact(meta_path)
        print(f"MLflow: experimento '{EXPERIMENT}', run {run.info.run_id}")
    assert not sin_spark.cargados(), sin_spark.cargados()
    print(f"\nTotal: {time.time() - t0:.1f} s | pyspark importado: {'pyspark' in sys.modules}")


if __name__ == "__main__":
    main()
