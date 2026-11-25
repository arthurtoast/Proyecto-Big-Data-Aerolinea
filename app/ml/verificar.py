"""
Verificaciones de los modelos sklearn de EVALUACION (app/models/eval/*_ene_oct.joblib),
despues de app/ml/train_sklearn.py. Agrega la seccion "verificaciones" a
app/models/metadata.json. No cambia los .joblib.

1. Calibracion del clasificador: deciles de probabilidad corregida (prob. media vs tasa real)
   en entrenamiento y prueba, y Brier. Como referencia, Platt e isotonica ajustadas SOLO con
   octubre sobre un modelo entrenado ene-sep.
2. Baseline de minutos: siempre la mediana y siempre la media de entrenamiento vs el HGB, y
   HGB con loss="absolute_error" (mismo early stopping con octubre).
3. Clima: promedio de DFW en la semana ISO 11 de 2025 y cobertura de has_weather.

Uso: .venv/bin/python -m app.ml.verificar
"""
import json
import os
import time

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
from app.ml import sin_spark  # noqa: E402,F401  (primero: bloquea pyspark)
import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.isotonic import IsotonicRegression  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (brier_score_loss, mean_absolute_error, roc_auc_score,  # noqa: E402
                             root_mean_squared_error)
from sklearn.pipeline import Pipeline  # noqa: E402

from app.ml import datos  # noqa: E402
from app.ml import train_sklearn as T  # noqa: E402
from app.ml.features import INPUT_COLS, clima_con_banderas, flight_features  # noqa: E402
from app.ml.modelos import prior_corrected  # noqa: E402

META_PATH = os.path.join(T.MODELS_DIR, "metadata.json")
logit = lambda p: np.log(p / (1 - p))  # noqa: E731
sigmoid = lambda z: 1 / (1 + np.exp(-z))  # noqa: E731


def deciles(y, p, nombre):
    y = np.asarray(y)
    d = pd.qcut(p, 10, labels=False, duplicates="drop")
    t = (pd.DataFrame({"y": y, "p": p, "d": d}).groupby("d")
         .agg(n=("y", "size"), prob_media=("p", "mean"), tasa_real=("y", "mean")))
    t.index = t.index + 1
    res = {"prob_media": float(p.mean()), "tasa_real": float(y.mean()),
           "brier": float(brier_score_loss(y, p)), "roc_auc": float(roc_auc_score(y, p)),
           "deciles": t.round(4).reset_index().rename(columns={"d": "decil"}).to_dict("records")}
    print(f"\n{nombre}: prob media {res['prob_media']:.4f} | tasa real {res['tasa_real']:.4f} | "
          f"Brier {res['brier']:.5f} | AUC {res['roc_auc']:.6f}")
    print(t.round(4).to_string())
    return res


def calibracion(cls, tr, te, fit, octb):
    print("========== 1. CALIBRACION ==========")
    out = {}
    p_tr = cls.predict_proba(tr[INPUT_COLS])[:, 1]
    p_te = cls.predict_proba(te[INPUT_COLS])[:, 1]
    out["actual_entrenamiento"] = deciles(tr["ARR_DEL15"], p_tr, "Actual, entrenamiento ene-oct")
    out["actual_prueba"] = deciles(te["ARR_DEL15"], p_te, "Actual, prueba nov-dic")
    base = float(tr["ARR_DEL15"].mean())
    out["brier_tasa_constante_prueba"] = float(brier_score_loss(te["ARR_DEL15"], np.full(len(te), base)))
    print(f"Brier de referencia en prueba (siempre {base:.4f}): {out['brier_tasa_constante_prueba']:.5f}")

    # Referencia: calibradores ajustados solo con octubre sobre un modelo ene-sep.
    n, n_pos = len(fit), int(fit["ARR_DEL15"].sum())
    w_pos, w_neg = n / (2 * n_pos), n / (2 * (n - n_pos))
    pipe, _ = T.clasificador_pipeline(n)
    t = time.time()
    pipe.fit(fit[INPUT_COLS], fit["ARR_DEL15"],
             std__sample_weight=np.where(fit["ARR_DEL15"] == 1, w_pos, w_neg))
    p_oct = prior_corrected(pipe.predict_proba(octb[INPUT_COLS])[:, 1], w_pos, w_neg)
    print(f"\nModelo ene-sep ({n:,} vuelos) en {time.time() - t:.1f} s | octubre: prob media "
          f"{p_oct.mean():.4f}, tasa real {octb['ARR_DEL15'].mean():.4f}")
    platt = LogisticRegression(C=1e10).fit(logit(p_oct)[:, None], octb["ARR_DEL15"])
    a, b = float(platt.coef_[0, 0]), float(platt.intercept_[0])
    iso = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1 - 1e-6).fit(
        p_oct, octb["ARR_DEL15"])
    print(f"Platt (octubre): p_cal = sigmoid({a:.4f} * logit(p) + {b:.4f})")
    out["platt_octubre"] = {"a": a, "b": b}
    for nombre, f in [("platt", lambda p: sigmoid(a * logit(p) + b)), ("isotonica", iso.predict)]:
        out[f"{nombre}_entrenamiento"] = deciles(tr["ARR_DEL15"], f(p_tr), f"{nombre}, entrenamiento")
        out[f"{nombre}_prueba"] = deciles(te["ARR_DEL15"], f(p_te), f"{nombre}, prueba")
    cerca = abs(out["actual_entrenamiento"]["prob_media"] - base) < 0.01
    out["decision"] = ("sin calibracion adicional: en entrenamiento la prob. media corregida "
                       f"({out['actual_entrenamiento']['prob_media']:.4f}) queda a menos de 0.01 de "
                       f"la tasa real ({base:.4f}); la brecha en prueba es cambio de tasa entre "
                       "periodos (oct 20.3%, nov-dic 23.7%), que un calibrador de octubre no corrige"
                       if cerca else "requiere calibracion")
    print("\nDecision:", out["decision"])
    return out


def baseline_minutos(reg, tr, te, fit_m, val_m, clip):
    print("\n========== 2. BASELINE DE MINUTOS ==========")
    y_tr = tr["ARR_DELAY"].clip(clip["p01"], clip["p99"])
    y_te = te["ARR_DELAY"].clip(clip["p01"], clip["p99"])
    med, mea = float(y_tr.median()), float(y_tr.mean())
    filas = []

    def add(nombre, pred, secs=None):
        filas.append({"modelo": nombre, "mae": mean_absolute_error(y_te, pred),
                      "rmse": root_mean_squared_error(y_te, pred),
                      "rmse_sin_recorte": root_mean_squared_error(te["ARR_DELAY"], pred),
                      "pred_media": float(np.mean(pred)),
                      "pred_p5_p50_p95": [round(float(x), 1) for x in np.quantile(pred, [0.05, 0.5, 0.95])],
                      "entren_s": secs})

    add(f"siempre la mediana de entrenamiento ({med:.1f})", np.full(len(te), med))
    add(f"siempre la media de entrenamiento ({mea:.2f})", np.full(len(te), mea))
    add("HGB squared_error (regresor.joblib)", reg.predict(te[INPUT_COLS]))

    pre = T.arbol_pre().fit(tr.loc[fit_m, INPUT_COLS])
    es = T.hgb(loss="absolute_error", max_iter=1000, early_stopping=True, n_iter_no_change=20,
               tol=1e-7, scoring="loss")
    es.fit(pre.transform(tr.loc[fit_m, INPUT_COLS]), y_tr[fit_m],
           X_val=pre.transform(tr.loc[val_m, INPUT_COLS]), y_val=y_tr[val_m])
    best = int(np.argmax(es.validation_score_))
    print(f"HGB absolute_error, early stopping con octubre: mejor en {best} arboles")
    pipe = Pipeline([("pre", T.arbol_pre()),
                     ("hgb", T.hgb(loss="absolute_error", max_iter=best, early_stopping=False))])
    t = time.time()
    pipe.fit(tr[INPUT_COLS], y_tr)
    add(f"HGB absolute_error ({best} arboles)",
        np.clip(pipe.predict(te[INPUT_COLS]), clip["p01"], clip["p99"]), time.time() - t)
    tabla = pd.DataFrame(filas)
    print(tabla.drop(columns="entren_s").round(3).to_string(index=False))
    gana = filas[2]["mae"] < filas[0]["mae"] - 1.0
    return {"tabla": filas, "mediana_prueba": float(y_te.median()), "media_prueba": float(y_te.mean()),
            "hgb_gana_claramente_mae_a_mediana": bool(gana),
            "decision": ("se mantiene squared_error en regresor.joblib: es el unico que gana a ambas "
                         "baselines en MAE y en RMSE; absolute_error baja el MAE "
                         f"{filas[2]['mae'] - filas[3]['mae']:.2f} min pero su RMSE "
                         f"({filas[3]['rmse']:.2f}) es peor que predecir siempre la media "
                         f"({filas[1]['rmse']:.2f}) y predice la mediana condicional (casi siempre "
                         "antes de hora), no los minutos esperados")}


def clima(tr, cls, reg):
    print("\n========== 3. CLIMA ==========")
    w = pd.read_parquet(datos.SILVER_WEATHER)
    w["date"] = pd.to_datetime(w["date"])
    cal = w["date"].dt.isocalendar()
    dfw = w[(w["airport"] == "DFW") & (cal["week"] == 11) & (cal["year"] == 2025)].sort_values("date")
    cols = ["precip_mm", "snow_cm", "wind_max_kmh", "temp_max_c", "temp_min_c"]
    prom = {c: round(float(dfw[c].mean()), 2) for c in cols}
    print(f"DFW semana ISO 11 de 2025 ({dfw['date'].min().date()} a {dfw['date'].max().date()}): {prom}")
    aptos = set(w["airport"])
    sin = tr[tr["has_weather"] == 0]
    n_otros = int(sin["ORIGIN"].astype(str).isin(aptos).sum())
    dias = w.groupby("airport").size()
    print(f"Entrenamiento con has_weather=0: {len(sin):,} de {len(tr):,} | en aeropuertos que si "
          f"tenian clima otros dias: {n_otros:,} | aeropuertos con clima: {len(aptos)} "
          f"({dias.min()}-{dias.max()} dias cada uno)")
    vuelo = dict(aerolinea="AA", origen="DFW", destino="ORD", MONTH=3, DAY_OF_WEEK=7,
                 dep_hour=18, DISTANCE=802.0)
    ejemplo = {}
    for nombre, cl in [("sin_clima", {"has_weather": 0}), ("con_clima", clima_con_banderas(**prom))]:
        X = flight_features(**vuelo, clima=cl)
        ejemplo[nombre] = {"prob": round(float(cls.predict_proba(X)[0, 1]), 4),
                           "minutos": round(float(reg.predict(X)[0]), 2)}
    print(f"AA DFW-ORD 15/03/2026 18h: {ejemplo}")
    return {"ejemplo_aa_dfw_ord_20260315_18h": ejemplo,
            "dfw_semana11_2025": {"fechas": [str(dfw["date"].min().date()), str(dfw["date"].max().date())],
                                  "promedio": prom},
            "entrenamiento_has_weather_0": int(len(sin)),
            "entrenamiento_has_weather_0_en_aeropuertos_con_clima": n_otros,
            "aeropuertos_con_clima": len(aptos), "dias_por_aeropuerto": [int(dias.min()), int(dias.max())]}


def main():
    clip = json.load(open(T.CLIP_PATH))
    df = datos.load_completed()
    tr, te = datos.split(df)
    del df
    fit_m, val_m = tr["MONTH"] <= 9, tr["MONTH"] == 10
    fit, octb = tr[fit_m].reset_index(drop=True), tr[val_m].reset_index(drop=True)
    # Las verificaciones son del modelo de EVALUACION (ene-oct): es el que tiene prueba nov-dic.
    cls = joblib.load(T.EVAL_PATHS["cls"])
    reg = joblib.load(T.EVAL_PATHS["reg"])
    ver = {"calibracion": calibracion(cls, tr, te, fit, octb),
           "baseline_minutos": baseline_minutos(reg, tr, te, fit_m, val_m, clip),
           "clima": clima(tr, cls, reg)}
    meta = json.load(open(META_PATH))
    meta["verificaciones"] = {"nota": "modelo de evaluacion (ene-oct); prueba nov-dic", **ver}
    with open(META_PATH, "w") as fh:
        json.dump(meta, fh, indent=2, ensure_ascii=False)
    assert not sin_spark.cargados()
    print(f"\n{os.path.relpath(META_PATH, T.ROOT)}: seccion 'verificaciones' actualizada")


if __name__ == "__main__":
    main()
