"""
Paso 4: datos de apoyo del modelo para la app (app/data/referencia_modelo.json).

Para cada clasificador de produccion (clasificador.joblib y clasificador_mes.joblib), sobre
los vuelos completados de 2025 (Silver, con su clima real):
- Distribucion de la probabilidad corregida y umbrales del semaforo de riesgo: percentiles
  50 y 80 (bajo = mitad inferior, alto = 20% de mayor riesgo), con la tasa real por nivel.
- Vuelo de referencia para explicar factores: promedio de las columnas transformadas z que
  usa la regresion logistica (one-hot = frecuencia de cada categoria; numericas = su media
  escalada), sin ponderar. Ver app/ml/explicacion.py.

Corre despues de make sklearn (necesita los modelos) y lee Silver: la app solo lee el JSON.
Uso: .venv/bin/python app/data/build_referencia.py
"""
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from app.ml import sin_spark  # noqa: E402,F401

import joblib  # noqa: E402
import numpy as np  # noqa: E402

from app.ml import datos  # noqa: E402
from app.ml.explicacion import grupos_columnas, transformar  # noqa: E402
from app.ml.features import INPUT_COLS  # noqa: E402

MODELS_DIR = os.path.join(ROOT, "app", "models")
OUT = os.path.join(ROOT, "app", "data", "referencia_modelo.json")
CLASIFICADORES = ["clasificador.joblib", "clasificador_mes.joblib"]
PCTS = [5, 10, 25, 33.3, 50, 66.7, 75, 80, 90, 95, 99]


def main():
    t0 = time.time()
    df = datos.load_completed()
    X, y = df[INPUT_COLS], df["ARR_DEL15"].to_numpy()
    print(f"Vuelos completados 2025: {len(df):,} | tasa real {y.mean():.4f}")
    out = {"nota": "calculado sobre los vuelos completados de 2025 (ene-dic) con su clima real",
           "vuelos": int(len(df)), "tasa_real": float(y.mean()), "clasificadores": {}}
    for nombre in CLASIFICADORES:
        cls = joblib.load(os.path.join(MODELS_DIR, nombre))
        p = cls.predict_proba(X)[:, 1]
        u = {"p50": float(np.percentile(p, 50)), "p80": float(np.percentile(p, 80))}
        nivel = np.digitize(p, [u["p50"], u["p80"]])
        por_nivel = {n: {"vuelos_pct": float(100 * (nivel == k).mean()),
                         "tasa_real": float(y[nivel == k].mean()),
                         "prob_media": float(p[nivel == k].mean())}
                     for k, n in enumerate(["bajo", "medio", "alto"])}
        z = transformar(cls, X)
        z_ref = np.asarray(z.mean(axis=0)).ravel()
        grupos = grupos_columnas(cls)
        assert len(grupos) == len(z_ref) == cls.pipeline.named_steps["lr"].coef_.shape[1]
        out["clasificadores"][nombre] = {
            "umbrales": u, "por_nivel": por_nivel,
            "percentiles": {str(q): float(np.percentile(p, q)) for q in PCTS},
            "prob_media": float(p.mean()), "z_ref": [float(v) for v in z_ref]}
        print(f"\n{nombre}: umbrales p50 {u['p50']:.4f} | p80 {u['p80']:.4f}")
        for n, d in por_nivel.items():
            print(f"  {n:<6} {d['vuelos_pct']:5.1f}% de vuelos | prob media {d['prob_media']:.3f} | "
                  f"tasa real {d['tasa_real']:.3f}")
        print("  percentiles:", {q: round(v, 3) for q, v in out["clasificadores"][nombre]["percentiles"].items()})
    with open(OUT, "w") as fh:
        json.dump(out, fh)
    print(f"\n{os.path.relpath(OUT, ROOT)}: {os.path.getsize(OUT) / 2**20:.3f} MB | {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
