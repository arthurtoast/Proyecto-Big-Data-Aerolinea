"""
Paso C (cierre Fase 3): figuras finales para el reporte en Word, en deliverables/figuras/.

PNG a 150 dpi, titulos y ejes en espanol. Reusa las 5 graficas de src/report.py (mismo
estilo) y agrega: causas de retraso, curva ROC vs baselines, matriz de confusion, importancia
de variables y real vs predicho (regresion).

Colores: serie principal azul; segunda serie naranja (slots 1 y 2 de la paleta de referencia,
validados para daltonismo); importancia con par divergente rojo (aumenta riesgo) y azul
(reduce riesgo). Las series tambien se distinguen por estilo de linea y leyenda.

Uso: python -m src.figures   (requiere python -m src.evaluate)
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from sklearn.metrics import roc_curve

from src.config import ROOT
from src.evaluate import KPIS_OUT, TEST_PREDS
from src.export_powerbi import OUT_DIR as POWERBI_DIR
from src.report import BLUE, GRAY_REF, GRID, INK, INK_2, SURFACE, _finish, build_charts
from src.spark_session import get_spark

FIG_DIR = os.path.join(ROOT, "deliverables", "figuras")
ORANGE, RED = "#eb6834", "#e34948"
BLUES = LinearSegmentedColormap.from_list("azules", ["#cde2fb", "#6da7ec", "#2a78d6", "#184f95", "#0d366b"])

NOMBRES = {
    "temp_max_c": "Temperatura maxima (C)", "temp_min_c": "Temperatura minima (C)",
    "month_cos": "Mes (componente coseno)", "month_sin": "Mes (componente seno)",
    "rain_flag": "Dia con lluvia", "snow_flag": "Dia con nieve", "high_wind_flag": "Dia con viento fuerte",
    "wind_max_kmh": "Viento maximo (km/h)", "precip_mm": "Precipitacion (mm)", "snow_cm": "Nieve (cm)",
    "has_weather": "Tiene datos de clima", "DISTANCE": "Distancia (millas)", "is_weekend": "Fin de semana",
    "dep_hour": "Hora de salida", "DAY_OF_WEEK": "Dia de la semana", "OP_UNIQUE_CARRIER": "Aerolinea",
    "ORIGIN": "Aeropuerto de origen", "DEST": "Aeropuerto de destino",
}


def nice(var: str) -> str:
    if " = " in var:
        k, v = var.split(" = ", 1)
        return f"{NOMBRES.get(k, k)} = {v}" + (" h" if k == "dep_hour" else "")
    return NOMBRES.get(var, var)


def fig_causas(path):
    c = pd.read_csv(os.path.join(POWERBI_DIR, "causas_retraso.csv")).sort_values("minutos_totales")
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.barh(c["causa"], c["pct_minutos_de_retraso"], color=BLUE, height=0.6)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("% de los minutos de retraso atribuidos")
    xmax = c["pct_minutos_de_retraso"].max()
    for i, r in enumerate(c.itertuples()):
        ax.text(r.pct_minutos_de_retraso + xmax * 0.01, i,
                f"{r.pct_minutos_de_retraso:.1f}%  ({r.minutos_totales / 1e6:.1f} M min, {r.vuelos_afectados:,} vuelos)",
                va="center", fontsize=8, color=INK_2)
    ax.set_xlim(0, xmax * 1.6)
    return _finish(fig, ax, "Causas de retraso (minutos atribuidos por el BTS)",
                   "Vuelos completados 2025 con llegada 15+ min tarde. Un vuelo puede tener varias causas.", path)


def fig_roc(test, k, path):
    y = test["ARR_DEL15"].astype(int)
    clf = k["evaluacion"]["clasificacion"]
    thr = k["evaluacion"]["conjunto_prueba"]["umbral_clasificacion"]
    fig, ax = plt.subplots(figsize=(7, 6))
    for col, color, ls, label, key in [
        ("pred_prob", BLUE, "-", "Modelo: regresion logistica", "modelo_regresion_logistica"),
        ("route_rate", ORANGE, "--", "Baseline: tasa historica por ruta", "baseline_tasa_historica_ruta")]:
        fpr, tpr, _ = roc_curve(y, test[col])
        ax.plot(fpr, tpr, color=color, lw=2, ls=ls, label=f"{label} (AUC {clf[key]['roc_auc']:.3f})")
        # Punto de operacion con el umbral usado (tasa base de entrenamiento).
        pred = test[col] >= thr
        op_fpr = (pred & (y == 0)).sum() / (y == 0).sum()
        op_tpr = (pred & (y == 1)).sum() / (y == 1).sum()
        ax.plot(op_fpr, op_tpr, "o", ms=8, color=color, mec=SURFACE, mew=2, zorder=4)
    ax.plot([0, 1], [0, 1], color=GRAY_REF, lw=1.5, ls=":",
            label="Baseline: siempre a tiempo (AUC 0.500)")
    ax.set_xlabel("Tasa de falsos positivos (vuelos a tiempo marcados como retraso)")
    ax.set_ylabel("Tasa de verdaderos positivos (recall)")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.0)
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    ax.text(0.98, 0.30, f"Puntos: umbral de operacion\n(prob >= {thr:.3f}, tasa base)", ha="right",
            fontsize=8, color=INK_2, transform=ax.transAxes)
    return _finish(fig, ax, "Curva ROC: modelo vs baselines",
                   f"Prueba: {len(test):,} vuelos completados de noviembre y diciembre 2025.", path)


def fig_confusion(k, path):
    cm = k["evaluacion"]["matriz_confusion_modelo"]
    m = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
    row_pct = m / m.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    ax.imshow(row_pct, cmap=BLUES, vmin=0, vmax=1)
    ax.grid(False)
    labels = ["A tiempo", "Con retraso"]
    ax.set_xticks([0, 1], [f"Predicho:\n{l}" for l in labels])
    ax.set_yticks([0, 1], [f"Real:\n{l}" for l in labels])
    for i in range(2):
        for j in range(2):
            # Color del texto segun la luminancia real de la celda (contraste legible).
            r_, g_, b_, _ = BLUES(row_pct[i, j])
            dark = 0.2126 * r_ + 0.7152 * g_ + 0.0722 * b_ < 0.55
            ax.text(j, i, f"{m[i, j]:,}\n{100 * row_pct[i, j]:.1f}% de la fila", ha="center", va="center",
                    fontsize=11, color="white" if dark else INK, fontweight="bold" if i == j else "normal")
    for s in ax.spines.values():
        s.set_visible(False)
    lr = k["evaluacion"]["clasificacion"]["modelo_regresion_logistica"]
    return _finish(fig, ax, "Matriz de confusion (regresion logistica)",
                   f"Prueba nov-dic. Precision {lr['precision']:.3f} | Recall {lr['recall']:.3f} | "
                   f"F1 {lr['f1']:.3f}", path)


def fig_importancia(k, path):
    top = pd.DataFrame(k["importancia_variables"]["regresion_logistica_top15"]).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 6.2))
    colors = [RED if c > 0 else BLUE for c in top["coeficiente"]]
    ax.barh([nice(v) for v in top["variable"]], top["importancia_estandarizada"], color=colors, height=0.65)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Importancia (|coeficiente| x desviacion estandar de la variable)")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=RED, label="Aumenta el riesgo de retraso"),
                       Patch(color=BLUE, label="Reduce el riesgo de retraso")],
              loc="lower right", frameon=False, fontsize=9)
    return _finish(fig, ax, "Top 15 variables del modelo de clasificacion",
                   "Regresion logistica. Horas y aeropuertos se comparan contra su categoria de referencia.", path)


def fig_real_vs_pred(test, k, path):
    reg = k["evaluacion"]["regresion"]["modelo_gbt"]
    lo, hi = k["evaluacion"]["recorte_arr_delay_min"]
    x, y = test["label_clip"].to_numpy(), test["pred_min"].to_numpy()
    fig, ax = plt.subplots(figsize=(8, 6))
    hb = ax.hexbin(x, y, gridsize=60, cmap=BLUES, norm=LogNorm(), mincnt=1, linewidths=0)
    cb = fig.colorbar(hb, ax=ax, pad=0.02)
    cb.set_label("Vuelos por celda (escala log)", color=INK_2)
    ax.plot([lo, hi], [lo, hi], color=GRAY_REF, ls="--", lw=1.2, label="Prediccion perfecta (y = x)")
    bins = np.arange(lo, hi + 10, 10)
    idx = np.digitize(x, bins)
    centers = [(bins[i - 1] + bins[i]) / 2 for i in range(1, len(bins)) if (idx == i).sum() > 200]
    means = [y[idx == i].mean() for i in range(1, len(bins)) if (idx == i).sum() > 200]
    ax.plot(centers, means, color=ORANGE, lw=2, label="Prediccion promedio por rango real")
    ax.set_xlabel("Retraso real de llegada (min, recortado)")
    ax.set_ylabel("Retraso predicho (min)")
    ax.grid(False)
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    return _finish(fig, ax, "Regresion: retraso real vs predicho (GBT)",
                   f"Prueba nov-dic, {len(test):,} vuelos. MAE {reg['mae']:.2f} min | RMSE {reg['rmse']:.2f} min.",
                   path)


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    with open(KPIS_OUT) as f:
        k = json.load(f)
    test = pd.read_parquet(TEST_PREDS)
    spark = get_spark("figures")
    charts = build_charts(spark, FIG_DIR)
    spark.stop()
    out = [v for kk, v in charts.items() if not kk.startswith("_")]
    out.append(fig_causas(os.path.join(FIG_DIR, "causas_retraso.png")))
    out.append(fig_roc(test, k, os.path.join(FIG_DIR, "curva_roc_modelo_vs_baselines.png")))
    out.append(fig_confusion(k, os.path.join(FIG_DIR, "matriz_confusion.png")))
    out.append(fig_importancia(k, os.path.join(FIG_DIR, "importancia_variables.png")))
    out.append(fig_real_vs_pred(test, k, os.path.join(FIG_DIR, "regresion_real_vs_predicho.png")))
    from PIL import Image
    print(f"{'figura':<52}{'pixeles':>12}{'dpi':>6}{'KB':>8}")
    for p in out:
        im = Image.open(p)
        dpi = round(im.info.get("dpi", (0, 0))[0])
        print(f"{p:<52}{f'{im.width}x{im.height}':>12}{dpi:>6}{os.path.getsize(p) / 1024:>8.0f}")


if __name__ == "__main__":
    main()
