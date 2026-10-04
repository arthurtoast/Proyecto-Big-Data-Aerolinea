"""
Paso 7: KPIs (reports/kpis.json) y graficas PNG (reports/).

Las agregaciones se hacen sobre el esquema estrella de Gold (FACT_VUELOS + dimensiones), igual
que las haria Power BI. Tasa de retraso = % de vuelos completados con ARR_DEL15 = 1
(llegada con 15+ minutos de retraso).

Uso: python -m src.report
"""
import json
import os

import matplotlib

matplotlib.use("Agg")  # sin ventana: solo archivos PNG
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from pyspark.sql import functions as F

from src.config import REPORTS_DIR
from src.gold import GOLD_PATHS
from src.spark_session import get_spark
from src.utils import STAGE_METRICS_PATH, timed, update_stage_metrics

KPIS_PATH = os.path.join(REPORTS_DIR, "kpis.json")
MIN_ROUTE_FLIGHTS = 1000  # rutas con menos vuelos dan tasas inestables

# Estilo: una serie en azul, texto en tintas neutras, cuadricula discreta.
BLUE, GRAY_REF = "#2a78d6", "#8a8984"
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
    "text.color": INK, "font.size": 10, "axes.titlesize": 13, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
})


def _finish(fig, ax, title, subtitle, path, top_label=False):
    # top_label: hay una etiqueta de referencia arriba del area; se sube el subtitulo.
    # Posiciones en puntos (no en fraccion del eje) para que no dependan del alto de la figura.
    ax.set_title(title, loc="left", pad=36 if top_label else 22)
    ax.annotate(subtitle, xy=(0, 1), xycoords="axes fraction",
                xytext=(0, 18 if top_label else 6), textcoords="offset points",
                color=INK_2, fontsize=9, va="bottom")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return os.path.relpath(path)


def _ref_line(ax, value, label, horizontal=True):
    if horizontal:
        ax.axhline(value, color=GRAY_REF, lw=1, ls="--")
        ax.text(1.0, value, f" {label}", transform=ax.get_yaxis_transform(), color=INK_2,
                fontsize=8, va="center", ha="left")
    else:
        ax.axvline(value, color=GRAY_REF, lw=1, ls="--")
        ax.text(value, 1.0, label, transform=ax.get_xaxis_transform(), color=INK_2,
                fontsize=8, va="bottom", ha="center")


def rate_chart(df_pd, x, path, title, subtitle, overall, xlabel, highlight_extremes=True):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars = ax.bar(df_pd[x].astype(str), df_pd["tasa"], color=BLUE, width=0.7)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Vuelos con retraso")
    ax.grid(axis="x", visible=False)
    _ref_line(ax, overall, f"promedio {overall:.1%}")
    if highlight_extremes:  # solo max y min, no un numero en cada barra
        for i in {df_pd["tasa"].idxmax(), df_pd["tasa"].idxmin()}:
            b = bars[i]
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{df_pd['tasa'][i]:.1%}",
                    ha="center", va="bottom", fontsize=8, color=INK)
    return _finish(fig, ax, title, subtitle, path)


def hbar_chart(df_pd, label_col, path, title, subtitle, overall, n_col="n"):
    df_pd = df_pd.iloc[::-1].reset_index(drop=True)  # mayor arriba
    fig, ax = plt.subplots(figsize=(9, 0.42 * len(df_pd) + 1.6))
    ax.barh(df_pd[label_col], df_pd["tasa"], color=BLUE, height=0.65)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("Vuelos con retraso")
    ax.grid(axis="y", visible=False)
    _ref_line(ax, overall, f"promedio {overall:.1%}", horizontal=False)
    xmax = df_pd["tasa"].max()
    for i, r in df_pd.iterrows():
        # Fondo del color de la superficie: la linea de referencia no cruza el texto.
        ax.text(r["tasa"] + xmax * 0.01, i, f"{r['tasa']:.1%}  ({r[n_col]:,} vuelos)",
                va="center", fontsize=8, color=INK_2, zorder=3,
                bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1.5))
    ax.set_xlim(0, xmax * 1.32)
    return _finish(fig, ax, title, subtitle, path, top_label=True)


def build_charts(spark, out_dir: str = REPORTS_DIR) -> dict:
    fact = spark.read.parquet(GOLD_PATHS["FACT_VUELOS"]).filter("CANCELLED = 0 AND DIVERTED = 0")
    aer = spark.read.parquet(GOLD_PATHS["DIM_AEROLINEA"])
    apt = spark.read.parquet(GOLD_PATHS["DIM_AEROPUERTO"])
    cli = spark.read.parquet(GOLD_PATHS["DIM_CLIMA"])
    rate = [F.avg("ARR_DEL15").alias("tasa"), F.count("*").alias("n")]
    overall = fact.agg(F.avg("ARR_DEL15")).first()[0]
    sub = "Vuelos domesticos completados en EE. UU., 2025. Retraso = llegada con 15+ min."
    out = {}

    meses = ["Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]
    m = fact.groupBy("MONTH").agg(*rate).orderBy("MONTH").toPandas()
    m["mes"] = [meses[i - 1] for i in m["MONTH"]]
    out["por_mes"] = rate_chart(m, "mes", os.path.join(out_dir, "retraso_por_mes.png"),
                                "Tasa de retraso por mes", sub, overall, "Mes")

    h = fact.groupBy("dep_hour").agg(*rate).orderBy("dep_hour").toPandas()
    out["por_hora"] = rate_chart(h, "dep_hour", os.path.join(out_dir, "retraso_por_hora.png"),
                                 "Tasa de retraso por hora programada de salida",
                                 sub + " Hora local del aeropuerto de origen.", overall, "Hora")

    a = (fact.join(aer, "aerolinea_key").groupBy("nombre").agg(*rate)
         .orderBy(F.desc("tasa")).toPandas())
    out["por_aerolinea"] = hbar_chart(a, "nombre", os.path.join(out_dir, "retraso_por_aerolinea.png"),
                                      "Tasa de retraso por aerolinea", sub, overall)

    o = apt.select(F.col("aeropuerto_key").alias("origen_key"), F.col("iata").alias("o"))
    d = apt.select(F.col("aeropuerto_key").alias("destino_key"), F.col("iata").alias("d"))
    r = (fact.join(o, "origen_key").join(d, "destino_key")
         .groupBy(F.concat_ws(" - ", "o", "d").alias("ruta")).agg(*rate)
         .filter(F.col("n") >= MIN_ROUTE_FLIGHTS).orderBy(F.desc("tasa")).limit(10).toPandas())
    out["top_rutas"] = hbar_chart(r, "ruta", os.path.join(out_dir, "top10_rutas_retrasadas.png"),
                                  "Top 10 rutas con mayor tasa de retraso",
                                  f"Solo rutas con {MIN_ROUTE_FLIGHTS:,}+ vuelos completados en 2025.",
                                  overall)

    etiquetas = {"despejado": "Despejado", "lluvia": "Lluvia", "nieve": "Nieve",
                 "viento_fuerte": "Viento fuerte"}
    c = (fact.join(cli.select("clima_key", "condicion"), "clima_key")
         .filter("clima_key > 0").groupBy("condicion").agg(*rate).toPandas())
    c["condicion"] = c["condicion"].map(etiquetas)
    c = c.set_index("condicion").loc[list(etiquetas.values())].reset_index()
    base = float(c.loc[c["condicion"] == "Despejado", "tasa"].iloc[0])
    fig, ax = plt.subplots(figsize=(10, 4.8))
    bars = ax.bar(c["condicion"], c["tasa"], color=BLUE, width=0.6)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_ylabel("Vuelos con retraso")
    ax.grid(axis="x", visible=False)
    _ref_line(ax, base, f"despejado {base:.1%}")
    for b, (_, row) in zip(bars, c.iterrows()):
        delta = "" if row["condicion"] == "Despejado" else f"\n{(row['tasa'] / base - 1):+.0%} vs despejado"
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{row['tasa']:.1%}{delta}",
                ha="center", va="bottom", fontsize=8, color=INK)
    ax.set_ylim(0, c["tasa"].max() * 1.25)
    ax.set_xticks(range(len(c)), [f"{k}\n({n:,} vuelos)" for k, n in zip(c["condicion"], c["n"])])
    out["efecto_clima"] = _finish(
        fig, ax, "Efecto del clima en el aeropuerto de origen",
        "Condicion del dia: nieve > lluvia (>= 1 mm) > viento (>= 30 km/h) > despejado. "
        "Vuelos con clima (top 50 aeropuertos).",
        os.path.join(out_dir, "efecto_clima.png"))

    out["_tablas"] = {"por_mes": m[["mes", "tasa", "n"]].round(4).to_dict("records"),
                      "efecto_clima": c[["condicion", "tasa", "n"]].round(4).to_dict("records"),
                      "top_rutas": r.round(4).to_dict("records"),
                      "tasa_global": round(overall, 4)}
    return out


def build_kpis(charts: dict) -> dict:
    with open(STAGE_METRICS_PATH) as f:
        s = json.load(f)
    mdl = s.get("models", {})
    best = mdl.get("best_regressor", "gbt")
    stages = ["organize", "bronze", "weather", "silver", "gold", "mysql", "models", "report"]
    times = {}
    for st in stages:
        v = s.get(st, {})
        if st == "models":
            times[st] = round(sum(v.get(k, 0) for k in ["logreg_s", "linreg_s", "gbt_s", "scoring_s"]), 1)
        elif "seconds" in v:
            times[st] = v["seconds"]
    kpis = {
        "registros_procesados": {
            "bronze_total": s["bronze"]["rows"],
            "silver_completados": s["silver"]["silver_rows"],
            "silver_cancelados_desviados": s["silver"]["disrupted_rows"],
            "gold_fact_vuelos": s["gold"]["table_rows"]["FACT_VUELOS"],
            "mysql_fact_vuelos": s.get("mysql", {}).get("table_rows", {}).get("FACT_VUELOS"),
        },
        "tiempo_por_etapa_s": times,
        "tiempo_total_s": round(sum(times.values()), 1),
        "cobertura_clima_pct": {
            "vuelos_completados_silver": s["silver"]["weather_coverage_pct"],
            "aeropuertos_con_clima": s["weather"]["airports_ok"],
        },
        "clasificacion_arr_del15": {k: round(mdl["logreg"][k], 4) for k in
                                    ["roc_auc", "precision", "recall", "f1", "accuracy"]},
        "matriz_confusion": mdl["logreg"]["confusion_matrix"],
        "regresion_arr_delay_min": {
            "modelo_en_produccion": best,
            "mae": round(mdl[best]["mae"], 2),
            "rmse": round(mdl[best]["rmse"], 2),
            "baseline_lineal": {"mae": round(mdl["linreg"]["mae"], 2), "rmse": round(mdl["linreg"]["rmse"], 2)},
            "gbt": {"mae": round(mdl["gbt"]["mae"], 2), "rmse": round(mdl["gbt"]["rmse"], 2)},
            "recorte_min": mdl.get("clip"),
        },
        "tasa_retraso_global": charts["_tablas"]["tasa_global"],
        "graficas": {k: v for k, v in charts.items() if not k.startswith("_")},
    }
    with open(KPIS_PATH, "w") as f:
        json.dump(kpis, f, indent=2, ensure_ascii=False)
    return kpis


def main():
    spark = get_spark("report")
    with timed("reporte") as t:
        charts = build_charts(spark)
    update_stage_metrics("report", seconds=t["seconds"])
    kpis = build_kpis(charts)
    spark.stop()

    tb = charts["_tablas"]
    print("\nGraficas generadas:")
    for k, v in kpis["graficas"].items():
        print(f"  {k:<14} {v}")
    print("\nEfecto del clima:", [(r["condicion"], f"{r['tasa']:.1%}", r["n"]) for r in tb["efecto_clima"]])
    print("Top 3 rutas:", [(r["ruta"], f"{r['tasa']:.1%}", r["n"]) for r in tb["top_rutas"][:3]])
    print(f"\n{os.path.relpath(KPIS_PATH)}:")
    print(json.dumps({k: v for k, v in kpis.items() if k != "graficas"}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
