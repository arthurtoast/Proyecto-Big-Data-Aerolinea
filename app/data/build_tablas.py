"""
Paso 2: tablas de apoyo para la app 2026 en app/data/ (pandas + pyarrow sobre Silver y Gold,
sin Spark). La app solo lee estas tablas; nunca Silver ni Gold.

Tablas (Parquet):
- aerolineas      codigo, nombre, tipo (DIM_AEROLINEA).
- aeropuertos     iata, nombre, ciudad, estado, latitud, longitud, zona_horaria, tiene_clima.
- rutas           origen, destino, aerolinea, vuelos 2025 y DISTANCE (mediana de la ruta).
- horas_ruta      origen, destino, aerolinea, dep_hour y vuelos 2025.
- historial       por origen, destino y aerolinea: tasa de retraso, retraso medio y tasa de
                  cancelacion 2025, con bandera pocos_datos.
- clima_diario    clima REAL de cada dia de 2025 en los 50 aeropuertos con clima (variables
                  y banderas tal como las calculo Silver). La app arma con el los 21 escenarios
                  de la ventana de clima tipico (app/ml/entrada.py).

"vuelos" = vuelos programados en 2025: completados (Silver flights) + cancelados y desviados
(Silver flights_disrupted).

Uso: .venv/bin/python app/data/build_tablas.py
"""
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from app.ml import sin_spark  # noqa: E402,F401  (primero: bloquea pyspark)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow.dataset as ds  # noqa: E402

from app.ml.features import WEATHER_NUM  # noqa: E402

DATA_DIR = os.path.join(ROOT, "app", "data")
SILVER = os.path.join(ROOT, "data", "silver")
GOLD = os.path.join(ROOT, "data", "gold")
POCOS_DATOS = 30      # vuelos completados minimos para presentar una tasa como confiable
KEY = ["origen", "destino", "aerolinea"]


def leer_vuelos():
    cols = ["FL_DATE", "OP_UNIQUE_CARRIER", "ORIGIN", "DEST", "dep_hour", "DISTANCE",
            "ARR_DEL15", "ARR_DELAY", "CANCELLED", "DIVERTED"]
    comp = ds.dataset(os.path.join(SILVER, "flights"), format="parquet",
                      partitioning="hive").to_table(columns=cols).to_pandas()
    dis = ds.dataset(os.path.join(SILVER, "flights_disrupted"), format="parquet",
                     partitioning="hive").to_table(columns=cols).to_pandas()
    comp["estado"], dis["estado"] = "completado", np.where(dis["CANCELLED"] == 1, "cancelado", "desviado")
    v = pd.concat([comp, dis], ignore_index=True)
    v = v.rename(columns={"ORIGIN": "origen", "DEST": "destino", "OP_UNIQUE_CARRIER": "aerolinea"})
    for c in KEY + ["estado"]:
        v[c] = v[c].astype("category")
    print(f"Vuelos 2025: {len(v):,} (completados {len(comp):,}, cancelados "
          f"{int((dis['CANCELLED'] == 1).sum()):,}, desviados {int((dis['DIVERTED'] == 1).sum()):,})")
    return v


def t_aerolineas():
    a = pd.read_parquet(os.path.join(GOLD, "dim_aerolinea"))
    return a[["codigo", "nombre", "tipo"]].sort_values("codigo").reset_index(drop=True)


def t_aeropuertos():
    a = pd.read_parquet(os.path.join(GOLD, "dim_aeropuerto"))
    w = pd.read_parquet(os.path.join(SILVER, "weather_daily"), columns=["airport"])
    con_clima = set(w["airport"])
    a["tiene_clima"] = a["iata"].isin(con_clima).astype("int8")
    assert a["tiene_clima"].sum() == len(con_clima) == 50
    return (a[["iata", "nombre", "ciudad", "estado", "latitud", "longitud", "zona_horaria",
               "tiene_clima"]].sort_values("iata").reset_index(drop=True))


def t_rutas(v):
    r = (v.groupby(KEY, observed=True)
         .agg(vuelos=("estado", "size"), DISTANCE=("DISTANCE", "median")).reset_index())
    return r.sort_values(KEY).reset_index(drop=True)


def t_horas_ruta(v):
    h = v.groupby(KEY + ["dep_hour"], observed=True).size().rename("vuelos").reset_index()
    h["dep_hour"] = h["dep_hour"].astype("int8")
    return h.sort_values(KEY + ["dep_hour"]).reset_index(drop=True)


def t_historial(v):
    comp = v[v["estado"] == "completado"]
    g = comp.groupby(KEY, observed=True).agg(
        completados=("ARR_DEL15", "size"), tasa_retraso=("ARR_DEL15", "mean"),
        retraso_promedio_min=("ARR_DELAY", "mean"))
    prog = v.groupby(KEY, observed=True).agg(
        vuelos_programados=("estado", "size"),
        cancelados=("CANCELLED", "sum"), desviados=("DIVERTED", "sum"))
    h = prog.join(g, how="left").reset_index()
    h["completados"] = h["completados"].fillna(0).astype("int64")
    h["tasa_cancelacion"] = h["cancelados"] / h["vuelos_programados"]
    # pocos_datos: la tasa de retraso y el promedio salen de los completados.
    h["pocos_datos"] = (h["completados"] < POCOS_DATOS).astype("int8")
    cols = KEY + ["vuelos_programados", "completados", "cancelados", "desviados", "tasa_retraso",
                  "retraso_promedio_min", "tasa_cancelacion", "pocos_datos"]
    return h[cols].sort_values(KEY).reset_index(drop=True)


def t_clima_diario():
    w = pd.read_parquet(os.path.join(SILVER, "weather_daily"))
    w["date"] = pd.to_datetime(w["date"])
    for apt, g in w.groupby("airport"):
        assert len(g) == 365 and g["date"].dt.year.eq(2025).all() and g["date"].is_unique, apt
    c = pd.DataFrame({"aeropuerto": w["airport"], "fecha": w["date"].dt.date,
                      "mes": w["date"].dt.month.astype("int8"),
                      "dia": w["date"].dt.day.astype("int8"),
                      "dia_del_anio": w["date"].dt.dayofyear.astype("int16")})
    for col in WEATHER_NUM:
        c[col] = w[col]
    for col in ["rain_flag", "snow_flag", "high_wind_flag"]:
        c[col] = w[col].astype("int8")
    return c.sort_values(["aeropuerto", "dia_del_anio"]).reset_index(drop=True)


def guardar(nombre, df, comparar_csv):
    p = os.path.join(DATA_DIR, f"{nombre}.parquet")
    df.to_parquet(p, index=False, compression="zstd")
    mb = os.path.getsize(p) / 2**20
    t = time.perf_counter()
    pd.read_parquet(p)
    t_pq = time.perf_counter() - t
    linea = f"{nombre + '.parquet':<26} {len(df):>9,} filas  {mb:>7.3f} MB  carga {1000 * t_pq:6.1f} ms"
    if comparar_csv:
        csv = os.path.join(comparar_csv, f"{nombre}.csv")
        df.to_csv(csv, index=False)
        t = time.perf_counter()
        pd.read_csv(csv)
        linea += (f"  | CSV {os.path.getsize(csv) / 2**20:7.3f} MB, carga "
                  f"{1000 * (time.perf_counter() - t):6.1f} ms")
    print(linea)
    assert mb < 10, f"{nombre} pesa {mb:.1f} MB"
    return p


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--comparar-csv", help="carpeta temporal donde escribir CSV para comparar")
    args = ap.parse_args()
    os.makedirs(DATA_DIR, exist_ok=True)
    t0 = time.time()
    v = leer_vuelos()
    tablas = {"aerolineas": t_aerolineas(), "aeropuertos": t_aeropuertos(), "rutas": t_rutas(v),
              "horas_ruta": t_horas_ruta(v), "historial": t_historial(v),
              "clima_diario": t_clima_diario()}
    # Controles: los conteos cuadran con Silver.
    assert tablas["rutas"]["vuelos"].sum() == len(v) == tablas["horas_ruta"]["vuelos"].sum()
    assert tablas["historial"]["completados"].sum() == int((v["estado"] == "completado").sum())
    assert set(tablas["rutas"]["aerolinea"].astype(str)) <= set(tablas["aerolineas"]["codigo"])
    apts = set(tablas["aeropuertos"]["iata"])
    assert set(tablas["rutas"]["origen"].astype(str)) <= apts
    assert set(tablas["rutas"]["destino"].astype(str)) <= apts
    assert len(tablas["clima_diario"]) == 50 * 365
    print(f"Construidas en {time.time() - t0:.1f} s\n")
    for nombre, df in tablas.items():
        for c in df.select_dtypes("category"):
            df[c] = df[c].astype(str)
        guardar(nombre, df, args.comparar_csv)
    print(f"\nTotal: {time.time() - t0:.1f} s | pyspark cargado: {bool(sin_spark.cargados())}")


if __name__ == "__main__":
    main()
