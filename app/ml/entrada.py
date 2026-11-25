"""
Arma la entrada del modelo para un vuelo 2026 usando SOLO las tablas de app/data/ (nunca
Silver ni Gold). Sin pyspark.

Regla de clima (Paso 2):
- Origen con clima (aeropuertos.tiene_clima = 1) y con pronostico: 1 escenario con el
  pronostico (banderas con los umbrales de Silver, clima_con_banderas).
- Origen con clima y SIN pronostico (clima tipico): 21 escenarios, uno por cada dia real de
  2025 en la ventana centrada de +-10 dias alrededor del mismo dia del ano (dando la vuelta
  entre diciembre y enero), con las variables y banderas tal como las calculo Silver. La
  prediccion es el PROMEDIO de las 21 predicciones, y el rango (minimo y maximo) se muestra
  como "segun el clima, entre X% y Y%".
- Origen sin clima: 1 escenario con has_weather = 0.
El 29 de febrero usa el dia 28.
"""
import os
from datetime import date

import numpy as np
import pandas as pd

from app.ml.features import WEATHER_NUM, clima_con_banderas, make_features

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
TABLAS = ["aerolineas", "aeropuertos", "rutas", "horas_ruta", "historial", "clima_diario"]
VENTANA = 10                     # dias a cada lado: 21 escenarios
FLAGS = ["rain_flag", "snow_flag", "high_wind_flag"]
_DOY_2025 = {(d.month, d.day): d.dayofyear for d in pd.date_range("2025-01-01", "2025-12-31")}


def cargar_tablas(data_dir=DATA_DIR):
    t = {n: pd.read_parquet(os.path.join(data_dir, f"{n}.parquet")) for n in TABLAS}
    # Indices para busquedas rapidas.
    t["_rutas"] = t["rutas"].set_index(["origen", "destino", "aerolinea"])
    t["_con_clima"] = set(t["aeropuertos"].loc[t["aeropuertos"]["tiene_clima"] == 1, "iata"])
    # Clima por aeropuerto como matriz 365 x columnas, en orden de dia del ano.
    cols = WEATHER_NUM + FLAGS
    t["_clima"] = {a: (g.sort_values("dia_del_anio")[cols].to_numpy(dtype="float64"),
                       g.sort_values("dia_del_anio")["fecha"].to_numpy())
                   for a, g in t["clima_diario"].groupby("aeropuerto")}
    return t


def ventana_clima(t, aeropuerto, fecha: date):
    """21 dias reales de 2025 alrededor del mismo dia del ano -> (DataFrame de clima, fechas)."""
    mes, dia = fecha.month, fecha.day
    if (mes, dia) == (2, 29):
        dia = 28
    doy = _DOY_2025[(mes, dia)]
    vals, fechas = t["_clima"][aeropuerto]
    idx = np.arange(doy - 1 - VENTANA, doy + VENTANA) % 365   # circular entre dic y ene
    clima = pd.DataFrame(vals[idx], columns=WEATHER_NUM + FLAGS)
    clima["has_weather"] = 1
    return clima, fechas[idx]


def armar_entrada(t, aerolinea, origen, destino, fecha: date, dep_hour, clima=None):
    """-> (X, info). X es un LOTE con las 18 variables: 1 fila con pronostico o sin clima, 21
    filas con clima tipico. `clima`: pronostico opcional como dict para clima_con_banderas
    (precip_mm, snow_cm, wind_max_kmh, temp_max_c, temp_min_c)."""
    try:
        ruta = t["_rutas"].loc[(origen, destino, aerolinea)]
    except KeyError:
        raise ValueError(f"{aerolinea} {origen}-{destino} no opero en 2025") from None
    fechas = None
    if origen in t["_con_clima"]:
        if clima is not None:
            cl, fuente = pd.DataFrame([clima_con_banderas(**clima)]), "pronostico"
        else:
            cl, fechas = ventana_clima(t, origen, fecha)
            fuente = f"tipico: {len(cl)} dias reales de 2025 ({fechas[0]} a {fechas[-1]})"
    else:
        cl, fuente = pd.DataFrame([{"has_weather": 0}]), "sin clima (aeropuerto sin datos)"
    n = len(cl)
    raw = pd.DataFrame({"aerolinea": [aerolinea] * n, "origen": [origen] * n,
                        "destino": [destino] * n, "MONTH": fecha.month,
                        "DAY_OF_WEEK": fecha.isoweekday(), "dep_hour": dep_hour,
                        "DISTANCE": float(ruta["DISTANCE"])})
    X = make_lote(raw, cl)
    return X, {"distancia": float(ruta["DISTANCE"]), "vuelos_2025": int(ruta["vuelos"]),
               "fuente_clima": fuente, "escenarios": n,
               "fechas_clima": None if fechas is None else [str(f) for f in fechas]}


def make_lote(raw, clima):
    """Entradas crudas (una fila por escenario) + clima de cada escenario -> 18 variables."""
    return make_features(pd.concat([raw.reset_index(drop=True), clima.reset_index(drop=True)],
                                   axis=1))


def predecir(t, cls, reg, aerolinea, origen, destino, fecha: date, dep_hour, clima=None):
    """Prediccion de alto nivel: arma el lote, predice todos los escenarios de una vez y
    devuelve el promedio, mas el rango de probabilidad y de minutos entre escenarios."""
    X, info = armar_entrada(t, aerolinea, origen, destino, fecha, dep_hour, clima)
    p = cls.predict_proba(X)[:, 1]
    m = reg.predict(X)
    return {"prob": float(p.mean()), "minutos": float(m.mean()),
            "prob_min": float(p.min()), "prob_max": float(p.max()),
            "minutos_min": float(m.min()), "minutos_max": float(m.max()),
            "texto_rango": (f"segun el clima, entre {100 * p.min():.0f}% y {100 * p.max():.0f}%"
                            if len(p) > 1 else None),
            **info}
