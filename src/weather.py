"""
Paso 3: clima diario historico de Open-Meteo para los aeropuertos de origen con mas vuelos.

- Ranking de aeropuertos desde Bronze (conteo de vuelos por ORIGIN).
- Coordenadas desde el catalogo OpenFlights (airports.dat) por codigo IATA.
- Una sola llamada por aeropuerto para todo el ano, con timezone=auto: el dia del clima
  es el dia local del aeropuerto, igual que FL_DATE del BTS.
- Cada respuesta se cachea en data/raw/weather/{IATA}.json; si el cache esta completo
  (365 dias) no se vuelve a llamar.
- Pausa entre llamadas reales y reintentos con espera exponencial (src.utils.http_get).
- Consolida todo en data/raw/weather/weather_daily.csv (una fila por aeropuerto y dia).

Unidades por defecto de Open-Meteo: precipitacion mm, nieve cm, viento km/h, temperatura C.

Uso: python -m src.weather [--top 50] [--pause 1.5]
"""
import argparse
import json
import os
import sys
import time

import pandas as pd
from pyspark.sql import functions as F

from src.bronze import AIRPORTS_PATH, BRONZE_FLIGHTS, download_airports
from src.config import RAW_WEATHER_DIR, WEATHER_DAILY_VARS, WEATHER_TOP_N, YEAR
from src.spark_session import get_spark
from src.utils import http_get, timed, update_stage_metrics

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
WEATHER_DAILY_CSV = os.path.join(RAW_WEATHER_DIR, "weather_daily.csv")
DAYS_IN_YEAR = 365

# Columnas de airports.dat (el archivo no trae header).
OPENFLIGHTS_COLS = ["airport_id", "name", "city", "country", "iata", "icao", "lat", "lon",
                    "altitude_ft", "utc_offset", "dst", "tz", "type", "source"]


def load_openflights() -> pd.DataFrame:
    """Catalogo OpenFlights indexado por IATA. '\\N' es el nulo del archivo."""
    download_airports()
    df = pd.read_csv(AIRPORTS_PATH, header=None, names=OPENFLIGHTS_COLS,
                     na_values=["\\N"], keep_default_na=False)
    df = df[df["iata"].notna() & (df["iata"].str.len() == 3)]
    # Si un IATA aparece repetido, se prefiere el de Estados Unidos.
    df = df.sort_values("country", key=lambda s: s != "United States")
    return df.drop_duplicates("iata").set_index("iata")


def top_origins(spark, n: int):
    """Top N aeropuertos de origen por numero de vuelos, y el total de vuelos."""
    bronze = spark.read.parquet(BRONZE_FLIGHTS)
    counts = bronze.groupBy("ORIGIN").count().orderBy(F.desc("count"))
    total = bronze.count()
    return [(r["ORIGIN"], r["count"]) for r in counts.limit(n).collect()], total


def cache_path(iata: str) -> str:
    return os.path.join(RAW_WEATHER_DIR, f"{iata}.json")


def cache_is_valid(path: str) -> bool:
    try:
        with open(path) as f:
            d = json.load(f)
        daily = d["daily"]
        return all(len(daily[v]) == DAYS_IN_YEAR for v in ["time"] + WEATHER_DAILY_VARS)
    except (OSError, ValueError, KeyError, TypeError):
        return False


def fetch_airport(iata: str, lat: float, lon: float) -> dict:
    params = {
        "latitude": round(lat, 4),
        "longitude": round(lon, 4),
        "start_date": f"{YEAR}-01-01",
        "end_date": f"{YEAR}-12-31",
        "daily": ",".join(WEATHER_DAILY_VARS),
        "timezone": "auto",
    }
    data = http_get(ARCHIVE_URL, params=params).json()
    if "daily" not in data or len(data["daily"].get("time", [])) != DAYS_IN_YEAR:
        raise ValueError(f"respuesta incompleta: {str(data)[:200]}")
    tmp = cache_path(iata) + ".part"
    with open(tmp, "w") as f:
        json.dump(data, f)
    os.replace(tmp, cache_path(iata))
    return data


def consolidate(iatas: list[str]) -> pd.DataFrame:
    """Une los JSON cacheados en una tabla larga (airport, date, variables)."""
    frames = []
    for iata in iatas:
        with open(cache_path(iata)) as f:
            daily = json.load(f)["daily"]
        df = pd.DataFrame(daily).rename(columns={"time": "date"})
        df.insert(0, "airport", iata)
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out.to_csv(WEATHER_DAILY_CSV, index=False)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=WEATHER_TOP_N)
    ap.add_argument("--pause", type=float, default=1.5, help="segundos entre llamadas reales")
    args = ap.parse_args()
    os.makedirs(RAW_WEATHER_DIR, exist_ok=True)

    with timed("clima") as t:
        spark = get_spark("weather")
        top, total_flights = top_origins(spark, args.top)
        spark.stop()
        catalog = load_openflights()

        ok, failed, no_coords, cached, fetched = [], [], [], 0, 0
        for i, (iata, n_flights) in enumerate(top, 1):
            if iata not in catalog.index:
                no_coords.append(iata)
                print(f"[{i:>2}/{len(top)}] {iata}: sin coordenadas en OpenFlights, se omite")
                continue
            if cache_is_valid(cache_path(iata)):
                ok.append(iata); cached += 1
                print(f"[{i:>2}/{len(top)}] {iata}: cache")
                continue
            row = catalog.loc[iata]
            try:
                if fetched:  # pausa solo entre llamadas reales
                    time.sleep(args.pause)
                fetch_airport(iata, row["lat"], row["lon"])
                ok.append(iata); fetched += 1
                print(f"[{i:>2}/{len(top)}] {iata}: descargado ({row['lat']:.3f}, {row['lon']:.3f}) {row['name']}")
            except Exception as e:
                failed.append(iata)
                print(f"[{i:>2}/{len(top)}] {iata}: ERROR {e}")

        if not ok:
            print("No se obtuvo clima de ningun aeropuerto.")
            sys.exit(1)
        weather = consolidate(ok)

    flights_by_airport = dict(top)
    covered = sum(flights_by_airport[a] for a in ok)
    coverage = 100 * covered / total_flights

    print(f"\nAeropuertos solicitados: {len(top)} | OK: {len(ok)} "
          f"(cache {cached}, descargados {fetched}) | fallidos: {len(failed)} {failed} "
          f"| sin coordenadas: {len(no_coords)} {no_coords}")
    print(f"Tabla de clima: {len(weather):,} filas ({len(ok)} aeropuertos x {DAYS_IN_YEAR} dias) "
          f"-> {os.path.relpath(WEATHER_DAILY_CSV)}")
    print(f"Nulos por variable: {weather[WEATHER_DAILY_VARS].isna().sum().to_dict()}")
    print(f"Cobertura: {covered:,} de {total_flights:,} vuelos = {coverage:.2f}% tendran clima de origen")

    update_stage_metrics("weather", seconds=t["seconds"], airports_requested=len(top),
                         airports_ok=len(ok), airports_failed=failed, airports_no_coords=no_coords,
                         flight_coverage_pct=round(coverage, 2))


if __name__ == "__main__":
    main()
