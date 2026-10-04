"""Configuracion central del proyecto: rutas, constantes y parametros ajustables."""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
RAW_FLIGHTS_DIR = os.path.join(RAW_DIR, "flights")
RAW_WEATHER_DIR = os.path.join(RAW_DIR, "weather")
RAW_AIRPORTS_DIR = os.path.join(RAW_DIR, "airports")
BRONZE_DIR = os.path.join(DATA_DIR, "bronze")
SILVER_DIR = os.path.join(DATA_DIR, "silver")
GOLD_DIR = os.path.join(DATA_DIR, "gold")
REPORTS_DIR = os.path.join(ROOT, "reports")

YEAR = 2025

# Clima: cuantos aeropuertos de origen (los de mas vuelos) se descargan de Open-Meteo.
WEATHER_TOP_N = int(os.getenv("WEATHER_TOP_N", "50"))
WEATHER_DAILY_VARS = ["precipitation_sum", "snowfall_sum", "wind_speed_10m_max",
                      "temperature_2m_max", "temperature_2m_min"]

# Header real del CSV del BTS (no se inventan columnas).
EXPECTED_HEADER = [
    "YEAR", "MONTH", "DAY_OF_MONTH", "DAY_OF_WEEK", "FL_DATE", "OP_UNIQUE_CARRIER",
    "OP_CARRIER_FL_NUM", "ORIGIN", "ORIGIN_CITY_NAME", "ORIGIN_STATE_ABR", "DEST",
    "DEST_CITY_NAME", "DEST_STATE_ABR", "CRS_DEP_TIME", "DEP_DELAY", "DEP_DEL15",
    "DEP_TIME_BLK", "CRS_ARR_TIME", "ARR_DELAY", "ARR_DEL15", "ARR_TIME_BLK", "CANCELLED",
    "CANCELLATION_CODE", "DIVERTED", "DISTANCE", "DISTANCE_GROUP", "CARRIER_DELAY",
    "WEATHER_DELAY", "NAS_DELAY", "SECURITY_DELAY", "LATE_AIRCRAFT_DELAY",
]
