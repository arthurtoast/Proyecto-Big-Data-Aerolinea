"""
Paso 3: clima para fechas futuras (pronostico de Open-Meteo) con caida a clima tipico.

Solo usa requests y la biblioteca estandar (mas app/ml/umbrales.py, que tambien es solo
biblioteca estandar). No importa streamlit ni pandas: la app lo envuelve despues.

Regla de decision (ClienteClima.obtener_clima(origen, fecha)):
- Fecha anterior a hoy (fecha local del aeropuerto): error FechaPasadaError.
- Origen sin clima en 2025: no se llama a la API; has_weather=0, fuente "sin datos de clima".
  El modelo nunca vio esos aeropuertos con clima (en 2025 todos sus vuelos tienen
  has_weather=0), asi que un pronostico ahi seria una combinacion fuera del entrenamiento.
- Origen con clima y fecha dentro de los 16 dias de pronostico: fuente "pronóstico", 1 fila.
- Origen con clima y fecha mas lejana, o si el pronostico no trae ese dia: fuente
  "clima típico" (clima=None: la entrada usa los 21 dias reales de la ventana de 2025).
- Si falla la red, la API responde error, cambian las unidades o vienen valores nulos: fuente
  "clima típico (pronóstico no disponible)", con el motivo registrado (logging) y devuelto.

Pronostico: una sola llamada por aeropuerto trae los 16 dias, con timezone=auto, asi que las
fechas de la respuesta son fechas locales del aeropuerto (igual que FL_DATE y que el clima de
entrenamiento, src/weather.py). Cache en memoria por aeropuerto: 3 horas de vigencia y como
maximo 128 aeropuertos (se descarta el mas antiguo).
"""
import logging
import threading
import time
from collections import OrderedDict
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import requests

from app.ml.umbrales import clima_con_banderas

log = logging.getLogger(__name__)

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
# Mismas variables diarias y unidades que el clima de entrenamiento (src/config.py
# WEATHER_DAILY_VARS y daily_units de data/raw/weather/*.json), con el nombre de Silver.
VARIABLES = {"precipitation_sum": "precip_mm", "snowfall_sum": "snow_cm",
             "wind_speed_10m_max": "wind_max_kmh", "temperature_2m_max": "temp_max_c",
             "temperature_2m_min": "temp_min_c"}
UNIDADES = {"precipitation_sum": "mm", "snowfall_sum": "cm", "wind_speed_10m_max": "km/h",
            "temperature_2m_max": "°C", "temperature_2m_min": "°C"}
DIAS_PRONOSTICO = 16
VIGENCIA_S = 3 * 3600
MAX_CACHE = 128
TIMEOUT_S = 5
REINTENTOS = 1
# Zona horaria de respaldo si el aeropuerto no tiene (BIH, EAR, XWA): UTC-10, la mas atrasada
# de EE. UU., para no rechazar como "pasada" una fecha que todavia es hoy en el aeropuerto.
ZONA_RESPALDO = "Pacific/Honolulu"

FUENTE_PRONOSTICO = "pronóstico"
FUENTE_TIPICO = "clima típico"
FUENTE_TIPICO_FALLA = "clima típico (pronóstico no disponible)"
FUENTE_SIN_DATOS = "sin datos de clima"


class FechaPasadaError(ValueError):
    """La app es para vuelos futuros: la fecha es anterior a hoy en el aeropuerto."""


class ErrorPronostico(Exception):
    """El pronostico no se pudo obtener o no es valido (motivo en el mensaje)."""


class ClienteClima:
    """aeropuertos: {iata: {"lat": float, "lon": float, "tiene_clima": bool,
    "zona_horaria": str | None}} (ver app.ml.entrada.aeropuertos_para_clima).
    session, ahora y reloj se pueden reemplazar en pruebas."""

    def __init__(self, aeropuertos, session=None, ahora=None, reloj=time.monotonic,
                 vigencia_s=VIGENCIA_S, max_cache=MAX_CACHE, timeout_s=TIMEOUT_S,
                 reintentos=REINTENTOS, pausa_reintento_s=0.5):
        self.aeropuertos = aeropuertos
        self.session = session or requests.Session()
        self.ahora = ahora or (lambda: datetime.now(timezone.utc))
        self.reloj = reloj
        self.vigencia_s = vigencia_s
        self.max_cache = max_cache
        self.timeout_s = timeout_s
        self.reintentos = reintentos
        self.pausa_reintento_s = pausa_reintento_s
        self._cache = OrderedDict()   # iata -> (instante reloj, consultado_en, dias)
        self._lock = threading.Lock()

    # ---------------------------------------------------------------- fechas
    def hoy_local(self, iata) -> date:
        zona = self.aeropuertos[iata].get("zona_horaria") or ZONA_RESPALDO
        return self.ahora().astimezone(ZoneInfo(zona)).date()

    # ---------------------------------------------------------------- decision
    def obtener_clima(self, origen, fecha: date) -> dict:
        """-> {"fuente", "clima", "consultado_en", "desde_cache", "motivo"}.
        clima: dict para el modelo (pronostico con banderas o has_weather=0), o None cuando la
        fuente es clima tipico (la entrada se arma con los 21 dias reales de la ventana)."""
        if origen not in self.aeropuertos:
            raise ValueError(f"aeropuerto desconocido: {origen}")
        hoy = self.hoy_local(origen)
        if fecha < hoy:
            raise FechaPasadaError(f"la fecha {fecha} ya paso en {origen} (hoy es {hoy}); "
                                   "la app es para vuelos futuros")
        res = {"fuente": None, "clima": None, "consultado_en": None, "desde_cache": False,
               "motivo": None}
        if not self.aeropuertos[origen].get("tiene_clima"):
            return {**res, "fuente": FUENTE_SIN_DATOS, "clima": {"has_weather": 0}}
        ultimo = hoy + timedelta(days=DIAS_PRONOSTICO - 1)
        if fecha > ultimo:
            return {**res, "fuente": FUENTE_TIPICO,
                    "motivo": f"fecha posterior al ultimo dia de pronostico ({ultimo})"}
        try:
            consultado_en, dias, desde_cache = self._pronostico(origen)
        except ErrorPronostico as e:
            log.warning("Pronostico no disponible para %s (%s): %s", origen, fecha, e)
            return {**res, "fuente": FUENTE_TIPICO_FALLA, "motivo": str(e)}
        except Exception as e:  # noqa: BLE001  (nunca romper la prediccion por el clima)
            log.exception("Error inesperado al pedir el pronostico de %s", origen)
            return {**res, "fuente": FUENTE_TIPICO_FALLA,
                    "motivo": f"error inesperado: {type(e).__name__}"}
        dia = dias.get(fecha.isoformat())
        if dia is None:
            return {**res, "fuente": FUENTE_TIPICO, "consultado_en": consultado_en,
                    "desde_cache": desde_cache,
                    "motivo": f"el pronostico no trae el dia {fecha}"}
        nulos = [k for k, v in dia.items() if v is None]
        if nulos:
            motivo = f"valores nulos en el pronostico de {fecha}: {nulos}"
            log.warning("Pronostico no disponible para %s: %s", origen, motivo)
            return {**res, "fuente": FUENTE_TIPICO_FALLA, "consultado_en": consultado_en,
                    "desde_cache": desde_cache, "motivo": motivo}
        return {**res, "fuente": FUENTE_PRONOSTICO, "clima": clima_con_banderas(**dia),
                "consultado_en": consultado_en, "desde_cache": desde_cache}

    # ---------------------------------------------------------------- API + cache
    def _pronostico(self, iata):
        """-> (consultado_en ISO UTC, {fecha ISO local: {variable: valor}}, desde_cache)."""
        with self._lock:
            hit = self._cache.get(iata)
            if hit and self.reloj() - hit[0] < self.vigencia_s:
                self._cache.move_to_end(iata)
                return hit[1], hit[2], True
        consultado_en, dias = self._descargar(iata)
        with self._lock:
            self._cache[iata] = (self.reloj(), consultado_en, dias)
            self._cache.move_to_end(iata)
            while len(self._cache) > self.max_cache:
                self._cache.popitem(last=False)
        return consultado_en, dias, False

    def _descargar(self, iata):
        a = self.aeropuertos[iata]
        params = {"latitude": round(a["lat"], 4), "longitude": round(a["lon"], 4),
                  "daily": ",".join(VARIABLES), "timezone": "auto",
                  "forecast_days": DIAS_PRONOSTICO, "precipitation_unit": "mm",
                  "wind_speed_unit": "kmh", "temperature_unit": "celsius"}
        ultimo_error = None
        for intento in range(1 + self.reintentos):
            if intento:
                time.sleep(self.pausa_reintento_s)
            try:
                r = self.session.get(FORECAST_URL, params=params, timeout=self.timeout_s)
                if r.status_code != 200:
                    try:
                        razon = r.json().get("reason", "")
                    except (ValueError, AttributeError):
                        razon = ""
                    raise ErrorPronostico(f"la API respondio {r.status_code} {razon}".strip())
                datos = r.json()
            except requests.exceptions.JSONDecodeError:
                ultimo_error = ErrorPronostico("respuesta que no es JSON")
                continue
            except requests.Timeout:
                ultimo_error = ErrorPronostico(f"timeout de {self.timeout_s} s")
                continue
            except requests.RequestException as e:
                ultimo_error = ErrorPronostico(f"error de red: {type(e).__name__}")
                continue
            except ErrorPronostico as e:
                ultimo_error = e
                continue
            consultado_en = self.ahora().astimezone(timezone.utc).isoformat(timespec="seconds")
            return consultado_en, self._parsear(datos)
        raise ultimo_error

    @staticmethod
    def _parsear(datos):
        daily, unidades = datos.get("daily"), datos.get("daily_units", {})
        if not isinstance(daily, dict) or "time" not in daily:
            raise ErrorPronostico("respuesta sin datos diarios")
        faltan = [v for v in VARIABLES if v not in daily]
        if faltan:
            raise ErrorPronostico(f"faltan variables en la respuesta: {faltan}")
        distintas = {v: unidades.get(v) for v in VARIABLES if unidades.get(v) != UNIDADES[v]}
        if distintas:
            raise ErrorPronostico(f"unidades distintas a las del entrenamiento: {distintas}")
        return {f: {VARIABLES[v]: daily[v][i] for v in VARIABLES}
                for i, f in enumerate(daily["time"])}
