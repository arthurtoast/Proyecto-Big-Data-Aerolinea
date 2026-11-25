"""
Pruebas de app/weather.py SIN red: la sesion HTTP es un doble que devuelve respuestas
simuladas o lanza excepciones. Uso: make test (o .venv/bin/python -m pytest app/tests).
"""
import os
import subprocess
import sys
from datetime import date, datetime, timedelta, timezone

import pytest
import requests

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from app import weather as W  # noqa: E402

# "Ahora" fijo: 25/11/2026 03:00 UTC = 24/11/2026 21:00 en Dallas (America/Chicago).
AHORA_UTC = datetime(2026, 11, 25, 3, 0, tzinfo=timezone.utc)
HOY_DFW = date(2026, 11, 24)
AEROPUERTOS = {
    "DFW": {"lat": 32.8968, "lon": -97.038, "tiene_clima": True, "zona_horaria": "America/Chicago"},
    "ORD": {"lat": 41.9786, "lon": -87.9048, "tiene_clima": True, "zona_horaria": "America/Chicago"},
    "TUL": {"lat": 36.1984, "lon": -95.8881, "tiene_clima": False, "zona_horaria": "America/Chicago"},
}


def respuesta_ok(inicio=HOY_DFW, dias=16, nulos_en=None, unidades=None):
    fechas = [(inicio + timedelta(days=i)).isoformat() for i in range(dias)]
    daily = {"time": fechas,
             "precipitation_sum": [2.5 + i for i in range(dias)],
             "snowfall_sum": [0.0] * dias,
             "wind_speed_10m_max": [31.0] * dias,
             "temperature_2m_max": [15.0] * dias,
             "temperature_2m_min": [5.0] * dias}
    if nulos_en is not None:
        daily["precipitation_sum"][nulos_en] = None
    return {"timezone": "America/Chicago", "daily": daily,
            "daily_units": {"time": "iso8601", **(unidades or W.UNIDADES)}}


class Resp:
    def __init__(self, status=200, datos=None):
        self.status_code, self._datos = status, datos

    def json(self):
        if isinstance(self._datos, Exception):
            raise self._datos
        return self._datos


class Sesion:
    """Devuelve (o lanza) los elementos de `respuestas` en orden y registra las llamadas."""

    def __init__(self, *respuestas):
        self.respuestas, self.llamadas = list(respuestas), []

    def get(self, url, params=None, timeout=None):
        self.llamadas.append({"url": url, "params": params, "timeout": timeout})
        r = self.respuestas.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


class Reloj:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def cliente(sesion, reloj=None, **kw):
    return W.ClienteClima(AEROPUERTOS, session=sesion, ahora=lambda: AHORA_UTC,
                          reloj=reloj or Reloj(), pausa_reintento_s=0, **kw)


def test_pronostico_normal_y_fecha_local():
    s = Sesion(Resp(200, respuesta_ok()))
    c = cliente(s)
    # En UTC ya es 25/11, pero en Dallas sigue siendo 24/11: "hoy" se toma en hora local.
    assert c.hoy_local("DFW") == HOY_DFW
    r = c.obtener_clima("DFW", date(2026, 11, 27))
    assert r["fuente"] == W.FUENTE_PRONOSTICO
    # 27/11 es el dia 3 de la respuesta (indice 3 desde el 24/11): precip 2.5 + 3.
    assert r["clima"] == {"precip_mm": 5.5, "snow_cm": 0.0, "wind_max_kmh": 31.0,
                          "temp_max_c": 15.0, "temp_min_c": 5.0, "rain_flag": 1,
                          "snow_flag": 0, "high_wind_flag": 1, "has_weather": 1}
    assert r["consultado_en"] == "2026-11-25T03:00:00+00:00" and r["desde_cache"] is False
    p = s.llamadas[0]["params"]
    assert s.llamadas[0]["url"] == W.FORECAST_URL and s.llamadas[0]["timeout"] == 5
    assert p["timezone"] == "auto" and p["forecast_days"] == 16
    assert p["daily"] == ("precipitation_sum,snowfall_sum,wind_speed_10m_max,"
                          "temperature_2m_max,temperature_2m_min")
    # El vuelo de hoy (hora local) es valido y tiene pronostico.
    assert c.obtener_clima("DFW", HOY_DFW)["fuente"] == W.FUENTE_PRONOSTICO


def test_fecha_fuera_de_rango_no_llama_a_la_api():
    s = Sesion()
    r = cliente(s).obtener_clima("DFW", HOY_DFW + timedelta(days=16))
    assert r["fuente"] == W.FUENTE_TIPICO and r["clima"] is None
    assert "ultimo dia de pronostico" in r["motivo"] and s.llamadas == []


def test_pronostico_que_no_trae_el_dia():
    s = Sesion(Resp(200, respuesta_ok(dias=10)))   # la API devolvio solo 10 dias
    r = cliente(s).obtener_clima("DFW", HOY_DFW + timedelta(days=12))
    assert r["fuente"] == W.FUENTE_TIPICO and r["clima"] is None
    assert "no trae el dia" in r["motivo"]


def test_aeropuerto_sin_clima_no_llama_a_la_api():
    s = Sesion()
    r = cliente(s).obtener_clima("TUL", date(2026, 11, 27))
    assert r["fuente"] == W.FUENTE_SIN_DATOS and r["clima"] == {"has_weather": 0}
    assert s.llamadas == []


def test_timeout_con_reintento_cae_a_clima_tipico(caplog):
    s = Sesion(requests.Timeout(), requests.Timeout())
    r = cliente(s).obtener_clima("DFW", date(2026, 11, 27))
    assert r["fuente"] == W.FUENTE_TIPICO_FALLA and r["clima"] is None
    assert "timeout de 5 s" in r["motivo"]
    assert len(s.llamadas) == 2            # una llamada mas un reintento
    assert "Pronostico no disponible para DFW" in caplog.text


def test_error_inesperado_no_rompe():
    s = Sesion(RuntimeError("falla rara"))
    r = cliente(s).obtener_clima("DFW", date(2026, 11, 27))
    assert r["fuente"] == W.FUENTE_TIPICO_FALLA and "RuntimeError" in r["motivo"]


def test_red_real_bloqueada_cae_a_clima_tipico():
    # Sesion real de requests: el conftest bloquea el socket con OSError.
    c = W.ClienteClima(AEROPUERTOS, ahora=lambda: AHORA_UTC, pausa_reintento_s=0)
    r = c.obtener_clima("DFW", date(2026, 11, 27))
    assert r["fuente"] == W.FUENTE_TIPICO_FALLA and "ConnectionError" in r["motivo"]


def test_reintento_exitoso():
    s = Sesion(requests.ConnectionError(), Resp(200, respuesta_ok()))
    r = cliente(s).obtener_clima("DFW", date(2026, 11, 27))
    assert r["fuente"] == W.FUENTE_PRONOSTICO and len(s.llamadas) == 2


def test_error_de_la_api():
    s = Sesion(Resp(400, {"error": True, "reason": "Latitude must be in range"}),
               Resp(400, {"error": True, "reason": "Latitude must be in range"}))
    r = cliente(s).obtener_clima("DFW", date(2026, 11, 27))
    assert r["fuente"] == W.FUENTE_TIPICO_FALLA
    assert "400" in r["motivo"] and "Latitude" in r["motivo"]


def test_respuesta_que_no_es_json():
    err = requests.exceptions.JSONDecodeError("no es json", "<html>", 0)
    s = Sesion(Resp(200, err), Resp(200, err))
    r = cliente(s).obtener_clima("DFW", date(2026, 11, 27))
    assert r["fuente"] == W.FUENTE_TIPICO_FALLA and "no es JSON" in r["motivo"]


def test_respuesta_con_nulos():
    s = Sesion(Resp(200, respuesta_ok(nulos_en=3)))
    r = cliente(s).obtener_clima("DFW", date(2026, 11, 27))
    assert r["fuente"] == W.FUENTE_TIPICO_FALLA and r["clima"] is None
    assert "nulos" in r["motivo"] and "precip_mm" in r["motivo"]


def test_unidades_distintas():
    s = Sesion(Resp(200, respuesta_ok(unidades={**W.UNIDADES, "snowfall_sum": "inch"})),
               Resp(200, respuesta_ok(unidades={**W.UNIDADES, "snowfall_sum": "inch"})))
    r = cliente(s).obtener_clima("DFW", date(2026, 11, 27))
    assert r["fuente"] == W.FUENTE_TIPICO_FALLA and "unidades" in r["motivo"]


def test_cache_segunda_llamada_sin_red_y_vigencia():
    reloj = Reloj()
    s = Sesion(Resp(200, respuesta_ok()), Resp(200, respuesta_ok()))
    c = cliente(s, reloj=reloj)
    r1 = c.obtener_clima("DFW", date(2026, 11, 27))
    r2 = c.obtener_clima("DFW", date(2026, 12, 5))      # otra fecha, mismo aeropuerto
    assert len(s.llamadas) == 1 and r2["desde_cache"] is True
    assert r2["consultado_en"] == r1["consultado_en"]   # hora de la consulta original
    reloj.t += 3 * 3600 - 1                             # aun vigente
    c.obtener_clima("DFW", date(2026, 11, 27))
    assert len(s.llamadas) == 1
    reloj.t += 2                                        # vencio (3 h)
    assert c.obtener_clima("DFW", date(2026, 11, 27))["desde_cache"] is False
    assert len(s.llamadas) == 2


def test_cache_tamano_maximo():
    s = Sesion(*[Resp(200, respuesta_ok()) for _ in range(3)])
    c = cliente(s, max_cache=1)
    c.obtener_clima("DFW", date(2026, 11, 27))
    c.obtener_clima("ORD", date(2026, 11, 27))          # expulsa a DFW
    assert list(c._cache) == ["ORD"]
    c.obtener_clima("DFW", date(2026, 11, 27))
    assert len(s.llamadas) == 3


def test_fallas_no_se_guardan_en_cache():
    s = Sesion(requests.Timeout(), requests.Timeout(), Resp(200, respuesta_ok()))
    c = cliente(s)
    assert c.obtener_clima("DFW", date(2026, 11, 27))["fuente"] == W.FUENTE_TIPICO_FALLA
    assert c.obtener_clima("DFW", date(2026, 11, 27))["fuente"] == W.FUENTE_PRONOSTICO


def test_fecha_pasada():
    s = Sesion()
    with pytest.raises(W.FechaPasadaError, match="vuelos futuros"):
        cliente(s).obtener_clima("DFW", HOY_DFW - timedelta(days=1))
    assert s.llamadas == []


def test_weather_no_importa_pandas_ni_streamlit():
    code = ("import sys; sys.path.insert(0, %r); import app.weather; "
            "print(sorted(m for m in ('pandas', 'numpy', 'streamlit', 'pyspark') "
            "if m in sys.modules))" % ROOT)
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]"


def test_predecir_vuelo_con_pronostico_y_tipico():
    """Integracion con app/ml/entrada.py: pronostico -> 1 escenario; tipico -> 21."""
    from app.ml.entrada import aeropuertos_para_clima, cargar_tablas, predecir_vuelo
    from app.ml.modelos import cargar_modelos
    t = cargar_tablas()
    cls, reg = cargar_modelos(os.path.join(ROOT, "app", "models"))
    s = Sesion(Resp(200, respuesta_ok()))
    c = W.ClienteClima(aeropuertos_para_clima(t), session=s, ahora=lambda: AHORA_UTC)
    r = predecir_vuelo(t, cls, reg, c, "AA", "DFW", "ORD", date(2026, 11, 27), 18)
    assert r["fuente_clima"] == W.FUENTE_PRONOSTICO and r["escenarios"] == 1
    assert r["clima_pronostico"]["rain_flag"] == 1
    r = predecir_vuelo(t, cls, reg, c, "AA", "DFW", "ORD", date(2026, 12, 20), 18)
    assert r["fuente_clima"] == W.FUENTE_TIPICO and r["escenarios"] == 21
    assert r["prob_min"] <= r["prob"] <= r["prob_max"]
    r = predecir_vuelo(t, cls, reg, c, "AA", "TUL", "DFW", date(2026, 11, 27), 6)
    assert r["fuente_clima"] == W.FUENTE_SIN_DATOS and r["escenarios"] == 1
    assert len(s.llamadas) == 1
