"""Utilidades compartidas: descargas con reintentos, cronometro de etapas y metricas."""
import json
import os
import time
from contextlib import contextmanager

import requests

from src.config import REPORTS_DIR

STAGE_METRICS_PATH = os.path.join(REPORTS_DIR, "stage_metrics.json")


def http_get(url: str, params: dict | None = None, retries: int = 4,
             backoff: float = 2.0, timeout: int = 60) -> requests.Response:
    """GET con reintentos y espera exponencial. Reintenta errores de red, 429 y 5xx;
    un 4xx distinto de 429 es error del cliente y se lanza de inmediato."""
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            if r.status_code == 429 or r.status_code >= 500:
                raise requests.HTTPError(f"HTTP {r.status_code}: {r.text[:200]}", response=r)
            r.raise_for_status()
            return r
        except requests.HTTPError as e:
            code = e.response.status_code if e.response is not None else None
            if code is not None and 400 <= code < 500 and code != 429:
                raise
            last_err = e
        except (requests.ConnectionError, requests.Timeout) as e:
            last_err = e
        if attempt < retries:
            wait = backoff ** attempt
            # 429 = limite de tasa: esperar mas que en un error transitorio.
            if isinstance(last_err, requests.HTTPError) and last_err.response is not None \
                    and last_err.response.status_code == 429:
                wait = max(wait, 30)
            print(f"  intento {attempt}/{retries} fallo ({last_err}); reintento en {wait:.0f}s")
            time.sleep(wait)
    raise RuntimeError(f"GET {url} fallo tras {retries} intentos: {last_err}")


def update_stage_metrics(stage: str, **values) -> None:
    """Guarda metricas de una etapa en reports/stage_metrics.json (se fusiona con lo existente).
    El Paso 7 consolida este archivo en reports/kpis.json."""
    os.makedirs(REPORTS_DIR, exist_ok=True)
    data = {}
    if os.path.exists(STAGE_METRICS_PATH):
        with open(STAGE_METRICS_PATH) as f:
            data = json.load(f)
    data.setdefault(stage, {}).update(values)
    with open(STAGE_METRICS_PATH, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


@contextmanager
def timed(label: str):
    """Mide el tiempo de un bloque. Uso: with timed('bronze') as t: ...; t['seconds']"""
    out = {}
    t0 = time.perf_counter()
    try:
        yield out
    finally:
        out["seconds"] = round(time.perf_counter() - t0, 2)
        print(f"[tiempo] {label}: {out['seconds']:.1f} s")
