"""
Prueba REAL con red (no la corre pytest): pronostico de Open-Meteo para AA DFW-ORD a las 18 h.
- 27/11/2026 y 05/12/2026: dentro de los 16 dias de pronostico.
- 20/12/2026: fuera del pronostico, clima tipico.
Para los dos con pronostico tambien muestra la prediccion con clima tipico de esa fecha.

Uso: make test-red (o .venv/bin/python app/tests/prueba_red_clima.py)
"""
import logging
import os
import sys
import time
from datetime import date

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from app.ml import sin_spark  # noqa: E402,F401

from app.ml.entrada import (aeropuertos_para_clima, cargar_tablas, predecir,  # noqa: E402
                            predecir_vuelo)
from app.ml.modelos import cargar_modelos  # noqa: E402
from app.weather import ClienteClima  # noqa: E402

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")


def main():
    t = cargar_tablas()
    cls, reg = cargar_modelos(os.path.join(ROOT, "app", "models"))
    cliente = ClienteClima(aeropuertos_para_clima(t))
    print(f"Hoy en DFW (hora local): {cliente.hoy_local('DFW')}")
    for f in [date(2026, 11, 27), date(2026, 12, 5), date(2026, 12, 20)]:
        r = predecir_vuelo(t, cls, reg, cliente, "AA", "DFW", "ORD", f, 18)
        print(f"\n=== AA DFW-ORD {f} 18 h ===")
        print(f"fuente: {r['fuente_clima']} | consultado_en: {r['consultado_en']} | "
              f"desde_cache: {r['desde_cache']} | motivo: {r['motivo_clima']}")
        if r["clima_pronostico"]:
            print(f"clima recibido: {r['clima_pronostico']}")
        else:
            print(f"clima: {r['detalle_clima']}")
        rango = f" | {r['texto_rango']}" if r["texto_rango"] else ""
        print(f"prob {r['prob']:.4f} | minutos {r['minutos']:.2f}{rango}")
        print(f"tiempo clima (API o cache): {1000 * r['tiempo_clima_s']:.1f} ms | "
              f"tiempo prediccion: {1000 * r['tiempo_prediccion_s']:.1f} ms")
        if r["fuente_clima"] == "pronóstico":
            t0 = time.perf_counter()
            tip = predecir(t, cls, reg, "AA", "DFW", "ORD", f, 18)     # clima tipico (21 dias)
            ms = 1000 * (time.perf_counter() - t0)
            print(f"misma fecha con clima tipico: prob {tip['prob']:.4f} | minutos "
                  f"{tip['minutos']:.2f} | {tip['texto_rango']} ({ms:.1f} ms)")
            print(f"efecto del pronostico: prob {r['prob'] - tip['prob']:+.4f} | minutos "
                  f"{r['minutos'] - tip['minutos']:+.2f}")


if __name__ == "__main__":
    main()
