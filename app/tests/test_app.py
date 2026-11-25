"""
Pruebas de la app (app/streamlit_app.py) con streamlit.testing, SIN red (conftest.py bloquea
los sockets: el pronostico falla y la app debe caer a clima tipico sin romperse).
"""
import os
import sys
from datetime import date, timedelta

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from streamlit.testing.v1 import AppTest  # noqa: E402

APP = os.path.join(ROOT, "app", "streamlit_app.py")


def correr(**q):
    at = AppTest.from_file(APP, default_timeout=60)
    at.query_params.update({**{k: str(v) for k, v in q.items()}, "calcular": "1"})
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def textos(at):
    return " ".join([m.value for m in at.markdown] + [c.value for c in at.caption]
                    + [e.value for k in ("success", "warning", "error", "info")
                       for e in getattr(at, k)])


def test_pronostico_sin_red_cae_a_clima_tipico():
    f = date.today() + timedelta(days=3)
    at = correr(origen="DFW", destino="ORD", aerolinea="AA", fecha=f.isoformat(), hora=18)
    txt = textos(at)
    assert "clima típico (pronóstico no disponible)" in txt
    assert "según el clima, entre" in txt.replace("**", "")
    assert "Riesgo" in txt and "Mejor hora para volar" in txt
    assert "No sustituye la información oficial de la aerolínea" in txt
    assert sum(("aumenta el riesgo" in m.value or "reduce el riesgo" in m.value)
               for m in at.markdown) == 3


def test_fecha_lejana_clima_tipico():
    at = correr(origen="DFW", destino="ORD", aerolinea="AA",
                fecha=(date.today() + timedelta(days=60)).isoformat(), hora=18)
    txt = textos(at)
    assert "Fuente: clima típico." in txt and "16 días" in txt


def test_aeropuerto_sin_clima():
    at = correr(origen="TUL", destino="DFW", aerolinea="AA",
                fecha=(date.today() + timedelta(days=3)).isoformat(), hora=6)
    txt = textos(at)
    assert "Sin datos de clima para este aeropuerto" in txt
    assert "aeropuerto sin datos de clima" in txt or "datos de clima" in txt


def test_pocos_datos():
    at = correr(origen="ABE", destino="RSW", aerolinea="G4",
                fecha=(date.today() + timedelta(days=30)).isoformat(), hora=7)
    assert "Pocos datos" in textos(at)


def test_contribuciones_suman_la_diferencia_de_logit():
    from app.ml import explicacion as E
    from app.ml.entrada import cargar_tablas, predecir_horas
    from app.ml.modelos import cargar_modelos
    t = cargar_tablas()
    cls, reg = cargar_modelos(os.path.join(ROOT, "app", "models"))
    ref = E.cargar_referencia(os.path.join(ROOT, "app", "data"), "clasificador.joblib")
    grupos = E.grupos_columnas(cls)
    _, X, _ = predecir_horas(t, cls, reg, "AA", "DFW", "ORD", date(2026, 12, 20), [18])
    cs = E.contribuciones(cls, X, ref["z_ref"], grupos)
    lr = cls.pipeline.named_steps["lr"]
    L = lr.intercept_[0] + np.asarray(E.transformar(cls, X) @ lr.coef_.ravel()).ravel()
    L_ref = lr.intercept_[0] + ref["z_ref"] @ lr.coef_.ravel()
    assert abs(sum(c for _, c, _ in cs) - np.mean(L - L_ref)) < 1e-9


def test_lote_de_horas_igual_a_prediccion_individual():
    from app.ml.entrada import cargar_tablas, predecir, predecir_horas
    from app.ml.modelos import cargar_modelos
    t = cargar_tablas()
    cls, reg = cargar_modelos(os.path.join(ROOT, "app", "models"))
    res, _, _ = predecir_horas(t, cls, reg, "AA", "DFW", "ORD", date(2026, 12, 20), [6, 18, 21])
    for h in [6, 18, 21]:
        r = predecir(t, cls, reg, "AA", "DFW", "ORD", date(2026, 12, 20), h)
        fila = res[res["dep_hour"] == h].iloc[0]
        assert abs(fila["prob"] - r["prob"]) < 1e-12 and abs(fila["minutos"] - r["minutos"]) < 1e-9
