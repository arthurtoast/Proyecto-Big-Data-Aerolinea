"""
App de la Fase 4: "¿Llegará a tiempo tu vuelo?" para vuelos de 2026.

Usa solo app/ml (modelos scikit-learn), app/data (tablas de apoyo) y app/weather.py
(pronostico de Open-Meteo con caida a clima tipico). No usa Spark ni lee Silver/Gold.
El texto visible lleva acentos; el codigo y los comentarios no.

Parametros opcionales en la URL para abrir una consulta directa (demo y capturas):
  ?origen=DFW&destino=ORD&aerolinea=AA&fecha=2026-11-27&hora=18&calcular=1

Uso: make app   (o: .venv/bin/streamlit run app/streamlit_app.py)
"""
import os
import sys
import time
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)  # Streamlit solo agrega app/ al path
from app.ml import sin_spark  # noqa: E402,F401  (primero: bloquea pyspark)

import altair as alt  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from app.ml import explicacion as E  # noqa: E402
from app.ml.entrada import (aeropuertos_para_clima, cargar_tablas, horas_de_ruta,  # noqa: E402
                            predecir_horas)
from app.ml.features import WEATHER_NUM  # noqa: E402
from app.ml.modelos import cargar_modelos  # noqa: E402
from app.weather import (FUENTE_PRONOSTICO, FUENTE_SIN_DATOS, FUENTE_TIPICO,  # noqa: E402
                         FUENTE_TIPICO_FALLA, ZONA_RESPALDO, ClienteClima, FechaPasadaError)

MODELS_DIR = os.path.join(ROOT, "app", "models")
DATA_DIR = os.path.join(ROOT, "app", "data")
AZUL, NARANJA = "#2a78d6", "#eb6834"
AVISO = ("Estimación basada en datos históricos de vuelos y clima de 2025. No sustituye la "
         "información oficial de la aerolínea.")

st.set_page_config(page_title="A Tiempo", page_icon=":material/flight_takeoff:",
                   layout="centered")


# ---------------------------------------------------------------- cargas (una vez)
@st.cache_resource(show_spinner="Cargando modelos y tablas...")
def cargar():
    import json
    t = cargar_tablas(DATA_DIR)
    cls, reg = cargar_modelos(MODELS_DIR)
    with open(os.path.join(MODELS_DIR, "metadata.json")) as fh:
        meta = json.load(fh)
    activo = meta.get("clasificador_activo", "clasificador.joblib")
    ref = E.cargar_referencia(DATA_DIR, activo)
    aer = dict(zip(t["aerolineas"]["codigo"], t["aerolineas"]["nombre"]))
    a = t["aeropuertos"].set_index("iata")
    ciudad = {i: (r.ciudad if isinstance(r.ciudad, str) else r.nombre) for i, r in a.iterrows()}
    hist = t["historial"].set_index(["origen", "destino", "aerolinea"])
    rutas = t["rutas"]
    return {"t": t, "cls": cls, "reg": reg, "meta": meta, "activo": activo, "ref": ref,
            "grupos": E.grupos_columnas(cls), "aerolinea": aer, "ciudad": ciudad,
            "hist": hist, "rutas": rutas}


@st.cache_resource
def cliente_clima():
    # Un solo cliente para toda la app: su cache de pronosticos (3 h) es compartida.
    return ClienteClima(aeropuertos_para_clima(cargar()["t"]))


@st.cache_data
def opciones():
    r = cargar()["rutas"]
    return {"origenes": sorted(r["origen"].unique()),
            "destinos": r.groupby("origen")["destino"].apply(lambda s: sorted(s.unique())).to_dict(),
            "aerolineas": r.groupby(["origen", "destino"])["aerolinea"]
                           .apply(lambda s: sorted(s.unique())).to_dict()}


def etiqueta_apto(iata):
    return f"{cargar()['ciudad'].get(iata, iata)} ({iata})"


def etiqueta_aer(cod):
    return f"{cargar()['aerolinea'].get(cod, cod)} ({cod})"


def indice(lista, valor, defecto=0):
    return lista.index(valor) if valor in lista else defecto


# ---------------------------------------------------------------- calculo
def calcular(aerolinea, origen, destino, fecha, hora):
    c = cargar()
    tiempos = {}
    t0 = time.perf_counter()
    clima = cliente_clima().obtener_clima(origen, fecha)   # FechaPasadaError si ya paso
    tiempos["clima"] = time.perf_counter() - t0
    pron = ({k: clima["clima"][k] for k in WEATHER_NUM}
            if clima["fuente"] == FUENTE_PRONOSTICO else None)
    horas = horas_de_ruta(c["t"], aerolinea, origen, destino)
    t0 = time.perf_counter()
    res, X, info = predecir_horas(c["t"], c["cls"], c["reg"], aerolinea, origen, destino, fecha,
                                  horas["dep_hour"].tolist(), clima=pron)
    tiempos["prediccion"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    Xh = X[X["dep_hour"] == hora]
    factores = E.contribuciones(c["cls"], Xh, c["ref"]["z_ref"], c["grupos"])
    tiempos["factores"] = time.perf_counter() - t0
    res = res.merge(horas, on="dep_hour")
    principal = res[res["dep_hour"] == hora].iloc[0]
    try:
        hist = c["hist"].loc[(origen, destino, aerolinea)]
    except KeyError:
        hist = None
    return {"clima": clima, "pron": clima["clima"] if pron else None, "res": res,
            "principal": principal, "factores": factores, "info": info, "hist": hist,
            "tiempos": tiempos, "filas": len(X), "escenarios": info["escenarios"]}


# ---------------------------------------------------------------- textos
def texto_minutos(m):
    if m >= 1:
        return f"Unos {m:.0f} min de retraso a la llegada"
    return "Llegaría a tiempo o antes"


def de_cada_diez(p):
    n = round(10 * p)
    if n == 0:
        return "Menos de 1 de cada 10 vuelos como este llega con 15 min o más de retraso."
    return (f"De cada 10 vuelos como este, alrededor de {n} "
            f"{'llega' if n == 1 else 'llegan'} con 15 min o más de retraso.")


def mostrar_semaforo(p, nivel, umbrales):
    titulo = {"bajo": "Riesgo bajo", "medio": "Riesgo medio", "alto": "Riesgo alto"}[nivel]
    caja = {"bajo": st.success, "medio": st.warning, "alto": st.error}[nivel]
    icono = {"bajo": ":material/check_circle:", "medio": ":material/warning:",
             "alto": ":material/error:"}[nivel]
    caja(f"**{titulo}** · {100 * p:.0f}% de probabilidad de llegar con 15 min o más de retraso",
         icon=icono)
    st.caption(f"{de_cada_diez(p)} Bajo: menos de {100 * umbrales['p50']:.0f}% (la mitad de los "
               f"vuelos de 2025). Alto: {100 * umbrales['p80']:.0f}% o más (el 20% con más riesgo).")


# ---------------------------------------------------------------- pagina
c = cargar()
op = opciones()
qp = st.query_params

st.title(":material/flight_takeoff: ¿Llegará a tiempo tu vuelo?")
st.caption("Estimación antes de volar con lo que se sabe de antemano: aerolínea, ruta, fecha, "
           "hora programada y clima del aeropuerto de origen.")

with st.container(border=True):
    col1, col2 = st.columns(2)
    origenes = op["origenes"]
    origen = col1.selectbox("Origen", origenes, format_func=etiqueta_apto,
                            index=indice(origenes, qp.get("origen", "DFW")))
    destinos = op["destinos"][origen]
    destino = col2.selectbox("Destino", destinos, format_func=etiqueta_apto,
                             index=indice(destinos, qp.get("destino", "ORD")))
    aerolineas = op["aerolineas"][(origen, destino)]
    aerolinea = st.selectbox("Aerolínea", aerolineas, format_func=etiqueta_aer,
                             index=indice(aerolineas, qp.get("aerolinea", "AA")))
    col3, col4 = st.columns(2)
    # El servidor (Streamlit Cloud) corre en UTC: "hoy" para el calendario se toma en la zona
    # mas atrasada de EE. UU. para no ocultar vuelos de hoy por la tarde. Si la fecha ya paso
    # en el aeropuerto de origen, obtener_clima() lo detecta con su zona horaria.
    hoy = datetime.now(ZoneInfo(ZONA_RESPALDO)).date()
    try:
        f_qp = date.fromisoformat(qp.get("fecha", ""))
    except ValueError:
        f_qp = hoy + timedelta(days=1)
    fecha = col3.date_input("Fecha del vuelo", value=max(f_qp, hoy), min_value=hoy,
                            max_value=hoy + timedelta(days=365), format="DD/MM/YYYY")
    horas = horas_de_ruta(c["t"], aerolinea, origen, destino)
    lista_h = horas["dep_hour"].tolist()
    vuelos_h = dict(zip(horas["dep_hour"], horas["vuelos"]))
    try:
        h_qp = int(qp.get("hora", 18))
    except ValueError:
        h_qp = 18
    hora = col4.selectbox("Hora de salida", lista_h,
                          index=indice(lista_h, h_qp, int(horas["vuelos"].idxmax())),
                          format_func=lambda h: f"{h:02d}:00 ({vuelos_h[h]:,} vuelos en 2025)",
                          help="Solo las horas con salidas programadas de esta ruta y aerolínea en 2025.")
    pulsado = st.button("Estimar retraso", type="primary", icon=":material/search:",
                        width="stretch")

consulta = (aerolinea, origen, destino, fecha, hora)
if pulsado or (qp.get("calcular") == "1" and st.session_state.get("consulta") is None):
    try:
        st.session_state["resultado"] = calcular(*consulta)
        st.session_state["consulta"] = consulta
        st.session_state["error"] = None
    except FechaPasadaError as e:
        st.session_state["resultado"], st.session_state["consulta"] = None, consulta
        st.session_state["error"] = str(e)

st.caption(f":material/info: {AVISO}")

r = st.session_state.get("resultado")
if st.session_state.get("consulta") == consulta and st.session_state.get("error"):
    st.error(f"No se puede estimar: {st.session_state['error']}", icon=":material/event_busy:")
elif r is None or st.session_state.get("consulta") != consulta:
    st.info("Elige tu vuelo y pulsa **Estimar retraso**.", icon=":material/touch_app:")
else:
    p_ = r["principal"]
    umbrales = c["ref"]["umbrales"]
    st.subheader(f"{aerolinea} {origen} a {destino} · {fecha:%d/%m/%Y} · {hora:02d}:00")
    mostrar_semaforo(p_["prob"], E.nivel_riesgo(p_["prob"], umbrales), umbrales)

    # ---- minutos y clima
    a, b = st.columns(2)
    with a.container(border=True):
        st.markdown("**:material/schedule: Minutos estimados**")
        st.markdown(f"### {texto_minutos(p_['minutos'])}")
        st.caption("Promedio esperado para vuelos con estas características; un vuelo concreto "
                   "puede variar mucho.")
    with b.container(border=True):
        cl = r["clima"]
        st.markdown("**:material/partly_cloudy_day: Clima en el origen**")
        if cl["fuente"] == FUENTE_PRONOSTICO:
            w = r["pron"]
            st.markdown(f"Pronóstico: lluvia {w['precip_mm']:.1f} mm · nieve {w['snow_cm']:.1f} cm  \n"
                        f"viento máx. {w['wind_max_kmh']:.0f} km/h · {w['temp_min_c']:.0f} a "
                        f"{w['temp_max_c']:.0f} °C")
            st.caption(f"Fuente: pronóstico de Open-Meteo, consultado {cl['consultado_en'][:16].replace('T', ' ')} UTC.")
        elif cl["fuente"] in (FUENTE_TIPICO, FUENTE_TIPICO_FALLA):
            st.markdown(f"Clima típico: según el clima, entre **{100 * p_['prob_min']:.0f}%** y "
                        f"**{100 * p_['prob_max']:.0f}%**")
            extra = (" El pronóstico no estuvo disponible en este momento."
                     if cl["fuente"] == FUENTE_TIPICO_FALLA else
                     " El pronóstico cubre solo los próximos 16 días.")
            st.caption(f"Fuente: {cl['fuente']}. Promedio de {r['escenarios']} días reales de 2025 "
                       f"alrededor de la misma fecha.{extra}")
        else:
            st.markdown("Sin datos de clima para este aeropuerto")
            st.caption(f"Fuente: {FUENTE_SIN_DATOS}. El modelo usa la historia de vuelos.")

    # ---- historial
    with st.container(border=True):
        st.markdown("**:material/history: Historial 2025 de esta ruta y aerolínea**")
        h = r["hist"]
        if h is None or h["completados"] == 0:
            st.markdown("No hay vuelos completados de esta ruta y aerolínea en 2025.")
        else:
            st.markdown(f"**{100 * h['tasa_retraso']:.0f}%** de sus {int(h['completados']):,} vuelos "
                        f"completados llegó con 15 min o más de retraso · retraso medio "
                        f"{h['retraso_promedio_min']:.0f} min · cancelados {100 * h['tasa_cancelacion']:.1f}%")
            if h["pocos_datos"]:
                st.warning("Pocos datos: menos de 30 vuelos completados en 2025. Toma esta tasa "
                           "como referencia aproximada.", icon=":material/info:")

    # ---- factores
    with st.container(border=True):
        st.markdown("**:material/tune: Qué mueve el riesgo de este vuelo**")
        vuelo = {"aerolinea": aerolinea, "origen": origen, "destino": destino, "fecha": fecha,
                 "dep_hour": hora, "distancia": r["info"]["distancia"], "clima": r["pron"],
                 "fuente_clima": r["clima"]["fuente"]}
        nombres = {"aerolinea": c["aerolinea"].get(aerolinea, aerolinea), "ciudad": etiqueta_apto}
        for g, _, ef in r["factores"][:3]:
            icono = ":material/arrow_upward:" if ef > 0 else ":material/arrow_downward:"
            color = "red" if ef > 0 else "green"
            st.markdown(f":{color}[{icono}] {E.frase(g, ef, vuelo, nombres)}")
        st.caption("Comparado con un vuelo promedio de 2025. Los puntos indican cuánto cambiaría "
                   "la probabilidad si ese factor fuera el de un vuelo promedio.")

    # ---- sugerencia de hora
    with st.container(border=True):
        st.markdown("**:material/insights: Mejor hora para volar ese día**")
        res = r["res"].copy()
        res["etiqueta"] = res["dep_hour"].map(lambda x: f"{x:02d}:00")
        res["tipo"] = res["dep_hour"].map(lambda x: "Tu hora" if x == hora else "Otras horas")
        mejor = res.loc[res["prob"].idxmin()]
        if mejor["dep_hour"] != hora:
            st.markdown(f"Salir a las **{mejor['etiqueta']}** baja el riesgo a "
                        f"**{100 * mejor['prob']:.0f}%** (desde {100 * p_['prob']:.0f}%).")
        else:
            st.markdown("Ya elegiste la hora con menor riesgo de ese día.")
        graf = (alt.Chart(res).mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
            x=alt.X("etiqueta:N", title="Hora de salida", sort=None, axis=alt.Axis(labelAngle=-45)),
            y=alt.Y("prob:Q", title="Probabilidad de retraso", axis=alt.Axis(format="%")),
            color=alt.Color("tipo:N", scale=alt.Scale(domain=["Tu hora", "Otras horas"],
                                                       range=[NARANJA, AZUL]),
                            legend=alt.Legend(title=None, orient="top")),
            tooltip=[alt.Tooltip("etiqueta:N", title="Hora"),
                     alt.Tooltip("prob:Q", title="Probabilidad", format=".0%"),
                     alt.Tooltip("minutos:Q", title="Minutos estimados", format=".0f"),
                     alt.Tooltip("vuelos:Q", title="Vuelos en 2025", format=",")])
            .properties(height=240))
        st.altair_chart(graf, width="stretch")
        st.caption("Solo horas con salidas programadas de esta ruta y aerolínea en 2025. "
                   "Confirma los horarios con la aerolínea.")

    # ---- detalle tecnico
    with st.expander("Detalle técnico"):
        tm = r["tiempos"]
        total = sum(tm.values())
        st.markdown(
            f"- Clima ({r['clima']['fuente']}): {1000 * tm['clima']:.0f} ms"
            f"{' (desde caché)' if r['clima']['desde_cache'] else ''}\n"
            f"- Predicción de {len(res)} horas × {r['escenarios']} escenarios = {r['filas']} filas "
            f"en un solo lote: {1000 * tm['prediccion']:.0f} ms\n"
            f"- Factores: {1000 * tm['factores']:.0f} ms\n"
            f"- Total: {1000 * total:.0f} ms (objetivo menor a 1 s sin clima: "
            f"{'cumple' if tm['prediccion'] + tm['factores'] < 1 else 'no cumple'})\n"
            f"- Clasificador: `{c['activo']}` · semáforo con percentiles 50 y 80 de 2025 "
            f"({100 * umbrales['p50']:.1f}% y {100 * umbrales['p80']:.1f}%)\n"
            f"- Distancia de la ruta: {r['info']['distancia']:,.0f} millas"
            + (f"\n- Motivo del clima típico: {r['clima']['motivo']}" if r["clima"]["motivo"] else ""))
        mp = c["meta"]["metricas_prueba"]["sklearn_clasificador"]
        mr = c["meta"]["metricas_prueba"]["sklearn_regresor"]
        st.caption(f"Modelos scikit-learn entrenados con 2025. Evaluación (entrenado ene-oct, "
                   f"probado nov-dic): ROC AUC {mp['roc_auc']:.2f}; error medio de minutos "
                   f"{mr['mae']:.0f} min. Sirve para comparar vuelos por riesgo, no para asegurar "
                   f"lo que pasará con un vuelo concreto.")

# ---------------------------------------------------------------- premium (maqueta)
st.divider()
st.subheader(":material/workspace_premium: Premium")
st.caption("Maqueta de la demo: estas funciones no están activas y no hay cobro.")
p1, p2, p3 = st.columns(3)
for col, (icono, titulo, texto) in zip([p1, p2, p3], [
        (":material/mail:", "Alertas por correo", "Aviso si el riesgo de tu vuelo cambia."),
        (":material/list_alt:", "Varios vuelos", "Sigue todos tus viajes en un solo lugar."),
        (":material/update:", "Pronóstico el día del vuelo", "Estimación actualizada con el clima de ese día.")]):
    with col.container(border=True):
        st.markdown(f"**:material/lock: {titulo}**")
        st.markdown(f"{icono} {texto}")
st.button("Próximamente", icon=":material/lock:", disabled=True, width="stretch")
st.caption(f"Demo académica, Fase 4. {AVISO}")
