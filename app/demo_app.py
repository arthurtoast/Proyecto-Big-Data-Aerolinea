"""
Demo de la app freemium: "¿Llegara a tiempo tu vuelo?".

- Plan gratis: probabilidad de retraso y nivel de riesgo, con la tasa historica de la ruta.
- Plan Premium (simulado con un interruptor): minutos estimados, clima del origen, mejor hora
  para volar ese dia y comparacion de aerolineas en la ruta.

Usa los modelos entrenados (src/predict.py). Todas las predicciones de una consulta se hacen
en un solo trabajo de Spark con predict_many(). El texto visible para el usuario lleva acentos;
el codigo y los comentarios no.

Uso: make app   (o: .venv/bin/streamlit run app/demo_app.py)
"""
import json
import os
import sys
from datetime import date

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)  # Streamlit solo agrega app/ al path; src/ vive en la raiz

import altair as alt  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from src.predict import catalog, predict_many  # noqa: E402

st.set_page_config(page_title="A Tiempo (demo)", page_icon=":material/flight_takeoff:", layout="centered")

# Umbrales del nivel de riesgo. La tasa nacional es ~22%: "moderado" la rodea.
RISK_LOW, RISK_HIGH = 0.15, 0.30
BLUE, ORANGE = "#2a78d6", "#eb6834"


@st.cache_resource(show_spinner="Cargando modelos y catálogos (solo la primera vez, ~10 s)...")
def load():
    c = catalog()
    kpis = {}
    for path in ["deliverables/kpis.json", "reports/kpis.json"]:
        try:
            with open(os.path.join(ROOT, path)) as f:
                kpis[path] = json.load(f)
        except (OSError, json.JSONDecodeError):
            kpis[path] = {}
    return c, kpis


@st.cache_data(show_spinner=False, max_entries=200)
def run(queries: tuple) -> list:
    return predict_many(list(queries))


def airport_label(c, iata: str) -> str:
    a = c["airports"].loc[iata]
    city = a["ciudad"] if isinstance(a["ciudad"], str) else ""
    return f"{iata} · {city}" if city else iata


def minutes_text(m: float) -> str:
    if m >= 1:
        return f"{m:.0f} min de retraso"
    if m <= -1:
        return f"{-m:.0f} min antes de lo programado"
    return "a la hora programada"


def risk(p: float):
    if p < RISK_LOW:
        return "Riesgo bajo", st.success, ":material/check_circle:"
    if p < RISK_HIGH:
        return "Riesgo moderado", st.warning, ":material/warning:"
    return "Riesgo alto", st.error, ":material/error:"


def out_of_ten(p: float) -> str:
    n = round(10 * p)
    if n == 0:
        return "Menos de 1 de cada 10 vuelos como este llega con 15+ min de retraso."
    verb = "llega" if n == 1 else "llegan"
    return f"De cada 10 vuelos como este, alrededor de {n} {verb} con 15+ min de retraso."


def locked(title: str, teaser: str):
    with st.container(border=True):
        st.markdown(f"**:material/lock: {title}**")
        st.caption(f"{teaser} Disponible en Premium.")


try:
    c, kpis = load()
except Exception as e:  # sin modelos o sin datos: explicar como generarlos
    st.error(f"No se pudieron cargar los modelos: {e}")
    st.info("Corre primero el pipeline: `python -m src.run_pipeline`")
    st.stop()

national = kpis["reports/kpis.json"].get("tasa_retraso_global")
ev = kpis["deliverables/kpis.json"].get("evaluacion", {})

# ---------------------------------------------------------------- barra lateral
with st.sidebar:
    st.header("Tu plan")
    premium = st.toggle("Premium (simulado)", value=False,
                        help="En la demo el plan se cambia con este interruptor; no hay pagos.")
    if premium:
        st.success("Plan Premium activo", icon=":material/workspace_premium:")
    else:
        st.info("Plan gratis", icon=":material/person:")
    st.markdown("**Premium incluye**\n\n- Minutos de retraso estimados\n- Clima del aeropuerto de origen"
                "\n- Mejor hora para volar ese día\n- Comparación de aerolíneas en la ruta")
    st.divider()
    st.caption("Demo académica con datos históricos del BTS y Open-Meteo de 2025. "
               "No es una aplicación comercial.")

# ---------------------------------------------------------------- formulario
st.title(":material/flight_takeoff: ¿Llegará a tiempo tu vuelo?")
st.caption("Estimación antes de volar, con lo que se sabe de antemano: aerolínea, ruta, fecha, "
           "hora programada y clima del origen.")

origins = sorted({o for o, _ in c["route_stats"]})
col1, col2 = st.columns(2)
origen = col1.selectbox("Origen", origins, index=origins.index("ORD") if "ORD" in origins else 0,
                        format_func=lambda x: airport_label(c, x))
dests = sorted({d for o, d in c["route_stats"] if o == origen})
destino = col2.selectbox("Destino", dests, index=dests.index("LGA") if "LGA" in dests else 0,
                         format_func=lambda x: airport_label(c, x))

carriers = c["route_carriers"].get((origen, destino), sorted(c["carrier_names"]))
col3, col4, col5 = st.columns([2, 1.2, 1])
aerolinea = col3.selectbox("Aerolínea", carriers, format_func=lambda x: f"{c['carrier_names'][x]} ({x})",
                           help="Aerolíneas que operaron esta ruta en 2025, de mayor a menor volumen.")
fecha = col4.date_input("Fecha", value=date(2025, 12, 22), min_value=date(2025, 1, 1),
                        max_value=date(2025, 12, 31), format="YYYY-MM-DD",
                        help="La demo usa 2025 porque es el año con clima histórico descargado.")
hora = col5.selectbox("Salida", list(range(24)), index=17, format_func=lambda h: f"{h:02d}:00")

# ---------------------------------------------------------------- prediccion
# Una sola consulta en lote: [0] el vuelo, [1..24] cada hora del dia, [25..] cada aerolinea.
queries = ([(aerolinea, origen, destino, fecha.isoformat(), hora)]
           + [(aerolinea, origen, destino, fecha.isoformat(), h) for h in range(24)]
           + [(a, origen, destino, fecha.isoformat(), hora) for a in carriers])
try:
    with st.spinner("Calculando..."):
        res = run(tuple(queries))
except ValueError as e:
    st.error(f"Consulta no válida: {e}")
    st.stop()

main, by_hour, by_carrier = res[0], res[1:25], res[25:]
p = main["prob_retraso"]
label, box, icon = risk(p)

st.divider()
st.subheader(f"{c['carrier_names'][aerolinea]} · {origen} → {destino} · {fecha:%d/%m/%Y} {hora:02d}:00")
m1, m2 = st.columns(2)
m1.metric("Probabilidad de llegar 15+ min tarde", f"{100 * p:.0f}%",
          delta=f"{100 * (p - national):+.0f} pp vs promedio nacional" if national else None,
          delta_color="inverse")
stats = c["route_stats"].get((origen, destino))
if stats:
    m2.metric("Tasa histórica de la ruta (2025)", f"{100 * stats['tasa_retraso']:.0f}%",
              help=f"{stats['vuelos']:,} vuelos completados en 2025, todas las aerolíneas.")
box(f"**{label}.** {out_of_ten(p)}", icon=icon)

# ---------------------------------------------------------------- premium
st.subheader("Detalle Premium")
if not premium:
    a, b = st.columns(2)
    with a:
        locked("Minutos estimados", "Cuánto retraso esperar en promedio.")
        locked("Clima del origen", "Lluvia, nieve, viento y temperatura del día.")
    with b:
        locked("Mejor hora para volar", "El riesgo de cada hora de salida ese mismo día.")
        locked("Comparar aerolíneas", f"Las {len(carriers)} aerolíneas de esta ruta, lado a lado.")
    st.button("Probar Premium", type="primary", disabled=True,
              help="En la demo, activa Premium con el interruptor de la barra lateral.")
else:
    a, b = st.columns(2)
    with a.container(border=True):
        st.markdown("**:material/schedule: Minutos estimados**")
        st.markdown(f"### {minutes_text(main['minutos_estimados'])}")
        st.caption("Promedio esperado para vuelos con estas características; un vuelo concreto puede variar mucho.")
    with b.container(border=True):
        st.markdown("**:material/partly_cloudy_day: Clima en el origen ese día**")
        w = main["clima_usado"]
        if isinstance(w, dict):
            st.markdown(f"Lluvia {w['precip_mm']:.1f} mm · Nieve {w['snow_cm']:.1f} cm  \n"
                        f"Viento máx. {w['wind_max_kmh']:.0f} km/h · {w['temp_min_c']:.0f} a {w['temp_max_c']:.0f} °C")
        else:
            st.markdown("Sin datos de clima para este aeropuerto")
        st.caption("Clima observado de 2025 (en producción sería el pronóstico). "
                   "Cubre los 50 aeropuertos con más vuelos.")

    with st.container(border=True):
        st.markdown("**:material/insights: Mejor hora para volar ese día**")
        # Solo horas con vuelos programados en esta ruta en 2025: no tiene sentido sugerir otras.
        valid = set(c["route_hours"].get((origen, destino), range(24))) | {hora}
        hours = pd.DataFrame([{"hora": r["hora"], "prob": r["prob_retraso"], "minutos": r["minutos_estimados"]}
                              for r in by_hour if r["hora"] in valid])
        hours["etiqueta"] = hours["hora"].map(lambda h: f"{h:02d}:00")
        hours["tipo"] = hours["hora"].map(lambda h: "Tu hora" if h == hora else "Otras horas")
        best = hours.loc[hours["prob"].idxmin()]
        chart = (alt.Chart(hours).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
            x=alt.X("etiqueta:N", title="Hora de salida", sort=None, axis=alt.Axis(labelAngle=-45)),
            y=alt.Y("prob:Q", title="Probabilidad de retraso", axis=alt.Axis(format="%")),
            color=alt.Color("tipo:N", scale=alt.Scale(domain=["Tu hora", "Otras horas"], range=[ORANGE, BLUE]),
                            legend=alt.Legend(title=None, orient="top")),
            tooltip=[alt.Tooltip("etiqueta:N", title="Hora"),
                     alt.Tooltip("prob:Q", title="Probabilidad", format=".0%"),
                     alt.Tooltip("minutos:Q", title="Minutos estimados", format=".0f")])
            .properties(height=260))
        st.altair_chart(chart, width="stretch")
        if best["hora"] != hora:
            st.markdown(f"Salir a las **{best['etiqueta']}** baja el riesgo a **{100 * best['prob']:.0f}%** "
                        f"(desde {100 * p:.0f}%).")
        else:
            st.markdown("Ya elegiste la hora con menor riesgo de ese día.")
        st.caption("Solo se muestran las horas con salidas programadas en esta ruta durante 2025. "
                   "Confirma los horarios con la aerolínea.")

    with st.container(border=True):
        st.markdown("**:material/compare_arrows: Aerolíneas en esta ruta, misma fecha y hora**")
        tab = pd.DataFrame([{
            "Aerolínea": c["carrier_names"][r["codigo_aerolinea"]],
            "Probabilidad de retraso": round(100 * r["prob_retraso"]),
            "Minutos estimados": r["minutos_estimados"],
        } for r in by_carrier]).sort_values("Probabilidad de retraso").reset_index(drop=True)
        st.dataframe(tab, hide_index=True, width="stretch", column_config={
            "Probabilidad de retraso": st.column_config.ProgressColumn(
                format="%d%%", min_value=0, max_value=100),
            "Minutos estimados": st.column_config.NumberColumn(format="%.0f min"),
        })

# ---------------------------------------------------------------- transparencia
with st.expander("Cómo funciona y qué tan confiable es"):
    clf = ev.get("clasificacion", {}).get("modelo_regresion_logistica", {})
    reg = ev.get("regresion", {}).get("modelo_gbt", {})
    st.markdown(
        "- Modelos entrenados con 5.75 millones de vuelos de enero a octubre de 2025 y evaluados con "
        "noviembre y diciembre.\n"
        + (f"- Clasificación: ROC AUC {clf['roc_auc']:.2f}. Sirve para comparar vuelos por riesgo, "
           "no para asegurar que un vuelo concreto llegará tarde.\n" if clf else "")
        + (f"- Minutos: error absoluto medio de {reg['mae']:.0f} min.\n" if reg else "")
        + "- No usa el retraso real ni sus causas: solo lo que se conoce antes de volar.\n"
        "- Limitación conocida: con un solo año de datos, subestima el riesgo de la temporada "
        "navideña (diciembre).")
    st.caption(f"Modelo de minutos: {main['modelo_minutos']} · Distancia: {main['distancia_millas']:.0f} millas "
               f"({main['distancia_fuente']})")
