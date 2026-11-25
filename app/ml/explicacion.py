"""
Factores que suben o bajan el riesgo de un vuelo (regresion logistica) y semaforo de riesgo.

Contribucion de cada variable ORIGINAL: la regresion logistica calcula
    logit = intercepto + suma_j coef_j * z_j
donde z son las columnas ya transformadas (one-hot y escaladas). Para un vuelo x y el vuelo de
REFERENCIA (el promedio de z sobre todos los vuelos completados de 2025, ver
app/data/build_referencia.py), la contribucion de la variable v es
    c_v = suma_{j en columnas de v} coef_j * (z_j(x) - z_ref_j)
y la suma de todas las c_v es exactamente logit(x) - logit(referencia). Para un modelo lineal
esto es la descomposicion exacta (equivale a SHAP lineal). En puntos de probabilidad se informa
cuanto bajaria (o subiria) la probabilidad corregida si esa variable tomara el valor de
referencia: p(L) - p(L - c_v). Con clima tipico (21 escenarios) se promedia entre escenarios.

Sin pandas en la ruta critica de calculo: solo numpy y scipy (via sklearn).
"""
import json
import os

import numpy as np

from app.ml.features import CATEGORICAL, CYCLIC_SMALL, NUMERIC

# Variable original (como la entiende el usuario) de cada variable del modelo.
GRUPO = {"OP_UNIQUE_CARRIER": "aerolinea", "ORIGIN": "origen", "DEST": "destino",
         "DAY_OF_WEEK": "dia", "is_weekend": "dia", "dep_hour": "hora",
         "month_sin": "mes", "month_cos": "mes", "MONTH": "mes", "DISTANCE": "distancia",
         "precip_mm": "lluvia", "rain_flag": "lluvia", "snow_cm": "nieve", "snow_flag": "nieve",
         "wind_max_kmh": "viento", "high_wind_flag": "viento",
         "temp_max_c": "temperatura", "temp_min_c": "temperatura",
         "has_weather": "datos_clima"}
GRUPOS_CLIMA = ["lluvia", "nieve", "viento", "temperatura", "datos_clima"]
MESES = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto",
         "septiembre", "octubre", "noviembre", "diciembre"]
DIAS = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]


def grupos_columnas(cls):
    """Grupo (variable original) de cada columna de salida del preprocesamiento. Se arma por
    posicion con output_indices_ y las categorias de cada OneHotEncoder, asi funciona tambien
    con la variante que agrega el mes categorico."""
    pre = cls.pipeline.named_steps["pre"]
    n = sum(s.stop - s.start for s in pre.output_indices_.values())
    grupos = [None] * n
    for nombre, sl in pre.output_indices_.items():
        if sl.stop == sl.start:
            continue
        if nombre == "ohe":
            enc = pre.named_transformers_["ohe"]
            k = sl.start
            for col, cats in zip(CATEGORICAL + CYCLIC_SMALL, enc.categories_):
                for _ in cats:
                    grupos[k] = GRUPO[col]
                    k += 1
        elif nombre == "mes":
            for k in range(sl.start, sl.stop):
                grupos[k] = "mes"
        elif nombre == "num":
            for k, col in zip(range(sl.start, sl.stop), NUMERIC):
                grupos[k] = GRUPO[col]
    assert None not in grupos
    return grupos


def transformar(cls, X):
    """Columnas finales z (las que multiplica la regresion logistica), como matriz dispersa."""
    pl = cls.pipeline.named_steps
    return pl["std"].transform(pl["pre"].transform(X))


def cargar_referencia(data_dir, nombre_clasificador):
    with open(os.path.join(data_dir, "referencia_modelo.json")) as fh:
        ref = json.load(fh)
    if nombre_clasificador not in ref["clasificadores"]:
        raise KeyError(f"sin referencia para {nombre_clasificador}: corre make tablas")
    r = ref["clasificadores"][nombre_clasificador]
    return {**r, "z_ref": np.asarray(r["z_ref"], dtype="float64")}


def contribuciones(cls, X, z_ref, grupos):
    """-> lista de (grupo, contribucion en logit, efecto en puntos de probabilidad),
    promediada sobre las filas de X (escenarios de clima). Ordenada por |efecto|."""
    lr = cls.pipeline.named_steps["lr"]
    coef = lr.coef_.ravel()
    z = transformar(cls, X)
    dz = np.asarray(z.todense()) - z_ref                       # filas x columnas
    por_col = dz * coef                                        # contribucion por columna
    nombres = sorted(set(grupos))
    idx = {g: [j for j, gg in enumerate(grupos) if gg == g] for g in nombres}
    c = {g: por_col[:, idx[g]].sum(axis=1) for g in nombres}   # por fila y grupo
    sin_clima = bool(np.all(X["has_weather"].to_numpy() == 0))
    if sin_clima:
        # Sin datos de clima las variables de clima valen 0 y has_weather=0: se informan juntas.
        c = {**{g: v for g, v in c.items() if g not in GRUPOS_CLIMA},
             "sin_clima": sum(c[g] for g in GRUPOS_CLIMA if g in c)}
    L = lr.intercept_[0] + np.asarray(z @ coef).ravel()
    shift = np.log(cls.w_neg / cls.w_pos)                      # correccion a la tasa base
    sig = lambda v: 1 / (1 + np.exp(-v))                       # noqa: E731
    out = [(g, float(v.mean()), float((sig(L + shift) - sig(L - v + shift)).mean()))
           for g, v in c.items()]
    return sorted(out, key=lambda r: -abs(r[2]))


def frase(grupo, efecto, vuelo, nombres):
    """Explicacion en lenguaje simple. vuelo: dict con aerolinea, origen, destino, fecha,
    dep_hour, distancia, clima (dict o None), fuente_clima. nombres: dict con
    'aerolinea' y 'ciudad' (por IATA)."""
    f = vuelo["fecha"]
    pron = vuelo.get("fuente_clima") == "pronóstico"
    cl = vuelo.get("clima") or {}
    clima_txt = "pronosticada" if pron else "típica de esas fechas"
    sujeto = {
        "hora": f"Salir a las {vuelo['dep_hour']} h",
        "dia": f"Volar en {DIAS[f.isoweekday() - 1]}",
        "mes": f"Volar en {MESES[f.month - 1]}",
        "aerolinea": f"Volar con {nombres['aerolinea']}",
        "origen": f"Salir de {nombres['ciudad'](vuelo['origen'])}",
        "destino": f"Llegar a {nombres['ciudad'](vuelo['destino'])}",
        "distancia": f"Una ruta de {vuelo['distancia']:,.0f} millas",
        "lluvia": (f"La lluvia pronosticada ({cl['precip_mm']:.1f} mm)" if pron and cl
                   else f"La lluvia {clima_txt}"),
        "nieve": (f"La nieve pronosticada ({cl['snow_cm']:.1f} cm)" if pron and cl
                  else f"La nieve {clima_txt}"),
        "viento": (f"El viento pronosticado ({cl['wind_max_kmh']:.0f} km/h)" if pron and cl
                   else "El viento típico de esas fechas"),
        "temperatura": (f"La temperatura pronosticada ({cl['temp_min_c']:.0f} a "
                        f"{cl['temp_max_c']:.0f} °C)" if pron and cl
                        else "La temperatura típica de esas fechas"),
        "datos_clima": "Tener datos de clima del origen",
        "sin_clima": "Salir de un aeropuerto sin datos de clima",
    }[grupo]
    verbo = "aumenta" if efecto > 0 else "reduce"
    pts = round(100 * efecto)
    if pts == 0:
        cuanto = "menos de 1 punto"
    else:
        cuanto = f"{pts:+d} punto{'s' if abs(pts) != 1 else ''}"
    return f"{sujeto} {verbo} el riesgo ({cuanto})."


def nivel_riesgo(p, umbrales):
    """Semaforo con los percentiles 50 y 80 de la probabilidad del modelo en 2025."""
    if p < umbrales["p50"]:
        return "bajo"
    if p < umbrales["p80"]:
        return "medio"
    return "alto"
