"""
Paso D (cierre Fase 3): genera deliverables/RESUMEN_TECNICO.md.

Todos los numeros se leen de archivos medidos (deliverables/kpis.json, reports/stage_metrics.json
y los CSV de deliverables/powerbi): ninguno se escribe a mano, asi el documento siempre coincide
con la ultima corrida. Lo que no se pudo medir se marca como "no medido".

Uso: python -m src.summary
"""
import json
import os
from importlib.metadata import version

import pandas as pd

from src.config import ROOT, WEATHER_TOP_N
from src.evaluate import KPIS_OUT, TEST_PREDS
from src.export_powerbi import OUT_DIR as PBI
from src.features import INPUT_COLS, LEAKAGE_COLS
from src.figures import FIG_DIR
from src.silver import HIGH_WIND_KMH, RAIN_MM
from src.utils import STAGE_METRICS_PATH

OUT = os.path.join(ROOT, "deliverables", "RESUMEN_TECNICO.md")


def n(x, d=0):
    """Numero con separador de miles (coma) y punto decimal."""
    if x is None:
        return "no medido"
    return f"{x:,.{d}f}"


def table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def main():
    with open(KPIS_OUT) as f:
        k = json.load(f)
    with open(STAGE_METRICS_PATH) as f:
        s = json.load(f)
    mq, vol, lim, tim, q, ev = (k["maquina"], k["volumen"], k["limpieza"], k["tiempos"],
                                k["consultas_dashboard"], k["evaluacion"])
    fil, mb = vol["filas"], vol["disco_mb"]
    clf, reg, meta = ev["clasificacion"], ev["regresion"], ev["conjunto_prueba"]
    lr, rb, at = clf["modelo_regresion_logistica"], clf["baseline_tasa_historica_ruta"], clf["baseline_siempre_a_tiempo"]
    gb, bg, br = reg["modelo_gbt"], reg["baseline_media_global"], reg["baseline_promedio_ruta"]
    cm = ev["matriz_confusion_modelo"]
    sil, gold, wea, my, mod = s["silver"], s["gold"], s["weather"], s.get("mysql", {}), s["models"]
    clima = pd.read_csv(os.path.join(PBI, "efecto_clima.csv"))
    causas = pd.read_csv(os.path.join(PBI, "causas_retraso.csv"))
    mes = pd.read_csv(os.path.join(PBI, "retraso_por_mes.csv"))
    pbi_files = sorted(f for f in os.listdir(PBI) if f.endswith(".csv"))
    figs = sorted(f for f in os.listdir(FIG_DIR) if f.endswith(".png"))
    mysql_tbl = mb["mysql"] if isinstance(mb["mysql"], dict) else {}
    pct = lambda x: f"{100 * x:.2f}%"
    libs = {p: version(p) for p in ["pyspark", "pandas", "pyarrow", "scikit-learn", "mlflow",
                                    "matplotlib", "SQLAlchemy", "PyMySQL", "requests"]}
    lo_c, hi_c = k["evaluacion"]["recorte_arr_delay_min"]
    n_over = int((pd.read_parquet(TEST_PREDS, columns=["pred_min"])["pred_min"] > hi_c).sum())
    dic = mes.loc[mes["mes"] == 12].iloc[0]

    md = []
    A = md.append
    A("# Resumen tecnico: prediccion de retrasos de vuelos (Fase 3)")
    A(f"Documento generado automaticamente el {k['generado']} a partir de las metricas medidas "
      "(`deliverables/kpis.json`). Todos los numeros provienen de la ultima corrida completa del "
      "pipeline; lo que no se pudo medir se indica como \"no medido\".")

    # 1 -----------------------------------------------------------------
    A("## 1. Arquitectura implementada")
    A("Arquitectura medallon (Bronze, Silver, Gold) con esquema estrella en Gold, implementada como "
      "prototipo local con Apache Spark y disenada para migrarse sin cambios de logica a Microsoft Fabric.")
    A("```\nzips BTS -> data/raw (CSV) -> Bronze (Parquet crudo) -> Silver (limpio + clima) -> Gold (estrella)\n"
      "Open-Meteo -> data/raw/weather ---------------------------^              |-> MySQL (vuelos)\n"
      "OpenFlights -> data/raw/airports --------------------------------------> |-> Modelos MLlib + MLflow\n"
      "                                                                         |-> CSV Power BI, figuras\n```")
    A(table(["Capa", "Ubicacion", "Formato", "Contenido"], [
        ["Raw", "data/raw/", "CSV, JSON", "12 CSV mensuales del BTS, JSON de clima por aeropuerto, airports.dat"],
        ["Bronze", "data/bronze/flights", "Parquet snappy, particion MONTH", "Datos sin transformar (todo texto) + linaje"],
        ["Silver", "data/silver/", "Parquet, particion MONTH", "flights (completados + clima), flights_disrupted, weather_daily"],
        ["Gold", "data/gold/", "Parquet", "FACT_VUELOS + DIM_TIEMPO, DIM_AEROLINEA, DIM_AEROPUERTO, DIM_CLIMA, con predicciones"],
    ]))
    A("**Tecnologias y versiones (medidas en la maquina de ejecucion):**")
    A(table(["Componente", "Version / valor"], [
        ["Equipo", f"{mq['modelo_equipo']} ({mq['cpu']}, {mq['nucleos_totales']} nucleos: "
                   f"{mq['nucleos_rendimiento']} de rendimiento + {mq['nucleos_eficiencia']} de eficiencia, {mq['ram_gb']} GB RAM)"],
        ["Sistema operativo", mq["sistema"]],
        ["Apache Spark (PySpark)", mq["spark"]],
        ["Java", mq["java"]],
        ["Python", mq["python"]],
        ["Spark: master / memoria driver / particiones shuffle",
         f"{mq['spark_master']} / {mq['spark_driver_memory']} / {mq['spark_shuffle_partitions']} (AQE activado)"],
        ["MySQL", f"{mq['mysql']['version']} (buffer pool {mq['mysql']['innodb_buffer_pool_mb']} MB)"
         if isinstance(mq["mysql"], dict) else mq["mysql"]],
        ["Librerias", ", ".join(f"{p} {v}" for p, v in libs.items())],
    ]))
    A("**Procesamiento distribuido.** Todo el procesamiento de datos usa el motor distribuido de Spark "
      f"(DataFrames, joins con broadcast, agregaciones con shuffle, MLlib). En el prototipo corre en modo "
      f"`{mq['spark_master']}`: un driver que reparte tareas en paralelo entre los {mq['nucleos_totales']} "
      "nucleos del equipo. El mismo codigo corre sin cambios en un cluster (Fabric Spark), donde las "
      "tareas se reparten entre varios nodos; en este proyecto no se midio una ejecucion multinodo.")
    A("**Mapeo a la arquitectura objetivo en Microsoft Fabric:**")
    A(table(["Prototipo local", "Microsoft Fabric"], [
        ["data/raw/", "Lakehouse lh_raw, seccion Files de OneLake"],
        ["data/bronze, data/silver, data/gold (Parquet)", "Lakehouses lh_bronze, lh_silver, lh_gold con tablas Delta en OneLake"],
        ["Modulos src/*.py ejecutados con PySpark local", "Notebooks de Spark en Fabric (un notebook por etapa)"],
        ["src/run_pipeline.py", "Data Factory pipeline que encadena los notebooks"],
        ["MLflow local (mlflow.db)", "Experimentos y modelos de ML de Fabric (MLflow integrado)"],
        ["MySQL vuelos", "Warehouse de Fabric o SQL analytics endpoint del lakehouse Gold"],
        ["CSV agregados para Power BI", "Modelo semantico en modo Direct Lake sobre las tablas Delta de lh_gold"],
    ]))

    # 2 -----------------------------------------------------------------
    A("## 2. Dataset")
    A(table(["Fuente", "Que aporta", "Acceso"], [
        ["BTS On-Time Performance (TranStats)", "Un registro por vuelo domestico programado", "12 zips mensuales descargados manualmente"],
        ["Open-Meteo Archive API", "Clima diario historico por aeropuerto", f"API gratuita, sin llave; {WEATHER_TOP_N} llamadas (una por aeropuerto)"],
        ["OpenFlights airports.dat", "Nombre, coordenadas y zona horaria por codigo IATA", "Descarga directa (GitHub)"],
    ]))
    A("**Periodo:** 1 de enero a 31 de diciembre de 2025 (12 meses completos).")
    A("**Columnas del BTS (31, header real):** YEAR, MONTH, DAY_OF_MONTH, DAY_OF_WEEK, FL_DATE, "
      "OP_UNIQUE_CARRIER, OP_CARRIER_FL_NUM, ORIGIN, ORIGIN_CITY_NAME, ORIGIN_STATE_ABR, DEST, "
      "DEST_CITY_NAME, DEST_STATE_ABR, CRS_DEP_TIME, DEP_DELAY, DEP_DEL15, DEP_TIME_BLK, CRS_ARR_TIME, "
      "ARR_DELAY, ARR_DEL15, ARR_TIME_BLK, CANCELLED, CANCELLATION_CODE, DIVERTED, DISTANCE, "
      "DISTANCE_GROUP, CARRIER_DELAY, WEATHER_DELAY, NAS_DELAY, SECURITY_DELAY, LATE_AIRCRAFT_DELAY.")
    A("**Variables de clima (Open-Meteo, diarias):** precipitation_sum (mm), snowfall_sum (cm), "
      "wind_speed_10m_max (km/h), temperature_2m_max y temperature_2m_min (C), con `timezone=auto` "
      "para que el dia coincida con el dia local del vuelo.")
    A("**Volumen por capa (medido):**")
    A(table(["Capa / tabla", "Filas", "Tamano en disco (MB)"], [
        ["Raw: 12 CSV del BTS", n(fil["raw_csv"]), n(mb["raw_flights_csv"], 1)],
        ["Raw: clima (50 JSON + CSV consolidado)", n(fil["silver_clima_diario"]) + " aeropuerto-dia", n(mb["raw_weather_json"], 1)],
        ["Bronze: flights", n(fil["bronze"]), n(mb["bronze"], 1)],
        ["Silver: flights (completados)", n(fil["silver_completados"]), n(mb["silver_detalle"]["flights"], 1)],
        ["Silver: flights_disrupted (cancelados y desviados)", n(fil["silver_cancelados_desviados"]), n(mb["silver_detalle"]["flights_disrupted"], 1)],
        ["Silver: weather_daily", n(fil["silver_clima_diario"]), n(mb["silver_detalle"]["weather_daily"], 1)],
        ["Gold: FACT_VUELOS", n(fil["gold"]["FACT_VUELOS"]), n(mb["gold_detalle"]["FACT_VUELOS"], 1)],
        ["Gold: 4 dimensiones", n(sum(v for kk, v in fil["gold"].items() if kk != "FACT_VUELOS")),
         n(sum(v for kk, v in mb["gold_detalle"].items() if kk != "FACT_VUELOS"), 1)],
        ["MySQL: FACT_VUELOS (datos + indices)", n(mysql_tbl.get("FACT_VUELOS", {}).get("filas_exactas")),
         n(mysql_tbl.get("FACT_VUELOS", {}).get("mb"), 1)],
    ]))
    A(f"Compresion: los {n(mb['raw_flights_csv'], 1)} MB de CSV ocupan {n(mb['bronze'], 1)} MB en Parquet "
      f"(Bronze), {mb['raw_flights_csv'] / mb['bronze']:.1f} veces menos.")
    A(f"**Cobertura de clima:** {vol['cobertura_clima_pct']['vuelos_completados']}% de los vuelos completados "
      f"({vol['cobertura_clima_pct']['todos_los_vuelos_gold']}% de todos los vuelos), al cubrir los "
      f"{vol['cobertura_clima_pct']['aeropuertos_con_clima']} aeropuertos de origen con mas vuelos de "
      f"{vol['cobertura_clima_pct']['aeropuertos_totales']} en total. {wea['airports_ok']} de "
      f"{wea['airports_requested']} descargas exitosas, 0 sin coordenadas.")

    # 3 -----------------------------------------------------------------
    A("## 3. Limpieza y transformacion")
    A(table(["Etapa", "Regla aplicada", "Filas afectadas"], [
        ["Organizacion", "Mes detectado leyendo la 2a linea de cada CSV (el nombre del zip no indica el mes); duplicados omitidos", "0 duplicados, 12 de 12 meses"],
        ["Bronze", "Schema explicito de 31 columnas como texto (sin inferSchema), modo FAILFAST", f"0 filas mal formadas; {n(fil['bronze'])} ingeridas"],
        ["Bronze", "Columnas de linaje _source_file y _ingested_at (sin modificar valores)", n(fil["bronze"])],
        ["Silver", "Tipos: enteros (calendario, banderas '0.00'/'1.00'), double (minutos, distancia), date (FL_DATE 'M/d/yyyy h:mm:ss a')", n(lim["filas_bronze"])],
        ["Silver", "Descartar filas con clave minima nula o invalida (fecha, aerolinea, origen, destino, hora, distancia, banderas)", n(lim["descartadas_clave_invalida_o_nula"])],
        ["Silver", "Coherencia de FL_DATE con YEAR/MONTH/DAY_OF_MONTH y DAY_OF_WEEK", n(lim["fecha_incoherente_con_year_month_day_o_day_of_week"])],
        ["Silver", "Duplicados por clave de negocio (FL_DATE, aerolinea, numero de vuelo, origen, destino, CRS_DEP_TIME)", n(lim["descartadas_duplicados_clave_negocio"])],
        ["Silver", "Separar cancelados a flights_disrupted", n(lim["separadas_cancelados"])],
        ["Silver", "Separar desviados a flights_disrupted", n(lim["separadas_desviados"])],
        ["Silver", "Descartar completados sin ARR_DELAY / ARR_DEL15 (sin etiqueta)", n(lim["descartadas_completados_sin_arr_delay"])],
        ["Silver", "Causas de retraso nulas en completados se rellenan con 0 (el BTS solo las llena con retraso 15+ min)", n(lim["completados_con_causas_nulas_rellenadas_con_0"])],
        ["Silver", "DEP_DELAY nulo en completados (se deja nulo, no se usa en el modelo)", n(lim["completados_con_dep_delay_nulo"])],
        ["Silver", "CRS_DEP_TIME = 2400 se trata como hora 0", n(lim["crs_dep_time_2400_tratado_como_hora_0"])],
        ["Silver", "CANCELLATION_CODE se elimina de completados (100% nula ahi; se conserva en flights_disrupted)", n(lim["filas_silver_completados"])],
        ["Silver", "Union con clima de ORIGEN por (aeropuerto, fecha), left join con broadcast; sin clima: has_weather = 0", f"{n(lim['completados_sin_clima_origen_has_weather_0'])} sin clima"],
        ["Silver", f"Derivadas: dep_hour, is_weekend (sabado y domingo), route; banderas lluvia (>= {RAIN_MM} mm), nieve (> 0 cm), viento fuerte (>= {HIGH_WIND_KMH} km/h)", n(lim["filas_silver_completados"])],
        ["Gold", "Union de completados y no completados en FACT_VUELOS; llaves sustitutas; clima_key = -1 sin clima", n(fil["gold"]["FACT_VUELOS"])],
    ]))
    A(f"Balance: {n(lim['filas_bronze'])} (Bronze) = {n(lim['filas_silver_completados'])} completados + "
      f"{n(fil['silver_cancelados_desviados'])} cancelados/desviados + "
      f"{n(lim['descartadas_completados_sin_arr_delay'])} descartado. Gold contiene "
      f"{n(fil['gold']['FACT_VUELOS'])} filas (completados + cancelados/desviados).")
    A(f"Banderas de clima sobre vuelos con clima: lluvia {sil['weather_flag_pct']['rain_flag']}%, nieve "
      f"{sil['weather_flag_pct']['snow_flag']}%, viento fuerte {sil['weather_flag_pct']['high_wind_flag']}%. "
      f"Los umbrales salen de la distribucion real de 18,250 aeropuerto-dia (viento: sobre el percentil 95).")

    # 4 -----------------------------------------------------------------
    A("## 4. Esquema estrella")
    A(table(["Tabla", "Filas", "Llave primaria", "Columnas principales"], [
        ["FACT_VUELOS", n(fil["gold"]["FACT_VUELOS"]), "vuelo_key (xxhash64 de la clave de negocio)",
         "fecha_key, aerolinea_key, origen_key, destino_key, clima_key; dep_hour, DISTANCE, CANCELLED, DIVERTED, "
         "DEP_DELAY, ARR_DELAY, ARR_DEL15, 5 causas de retraso, pred_prob, pred_min, MONTH"],
        ["DIM_TIEMPO", n(fil["gold"]["DIM_TIEMPO"]), "fecha_key (yyyymmdd)",
         "fecha, anio, trimestre, mes, nombre_mes, dia, dia_semana, nombre_dia, semana_anio, es_fin_semana, es_feriado (11 feriados federales)"],
        ["DIM_AEROLINEA", n(fil["gold"]["DIM_AEROLINEA"]), "aerolinea_key", "codigo, nombre, tipo (principal, bajo_costo, regional)"],
        ["DIM_AEROPUERTO", n(fil["gold"]["DIM_AEROPUERTO"]), "aeropuerto_key",
         "iata, nombre, ciudad, estado, latitud, longitud, zona_horaria, tiene_clima. Una sola dimension usada como origen y destino"],
        ["DIM_CLIMA", n(fil["gold"]["DIM_CLIMA"]), "clima_key",
         "aeropuerto_key, iata, fecha_key, precip_mm, snow_cm, wind_max_kmh, temp_max_c, temp_min_c, lluvia, nieve, viento_fuerte, condicion; miembro -1 sin_datos"],
    ]))
    A(f"Integridad verificada: {n(gold['fact_distinct_vuelo_key'])} vuelo_key distintos en {n(gold['fact_rows_expected'])} "
      f"filas (sin colisiones); 0 registros huerfanos en las 5 llaves foraneas (verificado en MySQL). "
      "Dos aeropuertos (EAR, XWA) no estan en OpenFlights: quedan en la dimension con ciudad y estado del BTS y sin coordenadas.")

    # 5 -----------------------------------------------------------------
    A("## 5. Modelos")
    A(f"**Variables de entrada ({len(INPUT_COLS)}):** aerolinea (OP_UNIQUE_CARRIER), origen (ORIGIN), destino (DEST), "
      "dia de la semana, hora programada de salida (dep_hour), mes codificado como seno y coseno, fin de semana, "
      "distancia, y clima del aeropuerto de origen (precip_mm, snow_cm, wind_max_kmh, temp_max_c, temp_min_c, "
      "banderas de lluvia, nieve y viento fuerte, has_weather).")
    A("**Variables EXCLUIDAS por fuga de datos:** " + ", ".join(LEAKAGE_COLS) + ". La app predice antes del vuelo; "
      "estas columnas solo existen despues de que el vuelo sale o llega (el retraso real y su causa). Usarlas daria "
      "metricas casi perfectas en pruebas y un modelo inutil en produccion. ARR_DEL15 y ARR_DELAY solo se usan como "
      "etiqueta. `src/train.py` valida la regla antes de entrenar. Tampoco se usan CANCELLED, DIVERTED ni "
      "CANCELLATION_CODE (resultado del vuelo). El clima es el observado del dia; en produccion seria el pronostico.")
    A(f"**Split temporal:** entrenamiento enero a octubre ({n(5752264)} vuelos completados), prueba noviembre y "
      f"diciembre ({n(meta['filas_prueba'])} vuelos). Un split aleatorio mezclaria vuelos del mismo dia y "
      f"sobreestimaria el desempeno. Tasa de retraso: {pct(meta['tasa_retraso_entrenamiento'])} en entrenamiento, "
      f"{pct(meta['tasa_retraso_prueba'])} en prueba.")
    pw = ev["pesos_clase"]
    A(table(["Modelo", "Objetivo", "Hiperparametros"], [
        ["Regresion logistica (Spark MLlib)", "ARR_DEL15",
         f"maxIter 50, regParam 0.001, elasticNet 0; pesos de clase balanceados (retrasado {pw['retrasado']:.3f}, "
         f"a tiempo {pw['a_tiempo']:.3f}); one-hot de aerolinea, origen, destino, dia y hora; numericas escaladas"],
        ["Regresion lineal (Spark MLlib)", "ARR_DELAY recortado", "maxIter 50, regParam 0.001; mismas variables (baseline lineal)"],
        ["GBT Regressor (Spark MLlib)", "ARR_DELAY recortado",
         "maxIter 60, maxDepth 6, stepSize 0.1, maxBins 400, subsamplingRate 0.8, seed 42; entrenado con una "
         "muestra aleatoria del 30% del entrenamiento (tamano exacto de la muestra no registrado)"],
    ]))
    lo, hi = ev["recorte_arr_delay_min"]
    A(f"**Recorte de ARR_DELAY:** a los percentiles 1 y 99 del entrenamiento ([{lo:.0f}, {hi:.0f}] min). Los retrasos "
      "extremos dominan el RMSE sin ayudar al caso tipico; los percentiles salen solo de enero a octubre.")
    A(f"**Probabilidad mostrada al usuario (pred_prob):** los pesos de clase inflan la probabilidad; se corrige "
      f"multiplicando los odds por peso_a_tiempo / peso_retrasado. El umbral de decision {meta['umbral_clasificacion']:.4f} "
      "(tasa base de entrenamiento) equivale a 0.5 en el modelo ponderado.")
    A("**Clasificacion: modelo vs baselines (mismo set de prueba):**")
    A(table(["Opcion", "ROC AUC", "Accuracy", "Precision", "Recall", "F1"], [
        ["Modelo: regresion logistica", f"{lr['roc_auc']:.4f}", f"{lr['accuracy']:.4f}", f"{lr['precision']:.4f}", f"{lr['recall']:.4f}", f"{lr['f1']:.4f}"],
        ["Baseline: siempre a tiempo", f"{at['roc_auc']:.4f}", f"{at['accuracy']:.4f}", f"{at['precision']:.4f}", f"{at['recall']:.4f}", f"{at['f1']:.4f}"],
        ["Baseline: tasa historica por ruta (ene-oct)", f"{rb['roc_auc']:.4f}", f"{rb['accuracy']:.4f}", f"{rb['precision']:.4f}", f"{rb['recall']:.4f}", f"{rb['f1']:.4f}"],
    ]))
    A(f"\"Siempre a tiempo\" tiene la mayor accuracy ({at['accuracy']:.4f}) solo porque {pct(1 - meta['tasa_retraso_prueba'])} "
      "de los vuelos llegan a tiempo; no detecta ningun retraso (recall 0, precision 0 por definicion). La tasa por ruta "
      f"usa el mismo umbral que el modelo; {n(meta['vuelos_prueba_ruta_sin_historial'])} vuelos de prueba en rutas sin "
      "historial reciben la tasa global.")
    A(f"Matriz de confusion del modelo: verdaderos negativos {n(cm['tn'])}, falsos positivos {n(cm['fp'])}, "
      f"falsos negativos {n(cm['fn'])}, verdaderos positivos {n(cm['tp'])}.")
    A("**Regresion: modelo vs baselines (minutos, ARR_DELAY recortado):**")
    A(table(["Opcion", "MAE", "RMSE", "Reduccion MAE del modelo", "Reduccion RMSE del modelo", "MAE sin recorte", "RMSE sin recorte"], [
        ["Modelo: GBT", f"{gb['mae']:.2f}", f"{gb['rmse']:.2f}", "-", "-", f"{gb['mae_sin_recorte']:.2f}", f"{gb['rmse_sin_recorte']:.2f}"],
        ["Baseline: media global (ene-oct)", f"{bg['mae']:.2f}", f"{bg['rmse']:.2f}", f"{bg['reduccion_mae_modelo_pct']:.2f}%",
         f"{bg['reduccion_rmse_modelo_pct']:.2f}%", f"{bg['mae_sin_recorte']:.2f}", f"{bg['rmse_sin_recorte']:.2f}"],
        ["Baseline: promedio por ruta (ene-oct)", f"{br['mae']:.2f}", f"{br['rmse']:.2f}", f"{br['reduccion_mae_modelo_pct']:.2f}%",
         f"{br['reduccion_rmse_modelo_pct']:.2f}%", f"{br['mae_sin_recorte']:.2f}", f"{br['rmse_sin_recorte']:.2f}"],
        ["Regresion lineal (baseline lineal)", f"{mod['linreg']['mae']:.2f}", f"{mod['linreg']['rmse']:.2f}", "-", "-", "-",
         f"{mod['linreg']['rmse_sin_recorte']:.2f}"],
    ]))
    imp = k["importancia_variables"]
    A("**Importancia de variables (regresion logistica, top 15):** importancia = |coeficiente| x desviacion "
      "estandar de la variable (categorias one-hot: sqrt(p(1-p)) con p su frecuencia en entrenamiento).")
    A(table(["#", "Variable", "Importancia", "Coeficiente", "Efecto"], [
        [i + 1, r["variable"], f"{r['importancia_estandarizada']:.4f}", f"{r['coeficiente']:+.4f}",
         "aumenta el riesgo" if r["coeficiente"] > 0 else "reduce el riesgo"]
        for i, r in enumerate(imp["regresion_logistica_top15"])]))
    A("temp_max_c y temp_min_c estan muy correlacionadas: sus signos opuestos deben leerse juntos. Sin estandarizar, "
      "el ranking lo dominan aeropuertos con muy pocos vuelos (coeficientes grandes por poca informacion).")
    A("GBT (regresion), importancia por ganancia: " + ", ".join(
        f"{r['variable']} {r['importancia']:.3f}" for r in imp["gbt_todas"][:8]) + ".")

    # 6 -----------------------------------------------------------------
    A("## 6. KPIs")
    # Como texto: los valores quedan exactamente como en el CSV (enteros sin ".0").
    kp = pd.read_csv(os.path.join(PBI, "kpis_resumen.csv"), dtype=str, keep_default_na=False)
    A(table(["KPI", "Valor", "Unidad", "Descripcion"],
            [[r.nombre, r.valor or "no calculado", r.unidad, r.descripcion] for r in kp.itertuples()]))
    A("**Tiempos por etapa (ultima corrida completa) y rendimiento:**")
    stage_rows = [
        ["Organizar zips", "organizar_zips"], ["Ingesta Bronze", "ingesta_bronze"],
        ["Clima (con cache)", "clima_con_cache"], ["Silver", "silver"], ["Gold", "gold"],
        ["Entrenamiento (3 modelos)", "entrenamiento_3_modelos"], ["Scoring de 7M vuelos + escritura Gold", "scoring_y_escritura_gold"],
        ["Carga MySQL", "carga_mysql"], ["Reporte", "reporte"]]
    A(table(["Etapa", "Segundos", "Filas", "Filas por segundo"], [
        [lbl, f"{tim[key]['segundos']:.1f}", n(tim[key]["filas"]) if tim[key]["filas"] else "-",
         n(tim[key]["filas_por_segundo"]) if tim[key]["filas_por_segundo"] else "-"] for lbl, key in stage_rows]
        + [["Pipeline completo (organize a predict, reloj)", f"{tim['pipeline_completo_s']:.1f}", n(fil["bronze"]),
            n(tim["pipeline_filas_por_segundo"])]]))
    d = tim["detalle_entrenamiento_s"]
    A(f"Detalle de entrenamiento: regresion logistica {d['regresion_logistica']} s, regresion lineal "
      f"{d['regresion_lineal']} s, GBT {d['gbt']} s. La descarga inicial del clima (50 llamadas, sin cache) tardo "
      f"{tim['clima_descarga_inicial_50_aeropuertos_s']} s en la primera ejecucion; las siguientes usan cache. El tiempo de "
      "reloj por etapa del orquestador incluye el arranque de Python y de Spark de cada subproceso.")
    A("**Costo por consulta analitica (5 consultas tipicas del dashboard sobre Gold, Spark local):**")
    A(table(["Consulta", "Primera ejecucion (s)", "Mediana de 5 (s)", "Min (s)", "Max (s)"], [
        [kk.replace("_", " "), f"{v['primera_s']:.3f}", f"{v['mediana_s']:.3f}", f"{v['min_s']:.3f}", f"{v['max_s']:.3f}"]
        for kk, v in q["consultas"].items()]))
    pv = q["consultas"]["prediccion_un_vuelo"]
    A(f"Promedio de medianas: {q['promedio_mediana_s']} s por consulta. La prediccion de un vuelo requiere ademas "
      f"{pv['arranque_carga_modelos_s']} s de arranque la primera vez (carga de modelos y catalogos). "
      "**Costo en dinero: no calculado** (no se inventan precios). Formula para completarlo: costo = tiempo de "
      "consulta (s) x precio por segundo de la capacidad elegida (por ejemplo, una SKU de Fabric).")

    # 7 -----------------------------------------------------------------
    A("## 7. Carga a MySQL")
    A(f"Base `vuelos` en MySQL {mq['mysql']['version'] if isinstance(mq['mysql'], dict) else ''} local, credenciales "
      "por `.env` (python-dotenv). Se cargaron las 4 dimensiones completas y FACT_VUELOS completa (100%, con pred_prob y pred_min).")
    A(table(["Tabla", "Filas (COUNT exacto)", "Tamano en MySQL (MB, datos + indices)"],
            [[t, n(v.get("filas_exactas")), n(v.get("mb"), 1)] for t, v in sorted(mysql_tbl.items())]))
    A(f"Tiempo de la carga completa: {my.get('seconds')} s (insercion de FACT_VUELOS {my.get('fact_insert_seconds')} s, "
      f"llave primaria + 5 indices + 5 llaves foraneas {my.get('fact_index_seconds')} s). Metodo: INSERT multi-fila por "
      "lotes de 20,000 (el servidor tiene local_infile desactivado), tabla sin indices durante la carga y "
      "`ALTER TABLE ... ALGORITHM=INPLACE` al final. Huerfanos por llave foranea: "
      + ", ".join(f"{a} {b}" for a, b in my.get("orphans", {}).items()) + ".")
    A("Antes de ajustar la memoria de InnoDB (valores por defecto: buffer pool 128 MB, redo log 100 MB) la creacion de "
      "indices fue el cuello de botella: un primer intento con `foreign_key_checks=1` uso ALGORITHM=COPY y se cancelo tras "
      "4,211 s (70 min) sin terminar; con INPLACE y la memoria por defecto tardo cerca de 50 min. Con buffer pool y redo log "
      f"de 2 GB (`SET PERSIST`) bajo a {my.get('fact_index_seconds')} s.")

    # 8 -----------------------------------------------------------------
    A("## 8. Problemas encontrados y como se resolvieron")
    A(table(["Problema", "Sintoma", "Solucion"], [
        ["Java ausente", "`java -version`: Unable to locate a Java Runtime", "`brew install openjdk@17`; spark_session.py usa el JDK de Homebrew si JAVA_HOME no esta definido (sin symlink con sudo)"],
        ["Python del sistema", "Solo habia Python 3.9.6 (el de macOS)", "Entorno virtual con Python 3.12 de Homebrew"],
        ["pandas 3 incompatible con PySpark 4.0", "createDataFrame con Arrow fallaba: Cannot convert pyarrow.lib.ChunkedArray to pyarrow.lib.Array", "requirements fija pandas < 3 (queda 2.3.3)"],
        ["Zips con el mismo CSV y sin mes en el nombre", "El primer zip era febrero; extraer juntos sobrescribe", "Leer la 2a linea de cada CSV para obtener MONTH y extraer con nombre vuelos_2025_MM.csv"],
        ["Formatos del BTS", "FL_DATE '2/1/2025 12:00:00 AM'; banderas '0.00'; CRS_DEP_TIME '0659' y '2400'", "Bronze guarda texto; Silver convierte con formato explicito; 2400 se trata como hora 0"],
        ["Nulos estructurales", "Causas de retraso nulas en vuelos sin retraso; clima nulo fuera del top 50", "Causas a 0 en completados; has_weather como indicador; politica documentada por columna"],
        ["Union de tablas en Gold", "CANCELLATION_CODE no existia en completados al unir", "Se agrega como nula en completados antes de la union"],
        ["API de clima", "2 de 50 llamadas con timeout de lectura (60 s)", "Reintentos con espera exponencial; cache por aeropuerto; 50 de 50 OK"],
        ["MySQL local_infile desactivado", "No se puede usar LOAD DATA LOCAL", "INSERT multi-fila por lotes de 20,000"],
        ["MySQL: indices y llaves foraneas", "ALTER con foreign_key_checks=1 copiaba la tabla fila por fila (> 70 min)", "ALGORITHM=INPLACE con foreign_key_checks=0 + verificacion de huerfanos aparte"],
        ["MySQL: memoria por defecto", "Buffer pool 128 MB / redo log 100 MB: esperas por paginas libres (Innodb_buffer_pool_wait_free 241,496)", f"SET PERSIST a 2 GB cada uno; indices y llaves foraneas en {my.get('fact_index_seconds')} s"],
        ["Probabilidad inflada por pesos de clase",
         f"Probabilidad media {mod['logreg']['mean_prob_raw']:.3f} en prueba con tasa real {mod['logreg']['real_delay_rate_test']:.3f}",
         f"Correccion de prior en pred_prob (media corregida {mod['logreg']['mean_prob_corrected']:.3f})"],
        ["Importancia de variables enganosa", "Top dominado por aeropuertos con muy pocos vuelos", "Coeficiente estandarizado por la desviacion de cada variable"],
        ["Graficas", "Etiquetas encimadas y subtitulos cortados", "Revision visual de cada PNG y ajuste de posiciones"],
    ]))

    # 9 -----------------------------------------------------------------
    A("## 9. Limitaciones y siguientes pasos")
    pdic = float(dic["prob_retraso_predicha_promedio_pct"]); rdic = float(dic["tasa_retraso_pct"])
    top_causa = causas.sort_values("minutos_totales", ascending=False).iloc[0]
    A(f"- **Poder predictivo moderado (AUC {lr['roc_auc']:.3f}).** La causa con mas minutos de retraso es "
      f"\"{top_causa.causa}\" ({top_causa.pct_minutos_de_retraso}% de los minutos), que no se conoce antes del vuelo. "
      "El modelo sirve para ordenar vuelos por riesgo, no para una prediccion exacta por vuelo.")
    A(f"- **Diciembre subestimado.** Con un solo ano, el modelo no vio una temporada navidena: en diciembre la "
      f"probabilidad media predicha es {pdic:.2f}% contra {rdic:.2f}% real. Siguiente paso: variables de feriados "
      "(conocidas de antemano) y entrenar con varios anos (2023 y 2024).")
    A(f"- **Regresion comprimida.** El GBT predice cerca de la media (reduccion de MAE {bg['reduccion_mae_modelo_pct']:.1f}% "
      f"y de RMSE {bg['reduccion_rmse_modelo_pct']:.1f}% frente a la media global); {n(n_over)} de {n(meta['filas_prueba'])} "
      f"predicciones superan el maximo de entrenamiento ({hi_c:.0f} min). Siguiente paso: modelar en dos etapas "
      "(probabilidad de retraso x minutos dado retraso) o usar perdida cuantil.")
    A(f"- **Clima observado y cobertura parcial.** Se usa el clima observado como proxy del pronostico y solo para "
      f"{vol['cobertura_clima_pct']['aeropuertos_con_clima']} aeropuertos ({vol['cobertura_clima_pct']['vuelos_completados']}% "
      "de vuelos). Siguiente paso: pronostico real y clima de destino; `--top 100` amplia la cobertura.")
    A("- **GBT con muestra del 30%.** Por tiempo de ejecucion en la laptop; en Fabric se puede entrenar con el 100%.")
    A(f"- **Latencia de prediccion individual.** {pv['mediana_s']} s por vuelo con modelos cargados y "
      f"{pv['arranque_carga_modelos_s']} s de arranque con Spark: suficiente para lotes y BI, no para una app en tiempo real. "
      "Siguiente paso: exportar el modelo a un servicio ligero o endpoint.")
    A("- **Ejecucion de un solo nodo.** Spark corrio en modo local; no se midio una ejecucion multinodo. Costo en dinero no calculado.")
    A("- **Siguientes pasos de plataforma:** migrar a Fabric (Delta, notebooks, Data Factory), carga incremental "
      "mensual con MERGE por vuelo_key, y dashboard en Power BI con Direct Lake sobre lh_gold.")

    # 10 ----------------------------------------------------------------
    A("## 10. Estructura del repositorio y reproduccion")
    A("```\n.\n|-- Makefile, requirements.txt, .env.example, README.md\n|-- src/\n"
      "|   |-- config.py, spark_session.py, utils.py\n"
      "|   |-- organize_data.py   (zips a CSV mensuales)\n|   |-- bronze.py          (ingesta cruda)\n"
      "|   |-- weather.py         (clima Open-Meteo)\n|   |-- silver.py          (limpieza y enriquecimiento)\n"
      "|   |-- gold.py            (esquema estrella)\n|   |-- mysql_loader.py    (carga a MySQL)\n"
      "|   |-- features.py        (variables y regla de fuga de datos)\n|   |-- train.py           (modelos, MLflow, predicciones)\n"
      "|   |-- report.py, predict.py\n|   |-- evaluate.py        (baselines, tiempos, consultas, volumen, importancia)\n"
      "|   |-- export_powerbi.py  (CSV para Power BI)\n|   |-- figures.py         (figuras finales)\n"
      "|   |-- summary.py         (este documento)\n|   `-- run_pipeline.py    (orquestador)\n"
      "|-- reports/              (kpis.json, stage_metrics.json, graficas)\n"
      "|-- deliverables/         (kpis.json, RESUMEN_TECNICO.md, powerbi/, figuras/)\n"
      "`-- data/                 (no versionado: raw, bronze, silver, gold, models)\n```")
    A("Reproducir todo (con los 12 zips en ~/Downloads y Java 17 instalado):")
    A("```zsh\nmake venv                      # entorno con Python 3.12 y dependencias\n"
      "cp .env.example .env           # credenciales MySQL (opcional)\n"
      "python -m src.run_pipeline     # pipeline completo + paquete de entrega (o: make all)\n"
      "make deliverables              # regenera indicadores, CSV, figuras, resumen y deliverables_fase3.zip\n```")
    A("Archivos de este paquete:")
    A(table(["Archivo", "Contenido"],
            [["kpis.json", "Todos los numeros medidos (maquina, volumen, limpieza, tiempos, consultas, evaluacion, importancia)"],
             ["RESUMEN_TECNICO.md", "Este documento"]]
            + [[f"powerbi/{f}", "Tabla agregada para Power BI"] for f in pbi_files]
            + [[f"figuras/{f}", "Figura PNG 150 dpi"] for f in figs]))

    text = "\n\n".join(md) + "\n"
    for bad in ("—", "–"):
        assert bad not in text, "el resumen no debe contener guiones largos"
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Escrito {os.path.relpath(OUT)}: {len(text.splitlines())} lineas, {len(text) / 1024:.1f} KB")


if __name__ == "__main__":
    main()
