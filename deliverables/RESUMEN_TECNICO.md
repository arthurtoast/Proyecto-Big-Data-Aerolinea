# Resumen tecnico: prediccion de retrasos de vuelos (Fase 3)

Documento generado automaticamente el 2026-10-04 14:58:09 a partir de las metricas medidas (`deliverables/kpis.json`). Todos los numeros provienen de la ultima corrida completa del pipeline; lo que no se pudo medir se indica como "no medido".

## 1. Arquitectura implementada

Arquitectura medallon (Bronze, Silver, Gold) con esquema estrella en Gold, implementada como prototipo local con Apache Spark y disenada para migrarse sin cambios de logica a Microsoft Fabric.

```
zips BTS -> data/raw (CSV) -> Bronze (Parquet crudo) -> Silver (limpio + clima) -> Gold (estrella)
Open-Meteo -> data/raw/weather ---------------------------^              |-> MySQL (vuelos)
OpenFlights -> data/raw/airports --------------------------------------> |-> Modelos MLlib + MLflow
                                                                         |-> CSV Power BI, figuras
```

| Capa | Ubicacion | Formato | Contenido |
|---|---|---|---|
| Raw | data/raw/ | CSV, JSON | 12 CSV mensuales del BTS, JSON de clima por aeropuerto, airports.dat |
| Bronze | data/bronze/flights | Parquet snappy, particion MONTH | Datos sin transformar (todo texto) + linaje |
| Silver | data/silver/ | Parquet, particion MONTH | flights (completados + clima), flights_disrupted, weather_daily |
| Gold | data/gold/ | Parquet | FACT_VUELOS + DIM_TIEMPO, DIM_AEROLINEA, DIM_AEROPUERTO, DIM_CLIMA, con predicciones |

**Tecnologias y versiones (medidas en la maquina de ejecucion):**

| Componente | Version / valor |
|---|---|
| Equipo | Mac16,13 (Apple M4, 10 nucleos: 4 de rendimiento + 6 de eficiencia, 16.0 GB RAM) |
| Sistema operativo | macOS 15.6 |
| Apache Spark (PySpark) | 4.0.4 |
| Java | openjdk version "17.0.20.1" 2026-08-18 |
| Python | 3.12.15 |
| Spark: master / memoria driver / particiones shuffle | local[*] / 6g / 32 (AQE activado) |
| MySQL | 26.7.0 (buffer pool 2048 MB) |
| Librerias | pyspark 4.0.4, pandas 2.3.3, pyarrow 25.0.1, scikit-learn 1.9.1, mlflow 3.16.1, matplotlib 3.11.2, SQLAlchemy 2.1.2, PyMySQL 1.2.3, requests 2.34.2 |

**Procesamiento distribuido.** Todo el procesamiento de datos usa el motor distribuido de Spark (DataFrames, joins con broadcast, agregaciones con shuffle, MLlib). En el prototipo corre en modo `local[*]`: un driver que reparte tareas en paralelo entre los 10 nucleos del equipo. El mismo codigo corre sin cambios en un cluster (Fabric Spark), donde las tareas se reparten entre varios nodos; en este proyecto no se midio una ejecucion multinodo.

**Mapeo a la arquitectura objetivo en Microsoft Fabric:**

| Prototipo local | Microsoft Fabric |
|---|---|
| data/raw/ | Lakehouse lh_raw, seccion Files de OneLake |
| data/bronze, data/silver, data/gold (Parquet) | Lakehouses lh_bronze, lh_silver, lh_gold con tablas Delta en OneLake |
| Modulos src/*.py ejecutados con PySpark local | Notebooks de Spark en Fabric (un notebook por etapa) |
| src/run_pipeline.py | Data Factory pipeline que encadena los notebooks |
| MLflow local (mlflow.db) | Experimentos y modelos de ML de Fabric (MLflow integrado) |
| MySQL vuelos | Warehouse de Fabric o SQL analytics endpoint del lakehouse Gold |
| CSV agregados para Power BI | Modelo semantico en modo Direct Lake sobre las tablas Delta de lh_gold |

## 2. Dataset

| Fuente | Que aporta | Acceso |
|---|---|---|
| BTS On-Time Performance (TranStats) | Un registro por vuelo domestico programado | 12 zips mensuales descargados manualmente |
| Open-Meteo Archive API | Clima diario historico por aeropuerto | API gratuita, sin llave; 50 llamadas (una por aeropuerto) |
| OpenFlights airports.dat | Nombre, coordenadas y zona horaria por codigo IATA | Descarga directa (GitHub) |

**Periodo:** 1 de enero a 31 de diciembre de 2025 (12 meses completos).

**Columnas del BTS (31, header real):** YEAR, MONTH, DAY_OF_MONTH, DAY_OF_WEEK, FL_DATE, OP_UNIQUE_CARRIER, OP_CARRIER_FL_NUM, ORIGIN, ORIGIN_CITY_NAME, ORIGIN_STATE_ABR, DEST, DEST_CITY_NAME, DEST_STATE_ABR, CRS_DEP_TIME, DEP_DELAY, DEP_DEL15, DEP_TIME_BLK, CRS_ARR_TIME, ARR_DELAY, ARR_DEL15, ARR_TIME_BLK, CANCELLED, CANCELLATION_CODE, DIVERTED, DISTANCE, DISTANCE_GROUP, CARRIER_DELAY, WEATHER_DELAY, NAS_DELAY, SECURITY_DELAY, LATE_AIRCRAFT_DELAY.

**Variables de clima (Open-Meteo, diarias):** precipitation_sum (mm), snowfall_sum (cm), wind_speed_10m_max (km/h), temperature_2m_max y temperature_2m_min (C), con `timezone=auto` para que el dia coincida con el dia local del vuelo.

**Volumen por capa (medido):**

| Capa / tabla | Filas | Tamano en disco (MB) |
|---|---|---|
| Raw: 12 CSV del BTS | 7,001,619 | 1,141.3 |
| Raw: clima (50 JSON + CSV consolidado) | 18,250 aeropuerto-dia | 1.4 |
| Bronze: flights | 7,001,619 | 123.2 |
| Silver: flights (completados) | 6,879,484 | 187.7 |
| Silver: flights_disrupted (cancelados y desviados) | 122,134 | 6.8 |
| Silver: weather_daily | 18,250 | 0.1 |
| Gold: FACT_VUELOS | 7,001,618 | 233.6 |
| Gold: 4 dimensiones | 18,982 | 0.3 |
| MySQL: FACT_VUELOS (datos + indices) | 7,001,618 | 1,865.3 |

Compresion: los 1,141.3 MB de CSV ocupan 123.2 MB en Parquet (Bronze), 9.3 veces menos.

**Cobertura de clima:** 78.44% de los vuelos completados (78.4% de todos los vuelos), al cubrir los 50 aeropuertos de origen con mas vuelos de 352 en total. 50 de 50 descargas exitosas, 0 sin coordenadas.

## 3. Limpieza y transformacion

| Etapa | Regla aplicada | Filas afectadas |
|---|---|---|
| Organizacion | Mes detectado leyendo la 2a linea de cada CSV (el nombre del zip no indica el mes); duplicados omitidos | 0 duplicados, 12 de 12 meses |
| Bronze | Schema explicito de 31 columnas como texto (sin inferSchema), modo FAILFAST | 0 filas mal formadas; 7,001,619 ingeridas |
| Bronze | Columnas de linaje _source_file y _ingested_at (sin modificar valores) | 7,001,619 |
| Silver | Tipos: enteros (calendario, banderas '0.00'/'1.00'), double (minutos, distancia), date (FL_DATE 'M/d/yyyy h:mm:ss a') | 7,001,619 |
| Silver | Descartar filas con clave minima nula o invalida (fecha, aerolinea, origen, destino, hora, distancia, banderas) | 0 |
| Silver | Coherencia de FL_DATE con YEAR/MONTH/DAY_OF_MONTH y DAY_OF_WEEK | 0 |
| Silver | Duplicados por clave de negocio (FL_DATE, aerolinea, numero de vuelo, origen, destino, CRS_DEP_TIME) | 0 |
| Silver | Separar cancelados a flights_disrupted | 102,876 |
| Silver | Separar desviados a flights_disrupted | 19,258 |
| Silver | Descartar completados sin ARR_DELAY / ARR_DEL15 (sin etiqueta) | 1 |
| Silver | Causas de retraso nulas en completados se rellenan con 0 (el BTS solo las llena con retraso 15+ min) | 5,344,847 |
| Silver | DEP_DELAY nulo en completados (se deja nulo, no se usa en el modelo) | 0 |
| Silver | CRS_DEP_TIME = 2400 se trata como hora 0 | 1 |
| Silver | CANCELLATION_CODE se elimina de completados (100% nula ahi; se conserva en flights_disrupted) | 6,879,484 |
| Silver | Union con clima de ORIGEN por (aeropuerto, fecha), left join con broadcast; sin clima: has_weather = 0 | 1,483,300 sin clima |
| Silver | Derivadas: dep_hour, is_weekend (sabado y domingo), route; banderas lluvia (>= 1.0 mm), nieve (> 0 cm), viento fuerte (>= 30.0 km/h) | 6,879,484 |
| Gold | Union de completados y no completados en FACT_VUELOS; llaves sustitutas; clima_key = -1 sin clima | 7,001,618 |

Balance: 7,001,619 (Bronze) = 6,879,484 completados + 122,134 cancelados/desviados + 1 descartado. Gold contiene 7,001,618 filas (completados + cancelados/desviados).

Banderas de clima sobre vuelos con clima: lluvia 29.2%, nieve 4.5%, viento fuerte 2.56%. Los umbrales salen de la distribucion real de 18,250 aeropuerto-dia (viento: sobre el percentil 95).

## 4. Esquema estrella

| Tabla | Filas | Llave primaria | Columnas principales |
|---|---|---|---|
| FACT_VUELOS | 7,001,618 | vuelo_key (xxhash64 de la clave de negocio) | fecha_key, aerolinea_key, origen_key, destino_key, clima_key; dep_hour, DISTANCE, CANCELLED, DIVERTED, DEP_DELAY, ARR_DELAY, ARR_DEL15, 5 causas de retraso, pred_prob, pred_min, MONTH |
| DIM_TIEMPO | 365 | fecha_key (yyyymmdd) | fecha, anio, trimestre, mes, nombre_mes, dia, dia_semana, nombre_dia, semana_anio, es_fin_semana, es_feriado (11 feriados federales) |
| DIM_AEROLINEA | 14 | aerolinea_key | codigo, nombre, tipo (principal, bajo_costo, regional) |
| DIM_AEROPUERTO | 352 | aeropuerto_key | iata, nombre, ciudad, estado, latitud, longitud, zona_horaria, tiene_clima. Una sola dimension usada como origen y destino |
| DIM_CLIMA | 18,251 | clima_key | aeropuerto_key, iata, fecha_key, precip_mm, snow_cm, wind_max_kmh, temp_max_c, temp_min_c, lluvia, nieve, viento_fuerte, condicion; miembro -1 sin_datos |

Integridad verificada: 7,001,618 vuelo_key distintos en 7,001,618 filas (sin colisiones); 0 registros huerfanos en las 5 llaves foraneas (verificado en MySQL). Dos aeropuertos (EAR, XWA) no estan en OpenFlights: quedan en la dimension con ciudad y estado del BTS y sin coordenadas.

## 5. Modelos

**Variables de entrada (18):** aerolinea (OP_UNIQUE_CARRIER), origen (ORIGIN), destino (DEST), dia de la semana, hora programada de salida (dep_hour), mes codificado como seno y coseno, fin de semana, distancia, y clima del aeropuerto de origen (precip_mm, snow_cm, wind_max_kmh, temp_max_c, temp_min_c, banderas de lluvia, nieve y viento fuerte, has_weather).

**Variables EXCLUIDAS por fuga de datos:** DEP_DELAY, DEP_DEL15, ARR_DELAY, ARR_DEL15, CARRIER_DELAY, WEATHER_DELAY, NAS_DELAY, SECURITY_DELAY, LATE_AIRCRAFT_DELAY. La app predice antes del vuelo; estas columnas solo existen despues de que el vuelo sale o llega (el retraso real y su causa). Usarlas daria metricas casi perfectas en pruebas y un modelo inutil en produccion. ARR_DEL15 y ARR_DELAY solo se usan como etiqueta. `src/train.py` valida la regla antes de entrenar. Tampoco se usan CANCELLED, DIVERTED ni CANCELLATION_CODE (resultado del vuelo). El clima es el observado del dia; en produccion seria el pronostico.

**Split temporal:** entrenamiento enero a octubre (5,752,264 vuelos completados), prueba noviembre y diciembre (1,127,220 vuelos). Un split aleatorio mezclaria vuelos del mismo dia y sobreestimaria el desempeno. Tasa de retraso: 22.03% en entrenamiento, 23.71% en prueba.

| Modelo | Objetivo | Hiperparametros |
|---|---|---|
| Regresion logistica (Spark MLlib) | ARR_DEL15 | maxIter 50, regParam 0.001, elasticNet 0; pesos de clase balanceados (retrasado 2.269, a tiempo 0.641); one-hot de aerolinea, origen, destino, dia y hora; numericas escaladas |
| Regresion lineal (Spark MLlib) | ARR_DELAY recortado | maxIter 50, regParam 0.001; mismas variables (baseline lineal) |
| GBT Regressor (Spark MLlib) | ARR_DELAY recortado | maxIter 60, maxDepth 6, stepSize 0.1, maxBins 400, subsamplingRate 0.8, seed 42; entrenado con una muestra aleatoria del 30% del entrenamiento (tamano exacto de la muestra no registrado) |

**Recorte de ARR_DELAY:** a los percentiles 1 y 99 del entrenamiento ([-37, 222] min). Los retrasos extremos dominan el RMSE sin ayudar al caso tipico; los percentiles salen solo de enero a octubre.

**Probabilidad mostrada al usuario (pred_prob):** los pesos de clase inflan la probabilidad; se corrige multiplicando los odds por peso_a_tiempo / peso_retrasado. El umbral de decision 0.2203 (tasa base de entrenamiento) equivale a 0.5 en el modelo ponderado.

**Clasificacion: modelo vs baselines (mismo set de prueba):**

| Opcion | ROC AUC | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Modelo: regresion logistica | 0.6589 | 0.6898 | 0.3711 | 0.4436 | 0.4041 |
| Baseline: siempre a tiempo | 0.5000 | 0.7629 | 0.0000 | 0.0000 | 0.0000 |
| Baseline: tasa historica por ruta (ene-oct) | 0.5687 | 0.5295 | 0.2716 | 0.5850 | 0.3710 |

"Siempre a tiempo" tiene la mayor accuracy (0.7629) solo porque 76.29% de los vuelos llegan a tiempo; no detecta ningun retraso (recall 0, precision 0 por definicion). La tasa por ruta usa el mismo umbral que el modelo; 1,387 vuelos de prueba en rutas sin historial reciben la tasa global.

Matriz de confusion del modelo: verdaderos negativos 658,944, falsos positivos 200,993, falsos negativos 148,704, verdaderos positivos 118,579.

**Regresion: modelo vs baselines (minutos, ARR_DELAY recortado):**

| Opcion | MAE | RMSE | Reduccion MAE del modelo | Reduccion RMSE del modelo | MAE sin recorte | RMSE sin recorte |
|---|---|---|---|---|---|---|
| Modelo: GBT | 24.81 | 42.68 | - | - | 26.86 | 61.02 |
| Baseline: media global (ene-oct) | 27.38 | 43.75 | 9.39% | 2.43% | 29.43 | 61.96 |
| Baseline: promedio por ruta (ene-oct) | 27.23 | 43.53 | 8.90% | 1.95% | 29.28 | 61.75 |
| Regresion lineal (baseline lineal) | 25.10 | 43.07 | - | - | - | 61.46 |

**Importancia de variables (regresion logistica, top 15):** importancia = |coeficiente| x desviacion estandar de la variable (categorias one-hot: sqrt(p(1-p)) con p su frecuencia en entrenamiento).

| # | Variable | Importancia | Coeficiente | Efecto |
|---|---|---|---|---|
| 1 | dep_hour = 6 | 0.2558 | -1.0093 | reduce el riesgo |
| 2 | temp_max_c | 0.2369 | -0.2369 | reduce el riesgo |
| 3 | temp_min_c | 0.2169 | +0.2169 | aumenta el riesgo |
| 4 | dep_hour = 7 | 0.1929 | -0.7428 | reduce el riesgo |
| 5 | dep_hour = 5 | 0.1901 | -1.1481 | reduce el riesgo |
| 6 | month_cos | 0.1618 | -0.1618 | reduce el riesgo |
| 7 | dep_hour = 8 | 0.1408 | -0.5538 | reduce el riesgo |
| 8 | dep_hour = 19 | 0.1241 | +0.5452 | aumenta el riesgo |
| 9 | rain_flag | 0.1236 | +0.1236 | aumenta el riesgo |
| 10 | dep_hour = 17 | 0.1210 | +0.4984 | aumenta el riesgo |
| 11 | dep_hour = 18 | 0.1197 | +0.5060 | aumenta el riesgo |
| 12 | wind_max_kmh | 0.1081 | +0.1081 | aumenta el riesgo |
| 13 | dep_hour = 20 | 0.1056 | +0.5133 | aumenta el riesgo |
| 14 | dep_hour = 9 | 0.1018 | -0.4407 | reduce el riesgo |
| 15 | dep_hour = 16 | 0.0989 | +0.4253 | aumenta el riesgo |

temp_max_c y temp_min_c estan muy correlacionadas: sus signos opuestos deben leerse juntos. Sin estandarizar, el ranking lo dominan aeropuertos con muy pocos vuelos (coeficientes grandes por poca informacion).

GBT (regresion), importancia por ganancia: DEST 0.275, ORIGIN 0.269, dep_hour 0.110, precip_mm 0.078, month_cos 0.040, OP_UNIQUE_CARRIER 0.039, DAY_OF_WEEK 0.036, temp_max_c 0.036.

## 6. KPIs

| KPI | Valor | Unidad | Descripcion |
|---|---|---|---|
| Vuelos procesados (Bronze) | 7001619 | vuelos | Registros BTS 2025 ingeridos sin transformar |
| Vuelos completados (Silver) | 6879484 | vuelos | Vuelos que llegaron a destino, usados para modelar |
| Vuelos cancelados o desviados | 122134 | vuelos | Separados en su propia tabla para analisis |
| Filas en FACT_VUELOS (Gold) | 7001618 | filas | Hechos del esquema estrella, con predicciones |
| Filas cargadas en MySQL (FACT_VUELOS) | 7001618 | filas | Conteo exacto verificado con COUNT(*) |
| Tamano CSV crudos | 1141.3 | MB | 12 CSV mensuales del BTS |
| Tamano Bronze (Parquet) | 123.2 | MB | Parquet snappy particionado por mes |
| Tamano Silver (Parquet) | 194.6 | MB | Completados + cancelados/desviados + clima |
| Tamano Gold (Parquet) | 233.8 | MB | Hecho + 4 dimensiones |
| Cobertura de clima (vuelos completados) | 78.44 | % | Vuelos cuyo origen esta en el top 50 con clima |
| Aeropuertos con clima | 50 | aeropuertos | Top 50 aeropuertos de origen por volumen |
| Tasa de retraso global | 22.31 | % | Vuelos completados con llegada 15+ min tarde (ARR_DEL15) |
| ROC AUC modelo | 0.6589 | 0-1 | Regresion logistica, prueba nov-dic |
| ROC AUC baseline tasa por ruta | 0.5687 | 0-1 | Tasa historica de la ruta (ene-oct) |
| Precision modelo | 0.3711 | 0-1 | De los vuelos marcados con retraso, cuantos lo tuvieron |
| Recall modelo | 0.4436 | 0-1 | De los vuelos con retraso, cuantos detecto |
| F1 modelo | 0.4041 | 0-1 | Media armonica de precision y recall |
| Accuracy modelo | 0.6898 | 0-1 | Aciertos totales con umbral = tasa base de entrenamiento |
| MAE modelo GBT | 24.81 | minutos | Error absoluto medio, ARR_DELAY recortado [-37, 222] |
| RMSE modelo GBT | 42.68 | minutos | Raiz del error cuadratico medio, ARR_DELAY recortado |
| MAE baseline media global | 27.38 | minutos | Predecir siempre la media de ene-oct |
| MAE baseline promedio por ruta | 27.23 | minutos | Promedio historico de la ruta (ene-oct) |
| Reduccion MAE vs media global | 9.39 | % | Mejora del modelo frente al metodo tradicional |
| Reduccion MAE vs promedio por ruta | 8.9 | % | Mejora del modelo frente al promedio por ruta |
| Reduccion RMSE vs media global | 2.43 | % | Mejora del modelo frente al metodo tradicional |
| Reduccion RMSE vs promedio por ruta | 1.95 | % | Mejora del modelo frente al promedio por ruta |
| Tiempo pipeline completo | 439.1 | segundos | De zips a reporte, un comando, MacBook Air M4 |
| Throughput pipeline | 15945 | filas/s | 7,001,619 filas / tiempo total |
| Tiempo ingesta Bronze | 19.34 | segundos | 12 CSV a Parquet |
| Throughput ingesta Bronze | 362028 | filas/s | Filas por segundo en la ingesta |
| Tiempo descarga clima (inicial) | 237.3 | segundos | 50 llamadas a Open-Meteo; luego se usa cache |
| Tiempo Silver | 53.31 | segundos | Limpieza, separacion y union con clima |
| Tiempo Gold | 12.83 | segundos | Esquema estrella en Parquet |
| Tiempo entrenamiento (3 modelos) | 114.31 | segundos | Logistica + lineal + GBT |
| Tiempo scoring 7M vuelos | 27.94 | segundos | pred_prob y pred_min escritos en Gold |
| Tiempo carga MySQL | 160.68 | segundos | 7M filas + indices + llaves foraneas |
| Tiempo promedio por consulta de dashboard | 0.348 | segundos | Promedio de medianas de 5 consultas sobre Gold con Spark |
| Costo por consulta | no calculado | USD | No calculado: completar con la tarifa de capacidad elegida (tiempo x precio) |

**Tiempos por etapa (ultima corrida completa) y rendimiento:**

| Etapa | Segundos | Filas | Filas por segundo |
|---|---|---|---|
| Organizar zips | 8.2 | 7,001,619 | 859,094 |
| Ingesta Bronze | 19.3 | 7,001,619 | 362,028 |
| Clima (con cache) | 4.4 | - | - |
| Silver | 53.3 | 7,001,619 | 131,338 |
| Gold | 12.8 | 7,001,618 | 545,722 |
| Entrenamiento (3 modelos) | 114.3 | 5,752,264 | 50,322 |
| Scoring de 7M vuelos + escritura Gold | 27.9 | 7,001,618 | 250,595 |
| Carga MySQL | 160.7 | 7,001,618 | 43,575 |
| Reporte | 5.8 | - | - |
| Pipeline completo (organize a predict, reloj) | 439.1 | 7,001,619 | 15,945 |

Detalle de entrenamiento: regresion logistica 27.93 s, regresion lineal 14.23 s, GBT 72.15 s. La descarga inicial del clima (50 llamadas, sin cache) tardo 237.3 s en la primera ejecucion; las siguientes usan cache. El tiempo de reloj por etapa del orquestador incluye el arranque de Python y de Spark de cada subproceso.

**Costo por consulta analitica (5 consultas tipicas del dashboard sobre Gold, Spark local):**

| Consulta | Primera ejecucion (s) | Mediana de 5 (s) | Min (s) | Max (s) |
|---|---|---|---|---|
| retraso por mes | 0.635 | 0.217 | 0.199 | 0.635 |
| retraso por aerolinea | 0.469 | 0.304 | 0.260 | 0.469 |
| top rutas | 0.693 | 0.398 | 0.338 | 0.693 |
| efecto clima | 0.568 | 0.222 | 0.194 | 0.568 |
| prediccion un vuelo | 0.642 | 0.597 | 0.573 | 0.642 |

Promedio de medianas: 0.348 s por consulta. La prediccion de un vuelo requiere ademas 8.641 s de arranque la primera vez (carga de modelos y catalogos). **Costo en dinero: no calculado** (no se inventan precios). Formula para completarlo: costo = tiempo de consulta (s) x precio por segundo de la capacidad elegida (por ejemplo, una SKU de Fabric).

## 7. Carga a MySQL

Base `vuelos` en MySQL 26.7.0 local, credenciales por `.env` (python-dotenv). Se cargaron las 4 dimensiones completas y FACT_VUELOS completa (100%, con pred_prob y pred_min).

| Tabla | Filas (COUNT exacto) | Tamano en MySQL (MB, datos + indices) |
|---|---|---|
| DIM_AEROLINEA | 14 | 0.0 |
| DIM_AEROPUERTO | 352 | 0.1 |
| DIM_CLIMA | 18,251 | 2.9 |
| DIM_TIEMPO | 365 | 0.0 |
| FACT_VUELOS | 7,001,618 | 1,865.3 |

Tiempo de la carga completa: 160.68 s (insercion de FACT_VUELOS 115.9 s, llave primaria + 5 indices + 5 llaves foraneas 43.6 s). Metodo: INSERT multi-fila por lotes de 20,000 (el servidor tiene local_infile desactivado), tabla sin indices durante la carga y `ALTER TABLE ... ALGORITHM=INPLACE` al final. Huerfanos por llave foranea: fecha 0, aerolinea 0, origen 0, destino 0, clima 0.

Antes de ajustar la memoria de InnoDB (valores por defecto: buffer pool 128 MB, redo log 100 MB) la creacion de indices fue el cuello de botella: un primer intento con `foreign_key_checks=1` uso ALGORITHM=COPY y se cancelo tras 4,211 s (70 min) sin terminar; con INPLACE y la memoria por defecto tardo cerca de 50 min. Con buffer pool y redo log de 2 GB (`SET PERSIST`) bajo a 43.6 s.

## 8. Problemas encontrados y como se resolvieron

| Problema | Sintoma | Solucion |
|---|---|---|
| Java ausente | `java -version`: Unable to locate a Java Runtime | `brew install openjdk@17`; spark_session.py usa el JDK de Homebrew si JAVA_HOME no esta definido (sin symlink con sudo) |
| Python del sistema | Solo habia Python 3.9.6 (el de macOS) | Entorno virtual con Python 3.12 de Homebrew |
| pandas 3 incompatible con PySpark 4.0 | createDataFrame con Arrow fallaba: Cannot convert pyarrow.lib.ChunkedArray to pyarrow.lib.Array | requirements fija pandas < 3 (queda 2.3.3) |
| Zips con el mismo CSV y sin mes en el nombre | El primer zip era febrero; extraer juntos sobrescribe | Leer la 2a linea de cada CSV para obtener MONTH y extraer con nombre vuelos_2025_MM.csv |
| Formatos del BTS | FL_DATE '2/1/2025 12:00:00 AM'; banderas '0.00'; CRS_DEP_TIME '0659' y '2400' | Bronze guarda texto; Silver convierte con formato explicito; 2400 se trata como hora 0 |
| Nulos estructurales | Causas de retraso nulas en vuelos sin retraso; clima nulo fuera del top 50 | Causas a 0 en completados; has_weather como indicador; politica documentada por columna |
| Union de tablas en Gold | CANCELLATION_CODE no existia en completados al unir | Se agrega como nula en completados antes de la union |
| API de clima | 2 de 50 llamadas con timeout de lectura (60 s) | Reintentos con espera exponencial; cache por aeropuerto; 50 de 50 OK |
| MySQL local_infile desactivado | No se puede usar LOAD DATA LOCAL | INSERT multi-fila por lotes de 20,000 |
| MySQL: indices y llaves foraneas | ALTER con foreign_key_checks=1 copiaba la tabla fila por fila (> 70 min) | ALGORITHM=INPLACE con foreign_key_checks=0 + verificacion de huerfanos aparte |
| MySQL: memoria por defecto | Buffer pool 128 MB / redo log 100 MB: esperas por paginas libres (Innodb_buffer_pool_wait_free 241,496) | SET PERSIST a 2 GB cada uno; indices y llaves foraneas en 43.6 s |
| Probabilidad inflada por pesos de clase | Probabilidad media 0.409 en prueba con tasa real 0.237 | Correccion de prior en pred_prob (media corregida 0.179) |
| Importancia de variables enganosa | Top dominado por aeropuertos con muy pocos vuelos | Coeficiente estandarizado por la desviacion de cada variable |
| Graficas | Etiquetas encimadas y subtitulos cortados | Revision visual de cada PNG y ajuste de posiciones |

## 9. Limitaciones y siguientes pasos

- **Poder predictivo moderado (AUC 0.659).** La causa con mas minutos de retraso es "Avion tardio (vuelo anterior)" (39.19% de los minutos), que no se conoce antes del vuelo. El modelo sirve para ordenar vuelos por riesgo, no para una prediccion exacta por vuelo.

- **Diciembre subestimado.** Con un solo ano, el modelo no vio una temporada navidena: en diciembre la probabilidad media predicha es 17.80% contra 26.77% real. Siguiente paso: variables de feriados (conocidas de antemano) y entrenar con varios anos (2023 y 2024).

- **Regresion comprimida.** El GBT predice cerca de la media (reduccion de MAE 9.4% y de RMSE 2.4% frente a la media global); 135 de 1,127,220 predicciones superan el maximo de entrenamiento (222 min). Siguiente paso: modelar en dos etapas (probabilidad de retraso x minutos dado retraso) o usar perdida cuantil.

- **Clima observado y cobertura parcial.** Se usa el clima observado como proxy del pronostico y solo para 50 aeropuertos (78.44% de vuelos). Siguiente paso: pronostico real y clima de destino; `--top 100` amplia la cobertura.

- **GBT con muestra del 30%.** Por tiempo de ejecucion en la laptop; en Fabric se puede entrenar con el 100%.

- **Latencia de prediccion individual.** 0.597 s por vuelo con modelos cargados y 8.641 s de arranque con Spark: suficiente para lotes y BI, no para una app en tiempo real. Siguiente paso: exportar el modelo a un servicio ligero o endpoint.

- **Ejecucion de un solo nodo.** Spark corrio en modo local; no se midio una ejecucion multinodo. Costo en dinero no calculado.

- **Siguientes pasos de plataforma:** migrar a Fabric (Delta, notebooks, Data Factory), carga incremental mensual con MERGE por vuelo_key, y dashboard en Power BI con Direct Lake sobre lh_gold.

## 10. Estructura del repositorio y reproduccion

```
.
|-- Makefile, requirements.txt, .env.example, README.md
|-- src/
|   |-- config.py, spark_session.py, utils.py
|   |-- organize_data.py   (zips a CSV mensuales)
|   |-- bronze.py          (ingesta cruda)
|   |-- weather.py         (clima Open-Meteo)
|   |-- silver.py          (limpieza y enriquecimiento)
|   |-- gold.py            (esquema estrella)
|   |-- mysql_loader.py    (carga a MySQL)
|   |-- features.py        (variables y regla de fuga de datos)
|   |-- train.py           (modelos, MLflow, predicciones)
|   |-- report.py, predict.py
|   |-- evaluate.py        (baselines, tiempos, consultas, volumen, importancia)
|   |-- export_powerbi.py  (CSV para Power BI)
|   |-- figures.py         (figuras finales)
|   |-- summary.py         (este documento)
|   `-- run_pipeline.py    (orquestador)
|-- reports/              (kpis.json, stage_metrics.json, graficas)
|-- deliverables/         (kpis.json, RESUMEN_TECNICO.md, powerbi/, figuras/)
`-- data/                 (no versionado: raw, bronze, silver, gold, models)
```

Reproducir todo (con los 12 zips en ~/Downloads y Java 17 instalado):

```zsh
make venv                      # entorno con Python 3.12 y dependencias
cp .env.example .env           # credenciales MySQL (opcional)
python -m src.run_pipeline     # pipeline completo + paquete de entrega (o: make all)
make deliverables              # regenera indicadores, CSV, figuras, resumen y deliverables_fase3.zip
```

Archivos de este paquete:

| Archivo | Contenido |
|---|---|
| kpis.json | Todos los numeros medidos (maquina, volumen, limpieza, tiempos, consultas, evaluacion, importancia) |
| RESUMEN_TECNICO.md | Este documento |
| powerbi/causas_retraso.csv | Tabla agregada para Power BI |
| powerbi/efecto_clima.csv | Tabla agregada para Power BI |
| powerbi/importancia_variables.csv | Tabla agregada para Power BI |
| powerbi/kpis_resumen.csv | Tabla agregada para Power BI |
| powerbi/modelo_vs_baseline.csv | Tabla agregada para Power BI |
| powerbi/predicciones_muestra.csv | Tabla agregada para Power BI |
| powerbi/retraso_por_aerolinea.csv | Tabla agregada para Power BI |
| powerbi/retraso_por_aeropuerto.csv | Tabla agregada para Power BI |
| powerbi/retraso_por_dia_semana.csv | Tabla agregada para Power BI |
| powerbi/retraso_por_hora.csv | Tabla agregada para Power BI |
| powerbi/retraso_por_mes.csv | Tabla agregada para Power BI |
| powerbi/top_rutas.csv | Tabla agregada para Power BI |
| figuras/causas_retraso.png | Figura PNG 150 dpi |
| figuras/curva_roc_modelo_vs_baselines.png | Figura PNG 150 dpi |
| figuras/efecto_clima.png | Figura PNG 150 dpi |
| figuras/importancia_variables.png | Figura PNG 150 dpi |
| figuras/matriz_confusion.png | Figura PNG 150 dpi |
| figuras/regresion_real_vs_predicho.png | Figura PNG 150 dpi |
| figuras/retraso_por_aerolinea.png | Figura PNG 150 dpi |
| figuras/retraso_por_hora.png | Figura PNG 150 dpi |
| figuras/retraso_por_mes.png | Figura PNG 150 dpi |
| figuras/top10_rutas_retrasadas.png | Figura PNG 150 dpi |
