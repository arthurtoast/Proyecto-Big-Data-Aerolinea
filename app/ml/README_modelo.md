# Modelo ligero (scikit-learn) para la app 2026

La app no puede depender de Spark (tarda unos 9 s en arrancar). Estos modelos replican los de
Spark MLlib (`src/train.py`) en scikit-learn, sin importar pyspark.

## Dos juegos de modelos

| juego | archivos | entrenamiento | para qué |
|---|---|---|---|
| **producción** | `app/models/clasificador.joblib`, `app/models/regresor.joblib` | ene-dic 2025 (6,879,484 vuelos) | lo usa la app |
| **variante de producción** | `app/models/clasificador_mes.joblib` | ene-dic 2025 | clasificador alternativo, con el mes categórico |
| **evaluación** | `app/models/eval/clasificador_ene_oct.joblib`, `app/models/eval/regresor_ene_oct.joblib` | ene-oct 2025 (5,752,264 vuelos) | métricas de prueba (nov-dic), comparación con Spark y verificaciones |

- **Métricas:** todas las de este documento y de `metadata.json` (`metricas_prueba`,
  `tabla_comparacion`, `verificaciones`) son del modelo de **evaluación**, probado en nov-dic.
- **Por qué se reentrena:** el de evaluación nunca vio noviembre ni diciembre, y la app va a
  recibir vuelos de esos meses en 2026.
- **Producción:** usa la misma configuración con los 12 meses, así que **no tiene métrica de
  prueba propia** (`metricas_prueba: null`): no queda ningún dato de 2025 fuera de su
  entrenamiento.

## Cómo se entrenó (modelo de evaluación)

- **Datos:** `data/silver/flights` (solo vuelos completados), leídos con pyarrow, más
  `data/silver/weather_daily` unido por ORIGIN y FL_DATE con un left join. Es lo mismo que hace
  `src.features.build_features()`.
- **Variables:** las mismas 18 de Spark, copiadas de `src/features.py` en
  `app/ml/features.py`. `make_features()` convierte las entradas crudas en las 18 variables y la
  usan tanto el entrenamiento como la app. Ninguna está en `LEAKAGE_COLS` (se valida).
- **Split:** de enero a octubre de 2025 para entrenamiento (5,752,264 vuelos) y noviembre y
  diciembre para prueba (1,127,220 vuelos). Son las mismas filas que usó Spark.
- **Clasificador ARR_DEL15:** `LogisticRegression` L2 con `class_weight='balanced'`, que da los
  mismos pesos que Spark (`n/(2*n_clase)`: w_pos 2.2694, w_neg 0.6413).
  - **Regularización:** `C = 1/(regParam*n) = 1.738e-4`, que equivale a `regParam=0.001` de
    Spark.
  - **Escalado:** un segundo escalado sin centrar y ponderado reproduce
    `standardization=True` de Spark.
  - **Artefacto:** `clasificador.joblib` es un `ClasificadorCorregido`. Su `predict_proba` ya
    devuelve la probabilidad corregida a la tasa base (`odds * w_neg/w_pos`), y su `predict`
    usa el umbral 0.5 del modelo ponderado.
- **Regresor de minutos:** `HistGradientBoostingRegressor` sobre ARR_DELAY recortado a
  [-37, 222], los mismos límites de Spark (`data/models/clip_and_prior.json`).
  - **Aeropuertos:** HGB admite hasta 255 categorías, así que ORIGIN y DEST conservan los 254
    aeropuertos más frecuentes y los 95 restantes (0.70% de los vuelos) se agrupan en una sola
    categoría OTRO.
  - **Número de árboles:** se fija con early stopping, entrenando de enero a septiembre y
    validando con octubre. Sale en 78 árboles. Después se reentrena con todo enero a octubre.
  - **Muestra:** se usó el entrenamiento completo (Spark usó el 30%).

## Modelo de producción (ene-dic 2025)

Usa la misma configuración que el de evaluación:

- **Pesos de clase:** se recalculan con ene-dic (w_pos 2.2414, w_neg 0.6436).
- **Corrección y regularización:** la misma corrección a la tasa base, y
  `C = 1/(0.001*6,879,484) = 1.454e-4` (el mismo `regParam` equivalente).
- **Recorte:** los mismos p01/p99 de ene-oct, [-37, 222].
- **Árboles:** 78 fijos, los que eligió el early stopping del modelo de evaluación, sin early
  stopping.

Ajuste **dentro del entrenamiento**. No es una métrica de prueba: compara el modelo con los
mismos datos con que se entrenó.

| mes | vuelos | tasa real | prob. media corregida | minutos reales (recortados) | minutos predichos |
|---|---|---|---|---|---|
| 1 | 522,269 | 0.188 | 0.203 | 2.2 | 2.7 |
| 2 | 496,476 | 0.208 | 0.213 | 4.1 | 4.2 |
| 3 | 592,301 | 0.196 | 0.219 | 3.9 | 4.9 |
| 4 | 577,730 | 0.197 | 0.221 | 4.3 | 4.5 |
| 5 | 597,574 | 0.236 | 0.240 | 8.4 | 7.6 |
| 6 | 599,472 | 0.283 | 0.246 | 13.2 | 12.6 |
| 7 | 612,811 | 0.289 | 0.241 | 14.4 | 14.0 |
| 8 | 593,733 | 0.226 | 0.236 | 7.3 | 7.8 |
| 9 | 558,328 | 0.166 | 0.220 | 1.1 | 2.2 |
| 10 | 601,570 | 0.203 | 0.213 | 4.3 | 4.1 |
| 11 | 555,296 | 0.206 | 0.207 | 4.9 | 4.6 |
| 12 | 571,924 | 0.268 | 0.207 | 10.7 | 9.7 |
| **total** | **6,879,484** | **0.2231** | **0.2227** | **6.70** | **6.70** |

- **Totales:** en todo el año la probabilidad media coincide con la tasa real, y los minutos
  también.
- **Por mes, en los minutos:** el regresor de árboles sigue bien la estacionalidad.
- **Por mes, en la probabilidad:** el clasificador la suaviza. Subestima junio, julio y
  diciembre (hasta 6 puntos) y sobrestima septiembre (5 puntos). La regresión logística solo ve
  el mes a través de `month_sin` y `month_cos` (dos términos lineales), que no alcanzan a
  formar picos como el de diciembre. Es la misma limitación del modelo de Spark, que usa las
  mismas variables.

### Variante del clasificador con el mes categórico (`clasificador_mes.joblib`)

- **Qué cambia:** además de `month_sin` y `month_cos`, agrega el mes como variable categórica
  (one-hot, 12 columnas más: 774 coeficientes contra 762). Todo lo demás es igual: los mismos
  pesos ene-dic, la corrección, el C y el escalado.
- **Entrada:** el mes se reconstruye de `month_sin` y `month_cos` dentro del pipeline, así que
  la variante recibe exactamente las mismas 18 variables de `flight_features()` y la app no
  cambia.
- **Modelo de evaluación:** no cambia. Sigue igual a Spark.

Ajuste **dentro del entrenamiento** (ene-dic). **No es una prueba**: los dos modelos se miden
sobre los mismos datos con que se entrenaron, y la variante tiene 12 parámetros más.

| mes | tasa real 2025 | prob. predicha actual | prob. predicha variante |
|---|---|---|---|
| 1 | 0.188 | 0.203 | 0.192 |
| 2 | 0.208 | 0.213 | 0.211 |
| 3 | 0.196 | 0.219 | 0.197 |
| 4 | 0.197 | 0.221 | 0.198 |
| 5 | 0.236 | 0.240 | 0.234 |
| 6 | 0.283 | 0.246 | 0.278 |
| 7 | 0.289 | 0.241 | 0.284 |
| 8 | 0.226 | 0.236 | 0.224 |
| 9 | 0.166 | 0.220 | 0.166 |
| 10 | 0.203 | 0.213 | 0.203 |
| 11 | 0.206 | 0.207 | 0.206 |
| 12 | 0.268 | 0.207 | 0.270 |
| **AUC / Brier (dentro del entrenamiento)** | | 0.6806 / 0.16047 | 0.6873 / 0.15932 |

- **Lo que muestra la tabla:** la variante reproduce la tasa de cada mes de 2025, con una
  diferencia máxima de 0.6 puntos. El actual se desvía hasta 6 puntos.
- **Lo que no muestra:** si eso generaliza a 2026. La variante memoriza la tasa de cada mes de
  un solo año; si 2026 tiene otra estacionalidad, puede empeorar. **Por eso no se elige
  todavía:** se decide con la validación 2026 (Paso 5).

**Cómo se elige:** `metadata.json` tiene la clave `"clasificador_activo"`, que hoy vale
`"clasificador.joblib"`; para usar la variante se cambia a `"clasificador_mes.joblib"`. La app
debe cargar los modelos con `app.ml.modelos.cargar_modelos("app/models")`, que lee esa clave.
El regresor es el mismo en los dos casos.

### Prueba de carga (`app/ml/smoke_test.py`)

AA DFW-ORD a las 18 h, con el clima promedio de DFW de la misma semana ISO de 2025 (semana 11
y semana 51):

| fecha | modelo | prob. retraso | minutos |
|---|---|---|---|
| 2026-03-15 | producción ene-dic | 0.496 | 30.1 |
| 2026-03-15 | variante con mes ene-dic | 0.465 | 30.1 |
| 2026-03-15 | evaluación ene-oct | 0.467 | 17.8 |
| 2026-12-20 | producción ene-dic | 0.488 | 42.7 |
| 2026-12-20 | variante con mes ene-dic | 0.586 | 42.7 |
| 2026-12-20 | evaluación ene-oct | 0.425 | 20.5 |

- **Variante:** en marzo baja la probabilidad y en diciembre la sube, igual que la tasa real de
  esos meses en 2025 (0.196 y 0.268).

- **Por qué cambian los minutos:** es un efecto de los datos. Los vuelos AA DFW-ORD de domingo
  entre las 17 y las 19 h tuvieron en 2025 una tasa de retraso de 0.553 y 38.1 min de media
  (94 vuelos). En noviembre y diciembre promediaron 41.5 y 26.5 min, y el modelo de evaluación
  no los vio.
- **Tiempos:** cada predicción tarda unos 8 ms. La carga (imports más los dos `.joblib`) tarda
  entre 0.8 y 1.2 s y ocurre una sola vez al arrancar la app.

## Comparación con Spark (modelo de evaluación, mismo set de prueba nov-dic)

Las métricas de Spark se calculan con el mismo código, a partir de `pred_prob` y `pred_min`
de `data/gold/fact_vuelos`.

| modelo | ROC AUC | F1 | MAE | RMSE | entrenamiento | disco |
|---|---|---|---|---|---|---|
| Spark LogisticRegression | 0.6589 | 0.4041 | - | - | 27.9 s | 0.02 MB |
| sklearn LogisticRegression | 0.6589 | 0.4041 | - | - | 19.4 s | 0.03 MB |
| Spark GBTRegressor (30%) | - | - | 24.806 | 42.684 | 72.2 s | 0.58 MB |
| sklearn HistGradientBoosting (100%) | - | - | 24.554 | 42.294 | 19.3 s | 0.27 MB |

- **Diferencias:** la de AUC es 0.0000 y la de MAE 0.25 min.
- **Correlación con Spark:** la probabilidad tiene correlación 1.0000 con la de Spark y los
  minutos 0.84.

Las métricas completas están en `app/models/metadata.json` y en MLflow (experimento
`retrasos_vuelos_sklearn`).

## Verificaciones

Las genera `app/ml/verificar.py` sobre el modelo de **evaluación** y quedan en
`metadata.json`, en la sección `verificaciones`.

### 1. Calibración del clasificador

Probabilidad corregida por deciles:

| decil | entren. prob. media | entren. tasa real | prueba prob. media | prueba tasa real |
|---|---|---|---|---|
| 1 | 0.069 | 0.080 | 0.054 | 0.101 |
| 2 | 0.103 | 0.105 | 0.081 | 0.132 |
| 3 | 0.132 | 0.129 | 0.103 | 0.158 |
| 4 | 0.160 | 0.151 | 0.126 | 0.182 |
| 5 | 0.189 | 0.177 | 0.150 | 0.207 |
| 6 | 0.219 | 0.207 | 0.174 | 0.230 |
| 7 | 0.252 | 0.241 | 0.201 | 0.263 |
| 8 | 0.290 | 0.288 | 0.232 | 0.295 |
| 9 | 0.340 | 0.353 | 0.276 | 0.343 |
| 10 | 0.446 | 0.471 | 0.391 | 0.460 |
| **total** | **0.2198** | **0.2203** | **0.1788** | **0.2371** |

- **Entrenamiento:** la probabilidad media corregida (0.2198) coincide con la tasa real
  (0.2203). No se agregó una calibración extra y `clasificador.joblib` no cambia.
- **Brier:** 0.1582 en entrenamiento y 0.1731 en prueba. Predecir siempre la tasa constante
  (0.2203) da 0.1812 en prueba.
- **Prueba:** el modelo subestima en todos los deciles porque la tasa de retraso cambió entre
  periodos (octubre 20.3%, nov-dic 23.7%), aunque el orden de riesgo se mantiene. Como
  referencia se probaron calibradores ajustados solo con octubre, sobre un modelo entrenado de
  enero a septiembre:

| calibración | prob. media entren. | Brier entren. | prob. media prueba | Brier prueba | AUC prueba |
|---|---|---|---|---|---|
| solo corrección (actual) | 0.2198 | 0.1582 | 0.1788 | 0.1731 | 0.658853 |
| + Platt (octubre) | 0.2399 | 0.1586 | 0.1976 | 0.1713 | 0.658853 |
| + isotónica (octubre) | 0.2398 | 0.1587 | 0.1975 | 0.1712 | 0.658770 |

Ninguno cierra la brecha de prueba (0.198 contra 0.237) y los dos descalibran el entrenamiento
(0.240 contra 0.220).

### 2. Baseline de minutos (prueba nov-dic, etiqueta recortada)

| modelo | MAE | RMSE | RMSE sin recortar | predicción p5 / p50 / p95 |
|---|---|---|---|---|
| siempre la mediana de entrenamiento (-6.0) | 24.859 | 45.860 | 63.847 | -6.0 / -6.0 / -6.0 |
| siempre la media de entrenamiento (6.49) | 27.378 | 43.749 | 61.958 | 6.5 / 6.5 / 6.5 |
| **HGB squared_error (regresor.joblib)** | 24.554 | **42.294** | **60.808** | -9.7 / -0.1 / 19.9 |
| HGB absolute_error (104 árboles) | **24.115** | 45.056 | 63.183 | -15.2 / -8.8 / 3.5 |

- **HGB squared_error contra la mediana:** solo le gana 0.31 min en MAE. La distribución es muy
  asimétrica: la mediana es -6 y la media 6.5. Este modelo predice la media condicional, así que
  el MAE no es la métrica que optimiza.
- **HGB absolute_error:** baja el MAE 0.44 min, pero su RMSE es peor que predecir siempre la
  media. Además predice la mediana condicional, que casi siempre cae antes de hora.
- **Decisión:** se mantiene squared_error, el único que gana a ambas baselines en MAE y en
  RMSE. Representa los minutos esperados.

### 3. Clima en la predicción

- **Ejemplo:** el vuelo AA DFW-ORD del 15/03/2026 a las 18 h da 0.475 y 16.8 min sin clima
  (`has_weather=0`). Con el clima promedio de DFW en la semana ISO 11 de 2025 (10 a 16 de
  marzo: 0.06 mm, 18.9 km/h, 25.2 / 11.4 °C, todas las banderas en 0) da 0.467 y 17.8 min.
- **Cobertura:** de los 5,752,264 vuelos de entrenamiento, 1,241,903 tienen `has_weather=0`.
  Ninguno sale de un aeropuerto que tenga clima otros días: los 50 aeropuertos con clima lo
  tienen los 365 días. Por eso `has_weather=0` en DFW es una combinación que el modelo nunca
  vio.
- **Para la app:** en esos 50 aeropuertos siempre se manda clima (el pronóstico o los 21 días
  reales de 2025, ver Paso 2), y `has_weather=0` solo se usa en los demás.

## Paso 2: tablas de apoyo de la app (`app/data/`)

`make tablas` las construye desde Silver y Gold con pandas y pyarrow (`app/data/build_tablas.py`,
unos 3 s) y las verifica sin Silver ni Gold (`app/data/verificar_tablas.py`). La app solo lee
estas tablas.

| tabla | filas | MB | contenido |
|---|---|---|---|
| `aerolineas.parquet` | 14 | 0.002 | código, nombre, tipo (DIM_AEROLINEA) |
| `aeropuertos.parquet` | 352 | 0.020 | IATA, nombre, ciudad, estado, latitud, longitud, zona horaria, `tiene_clima` (50 aeropuertos) |
| `rutas.parquet` | 13,206 | 0.065 | origen, destino, aerolínea, vuelos 2025, DISTANCE (mediana de la ruta) |
| `horas_ruta.parquet` | 99,641 | 0.191 | origen, destino, aerolínea, `dep_hour`, vuelos 2025 |
| `historial.parquet` | 13,206 | 0.308 | vuelos programados, completados, cancelados, desviados, tasa de retraso, retraso promedio, tasa de cancelación, `pocos_datos` |
| `clima_diario.parquet` | 18,250 | 0.101 | clima real de cada día de 2025 en los 50 aeropuertos con clima, con las banderas de Silver |

- **Formato:** Parquet (zstd). Todas las tablas juntas pesan 0.7 MB (en CSV serían unos 5 MB),
  cargan en 1 a 5 ms cada una y conservan los tipos.
- **Vuelos:** se cuentan los programados en 2025: completados, cancelados y desviados (7,001,618).
- **Tasas:** la de retraso y el retraso promedio (minutos sin recortar) salen de los vuelos
  completados. La de cancelación se calcula sobre los programados.
- **`pocos_datos`:** vale 1 cuando la combinación tiene menos de 30 vuelos completados (2,201 de
  13,206 combinaciones). Hay 21 rutas que solo tuvieron cancelados o desviados: no tienen tasa
  de retraso.
- **Datos incompletos:** BIH, EAR y XWA no tienen zona horaria, y EAR y XWA tampoco coordenadas.
  Así vienen de DIM_AEROPUERTO, y ninguno tiene clima.

### Entrada del modelo (`app/ml/entrada.py`)

`cargar_tablas()` carga las tablas una vez (20 ms) y `predecir()` hace la predicción de alto
nivel. Por dentro, `armar_entrada()` devuelve un **lote** de escenarios:

| caso | escenarios | clima |
|---|---|---|
| origen con clima y con pronóstico | 1 | el pronóstico, con banderas de `clima_con_banderas()` |
| origen con clima, sin pronóstico | 21 | los 21 días reales de 2025 alrededor del mismo día del año (±10 días, dando la vuelta entre diciembre y enero); el 29 de febrero usa el 28 |
| origen sin clima | 1 | `has_weather=0` |

- **Resultado:** `predecir()` predice todo el lote de una vez y devuelve el promedio de
  probabilidad y de minutos, el rango (mínimo y máximo) y el texto "según el clima, entre X% y
  Y%".
- **Ruta inexistente:** si la combinación no operó en 2025, da error.
- **Tiempo:** unos 7 ms por consulta.

**Por qué promediar predicciones y no armar un día promedio:** se probaron tres formas de
convertir el clima típico en entrada del modelo.

| vuelo | opción | banderas (lluvia, nieve, viento) | prob. | minutos |
|---|---|---|---|---|
| AA DFW-ORD 15/03/2026 18 h | moda (bandera = 1 si más de la mitad de los días) | 0, 0, 0 | 0.505 | 36.6 |
| | promedios con los umbrales de Silver | 0, 0, 0 | 0.505 | 36.6 |
| | **promedio de 21 predicciones** | días reales: 2 con lluvia, 2 con viento | **0.516** (45% a 60%) | **36.2** (28.7 a 76.6) |
| AA ORD-DFW 15/01/2026 18 h | moda | 0, 0, 0 | 0.420 | 19.1 |
| | promedios con los umbrales de Silver | 0, 1, 0 | 0.544 | 19.1 |
| | **promedio de 21 predicciones** | días reales: 5 con lluvia, 10 con nieve | **0.496** (37% a 72%) | **11.7** (0.5 a 47.2) |

- **El problema de un día promedio:** las banderas dependen de la regla. En ORD, con 10 de 21
  días con nieve, `moda` la apaga y `promedios` la prende, y la probabilidad salta de 0.420 a
  0.544. En todo el año, `promedios` marca nieve en el 22% de los días-aeropuerto, contra 4.9%
  de días reales con nieve.
- **Lo que hace el promedio de predicciones:** usa días que sí ocurrieron, con banderas
  coherentes con sus variables, y pondera cada situación por su frecuencia real. Además da un
  rango honesto.
- **Sensibilidad del regresor:** sube 4.6 min si solo la precipitación pasa de 0.06 a 0.34 mm,
  otra razón para no depender de un único día promedio.

### Revisión de minutos (regresor de producción)

- **El vuelo de ejemplo:** AA DFW-ORD del 15/03/2026 a las 18 h, con clima de la semana 11 de
  2025 y entradas idénticas, da 17.76 min con `eval/regresor_ene_oct.joblib` y 30.07 con
  `regresor.joblib`.
- **Por mes:** dentro de su entrenamiento, el regresor de producción reproduce los minutos
  reales (recortados) de cada mes de 2025 con una diferencia máxima de 1.07 min; el total es
  6.70 en los dos. Ni los 78 árboles fijos ni el recorte causan el cambio.
- **La causa:** el modelo de evaluación no vio noviembre ni diciembre y en ese periodo predice
  de menos (1.7 y 2.2 min de media, contra 4.9 y 10.7 reales).
- **Historia de la ruta:** AA DFW-ORD con salida entre las 17 y las 19 h tuvo 639 vuelos en
  2025: retraso medio 34.6 min, mediana 13.0, media recortada 32.8 (31.8 predicho). Solo a las
  18 h: media recortada 39.3. Los 30 min de producción son coherentes con esa historia.

## Cómo regenerar los modelos

```bash
make sklearn   # evaluación ene-oct + comparación con Gold + producción ene-dic + verificaciones + smoke test
make tablas    # tablas de apoyo de la app en app/data/ + verificación sin Silver/Gold
```

Requisitos:

- Deben existir `data/silver`, `data/gold/fact_vuelos` y `data/models/clip_and_prior.json`.
  Se generan con `make all` (`make clean-derived` los borra).
- scikit-learn 1.9.x, la misma versión menor con que se guardaron los `.joblib`.

## Uso en la app

```python
from datetime import date
from app.ml.entrada import cargar_tablas, predecir
from app.ml.modelos import cargar_modelos

t = cargar_tablas()                        # app/data/*.parquet, una vez al arrancar
cls, reg = cargar_modelos("app/models")    # producción; clasificador según "clasificador_activo"

# Sin pronóstico: clima típico (21 días reales de 2025) -> promedio y rango.
r = predecir(t, cls, reg, "AA", "DFW", "ORD", date(2026, 3, 15), 18)
r["prob"], r["minutos"], r["texto_rango"]  # 0.516, 36.2, "segun el clima, entre 45% y 60%"

# Con pronóstico del día (un solo escenario):
r = predecir(t, cls, reg, "AA", "DFW", "ORD", date(2026, 3, 15), 18,
             clima=dict(precip_mm=0.06, snow_cm=0.0, wind_max_kmh=18.9,
                        temp_max_c=25.19, temp_min_c=11.37))
```
