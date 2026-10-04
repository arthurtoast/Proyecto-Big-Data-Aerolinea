"""
SparkSession compartida, ajustada para una MacBook Air (Apple M4, 10 nucleos, 16 GB RAM).

Decisiones:
- local[*]: usa todos los nucleos; en modo local el driver es tambien el executor.
- driver.memory 6g: el dataset completo (~7M filas, ~1.2 GB en CSV) cabe holgado, y
  quedan ~10 GB para macOS, pandas/sklearn (que corren en el mismo equipo) y el navegador.
- shuffle.partitions 32: el default (200) genera cientos de tareas diminutas para este
  volumen; ~3 particiones por nucleo da paralelismo sin sobrecosto de planificacion.
- AQE activado: Spark fusiona particiones pequenas despues de cada shuffle y ajusta joins
  en tiempo de ejecucion, asi el valor anterior es solo un punto de partida.
- maxResultSize 2g: permite traer a pandas muestras grandes con toPandas() para sklearn.
- Arrow activado: toPandas()/createDataFrame(pandas) mucho mas rapidos.
- Zona horaria UTC: las fechas del BTS son locales sin zona; fijar UTC evita corrimientos
  de dia al convertir a timestamp.
"""
import os
import sys

from pyspark.sql import SparkSession

_spark = None

# Respaldo: si JAVA_HOME no esta definido, usa el JDK 17 de Homebrew (keg-only, no
# queda en el PATH por defecto). Asi no hace falta el symlink con sudo.
_BREW_JDK17 = "/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home"
if not os.environ.get("JAVA_HOME") and os.path.isdir(_BREW_JDK17):
    os.environ["JAVA_HOME"] = _BREW_JDK17


def get_spark(app_name: str = "vuelos-bigdata") -> SparkSession:
    """Crea (una vez) y devuelve la SparkSession del proyecto."""
    global _spark
    if _spark is not None:
        return _spark

    # Spark lanza workers de Python: que usen el mismo interprete del venv.
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)

    try:
        _spark = (
            SparkSession.builder.appName(app_name)
            .master(os.getenv("SPARK_MASTER", "local[*]"))
            .config("spark.driver.memory", os.getenv("SPARK_DRIVER_MEMORY", "6g"))
            .config("spark.driver.maxResultSize", "2g")
            .config("spark.sql.shuffle.partitions", os.getenv("SPARK_SHUFFLE_PARTITIONS", "32"))
            .config("spark.sql.adaptive.enabled", "true")
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
            .config("spark.sql.execution.arrow.pyspark.enabled", "true")
            .config("spark.sql.session.timeZone", "UTC")
            .config("spark.sql.parquet.compression.codec", "snappy")
            .config("spark.ui.showConsoleProgress", "false")
            .getOrCreate()
        )
    except Exception as e:  # tipicamente Java ausente o version incompatible
        raise RuntimeError(
            "No se pudo iniciar Spark. Verifica Java 17 con `java -version` "
            f"y JAVA_HOME. Detalle: {e}"
        ) from e

    _spark.sparkContext.setLogLevel("WARN")
    return _spark


if __name__ == "__main__":
    # Prueba rapida: python -m src.spark_session
    spark = get_spark("smoke-test")
    print("Spark", spark.version, "| master:", spark.sparkContext.master,
          "| paralelismo:", spark.sparkContext.defaultParallelism)
    for k in ["spark.driver.memory", "spark.sql.shuffle.partitions", "spark.sql.adaptive.enabled"]:
        print(f"  {k} = {spark.conf.get(k)}")
    print("  conteo de prueba:", spark.range(1_000_000).count())
    spark.stop()
