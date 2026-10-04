"""
Carga de las tablas Gold (Parquet) a MySQL local.

- Credenciales desde .env (DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD).
- Si MySQL no conecta: se avisa y se retorna sin lanzar error (el pipeline sigue).
- Cada corrida recrea las tablas (DROP + CREATE): carga idempotente.
- Dimensiones: completas.
- FACT_VUELOS: el servidor tiene local_infile=OFF, asi que no se puede usar LOAD DATA LOCAL;
  se usa INSERT multi-fila por lotes. Para acelerar:
    * la tabla se crea SIN llave primaria ni indices; se agregan al final en un solo ALTER
      (construir el indice una vez es mucho mas rapido que mantenerlo fila por fila, sobre
      todo con vuelo_key, que es un hash y llega en orden aleatorio).
    * unique_checks y foreign_key_checks apagados durante la carga.
  Modo del hecho: 'auto' (default) mide la velocidad con las primeras filas; si la proyeccion
  supera MYSQL_FACT_MAX_MIN minutos cambia a una muestra determinista (por hash de
  vuelo_key) que quepa en ese tiempo. 'full' fuerza todo; 'sample' usa MYSQL_FACT_SAMPLE_PCT.

Uso: python -m src.mysql_loader [--fact-mode auto|full|sample]
"""
import argparse
import math
import os
import time

import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
import pymysql
from dotenv import load_dotenv

from src.config import GOLD_DIR, ROOT
from src.utils import timed, update_stage_metrics

BATCH_ROWS = 20_000
PROBE_ROWS = 300_000  # filas usadas para medir la velocidad en modo auto

# DDL explicito: tipos y tamanos pensados para cada columna (no se infieren).
DDL = {
    "DIM_TIEMPO": """
        fecha_key INT NOT NULL, fecha DATE NOT NULL, anio SMALLINT, trimestre TINYINT,
        mes TINYINT, nombre_mes VARCHAR(12), dia TINYINT, dia_semana TINYINT,
        nombre_dia VARCHAR(10), semana_anio TINYINT, es_fin_semana TINYINT, es_feriado TINYINT,
        PRIMARY KEY (fecha_key)""",
    "DIM_AEROLINEA": """
        aerolinea_key INT NOT NULL, codigo CHAR(2) NOT NULL, nombre VARCHAR(60),
        tipo VARCHAR(20), PRIMARY KEY (aerolinea_key), UNIQUE KEY uk_codigo (codigo)""",
    "DIM_AEROPUERTO": """
        aeropuerto_key INT NOT NULL, iata CHAR(3) NOT NULL, nombre VARCHAR(120),
        ciudad VARCHAR(80), estado CHAR(2), latitud DOUBLE, longitud DOUBLE,
        zona_horaria VARCHAR(40), tiene_clima TINYINT,
        PRIMARY KEY (aeropuerto_key), UNIQUE KEY uk_iata (iata)""",
    "DIM_CLIMA": """
        clima_key INT NOT NULL, aeropuerto_key INT, iata CHAR(3), fecha_key INT,
        precip_mm DOUBLE, snow_cm DOUBLE, wind_max_kmh DOUBLE, temp_max_c DOUBLE,
        temp_min_c DOUBLE, lluvia TINYINT, nieve TINYINT, viento_fuerte TINYINT,
        condicion VARCHAR(15), PRIMARY KEY (clima_key),
        KEY ix_clima_aeropuerto_fecha (aeropuerto_key, fecha_key)""",
    # Sin indices al crear: se agregan despues de cargar (ver FACT_POST_LOAD).
    "FACT_VUELOS": """
        vuelo_key BIGINT NOT NULL, fecha_key INT NOT NULL, aerolinea_key INT NOT NULL,
        origen_key INT NOT NULL, destino_key INT NOT NULL, clima_key INT NOT NULL,
        OP_CARRIER_FL_NUM INT, CRS_DEP_TIME SMALLINT, dep_hour TINYINT, CRS_ARR_TIME SMALLINT,
        DISTANCE DOUBLE, DISTANCE_GROUP TINYINT, CANCELLED TINYINT, CANCELLATION_CODE CHAR(1),
        DIVERTED TINYINT, DEP_DELAY DOUBLE, DEP_DEL15 TINYINT, ARR_DELAY DOUBLE, ARR_DEL15 TINYINT,
        CARRIER_DELAY DOUBLE, WEATHER_DELAY DOUBLE, NAS_DELAY DOUBLE, SECURITY_DELAY DOUBLE,
        LATE_AIRCRAFT_DELAY DOUBLE, pred_prob DOUBLE, pred_min DOUBLE, MONTH TINYINT""",
}
FACT_POST_LOAD = """
    ALTER TABLE FACT_VUELOS
      ADD PRIMARY KEY (vuelo_key),
      ADD KEY ix_fecha (fecha_key), ADD KEY ix_aerolinea (aerolinea_key),
      ADD KEY ix_origen (origen_key), ADD KEY ix_destino (destino_key),
      ADD KEY ix_clima (clima_key),
      ADD CONSTRAINT fk_fecha FOREIGN KEY (fecha_key) REFERENCES DIM_TIEMPO (fecha_key),
      ADD CONSTRAINT fk_aerolinea FOREIGN KEY (aerolinea_key) REFERENCES DIM_AEROLINEA (aerolinea_key),
      ADD CONSTRAINT fk_origen FOREIGN KEY (origen_key) REFERENCES DIM_AEROPUERTO (aeropuerto_key),
      ADD CONSTRAINT fk_destino FOREIGN KEY (destino_key) REFERENCES DIM_AEROPUERTO (aeropuerto_key),
      ADD CONSTRAINT fk_clima FOREIGN KEY (clima_key) REFERENCES DIM_CLIMA (clima_key),
      ALGORITHM=INPLACE, LOCK=NONE
"""
# Importante: este ALTER se ejecuta con foreign_key_checks=0. Con checks=1, agregar FKs
# obliga a MySQL a usar ALGORITHM=COPY (copia fila por fila a tabla temporal): con 7M filas
# y un buffer pool de 128 MB tardo mas de 70 min. Con INPLACE tarda ~40 s. ALGORITHM=INPLACE
# explicito hace que falle de inmediato si alguna vez no se puede, en vez de copiar en silencio.
# Como las FK no validan filas existentes en este modo, se verifica la integridad aparte.
ORPHAN_CHECKS = {
    "fecha": ("fecha_key", "DIM_TIEMPO", "fecha_key"),
    "aerolinea": ("aerolinea_key", "DIM_AEROLINEA", "aerolinea_key"),
    "origen": ("origen_key", "DIM_AEROPUERTO", "aeropuerto_key"),
    "destino": ("destino_key", "DIM_AEROPUERTO", "aeropuerto_key"),
    "clima": ("clima_key", "DIM_CLIMA", "clima_key"),
}
DIM_ORDER = ["DIM_TIEMPO", "DIM_AEROLINEA", "DIM_AEROPUERTO", "DIM_CLIMA"]


def connect():
    load_dotenv(os.path.join(ROOT, ".env"))
    return pymysql.connect(
        host=os.getenv("DB_HOST", "127.0.0.1"), port=int(os.getenv("DB_PORT", "3306")),
        user=os.getenv("DB_USER"), password=os.getenv("DB_PASSWORD", ""),
        database=os.getenv("DB_NAME", "vuelos"), charset="utf8mb4",
        autocommit=False, connect_timeout=10)


def table_columns(name: str) -> list[str]:
    """Columnas en el orden del DDL (se omiten las lineas de llaves e indices)."""
    cols = []
    for part in DDL[name].split(","):
        tok = part.strip().split()
        if tok and tok[0].upper() not in ("PRIMARY", "UNIQUE", "KEY") and "(" not in tok[0] and ")" not in tok[0]:
            cols.append(tok[0])
    return cols


def recreate_tables(conn):
    with conn.cursor() as cur:
        cur.execute("SET FOREIGN_KEY_CHECKS = 0")
        for name in ["FACT_VUELOS"] + DIM_ORDER[::-1]:
            cur.execute(f"DROP TABLE IF EXISTS {name}")
        for name in DIM_ORDER + ["FACT_VUELOS"]:
            cur.execute(f"CREATE TABLE {name} ({DDL[name]}) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4")
        cur.execute("SET FOREIGN_KEY_CHECKS = 1")
    conn.commit()


def iter_batches(name: str, sample_pct: int | None = None):
    """Lee el Parquet de Gold por lotes (sin pasar por Spark) y devuelve listas de tuplas.
    En el hecho, MONTH sale del nombre de la carpeta de particion (estilo hive)."""
    dataset = ds.dataset(os.path.join(GOLD_DIR, name.lower()), format="parquet",
                         partitioning="hive")
    for batch in dataset.to_batches(columns=table_columns(name), batch_size=BATCH_ROWS):
        if sample_pct is not None:
            # Muestra determinista: vuelo_key mod 100 < pct (la misma en cada corrida).
            keys = batch.column("vuelo_key").to_numpy(zero_copy_only=False)
            batch = batch.filter(pa.array((keys % 100) < sample_pct))
        if batch.num_rows:
            yield list(zip(*(c.to_pylist() for c in batch.columns)))


def insert_rows(conn, name: str, rows: list[tuple]):
    cols = table_columns(name)
    sql = f"INSERT INTO {name} ({','.join(cols)}) VALUES ({','.join(['%s'] * len(cols))})"
    with conn.cursor() as cur:
        cur.executemany(sql, rows)  # pymysql lo convierte en INSERT multi-fila
    conn.commit()


def load_fact(conn, mode: str, max_min: float, sample_pct: int) -> dict:
    total = ds.dataset(os.path.join(GOLD_DIR, "fact_vuelos"), format="parquet",
                       partitioning="hive").count_rows()
    with conn.cursor() as cur:
        cur.execute("SET unique_checks = 0")
        cur.execute("SET foreign_key_checks = 0")

    pct = sample_pct if mode == "sample" else None
    loaded, t0, decided = 0, time.perf_counter(), mode != "auto"
    gen = iter_batches("FACT_VUELOS", pct)
    for rows in gen:
        insert_rows(conn, "FACT_VUELOS", rows)
        loaded += len(rows)
        if not decided and loaded >= PROBE_ROWS:
            rate = loaded / (time.perf_counter() - t0)
            projected_min = total / rate / 60
            print(f"  velocidad medida: {rate:,.0f} filas/s -> proyeccion total {projected_min:.1f} min")
            decided = True
            if projected_min > max_min:
                # Muestra que quepa en el tiempo maximo, recargando desde cero para que sea
                # una muestra limpia por hash (no "las primeras N filas", que sesgaria a enero).
                pct = max(1, math.floor(100 * max_min / projected_min))
                print(f"  supera {max_min} min: se cambia a muestra del {pct}% por hash de vuelo_key")
                with conn.cursor() as cur:
                    cur.execute("TRUNCATE TABLE FACT_VUELOS")
                conn.commit()
                loaded = 0
                gen.close()
                for r2 in iter_batches("FACT_VUELOS", pct):
                    insert_rows(conn, "FACT_VUELOS", r2)
                    loaded += len(r2)
                break
        if loaded % 1_000_000 < BATCH_ROWS:
            print(f"  {loaded:,} / {total:,} filas ({time.perf_counter() - t0:.0f} s)")
    load_s = time.perf_counter() - t0

    print("  creando llave primaria, indices y llaves foraneas...")
    t1 = time.perf_counter()
    with conn.cursor() as cur:
        cur.execute(FACT_POST_LOAD)  # todavia con foreign_key_checks=0 (ver nota arriba)
        cur.execute("SET unique_checks = 1")
        cur.execute("SET foreign_key_checks = 1")
        orphans = {}
        for k, (col, dim, pk) in ORPHAN_CHECKS.items():
            cur.execute(f"SELECT COUNT(*) FROM FACT_VUELOS f LEFT JOIN {dim} d "
                        f"ON f.{col} = d.{pk} WHERE d.{pk} IS NULL")
            orphans[k] = cur.fetchone()[0]
    conn.commit()
    index_s = time.perf_counter() - t1
    print(f"  huerfanos por llave foranea: {orphans}")
    if any(orphans.values()):
        raise RuntimeError(f"integridad referencial rota en FACT_VUELOS: {orphans}")
    return {"rows": loaded, "total_gold_rows": total, "sample_pct": pct or 100,
            "insert_seconds": round(load_s, 1), "index_seconds": round(index_s, 1),
            "orphans": orphans}


def load_gold_to_mysql(fact_mode: str | None = None) -> dict | None:
    """Carga Gold a MySQL. Devuelve un resumen, o None si MySQL no esta disponible."""
    fact_mode = fact_mode or os.getenv("MYSQL_FACT_MODE", "auto")
    max_min = float(os.getenv("MYSQL_FACT_MAX_MIN", "20"))
    sample_pct = int(os.getenv("MYSQL_FACT_SAMPLE_PCT", "10"))
    print("\n=== Carga a MySQL ===")
    try:
        conn = connect()
    except Exception as e:
        print(f"AVISO: MySQL no disponible ({type(e).__name__}: {e}). "
              "Se omite la carga y el pipeline continua.")
        update_stage_metrics("mysql", status="skipped", error=str(e)[:200])
        return None

    try:
        with timed("carga mysql") as t:
            recreate_tables(conn)
            summary = {}
            for name in DIM_ORDER:
                n = 0
                for rows in iter_batches(name):
                    insert_rows(conn, name, rows)
                    n += len(rows)
                summary[name] = n
                print(f"  {name:<15} {n:>10,} filas")
            fact = load_fact(conn, fact_mode, max_min, sample_pct)
            summary["FACT_VUELOS"] = fact["rows"]

            # Verificacion contra lo que quedo en MySQL.
            with conn.cursor() as cur:
                db_counts = {}
                for name in DIM_ORDER + ["FACT_VUELOS"]:
                    cur.execute(f"SELECT COUNT(*) FROM {name}")
                    db_counts[name] = cur.fetchone()[0]
    except Exception as e:
        print(f"AVISO: fallo la carga a MySQL ({type(e).__name__}: {e}). El pipeline continua.")
        update_stage_metrics("mysql", status="failed", error=str(e)[:200])
        return None
    finally:
        conn.close()

    print(f"  FACT_VUELOS     {fact['rows']:>10,} filas de {fact['total_gold_rows']:,} "
          f"({fact['sample_pct']}%) | insert {fact['insert_seconds']} s, "
          f"indices+FK {fact['index_seconds']} s")
    print(f"  Conteos en MySQL: {db_counts}")
    update_stage_metrics("mysql", status="ok", seconds=t["seconds"], table_rows=db_counts,
                         fact_mode=fact_mode, **{f"fact_{k}": v for k, v in fact.items()})
    return {"counts": db_counts, "fact": fact}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fact-mode", choices=["auto", "full", "sample"], default=None)
    load_gold_to_mysql(ap.parse_args().fact_mode)
