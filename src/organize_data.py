"""
Paso 0: organiza los zips mensuales del BTS en data/raw/flights/vuelos_2025_MM.csv.

Todos los zips traen un CSV con el mismo nombre (T_ONTIME_REPORTING.csv), por eso
no se descomprimen juntos: se lee la segunda linea de cada uno para saber el mes y
se copia el CSV en streaming al nombre final. Nunca se borra nada del origen.

Uso:
    python -m src.organize_data [--src ~/Downloads] [--force]
"""
import argparse
import csv
import glob
import io
import os
import shutil
import sys
import zipfile

from src.config import RAW_FLIGHTS_DIR, EXPECTED_HEADER, YEAR
from src.utils import timed, update_stage_metrics

ZIP_PATTERN = "T_ONTIME_REPORTING_2026*.zip"


def find_csv_member(zf: zipfile.ZipFile) -> str:
    """Devuelve el nombre del unico CSV dentro del zip."""
    csvs = [n for n in zf.namelist() if n.lower().endswith(".csv")]
    if len(csvs) != 1:
        raise ValueError(f"se esperaba 1 CSV y hay {len(csvs)}: {csvs}")
    return csvs[0]


def read_month(zf: zipfile.ZipFile, member: str) -> int:
    """Lee solo header + primera fila de datos y devuelve el valor de MONTH."""
    with zf.open(member) as raw:
        text = io.TextIOWrapper(raw, encoding="utf-8", newline="")
        reader = csv.reader(text)
        header = next(reader)
        if header != EXPECTED_HEADER:
            raise ValueError(f"header inesperado: {header}")
        row = next(reader)
    rec = dict(zip(header, row))
    if int(rec["YEAR"]) != YEAR:
        raise ValueError(f"YEAR={rec['YEAR']}, se esperaba {YEAR}")
    return int(rec["MONTH"])


def count_rows(path: str) -> int:
    """Cuenta filas de datos (sin header). Se cuentan registros CSV, no lineas,
    por si algun campo entrecomillado trae saltos de linea."""
    with open(path, encoding="utf-8", newline="") as f:
        return sum(1 for _ in csv.reader(f)) - 1


def organize(src_dir: str, force: bool = False) -> dict:
    zips = sorted(glob.glob(os.path.join(os.path.expanduser(src_dir), ZIP_PATTERN)))
    if not zips:
        sys.exit(f"No se encontraron {ZIP_PATTERN} en {src_dir}")
    os.makedirs(RAW_FLIGHTS_DIR, exist_ok=True)

    seen: dict[int, str] = {}
    duplicates, errors = [], []
    for zpath in zips:
        name = os.path.basename(zpath)
        try:
            with zipfile.ZipFile(zpath) as zf:
                member = find_csv_member(zf)
                month = read_month(zf, member)
                if month in seen:
                    duplicates.append((name, month, seen[month]))
                    print(f"[DUPLICADO] {name}: mes {month:02d} ya vino de {seen[month]}, se omite")
                    continue
                seen[month] = name
                dest = os.path.join(RAW_FLIGHTS_DIR, f"vuelos_{YEAR}_{month:02d}.csv")
                if os.path.exists(dest) and not force:
                    print(f"[OK] {name} -> {os.path.basename(dest)} (ya existia)")
                    continue
                # Copia en streaming a un temporal y renombra: si falla a medias
                # no queda un CSV truncado con el nombre definitivo.
                tmp = dest + ".part"
                with zf.open(member) as s, open(tmp, "wb") as d:
                    shutil.copyfileobj(s, d, length=4 * 1024 * 1024)
                os.replace(tmp, dest)
                print(f"[OK] {name} -> {os.path.basename(dest)}")
        except (zipfile.BadZipFile, ValueError, StopIteration, KeyError) as e:
            errors.append((name, str(e)))
            print(f"[ERROR] {name}: {e}")

    return {"seen": seen, "duplicates": duplicates, "errors": errors}


def verify() -> tuple[bool, int]:
    """Verifica 01..12, tamanos y filas por mes. Devuelve (todo_ok, total_filas)."""
    print("\nVerificacion de data/raw/flights:")
    print(f"{'archivo':<22}{'tamano (MB)':>13}{'filas':>12}")
    total, missing = 0, []
    for m in range(1, 13):
        path = os.path.join(RAW_FLIGHTS_DIR, f"vuelos_{YEAR}_{m:02d}.csv")
        if not os.path.exists(path):
            missing.append(f"{m:02d}")
            print(f"{os.path.basename(path):<22}{'FALTA':>13}")
            continue
        size_mb = os.path.getsize(path) / 1024**2
        rows = count_rows(path)
        total += rows
        print(f"{os.path.basename(path):<22}{size_mb:>13.1f}{rows:>12,}")
    print(f"{'TOTAL':<22}{'':>13}{total:>12,}")
    if missing:
        print(f"\nFALTAN meses: {', '.join(missing)}. Deteniendo.")
        return False, total
    print("\n12 de 12 meses presentes.")
    return True, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="~/Downloads")
    ap.add_argument("--force", action="store_true", help="re-extrae aunque ya exista")
    args = ap.parse_args()
    with timed("organizar datos") as t:
        res = organize(args.src, args.force)
        if res["duplicates"]:
            print(f"\nAVISO: {len(res['duplicates'])} zip(s) duplicados omitidos.")
        ok, total = verify()
    if not ok:
        sys.exit(1)
    update_stage_metrics("organize", seconds=t["seconds"], rows=total,
                         duplicates=len(res["duplicates"]), errors=len(res["errors"]))


if __name__ == "__main__":
    main()
