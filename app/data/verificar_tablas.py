"""
Paso 2D: verifica las tablas de app/data/ en un proceso nuevo SIN acceso a Silver ni Gold.

- Copia solo app/ (codigo, modelos y tablas) a una carpeta temporal que NO tiene data/, y se
  vuelve a ejecutar ahi (--aislado): Silver y Gold no existen para ese proceso. Ademas
  bloquea pyspark (app/ml/sin_spark.py).
- Muestra tamano, filas y 5 filas de muestra de cada tabla.
- Arma la entrada del modelo para 3 vuelos 2026 (origen con clima, origen sin clima y uno de
  diciembre) solo con las tablas, y comprueba que con las mismas entradas la prediccion es
  identica a la de app/ml/smoke_test.py.

Uso: .venv/bin/python app/data/verificar_tablas.py
"""
import os
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def relanzar_aislado():
    with tempfile.TemporaryDirectory(prefix="verif_tablas_") as tmp:
        shutil.copytree(os.path.join(ROOT, "app"), os.path.join(tmp, "app"),
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
        r = subprocess.run([sys.executable, os.path.join(tmp, "app", "data", "verificar_tablas.py"),
                            "--aislado"], cwd=tmp, env=env)
    sys.exit(r.returncode)


if __name__ == "__main__" and "--aislado" not in sys.argv:
    relanzar_aislado()

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from app.ml import sin_spark  # noqa: E402  (primero: bloquea pyspark)

from datetime import date  # noqa: E402

import pandas as pd  # noqa: E402

from app.ml.entrada import DATA_DIR, TABLAS, armar_entrada, cargar_tablas, predecir  # noqa: E402
from app.ml.features import flight_features, clima_con_banderas  # noqa: E402
from app.ml.modelos import cargar_modelos  # noqa: E402
from app.ml.smoke_test import BASE, CLIMA_DFW_SEM11_2025, CLIMA_DFW_SEM51_2025  # noqa: E402


def main():
    print(f"Proceso aislado en {ROOT}")
    for sub in ["data", os.path.join("data", "silver"), os.path.join("data", "gold")]:
        assert not os.path.exists(os.path.join(ROOT, sub)), sub
    print(f"Contenido: {sorted(os.listdir(ROOT))} (sin data/: Silver y Gold no existen aqui)\n")

    pd.set_option("display.width", 200)
    t = cargar_tablas()
    for n in TABLAS:
        p = os.path.join(DATA_DIR, f"{n}.parquet")
        print(f"== {n}.parquet | {os.path.getsize(p) / 2**20:.3f} MB | {len(t[n]):,} filas")
        print(t[n].sample(5, random_state=42).to_string(index=False), "\n")

    cls, reg = cargar_modelos(os.path.join(ROOT, "app", "models"))
    vuelos = [
        ("origen con clima", "AA", "DFW", "ORD", date(2026, 3, 15), 18),
        ("origen sin clima", "AA", "TUL", "DFW", date(2026, 6, 10), 6),
        ("diciembre", "AA", "DFW", "ORD", date(2026, 12, 20), 18),
    ]
    h = t["horas_ruta"]
    print("Entrada armada solo con app/data (sin pronostico: clima tipico = 21 dias reales):")
    for nombre, a, o, d, f, hora in vuelos:
        assert ((h.origen == o) & (h.destino == d) & (h.aerolinea == a) & (h.dep_hour == hora)).any()
        X, info = armar_entrada(t, a, o, d, f, hora)
        r = predecir(t, cls, reg, a, o, d, f, hora)
        print(f"\n[{nombre}] {a} {o}-{d} {f} {hora} h | distancia {info['distancia']:.0f} | "
              f"vuelos 2025 {info['vuelos_2025']:,} | clima: {info['fuente_clima']}")
        print(f"  lote: {len(X)} filas x {X.shape[1]} variables; primeras 3:")
        print(X.head(3).to_string(index=False))
        print(f"  -> prob {r['prob']:.4f} (rango {r['prob_min']:.4f} a {r['prob_max']:.4f}) | "
              f"minutos {r['minutos']:.2f} (rango {r['minutos_min']:.1f} a {r['minutos_max']:.1f})"
              + (f" | \"{r['texto_rango']}\"" if r["texto_rango"] else ""))
        assert len(X) == (21 if o in t["_con_clima"] else 1)

    print("\nMismas entradas que el smoke test (clima de la semana ISO de 2025):")
    for f, mes, clima in [(date(2026, 3, 15), 3, CLIMA_DFW_SEM11_2025),
                          (date(2026, 12, 20), 12, CLIMA_DFW_SEM51_2025)]:
        X_t, _ = armar_entrada(t, "AA", "DFW", "ORD", f, 18, clima=clima)   # pronostico: 1 fila
        X_s = flight_features(**BASE, MONTH=mes, DAY_OF_WEEK=7, clima=clima_con_banderas(**clima))
        pt, ps = cls.predict_proba(X_t)[0, 1], cls.predict_proba(X_s)[0, 1]
        mt, ms = reg.predict(X_t)[0], reg.predict(X_s)[0]
        iguales = X_t.equals(X_s) and pt == ps and mt == ms
        print(f"  {f}: tablas prob {pt:.4f} min {mt:.2f} | smoke test prob {ps:.4f} min {ms:.2f} | "
              f"{'IDENTICAS' if iguales else 'DIFERENTES'}")
        assert iguales

    assert not sin_spark.cargados()
    print("\nSin Silver/Gold en disco y sin pyspark: la entrada se arma solo con app/data. OK")


if __name__ == "__main__":
    main()
