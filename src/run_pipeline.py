"""
Orquestador: corre todas las etapas en orden con un solo comando.

    python -m src.run_pipeline                      # todo
    python -m src.run_pipeline --from silver        # desde una etapa en adelante
    python -m src.run_pipeline --only gold train    # solo esas etapas
    python -m src.run_pipeline --skip-mysql         # sin cargar MySQL

Cada etapa corre como subproceso (python -m src.<modulo>): tiene su propia SparkSession y
sus propios argumentos, y si falla el pipeline se detiene ahi (salvo MySQL, que solo avisa).
En la corrida completa Gold no carga MySQL: la carga se hace una vez al final del
entrenamiento, ya con pred_prob y pred_min, para no subir 7M filas dos veces.
"""
import argparse
import subprocess
import sys
import time

from src.utils import update_stage_metrics

# (nombre, modulo, argumentos extra)
STAGES = [
    ("organize", "src.organize_data", []),
    ("bronze", "src.bronze", []),
    ("weather", "src.weather", []),
    ("silver", "src.silver", []),
    ("gold", "src.gold", ["--skip-mysql"]),
    ("train", "src.train", []),
    ("report", "src.report", []),
    ("predict", "src.predict", []),
    # Paquete de entrega (Fase 3): indicadores, CSV para Power BI y figuras.
    ("evaluate", "src.evaluate", []),
    ("export", "src.export_powerbi", []),
    ("figures", "src.figures", []),
    ("summary", "src.summary", []),
]
NAMES = [s[0] for s in STAGES]
CORE_END = "predict"
CORE = NAMES[:NAMES.index(CORE_END) + 1]  # pipeline central, sin el paquete de entrega


def main():
    ap = argparse.ArgumentParser(description="Pipeline medallon de retrasos de vuelos")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--from", dest="start", choices=NAMES, help="empieza en esta etapa")
    g.add_argument("--only", nargs="+", choices=NAMES, help="corre solo estas etapas")
    ap.add_argument("--skip-mysql", action="store_true", help="no carga MySQL")
    ap.add_argument("--src", default=None, help="carpeta con los zips (default ~/Downloads)")
    args = ap.parse_args()

    if args.only:
        selected = [s for s in STAGES if s[0] in args.only]
    elif args.start:
        selected = STAGES[NAMES.index(args.start):]
    else:
        selected = STAGES

    run_all = len(selected) == len(STAGES)
    durations = {}
    t_total = time.perf_counter()
    for name, module, extra in selected:
        extra = list(extra)
        # Si se pide solo gold (sin train), gold si debe cargar MySQL.
        if name == "gold" and not run_all and "train" not in [s[0] for s in selected]:
            extra.remove("--skip-mysql")
        if args.skip_mysql and name in ("gold", "train") and "--skip-mysql" not in extra:
            extra.append("--skip-mysql")
        if name == "organize" and args.src:
            extra += ["--src", args.src]

        print(f"\n{'=' * 70}\n>>> Etapa: {name}  (python -m {module} {' '.join(extra)})\n{'=' * 70}",
              flush=True)
        t0 = time.perf_counter()
        code = subprocess.call([sys.executable, "-m", module, *extra])
        durations[name] = round(time.perf_counter() - t0, 1)
        if code != 0:
            print(f"\nPIPELINE DETENIDO: la etapa '{name}' termino con codigo {code} "
                  f"tras {durations[name]} s. Corrige y reanuda con: "
                  f"python -m src.run_pipeline --from {name}")
            sys.exit(code)
        # El tiempo del pipeline central (organize a predict) se registra en cuanto termina,
        # para que la etapa evaluate lo lea en esta misma corrida. Solo si se corrieron todas.
        if name == CORE_END and all(n in durations for n in CORE):
            update_stage_metrics("pipeline", seconds=round(time.perf_counter() - t_total, 1),
                                 stages_wall_s={n: durations[n] for n in CORE})

    total = round(time.perf_counter() - t_total, 1)
    print(f"\n{'=' * 70}\nPipeline completo en {total} s ({total / 60:.1f} min)")
    for name, s in durations.items():
        print(f"  {name:<10} {s:>8.1f} s")
    if run_all:
        update_stage_metrics("pipeline_con_entregables", seconds=total, stages_wall_s=durations)


if __name__ == "__main__":
    main()
