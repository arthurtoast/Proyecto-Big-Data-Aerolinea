"""
Bloquea pyspark en el proceso actual, como si no estuviera instalado.

Nuestro codigo no importa pyspark, pero algunas librerias lo cargan solas si lo encuentran
instalado (por ejemplo, `import mlflow` importa pyspark cuando esta en el entorno virtual).
Con el bloqueador, esas librerias siguen funcionando sin el (lo tratan como ausente) y queda
garantizado que ni el entrenamiento ni la app dependen de Spark.
Importar este modulo ANTES que cualquier otra cosa.
"""
import importlib.abc
import sys

BLOQUEADOS = ("pyspark", "py4j", "src")  # src/ importa pyspark


class _Bloqueador(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in BLOQUEADOS:
            raise ModuleNotFoundError(f"bloqueado (sin Spark): {name}")
        return None


def bloquear():
    if not any(isinstance(f, _Bloqueador) for f in sys.meta_path):
        sys.meta_path.insert(0, _Bloqueador())


def cargados():
    return [m for m in sys.modules if m.split(".")[0] in BLOQUEADOS]


bloquear()
