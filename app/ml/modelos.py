"""
Clases que se guardan con joblib. Viven en app/ml/ para que la app las pueda importar al
cargar los .joblib (joblib/pickle guarda la ruta app.ml.modelos.<Clase>). Sin pyspark.
"""
import numpy as np


def prior_corrected(p, w_pos, w_neg):
    """Misma formula que src.train.prior_corrected: ponderar positivos por w_pos y negativos
    por w_neg multiplica los odds por w_pos/w_neg; se deshace multiplicando por w_neg/w_pos."""
    p = np.asarray(p, dtype="float64")
    odds = p / (1 - p) * (w_neg / w_pos)
    return odds / (1 + odds)


class ClasificadorCorregido:
    """Pipeline de clasificacion ponderado + correccion de probabilidad a la tasa base.

    predict_proba(X)            -> [[1-p, p]] con p ya corregida (escala de la tasa real).
    predict_proba_ponderada(X)  -> la probabilidad del modelo ponderado, sin corregir.
    predict(X)                  -> 1 si la probabilidad ponderada > 0.5 (umbral de las
                                   metricas, igual que Spark). Equivale a p corregida >
                                   umbral_corregido = w_neg / (w_pos + w_neg).
    """

    def __init__(self, pipeline, w_pos, w_neg, umbral=0.5):
        self.pipeline = pipeline
        self.w_pos = float(w_pos)
        self.w_neg = float(w_neg)
        self.umbral = float(umbral)
        self.classes_ = np.array([0, 1])

    @property
    def umbral_corregido(self):
        odds = self.umbral / (1 - self.umbral) * (self.w_neg / self.w_pos)
        return odds / (1 + odds)

    def predict_proba_ponderada(self, X):
        return self.pipeline.predict_proba(X)[:, 1]

    def predict_proba(self, X):
        p = prior_corrected(self.predict_proba_ponderada(X), self.w_pos, self.w_neg)
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba_ponderada(X) > self.umbral).astype(int)


class RegresorRecortado:
    """Pipeline de regresion de minutos con los limites de recorte de la etiqueta. La
    prediccion ya cae en ese rango por construccion; se recorta por seguridad."""

    def __init__(self, pipeline, p01, p99):
        self.pipeline = pipeline
        self.p01 = float(p01)
        self.p99 = float(p99)

    def predict(self, X):
        return np.clip(self.pipeline.predict(X), self.p01, self.p99)
