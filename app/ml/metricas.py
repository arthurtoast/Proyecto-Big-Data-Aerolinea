"""Metricas comunes: se usan igual para las predicciones de sklearn y para las de Spark (Gold)."""
import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, mean_absolute_error,
                             precision_score, r2_score, recall_score, roc_auc_score,
                             root_mean_squared_error)


def clasificacion(y, p_corr, y_pred):
    """y: etiqueta ARR_DEL15. p_corr: probabilidad corregida (sirve para AUC porque la
    correccion es monotona). y_pred: decision 0/1 con umbral 0.5 del modelo ponderado."""
    tn, fp, fn, tp = confusion_matrix(y, y_pred, labels=[0, 1]).ravel()
    m = {"roc_auc": roc_auc_score(y, p_corr),
         "precision": precision_score(y, y_pred, zero_division=0),
         "recall": recall_score(y, y_pred, zero_division=0),
         "f1": f1_score(y, y_pred, zero_division=0),
         "accuracy": accuracy_score(y, y_pred),
         "mean_prob_corrected": float(np.mean(p_corr)),
         "real_delay_rate_test": float(np.mean(y))}
    return m, {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def regresion(y_clip, y_raw, pred):
    return {"mae": mean_absolute_error(y_clip, pred),
            "rmse": root_mean_squared_error(y_clip, pred),
            "r2": r2_score(y_clip, pred),
            "rmse_sin_recorte": root_mean_squared_error(y_raw, pred)}
