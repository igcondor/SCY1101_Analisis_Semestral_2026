import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score,
                             mean_squared_error, r2_score)


def evaluar_clasificacion_cv(modelos, X, y, kf):
    #Evalúa múltiples modelos de clasificación con validación cruzada.
    #parametros
    #  modelos: dict {nombre: modelo}
    #    X: features
    #    y: target
    #    kf: KFold object
    # Retorna:  DataFrame con accuracy medio y desviación estándar por modelo

    resultados = []
    for nombre, modelo in modelos.items():
        scores = cross_val_score(modelo, X, y, cv=kf, scoring='accuracy')
        resultados.append({
            'Modelo': nombre,
            'Accuracy Media': round(scores.mean(), 4),
            'Desviacion Std': round(scores.std(), 4)
        })
    return pd.DataFrame(resultados).sort_values('Accuracy Media', ascending=False)


def evaluar_regresion_cv(modelos, X, y, kf):

    #Evalúa múltiples modelos de regresión con validación cruzada.

    #Parámetros:
    #    modelos: dict {nombre: modelo}
    #    X: features
    #    y: target
    #    kf: KFold object
    resultados = []
    for nombre, modelo in modelos.items():
        r2_scores = cross_val_score(modelo, X, y, cv=kf, scoring='r2')
        neg_mse = cross_val_score(modelo, X, y, cv=kf, scoring='neg_mean_squared_error')
        rmse = np.sqrt(-neg_mse).mean()
        resultados.append({
            'Modelo': nombre,
            'RMSE Medio': round(rmse, 4),
            'R2 Medio': round(r2_scores.mean(), 4),
            'Desviacion Std R2': round(r2_scores.std(), 4)
        })
    return pd.DataFrame(resultados).sort_values('R2 Medio', ascending=False)


def tabla_comparacion_clf(modelos_clf, y_test):
    # Genera tabla comparativa de métricas de clasificación.
    resultados = []
    for nombre, (modelo, y_pred) in modelos_clf.items():
        resultados.append({
            'Modelo': nombre,
            'Accuracy': round(accuracy_score(y_test, y_pred), 4),
            'Precision': round(precision_score(y_test, y_pred), 4),
            'Recall': round(recall_score(y_test, y_pred), 4),
            'F1': round(f1_score(y_test, y_pred), 4)
        })
    return pd.DataFrame(resultados).sort_values('Accuracy', ascending=False)


def tabla_comparacion_reg(modelos_reg, y_test):
    # Genera tabla comparativa de métricas de regresión.
    resultados = []
    for nombre, (modelo, y_pred) in modelos_reg.items():
        resultados.append({
            'Modelo': nombre,
            'RMSE': round(np.sqrt(mean_squared_error(y_test, y_pred)), 4),
            'R2': round(r2_score(y_test, y_pred), 4)
        })
    return pd.DataFrame(resultados).sort_values('R2', ascending=False)
