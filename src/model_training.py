import numpy as np
from sklearn.metrics import (accuracy_score, classification_report,
                             mean_squared_error, r2_score)


def entrenar_clasificacion(modelo, X_train, X_test, y_train, y_test, nombre):
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    print(f"=== {nombre} ===")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred))
    return modelo, y_pred


def entrenar_regresion(modelo, X_train, X_test, y_train, y_test, nombre):
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    print(f"=== {nombre} ===")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
    print(f"R2:   {r2_score(y_test, y_pred):.4f}")
    return modelo, y_pred
