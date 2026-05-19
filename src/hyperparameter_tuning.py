import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

import sys
sys.path.append('.')
from src.preprocesamiento_data import pipeline_clasificacion

def cargaData():
  df_pipeline = pd.read_csv("clean_retail_store_sales.csv")
  df_auto = df_pipeline.copy()
  return df_auto

def definirxy(df_auto):
  X, y = pipeline_clasificacion(df_auto)

  # División: 80% entrenamiento, 20% prueba
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
  return X_train, X_test, y_train, y_test

def optimizacion(X_train, y_train):
  modelo = RandomForestClassifier(random_state=42)

  param_grid = {
      'n_estimators': [10, 50, 100],      # Cantidad de árboles
      'max_depth': [None, 3, 5, 10],      # Profundidad de los árboles
      'criterion': ['gini', 'entropy']    # Medida de calidad de división
    }

  grid_search = GridSearchCV(
        estimator=modelo,
        param_grid=param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,          # SOLO USAR EN COLAB, esto te usa todos los nucleos para calcular
        verbose=1
    )

  # Ejecutar la optimización
  grid_search.fit(X_train, y_train)
  return grid_search

def resultados(grid_search, X_test, y_test):
  print(f"Mejores Hiperparámetros: {grid_search.best_params_}")
  print(f"Mejor Precisión (Accuracy) en entrenamiento: {grid_search.best_score_:.4f}")

  y_pred = grid_search.predict(X_test)
  print("\nReporte de Clasificación Final:")
  print(classification_report(y_test, y_pred))

def ejecucion():
  df_auto = cargaData()
  X_train, X_test, y_train, y_test = definirxy(df_auto)
  grid_search = optimizacion(X_train, y_train)
  resultados(grid_search, X_test, y_test)
  return "Ejecucion de Hiperparametro, terminada"

# ejecucion()