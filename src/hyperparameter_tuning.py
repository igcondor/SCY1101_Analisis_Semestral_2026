import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def cargaData():
  df_pipeline = pd.read_csv("../clean_retail_store_sales.csv")
  df_auto = df_pipeline.copy()
  return df_auto

def definirxy(df_auto, columnaBuscada: str):
  y = df_auto[columnaBuscada]
  X = df_auto.drop(columns=['Transaction ID',	'Customer ID',
                            'Transaction Date', 'Timestamp', columnaBuscada])
  
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
  return X_train, X_test, y_train, y_test

def preprocesador(df_auto, columnaBuscada: str):
  df_auto = df_auto.drop(columns=['Transaction ID',	'Customer ID', 'Transaction Date', 
                                  'Timestamp', columnaBuscada]).copy()
  cateList = df_auto.select_dtypes(include=["object"]).columns.to_list()
  numList = df_auto.select_dtypes(include=["int64", "float64", "bool"]).columns.to_list()

  preprocessor = ColumnTransformer(transformers=[
  ('cat', OneHotEncoder(), cateList),
  ('num', StandardScaler(), numList)
  ])
  return preprocessor

def pipeline(preprocessor):
  full_pipeline = Pipeline(steps=[
      ('preprocessor', preprocessor),
      ('classifier', RandomForestClassifier(random_state=42))
  ])
  return full_pipeline

def entrenamiento(full_pipeline, X_train, y_train):
  param_grid = {
      'classifier__n_estimators': [10, 50, 100],      # Cantidad de árboles
      'classifier__max_depth': [None, 3, 5, 10],      # Profundidad de los árboles
      'classifier__criterion': ['gini', 'entropy']    # Medida de calidad de división
  }

  grid_search = GridSearchCV(
        estimator=full_pipeline,
        param_grid=param_grid,
        cv=5,
        scoring='accuracy',
        # n_jobs=-1,    # SOLO USAR EN COLAB, esto te usa todos los nucleos para calcular
        verbose=1
      )
  grid_search.fit(X_train, y_train)
  return grid_search

def resultados(grid_search, X_test, y_test):
  print(f"Mejores Hiperparámetros: {grid_search.best_params_}")
  print(f"Mejor Precisión (Accuracy) en entrenamiento: {grid_search.best_score_:.4f}")
  y_pred = grid_search.predict(X_test)
  print("\nReporte de Clasificación Final:")
  print(classification_report(y_test, y_pred))

def ejecucion(columnaBuscada: str):
  df_auto = cargaData()
  X_train, X_test, y_train, y_test = definirxy(df_auto, columnaBuscada)
  preprocessor = preprocesador(df_auto, columnaBuscada)
  full_pipeline = pipeline(preprocessor)
  grid_search = entrenamiento(full_pipeline, X_train, y_train)
  resultados(grid_search, X_test, y_test)
  return resultados

# ejecucion("Category")