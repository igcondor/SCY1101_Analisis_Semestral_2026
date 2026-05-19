# SCY1101 - Análisis Semestral 2026

Proyecto de análisis de datos y modelamiento de Machine Learning desarrollado para la asignatura **SCY1101**.

El proyecto aborda el análisis exploratorio, limpieza de datos, entrenamiento de modelos supervisados y evaluación de desempeño utilizando Python, Jupyter Notebook y librerías del ecosistema de ciencia de datos.

---

# Objetivos del Proyecto

* Realizar un análisis exploratorio de datos (EDA).
* Aplicar técnicas de preprocesamiento y limpieza de datos.
* Construir modelos supervisados de clasificación y regresión.
* Evaluar el desempeño de distintos modelos de Machine Learning.
* Optimizar hiperparámetros para mejorar resultados.
* Generar conclusiones finales basadas en métricas y visualizaciones.

---

# Tecnologías Utilizadas

## Lenguaje

* Python 3

## Librerías principales

* pandas
* numpy
* matplotlib
* seaborn
* scikit-learn

## Herramientas

* Jupyter Notebook
* Git
* GitHub
* Google Colab

---

# Estructura del Proyecto

```text
SCY1101_Analisis_Semestral_2026/
│
├── notebooks/
│   ├── 01_analisis_exploratorio.ipynb
│   ├── 02_modelamiento_supervisado.ipynb
│   ├── 03_model_evaluation.ipynb
│   ├── 04_hyperparameter_optimization_ipynb.ipynb
│   └── 05_final_analysis.ipynb
│
├── src/
│   ├── preprocesamiento_data.py
│   ├── model_training.py
│   ├── model_evaluation.py
│   └── hyperparameter_tuning.py
│
├── models/
│   └── trained_models/
│
├── results/
│   ├── metrics/
│   ├── plots/
│   └── reports/
│
└── README.md
```

---

# Descripción de los Notebooks

## 1. Análisis Exploratorio

Archivo: `01_analisis_exploratorio.ipynb`

En este notebook se realiza:

* Carga del dataset.
* Exploración inicial de variables.
* Identificación de valores nulos.
* Revisión de duplicados.
* Limpieza y transformación de datos.
* Visualización de patrones y relaciones.
* Aplicación de técnicas como PCA y K-Means.

---

## 2. Modelamiento Supervisado

Archivo: `02_modelamiento_supervisado.ipynb`

Se construyen modelos de:

### Clasificación

* Logistic Regression
* Decision Tree Classifier
* Random Forest Classifier

### Regresión

* Linear Regression
* Random Forest Regressor

Además:

* Se realiza separación train/test.
* Se preparan pipelines de clasificación y regresión.
* Se comparan métricas iniciales.

---

## 3. Evaluación de Modelos

Archivo: `03_model_evaluation.ipynb`

En este notebook se evalúa el rendimiento de los modelos utilizando:

### Métricas de Clasificación

* Accuracy
* Precision
* Recall
* F1-Score
* Matriz de confusión

### Métricas de Regresión

* MSE (Mean Squared Error)
* RMSE
* R² Score

También se aplican técnicas de validación cruzada utilizando `KFold` y `cross_val_score`.

---

## 4. Optimización de Hiperparámetros

Archivo: `04_hyperparameter_optimization_ipynb.ipynb`

Se implementa búsqueda de hiperparámetros utilizando:

* GridSearchCV
* Pipelines de transformación
* Random Forest

El objetivo es encontrar configuraciones que mejoren el desempeño de los modelos.

---

## 5. Análisis Final

Archivo: `05_final_analysis.ipynb`

Notebook consolidado donde:

* Se reutiliza el pipeline de limpieza.
* Se entrenan modelos finales.
* Se comparan resultados.
* Se presentan métricas finales.
* Se generan conclusiones del análisis.

---

# Preprocesamiento de Datos

El proyecto incluye un módulo dedicado al preprocesamiento dentro de `src/preprocesamiento_data.py`.

Entre las tareas realizadas se encuentran:

* Conversión de fechas.
* Eliminación de registros inválidos.
* Recuperación de valores faltantes.
* Limpieza de descuentos.
* Creación de pipelines de clasificación y regresión.
* Escalamiento y transformación de variables.

---

# Dataset

El proyecto trabaja con datasets relacionados a ventas retail y transacciones comerciales.

Algunos archivos utilizados:

* `retail_store_sales.csv`
* `clean_retail_store_sales.csv`
* `TransManual.csv`

Las variables incluyen información como:

* Categorías de productos
* Clientes
* Métodos de pago
* Ubicación
* Cantidad comprada
* Descuentos aplicados
* Total gastado

---

# Cómo Ejecutar el Proyecto

## Clonar repositorio

```bash
git clone <URL_DEL_REPOSITORIO>
cd SCY1101_Analisis_Semestral_2026
```

---

## Instalar dependencias

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

---

## Ejecutar notebooks

```bash
jupyter notebook
```

O abrir directamente en Google Colab.

---

# Resultados Esperados

El proyecto permite:

* Comprender patrones dentro del dataset.
* Detectar relaciones entre variables.
* Evaluar modelos predictivos.
* Comparar algoritmos supervisados.
* Analizar el impacto de la optimización de hiperparámetros.

---

# Posibles Mejoras Futuras

* Incorporar más modelos de Machine Learning.
* Implementar técnicas de Deep Learning.
* Automatizar pipelines de entrenamiento.
* Añadir visualizaciones interactivas.
* Desplegar modelos mediante API o aplicación web.
