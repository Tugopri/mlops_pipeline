V1.1.0 - Ingeniería de características y modelamiento supervisado
En esta versión se realiza la preparación de los datos, la ingeniería de características y el desarrollo de modelos de clasificación supervisada para predecir la variable Pago_atiempo.

Ingeniería de características
Se realizó:

Conversión y extracción de información de fecha_prestamo.

Conversión de puntaje a variable numérica.

Limpieza de valores inconsistentes en tendencia_ingresos.

Tratamiento de valores faltantes.

Separación de variables numéricas, categóricas y categóricas ordinales.

Codificación de variables categóricas.

Codificación ordinal de tendencia_ingresos.

Estandarización de variables numéricas.

División de los datos en conjuntos de entrenamiento y evaluación.

Modelamiento
Se implementaron modelos de clasificación supervisada utilizando pipelines:

Logistic Regression

Decision Tree

Random Forest

Para evitar repetir procesos se implementaron las funciones build_model() y summarize_classification().

Evaluación
Los modelos se comparan utilizando:

Accuracy

Precision

Recall

F1-score

También se generan gráficos comparativos y matrices de confusión para analizar el comportamiento de los modelos.

Resultado
La tabla de evaluación permite comparar objetivamente los modelos mediante las métricas obtenidas sobre el conjunto de evaluación.

El modelo seleccionado para continuar el proyecto deberá definirse a partir de las métricas obtenidas y de los objetivos del problema de clasificación.

