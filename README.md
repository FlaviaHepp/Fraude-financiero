# Fraude-financiero
Detección de Fraude Financiero en Transacciones (2010-2024)

Herramientas utilizadas: Python, panas, numpy, 
Este proyecto se centra en la detección de fraude financiero utilizando algoritmos de aprendizaje automático. A continuación, resumo las etapas y tareas principales:

**Descripción del conjunto de datos**
Contiene más de 6 millones de transacciones financieras, con detalles como:
-Cantidad de la transacción.
-Saldos antes y después de la transacción.
-Tipo de transacción (transferencia, pago, etc.).
-Indicadores de fraude (isFraud y isFlaggedFraud).

**Análisis Exploratorio de Datos (EDA)**
Se analizaron distribuciones y correlaciones entre variables.
Se visualizaron los datos con gráficos de barras, histogramas y mapas de calor para identificar patrones.
Por ejemplo, se observó cómo ciertas características están relacionadas con transacciones fraudulentas.

**Preprocesamiento de los datos**
Se eliminaron columnas irrelevantes (nameOrig, nameDest).
Se transformaron variables categóricas en numéricas utilizando codificación (OneHotEncoder y LabelEncoder).
Se probaron diferentes técnicas de escalado para las variables numéricas, como StandardScaler y MinMaxScaler.

**Entrenamiento de modelos**
Se probaron varios algoritmos para predecir si una transacción es fraudulenta: Regresión logística, Árbol de decisión, Bosque aleatorio y K-Vecinos más cercanos (KNN).
*Modelos avanzados:* CatBoost, XGBoost, LightGBM, entre otros.

**Evaluación de modelos**
Se evaluó el rendimiento de los modelos mediante métricas como: Precisión (Accuracy).
*AUC-ROC:* para medir la capacidad de los modelos de distinguir entre transacciones fraudulentas y no fraudulentas.
Se concluyó que el modelo de bosque aleatorio alcanzó el mejor rendimiento con una AUC-ROC de 1.00 (perfecto).

**Visualización de resultados**
Se generaron matrices de confusión, gráficos de curvas ROC y análisis de métricas para comparar los modelos.
Se identificaron patrones específicos en los datos, en las transacciones fraudulentas tienden a tener montos más altos que las legítimas.

***Conclusión***
El proyecto combina análisis exploratorio, preprocesamiento, entrenamiento de modelos y evaluación para construir una herramienta efectiva de detección de fraudes. El mejor modelo (bosque aleatorio) mostró una capacidad perfecta para identificar fraudes, lo que lo hace ideal para aplicaciones en sistemas financieros.
