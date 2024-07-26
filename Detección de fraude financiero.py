"""
Descripción del conjunto de datos sobre transacciones fraudulentas
Este conjunto de datos proporciona información completa sobre transacciones, con especial atención a la identificación de actividades fraudulentas. 
Con más de 6 millones de entradas, ofrece una colección rica y diversa de datos transaccionales para análisis y modelado.

Columnas:

paso : representa una unidad de tiempo en el proceso de transacción, aunque la unidad de tiempo específica no se especifica en el conjunto de datos. 
Podría denotar horas, días u otra unidad, según el contexto.
tipo : Describe el tipo de transacción, como transferencia, pago, etc. Esta variable categórica permite clasificar diferentes comportamientos de 
transacción.
Monto : Indica el valor monetario de la transacción, proporcionando información sobre la magnitud financiera de cada transacción.
nameOrig : sirve como identificador de la cuenta o entidad de origen que inicia la transacción. Esto ayuda a rastrear el origen de los fondos en 
cada transacción.
oldbalanceOrg : Representa el saldo en la cuenta de origen antes de que ocurriera la transacción, ofreciendo un punto de referencia para comprender 
los cambios en los saldos de las cuentas.
newbalanceOrig : refleja el saldo en la cuenta de origen después de que se haya procesado la transacción, proporcionando información sobre cómo la 
transacción afecta el saldo de la cuenta.
nameDest : Funciona como identificador de la cuenta o entidad de destino que recibe los fondos en cada transacción. Ayuda a rastrear a dónde se 
transfiere el dinero.
oldbalanceDest : Indica el saldo en la cuenta de destino antes de la transacción, ofreciendo una línea de base para evaluar los cambios en los 
saldos de las cuentas debido a los fondos entrantes.
newbalanceDest : representa el saldo en la cuenta de destino después de que se haya completado la transacción, lo que proporciona información sobre 
el impacto de los fondos entrantes en el saldo de la cuenta.
isFraud : indicador binario (0 o 1) que indica si la transacción es fraudulenta (1) o legítima (0). Esta es la variable objetivo para el modelado de 
detección de fraude.
isFlaggedFraud : otro indicador binario (0 o 1) que puede indicar si una transacción ha sido marcada como potencialmente fraudulenta. Esto podría 
servir como una característica adicional para los algoritmos de detección de fraude.
Uso:
Este conjunto de datos es particularmente valioso para desarrollar y evaluar algoritmos y modelos de detección de fraude. Al analizar patrones y 
anomalías dentro de los datos de las transacciones, los investigadores y analistas pueden identificar características asociadas con actividades 
fraudulentas y construir modelos predictivos para detectar automáticamente dichas transacciones en tiempo real.

Aplicaciones potenciales:

Detección de fraude: utilización de algoritmos de aprendizaje automático para identificar automáticamente transacciones fraudulentas en función de 
patrones históricos y características de comportamiento.
Gestión de riesgos: evaluar y mitigar los riesgos asociados con actividades fraudulentas, protegiendo así a las instituciones financieras y a sus 
clientes.
Cumplimiento Normativo: Garantizar el cumplimiento de las regulaciones y estándares antifraude mediante la implementación de mecanismos efectivos 
de detección de fraude.
Protección del cliente: proteger a los clientes de pérdidas financieras detectando y previniendo de forma proactiva transacciones fraudulentas.
Conclusión:
El conjunto de datos sobre transacciones fraudulentas constituye un recurso valioso para comprender y abordar los desafíos que plantean las 
actividades fraudulentas en las transacciones financieras. Al aprovechar las técnicas avanzadas de análisis y aprendizaje automático, las 
organizaciones pueden mejorar su capacidad para detectar y prevenir transacciones fraudulentas, promoviendo así la confianza, la seguridad y la 
integridad dentro del ecosistema financiero."""

#Cargar bibliotecas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('dark_background')
from skimpy import skim
from sklearn.preprocessing import (OneHotEncoder, StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer)
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import (balanced_accuracy_score as bas, confusion_matrix)
from tqdm.autonotebook import tqdm
from collections import Counter
import warnings
warnings.filterwarnings("ignore")
import os
import matplotlib.ticker as mticker # Para visualización de datos básicos
import seaborn as sns  # Para una visualización de datos mejorada
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split as tts #Para dividir los datos en tren y prueba.
from sklearn.neighbors import KNeighborsClassifier #Clasificador KNN
from sklearn.ensemble import RandomForestClassifier # Clasificador de bosque aleatorio
from sklearn.linear_model import LogisticRegression #Clasificador de regresión logística
from sklearn.tree import DecisionTreeClassifier #Clasificador de árbol de decisión
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix, roc_auc_score # Para evaluación de modelos
from sklearn.metrics import roc_curve, auc #gráfico de curva_roc
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split #Para dividir los datos en tren y prueba.


#Cargar datos
data = pd.read_csv("Transactions Data.csv")
data.head()

#EDA
skim(data)

#Preprocesamiento de datos
data_prep = data.drop(columns = ['nameOrig', 'nameDest'])
data_prep.head()

X = data_prep.drop('isFraud', axis = 1)
y = data_prep['isFraud']

SEED = 42
TEST_SIZE = 0.2

X_train, X_test, y_train, y_test = tts(X, 
                                       y, 
                                       test_size = TEST_SIZE, 
                                       random_state = SEED, 
                                       stratify = y)

print(f"X_tren: {len(y_train)} muestras")
print(f"X_prueba: {len(y_test)} muestras")

print(f"tren objetivo: {Counter(y_train)}\n")
print(f"prueba objetivo: {Counter(y_test)}")

NUM_WORKERS = os.cpu_count()
NUM_WORKERS

numerical_features = ['step', 
                      'amount', 
                      'oldbalanceOrg', 
                      'newbalanceOrig', 
                      'oldbalanceDest', 
                      'newbalanceDest']

categorical_feature = ['type']
#Probemos todos los transformadores para las variables numéricas y comprobemos cuál de ellos nos ofrece mejor rendimiento.

scaler_1 = StandardScaler()
scaler_2 = MinMaxScaler()
scaler_3 = RobustScaler()
scaler_4 = PowerTransformer()

SCALERS = [scaler_1, scaler_2, scaler_3, scaler_4]

X_train_prep = {}
X_test_prep = {}

for scaler in tqdm(SCALERS):
    name = type(scaler).__name__
    transformers = [(
                     'ohe', 
                     OneHotEncoder(
                                    drop = 'first', 
                                    sparse_output = False, 
                                    handle_unknown = 'ignore'
                                    ), 
                      categorical_feature
                     ), 
                     (
                      'scaler', 
                      scaler, 
                      numerical_features
                     )]
    
    preprocessor = ColumnTransformer(
                                     transformers, 
                                     remainder = 'passthrough', 
                                     n_jobs = NUM_WORKERS, 
                                     verbose_feature_names_out = False
                                     )
    
    preprocessor.set_output(transform = 'pandas')

    X_train_prep[name] = preprocessor.fit_transform(X_train)
    X_test_prep[name] = preprocessor.transform(X_test)
    
    print(f"{name} finalizado\n")
    
#Hagamos un preprocesamiento para los modelos que no necesitan escalarse.

transforms = [
                (
                'ohe', 
                OneHotEncoder(drop = 'first', 
                              handle_unknown = 'ignore', 
                              sparse_output = False), 
                categorical_feature)
             ]
preprocessor = ColumnTransformer(transforms, 
                                 remainder = 'passthrough', 
                                 n_jobs = NUM_WORKERS, 
                                 verbose_feature_names_out = False)
preprocessor.set_output(transform = 'pandas')

X_train_prep_others = preprocessor.fit_transform(X_train)
X_test_prep_others = preprocessor.transform(X_test)

print(f"X_train_prep_others shape: {X_train_prep_others.shape}")
print(f"X_test_prep_others shape: {X_test_prep_others.shape}")

#Clasificadores
#Elegimos el clasificador base, que es LogisticRegression.

for x_train, x_test in zip(X_train_prep, X_test_prep):

    clf_base = LogisticRegression(
                                  class_weight = 'balanced', 
                                  random_state = SEED, 
                                  max_iter = 1000, 
                                  n_jobs = NUM_WORKERS
                                 )
    
    clf_base.fit(X_train_prep[x_train], y_train)
    
    y_pred_train = clf_base.predict(X_train_prep[x_train])
    y_pred_test = clf_base.predict(X_test_prep[x_test])
    
    print(f"{x_train} | Acc Train: {bas(y_train, y_pred_train):.4f} | Acc Test: {bas(y_test, y_pred_test):.4f}\n")
    
#El mejor de todos los transformadores es PowerTransformer para nuestro clasificador base.

#Ahora probemos otros clasificadores, elijamos el mejor y compárelo con el clasificador base.

clf1 = DecisionTreeClassifier(random_state = SEED, 
                              class_weight = 'balanced')

clf2 = HistGradientBoostingClassifier(random_state = SEED, 
                                      class_weight = 'balanced')

clf3 = XGBClassifier(random_state = SEED, 
                     scale_pos_weight = Counter(y_train)[0] / Counter(y_train)[1],
                     n_jobs = NUM_WORKERS)

clf4 = LGBMClassifier(random_state = SEED, 
                      n_jobs = NUM_WORKERS, 
                      class_weight = 'balanced', 
                      verbosity = -1)

clf5 = CatBoostClassifier(random_state = SEED, 
                          thread_count = NUM_WORKERS, 
                          auto_class_weights = 'Balanced', 
                          verbose = 0)

MODELS = [clf1, clf2, clf3, clf4, clf5]

for model in tqdm(MODELS):
    name = type(model).__name__
    model.fit(X_train_prep_others, y_train)
    
    y_pred_train = model.predict(X_train_prep_others)
    y_pred_test = model.predict(X_test_prep_others)
    
    print(f"{name} | Tren Acc: {bas(y_train, y_pred_train):.4f} | Prueba de Acc: {bas(y_test, y_pred_test):.4f}\n")
    
#Métrica
#Entrenamos a nuestro clasificador ganador y calculamos su matriz de confusión.

model = CatBoostClassifier(random_state = SEED, 
                          thread_count = NUM_WORKERS, 
                          auto_class_weights = 'Balanced', 
                          verbose = 0)

model.fit(X_train_prep_others, y_train)

y_pred_train = model.predict(X_train_prep_others)
y_pred_prob_train = model.predict_proba(X_train_prep_others)[:, 1]

y_pred_test = model.predict(X_test_prep_others)
y_pred_prob_test = model.predict_proba(X_test_prep_others)[:, 1]

#Matriz de confusión
cf_mx_train = confusion_matrix(y_train, y_pred_train)
cf_mx_test = confusion_matrix(y_test, y_pred_test)
fig,ax = plt.subplots(nrows = 1, ncols = 2, figsize = (9, 4))
ax = ax.flat

sns.heatmap(cf_mx_train, 
            cmap = 'cool', 
            annot = True, 
            annot_kws = {'fontsize':10, 'fontweight':'bold'},
            fmt = '',
            cbar = False, 
            linewidths = 1.2,
            square = True, 
            ax = ax[0])
ax[0].set_title("Tren de la matriz de confusión\n", fontsize = 12, fontweight = 'bold', color = 'black')
ax[0].set_xlabel("Predicho", fontsize = 10, fontweight = 'bold', color = 'black')
ax[0].set_ylabel("Verdadero", fontsize = 10, fontweight = 'bold', color = 'black')

sns.heatmap(cf_mx_test, 
            cmap = 'cool', 
            annot = True, 
            annot_kws = {'fontsize':10, 'fontweight':'bold'},
            fmt = '',
            cbar = False, 
            linewidths = 1.2,
            square = True, 
            ax = ax[1])
ax[1].set_title("Prueba de matriz de confusión\n", fontsize = 12, fontweight = 'bold', color = 'black')
ax[1].set_xlabel("Predicho", fontsize = 10, fontweight = 'bold', color = 'black')
ax[1].set_ylabel("Verdadero", fontsize = 10, fontweight = 'bold', color = 'black')

fig.tight_layout()
fig.show()

#Detección de transacciones fraudulentas
#Importar bibliotecas necesarias
#Primero, necesitamos importar las bibliotecas necesarias, como pandas, numpy, matplotlib y seaborn para la visualización de datos, y sklearn para construir 
# nuestro modelo predictivo.

#cargar el conjunto de datos

#buscar el conjunto de datos

#columnas en el conjunto de datos
data.columns

#Explora el conjunto de datos
#Antes de construir el modelo, debemos explorar el conjunto de datos para comprender los datos, su estructura y las relaciones entre las 
# características. Podemos utilizar varios métodos de pandas como head(), info(), describe() y shape para obtener información básica sobre el 
# conjunto de datos.
# Ver las primeras 5 filas del conjunto de datos
data.head()

# Ver las últimas 5 filas del conjunto de datos
data.tail()

# Ver la información del conjunto de datos
data.info()

# Ver las estadísticas del conjunto de datos
data.describe()

#Limpieza y preprocesamiento de datos
#En este paso, limpiaremos y preprocesaremos el conjunto de datos manejando los valores faltantes, eliminando duplicados y convirtiendo variables 
# categóricas en numéricas.
# Comprobar valores faltantes
data.isnull().sum()

#verificar los tipos de datos que tenemos en nuestro conjunto de datos 
data.dtypes

print('El número de duplicados es: ', data.duplicated().sum())

#Análisis exploratorio de datos
#EDA nos ayuda a comprender la distribución de los datos, las relaciones entre las características y los conocimientos importantes que pueden ayudar a comprender el conjunto de datos con claridad. Utilizamos varios gráficos para visualizar el conjunto de datos.
data_h = data[['step', 'amount', 'nameOrig', 'oldbalanceOrg', 'newbalanceOrig',
                    'nameDest', 'oldbalanceDest', 'newbalanceDest', 'isFraud', 'isFlaggedFraud']]
plt.figure(figsize=(10, 8))
mask = np.triu(np.ones_like(data_h.corr(numeric_only=True), dtype=np.bool_))
heatmap = sns.heatmap(data_h.corr(numeric_only=True), 
                      mask=mask, 
                      vmin=-1, 
                      vmax=1, 
                      center=0, 
                      annot=True, 
                      cmap="inferno",  
                      linewidths=.5)
heatmap.set_title('Mapa de calor de correlación\n', pad=12, fontsize = '16', fontweight = 'bold')
heatmap.tick_params(axis='both', labelsize=11)
plt.show()

#Esta es una matriz de correlación que muestra los coeficientes de correlación entre diferentes pares de variables en su conjunto de datos.
#El coeficiente de correlación es una medida estadística que describe en qué medida dos variables cambian juntas. Los valores varían de -1 a 1, donde:

#El número 1 indica una correlación positiva perfecta (a medida que una variable aumenta, la otra también aumenta).
#-1 indica una correlación negativa perfecta (a medida que una variable aumenta, la otra disminuye).
#0 indica que no hay correlación.
#Aquí hay una interpretación de la matriz de correlación dada:

"""
#Paso:

#Correlación positiva débil con newbalanceDest (0,027665) e isFraud (0,031578).
#Correlación negativa débil con oldbalanceOrg (-0.010058), newbalanceOrig (-0.010299) y oldbalanceDest (0.027665).
#Cantidad:

Correlación positiva moderada con oldbalanceDest (0,294137) y newbalanceDest (0,459304).
Correlación positiva débil con isFraud (0,076688).
Antigua organización del equilibrio:

Correlación negativa muy débil con oldbalanceDest (0,066243) y newbalanceDest (0,042029).
Correlación positiva muy débil con isFraud (0,010154).
Origen de Newbalance:

Correlación negativa muy débil con oldbalanceDest (0,067812) y newbalanceDest (0,041837).
Correlación negativa muy débil con isFraud (-0,008148).
Antiguo equilibrioDest:

Fuerte correlación positiva con newbalanceDest (0,976569).
Correlación positiva muy débil con isFlaggedFraud (-0,000513).
Nuevo equilibrioDest:

Fuerte correlación positiva con oldbalanceDest (0,976569).
Correlación negativa muy débil con isFlaggedFraud (-0,000529).
Es fraude:

Débil correlación positiva con cantidad (0,076688) y newbalanceOrig (-0,008148).
Correlación positiva muy débil con otras variables.
Está marcado como fraude:

Correlación positiva débil con la cantidad (0,012295).
Correlación positiva muy débil con otras variables."""

#matriz de correlación
data_h.corr(numeric_only=True)

# Cuente el número de cada tipo de transacción
transaction_type_counts = data['type'].value_counts()

# Configurar la figura y los ejes
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Gráfico de barras
sns.barplot(x=transaction_type_counts.index, y=transaction_type_counts.values, ax=ax1, palette=['#044451','#0081a7', '#468896','#00afb9','#38c0db'])
ax1.set_xlabel('tipo de transacción', fontsize=12,fontweight = 'bold')
ax1.set_ylabel('Contar', fontsize=12,fontweight = 'bold')
ax1.set_xticklabels(transaction_type_counts.index,  fontsize=11)
ax1.yaxis.grid(True, linestyle='--', alpha=0.7)
sns.despine(ax=ax1)

# Gráfico circular
# colors = ['#f07167', '#f3b562']
colors = ['#044451','#0081a7',  '#468896','#00afb9','#38c0db']
wedges, texts, autotexts = ax2.pie(transaction_type_counts, labels=transaction_type_counts.index, autopct='%1.1f%%', colors=colors, startangle=0, wedgeprops={'edgecolor': 'white', 'linewidth': 1.5})
ax2.axis('equal')

# Embellecer las etiquetas del gráfico circular
for text in texts:
    text.set_fontsize(11)
    text.set_fontweight('bold')

for autotext in autotexts:
    autotext.set_fontsize(11)
    autotext.set_fontweight('bold')

# Agregar sombra al gráfico circular
for wedge in wedges:
    wedge.set_edgecolor('white')
    wedge.set_linewidth(1.5)
    wedge.set_alpha(0.9)

# Ajustar el espacio de diseño entre parcelas
plt.tight_layout()

# Establecer un único título para toda la figura
fig.suptitle('Distribución de tipos de transacciones\n', fontsize=16, fontweight='bold')
plt.show()

# Crea una figura y ejes
plt.figure(figsize=(12, 7))
ax = sns.countplot(data=data, x='type', hue='isFraud', palette=['#044451', '#00afb9'])

# Establecer etiquetas y título
ax.set_xlabel('Tipo\n', fontsize=12, fontweight='bold')
ax.set_ylabel('Conteo\n', fontsize=12, fontweight='bold')
ax.set_yscale('log')
ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
plt.title('Recuento de transacciones por tipo y estado de fraude\n', fontsize=16, fontweight='bold')

# Embellecer leyenda
legend = ax.legend(title='Es fraude\n', title_fontsize='11', loc='upper right')
legend.get_title().set_fontweight('bold')

# Embellecer ticks y grid
ax.tick_params(axis='x', labelsize=11)
ax.tick_params(axis='y', labelsize=11)
ax.yaxis.grid(True, linestyle='--', alpha=0.7)


# Agregue etiquetas de valor encima de cada barra
for p in ax.patches:
    ax.annotate(f"{p.get_height():,}", (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center',
                fontsize=11, color='black', xytext=(0, 10), textcoords='offset points')

# Agregue una cuadrícula de fondo sutil para mejorar la legibilidad
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.show()

plt.figure(figsize=(10, 6))

# Crear el gráfico de conteo
countplot = sns.countplot(x='isFraud', data=data, palette='pastel')

# # Establecer etiquetas y título
countplot.set_title('Recuento de transacciones fraudulentas (1) y transacciones no fraudulentas (0)\n', fontsize=16, fontweight='bold')
countplot.set_xlabel('Es fraudulento\n')
countplot.set_ylabel('Conteo\n')

# Embellecer ticks y grid
countplot.tick_params(axis='both', labelsize=11)
countplot.yaxis.grid(True, linestyle='--', alpha=0.7)

# Agregar etiquetas de datos
for p in countplot.patches:
    countplot.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', fontsize=12, color='black', xytext=(0, 5), textcoords='offset points')
plt.show()

plt.figure(figsize=(12, 8))

# Crear el gráfico del histograma
histplot = sns.histplot(data=data[:100000], 
                        x='amount', 
                        hue='isFraud', 
                        kde=True, 
                        element='step', 
                        palette='cool',
                        log_scale=True)

# Establecer etiquetas y título
histplot.set_ylabel('Número de observaciones\n', fontsize=12, fontweight='bold')
histplot.set_xlabel('Cantidad de transacciones\n', fontsize=12, fontweight='bold')
plt.title('Distribución de montos de transacciones por estado de fraude\n', fontsize=16, fontweight='bold')

# Agregue líneas verticales para los valores medios
mean_value_f = data[data['isFraud'] == False]['amount'].mean()
mean_value_t = data[data['isFraud'] == True]['amount'].mean()
histplot.axvline(x=mean_value_f, color='blue', linestyle='--', label=f'Mean (Regular): ${mean_value_f:,.2f}')
histplot.axvline(x=mean_value_t, color='orange', linestyle='--', label=f'Mean (Fraudulent): ${mean_value_t:,.2f}')

# Agregar anotaciones para valores medios
histplot.annotate(f'Importe medio de transacciones regulares: ${mean_value_f:,.2f}', 
                  xy=(0.5, 0.9),
                  xycoords='axes fraction',
                  fontsize=11, fontweight='bold', color='blue')
histplot.annotate(f'Importe medio de transacciones fraudulentas: ${mean_value_t:,.2f}', 
                  xy=(0.5, 0.85),
                  xycoords='axes fraction',
                  fontsize=11, fontweight='bold', color='orange')
histplot.xaxis.set_major_formatter(mticker.ScalarFormatter())
histplot.ticklabel_format(style='plain', axis='x')
histplot.tick_params(axis='both', labelsize=11)
histplot.yaxis.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.show()

plt.figure(figsize=(16, 10))
boxplot = sns.boxplot(x='type', y='amount', data=data, palette='pastel')
boxplot.set_title('Montos de transacciones por tipo de transacción\n', fontsize=16, fontweight='bold')
boxplot.set_xlabel('Tipo de transacción\n', fontsize=12, fontweight='bold')
boxplot.set_ylabel('Monto de la transacción (escala logarítmica)\n', fontsize=12, fontweight='bold')
boxplot.set_yscale('log')
boxplot.tick_params(axis='both', labelsize=11)
boxplot.yaxis.grid(True, linestyle='--', alpha=0.7)
plt.show()

# Ingeniería de funciones
#Los valores de los parámetros de las columnas se clasifican en el conjunto de datos numéricos para obtener la mejor actualidad para el 
# entrenamiento del modelo.
#tipos de datos de los atributos del conjunto de datos
data.dtypes

#codificar los objetos de cadena a los valores categóricos a valores numéricos
encoder = {}
for i in data.select_dtypes('object').columns:
    encoder[i] = LabelEncoder()
    data[i] = encoder[i].fit_transform(data[i])
x = data.drop(columns=['isFraud'])
y = data['isFraud']
#escalar el conjunto de datos
scaler = MinMaxScaler()
x = scaler.fit_transform(x)
# Dividir los datos de entrenamiento y pruebas en funciones dependientes e independientes con el 80 % de los datos de entrenamiento y el 
# 20 % de los datos para pruebas.
# Dividir los datos en características (X) y etiquetas (y)
X = data[['step', 'type', 'amount', 'oldbalanceOrg', 'newbalanceOrig','nameDest', 'oldbalanceDest', 'isFlaggedFraud']]
y= data['isFraud']
# Divida los datos en conjuntos de entrenamiento y prueba.
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.20,random_state=0)
print('Características del entrenamiento Forma:', X_train.shape)
print('Forma de las etiquetas de entrenamiento:', y_train.shape)
print('Características de prueba Forma:', X_test.shape)
print('Forma de las etiquetas de prueba:', y_test.shape)

# Inicialización y capacitación del modelo (aquí el modelo se inicializa y entrena) con la evaluación del modelo para todos los modelos de 
# aprendizaje automático.
#K-Clasificador de vecinos más cercanos

#Inicializar y capacitar uno de los modelos K-Vecinos más cercanos para la clasificación de transacciones fraudulentas
knn_classifier = KNeighborsClassifier()
knn_classifier.fit(X_train,y_train)

#predecir el modelo y clasificar los resultados según los datos de prueba
knn_predictions = knn_classifier.predict(X_test)
#calcular la precisión del modelo KNN
knn_accuracy = accuracy_score(y_test, knn_predictions)
print("Precisión para el modo K-Vecinos más cercanos: {:.2f}%".format(knn_accuracy * 100))

#Informe de clasificación
print("Informe de clasificación para el modelo KNN:\n ",classification_report(y_test,knn_predictions))

#matriz de confusión del modelo K-vecinos más cercanos
knn_confusion_matrix=confusion_matrix(y_test,knn_predictions)

# Clasificador de árbol de decisión

# Inicializar y entrenar uno de los modelos de árbol de decisión para la clasificación de transacciones fraudulentas
dt_classifier = DecisionTreeClassifier()
dt_classifier.fit(X_train,y_train)

#predecir el modelo y clasificar los resultados según los datos de prueba
dt_predictions = dt_classifier.predict(X_test)
#calcular la precisión del modelo de árbol de decisión
dt_accuracy = accuracy_score(y_test, dt_predictions)
print("Precisión del modelo de árbol de decisión: {:.2f}%".format(dt_accuracy * 100))

#Informe de clasificación
print("Informe de clasificación para el modelo de árbol de decisión:\n ",classification_report(y_test,dt_predictions))

#matriz de confusión del modelo de árbol de decisión
dt_confusion_matrix=confusion_matrix(y_test,dt_predictions)
# Clasificador de regresión logística

#Inicializar y entrenar uno de los (Modelo de Regresión Logística) para la Clasificación de Transacciones Fraude
logistic_regression_model = LogisticRegression()
logistic_regression_model.fit(X_train,y_train)

#predecir el modelo y clasificar los resultados según los datos de prueba
logistic_regression_predictions = logistic_regression_model.predict(X_test)
#calcular la precisión del modelo de regresión logística
logistic_regression_accuracy = accuracy_score(y_test, logistic_regression_predictions)
print("Precisión del modelo de regresión logística: {:.2f}%".format(logistic_regression_accuracy * 100))

#Informe de clasificación
print("Informe de clasificación para el modelo de regresión logística:\n ",classification_report(y_test,logistic_regression_predictions))

#matriz de confusión del modelo de regresión logística
logistic_regression_confusion_matrix=confusion_matrix(y_test,logistic_regression_predictions)
# Clasificador de bosque aleatorio

#Inicializar y capacitar uno de los (modelos de bosque aleatorio) para la clasificación de transacciones fraudulentas
random_forest_model = RandomForestClassifier()
random_forest_model.fit(X_train,y_train)

#predecir el modelo y clasificar los resultados según los datos de prueba
random_forest_predictions = random_forest_model.predict(X_test)

#calcular la precisión del modelo de bosque aleatorio
random_forest_accuracy = accuracy_score(y_test, random_forest_predictions)
print("Precisión para el modo de bosque aleatorio: {:.2f}%".format(random_forest_accuracy * 100))

#Informe de clasificación
print("Informe de clasificación para mod (bosque aleatorio):\n ",classification_report(y_test,random_forest_predictions))

#matriz de confusión del modelo de bosque aleatorio
rdf_confusion_matrix=confusion_matrix(y_test,random_forest_predictions)
# Comparación de los resultados de las métricas para todos los modelos de aprendizaje automático.
# modelos
models = {
    "K-Nearest Neighbors (KNN)": knn_classifier,
    "Decision Tree": dt_classifier,
    "Logistic Regression": logistic_regression_model,
    "Random Forest": random_forest_model
}
# Cree un diccionario para contener las matrices de confusión para cada modelo
confusion_matrices = {'K-Nearest Neighbors': knn_confusion_matrix,
                      'Decision Tree': dt_confusion_matrix,
                      'Logistic Regression': logistic_regression_confusion_matrix,
                      'Random Forest': rdf_confusion_matrix
                      }

# paleta de colores personalizada adecuada para visualización financiera
custom_palette = sns.color_palette("pastel", as_cmap=True)

# figura para las matrices de confusión
plt.figure(figsize=(18, 18))

# Iterar a través de las matrices de confusión y trazarlas
for i, (label, conf_matrix) in enumerate(confusion_matrices.items()):
    plt.subplot(4, 4, i + 1)
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap=custom_palette, cbar=False, square=True,
                xticklabels=['Predicted Non-Fraud', 'Predicted Fraud'],
                yticklabels=['Actual Non-Fraud', 'Actual Fraud'])
    
    plt.title(f'Matriz de confusión para {label}\n', fontsize=16, fontweight='bold')
    plt.xlabel('Predicho\n', fontsize=12, fontweight='bold')
    plt.ylabel('Actual\n', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.show()

# cifra para curvas ROC AUC
plt.figure(figsize=(12, 10))

# colores para la curva de cada modelo
colors = ['b', 'g', 'r', 'c']

# Iterar a través de cada modelo y trazar la curva ROC AUC
for i, (label, model) in enumerate(models.items()):
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    
    # Trazar las curvas ROC AUC con distintos colores
    plt.plot(fpr, tpr, lw=2, label=f'{label} (AUC = {roc_auc:.2f})', color=colors[i])
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Tasa de falsos positivos\n', fontsize=12, fontweight='bold')
plt.ylabel('Tasa de verdaderos positivos\n', fontsize=12, fontweight='bold')
plt.title('Curvas ROC para todos los modelos de aprendizaje automático\n', fontsize=16, fontweight='bold')
plt.legend(loc="lower right", fontsize=12)
plt.show()

"""AUC es una métrica común utilizada para evaluar el rendimiento de los modelos de clasificación binaria, que se utilizan para predecir uno de 
dos resultados posibles.

1. K-Vecinos más cercanos (KNN) (AUC = 0,89):
K-Nearest Neighbors es un algoritmo de aprendizaje automático simple e intuitivo que se utiliza para tareas de clasificación y regresión. Opera 
según el principio de que los puntos de datos similares están cerca entre sí en el espacio de características.
Una puntuación AUC de 0,89 sugiere que este modelo KNN logró un buen nivel de discriminación, aunque no perfecto, al distinguir entre las dos 
clases que predice. 
Cuanto más se acerque la puntuación AUC a 1,0, mejor será el rendimiento del modelo.
2. Árbol de decisión (AUC = 0,93):
Un árbol de decisión es una estructura similar a un árbol que se utiliza para tomar decisiones basadas en un conjunto de condiciones. Es un 
modelo popular tanto para tareas de clasificación como de regresión.
Una puntuación AUC de 0,93 indica que el modelo de árbol de decisión tiene una gran capacidad para diferenciar entre las clases que predice. 
Está funcionando bastante bien en este contexto.
3. Regresión logística (AUC = 0,96):
La regresión logística es un modelo estadístico utilizado para la clasificación binaria. Modela la probabilidad de un resultado binario.
Una puntuación AUC de 0,96 es bastante alta, lo que indica que el modelo de regresión logística es muy eficaz para clasificar datos. Es 
excelente para distinguir entre las dos clases.
4. Bosque aleatorio (AUC = 1,00):
Random Forest es un método de aprendizaje conjunto que combina múltiples árboles de decisión para mejorar la precisión predictiva y reducir 
el sobreajuste.
Una puntuación AUC de 1,00 significa que el modelo Random Forest ha logrado una discriminación perfecta. Puede distinguir perfectamente entre 
las dos clases en el conjunto de datos, lo que indica un rendimiento extremadamente sólido.
En resumen, las puntuaciones AUC reflejan el rendimiento de clasificación de estos modelos. Una puntuación AUC más alta generalmente implica 
un mejor rendimiento del modelo en términos de distinción entre las dos clases, y Random Forest, con un AUC de 1,00, destaca por haber logrado una discriminación 
perfecta en este contexto.
"""

print(data.isnull().sum())

print(data['type'].value_counts())
print(data['isFraud'].value_counts())
print(data['isFlaggedFraud'].value_counts())

categorical_cols = ['type']
numeric_cols = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(), categorical_cols)
    ])

X = data.drop(['isFraud'], axis=1)
y = data['isFraud']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(),
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"{name}")
    print(classification_report(y_test, y_pred))
    print("ROC-AUC:", roc_auc_score(y_test, y_pred))
    
