import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import re
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.neighbors import KNeighborsRegressor

from scikeras.wrappers import KerasRegressor
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from sklearn.preprocessing import FunctionTransformer

import os

'''
--------------------------------------------------------------------------------------
                         Análisis Exploratorio de Datos (EDA)
--------------------------------------------------------------------------------------
'''

# Cargar dataset
current_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(current_dir, "laptop_price.csv")
laptops = pd.read_csv(csv_path, encoding="latin-1")
laptops = laptops.set_index('laptop_ID')

# Vista inicial
print("\nPRIMERAS 5 FILAS (REGISTROS) DE LA TABLA: ")
print(laptops.head())

print("\nINFORMACIÓN SOBRE LAS COLUMNAS: ")
print(laptops.info())

# Estadísticos Descriptivos Básicos
print("\nESTADÍSTICOS DESCRIPTIVOS PARA LAS CARACTERÍSTICAS: ")
print(laptops.describe(include="all"))

# Distribuciones de Variables Numéricas
variables_numericas = ["Inches", "Weight", "Price_euros"]
for var in variables_numericas:
    plt.figure(figsize=(6,4))
    sb.histplot(laptops[var], kde=True, bins=30)
    plt.title(f"Distribución de {var}")
    plt.show()

# Distribuciones de Variables Categóricas
variables_categoricas = ["Company", "TypeName", "OpSys"]
for var in variables_categoricas:
    plt.figure(figsize=(8,4))
    sb.countplot(y=laptops[var], order=laptops[var].value_counts().index)
    plt.title(f"Distribución de {var}")
    plt.show()

# Limpiar y Preparar los Datos, para Correlaciones y para lo Siguiente (Preprocesamiento de Datos)
laptops["Ram"] = laptops["Ram"].str.replace('GB', '')
laptops["Weight"] = laptops["Weight"].str.replace('kg', '')
laptops["Memory"] = laptops["Memory"].astype(str).str.replace(r"\.0", "", regex=True)
laptops["Memory"] = laptops["Memory"].str.replace('GB', '')
laptops["Memory"] = laptops["Memory"].str.replace('TB', '000')
new2 = laptops["Memory"].str.split("+", n = 1, expand = True)
first_layer = new2[0].fillna('').str.strip()
second_layer = new2[1].fillna('').str.strip()

laptops["Layer1HDD"] = first_layer.str.contains("HDD", na=False).astype(int)
laptops["Layer1SSD"] = first_layer.str.contains("SSD", na=False).astype(int)
laptops["Layer1Hybrid"] = first_layer.str.contains("Hybrid", na=False).astype(int)
laptops["Layer1Flash_Storage"] = first_layer.str.contains("Flash Storage", na=False).astype(int)
laptops["Layer2HDD"] = second_layer.str.contains("HDD", na=False).astype(int)
laptops["Layer2SSD"] = second_layer.str.contains("SSD", na=False).astype(int)
laptops["Layer2Hybrid"] = second_layer.str.contains("Hybrid", na=False).astype(int)
laptops["Layer2Flash_Storage"] = second_layer.str.contains("Flash Storage", na=False).astype(int)

laptops["first"] = first_layer.str.extract(r"(\d+)")[0].fillna("0").astype(int)
laptops["second"] = second_layer.str.extract(r"(\d+)")[0].fillna("0").astype(int)
laptops["Total_Memory"]=(laptops["first"]*(laptops["Layer1HDD"]+laptops["Layer1SSD"]+laptops["Layer1Hybrid"]+laptops["Layer1Flash_Storage"])+laptops["second"]*(laptops["Layer2HDD"]+laptops["Layer2SSD"]+laptops["Layer2Hybrid"]+laptops["Layer2Flash_Storage"]))
laptops["Memory"]=laptops["Total_Memory"]
laptops["HDD"]=(laptops["first"]*laptops["Layer1HDD"]+laptops["second"]*laptops["Layer2HDD"])
laptops["SSD"]=(laptops["first"]*laptops["Layer1SSD"]+laptops["second"]*laptops["Layer2SSD"])
laptops["Hybrid"]=(laptops["first"]*laptops["Layer1Hybrid"]+laptops["second"]*laptops["Layer2Hybrid"])
laptops["Flash_Storage"]=(laptops["first"]*laptops["Layer1Flash_Storage"]+laptops["second"]*laptops["Layer2Flash_Storage"])
new = laptops["ScreenResolution"].str.split("x", n = 1, expand = True) 
laptops["X_res"]= new[0]
laptops["Y_res"]= new[1]
laptops["Y_res"]= pd.to_numeric(laptops["Y_res"])
laptops["Y_res"]= laptops["Y_res"].astype(float)
laptops["X_res"]=(laptops['X_res'].str.replace(',','').str.findall(r'(\d+\.?\d+)').apply(lambda x: pd.Series(x).astype(int)).mean(1))
laptops["X_res"]=pd.to_numeric(laptops["X_res"])
laptops["PPI"]=(((laptops["X_res"]**2+laptops["Y_res"]**2)**(1/2))/laptops["Inches"]).astype(float)
laptops["ScreenResolution"]=(laptops["X_res"]*laptops["Y_res"]).astype(float)
laptops["Ram"] = laptops["Ram"].astype(int)
laptops["Weight"] = laptops["Weight"].astype(float)
laptops=laptops.drop(['first','second','Layer1HDD','Layer1SSD','Layer1Hybrid','Layer1Flash_Storage','Layer2HDD','Layer2SSD','Layer2Hybrid','Layer2Flash_Storage','Total_Memory'],axis=1)

# Mostrar el Head con los cambios
print("\nPRIMERAS 5 FILAS (REGISTROS) DE LA TABLA LUEGO DE LA LIMPIEZA: ")
print(laptops.head(5))

# Mostrar todas las Columnas con los cambios
print("\nINFORMACIÓN SOBRE LAS COLUMNAS LUEGO DE LA LIMPIEZA: ")
print(laptops.info())

# Función con mapa de correlaciones entre variables númericas
def mapa_correlacion (datos):
    correlaciones = datos.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(16,16))
    sb.heatmap(correlaciones, vmax=1.0, center=0, fmt='.2f', square=True, linewidths=0.5, annot=True, cbar_kws={"shrink":.70})
    plt.title("Mapa de correlaciónes variables Numericas")
    plt.show()

mapa_correlacion(laptops)

# Precio según la Compañía
plt.figure(figsize=(12,6))
sb.boxplot(x="Company", y="Price_euros", data=laptops)
plt.xticks(rotation=90)
plt.title("Distribución del Precio según la Compañía")
plt.show()

# Precio según la RAM
plt.figure(figsize=(8,6))
sb.boxplot(x="Ram", y="Price_euros", data=laptops)
plt.title("Distribución del Precio según la RAM")
plt.show()

fig_dims = (20, 10)
fig, ax = plt.subplots(figsize=fig_dims)
sb.scatterplot(data=laptops, x="Price_euros", y="Ram", ax=ax, s=75)

# Precio según el SSD
plt.figure(figsize=(8,6))
sb.boxplot(x="SSD", y="Price_euros", data=laptops)
plt.title("Distribución del Precio según el SSD (GB)")
plt.show()

fig_dims = (20, 10)
fig, ax = plt.subplots(figsize=fig_dims)
sb.scatterplot(data=laptops, x="Price_euros", y="SSD", ax=ax, s=75)


'''
--------------------------------------------------------------------------------------
                             Procesamiento de Datos
--------------------------------------------------------------------------------------
'''


# 1. Definir X (features) e y (target) 
TARGET = "Price_euros"

# Columnas numéricas generadas en la limpieza y que vamos a usar
caracteristicas_numericas = [
    "Inches", "Weight", "Ram", "Memory",        
    "X_res", "Y_res", "PPI", "ScreenResolution",
    "HDD", "SSD", "Hybrid", "Flash_Storage"
]

# Categóricas a codificar
caracteristicas_categoricas = ["Company", "TypeName", "OpSys"]

# Verificación rápida numéricas
numericas_filtradas = []
for c in caracteristicas_numericas:
    if c in laptops.columns:
        numericas_filtradas.append(c)
caracteristicas_numericas = numericas_filtradas

# Verificación rápida categóricas
categoricas_filtradas = []
for c in caracteristicas_categoricas:
    if c in laptops.columns:
        categoricas_filtradas.append(c)
caracteristicas_categoricas = categoricas_filtradas

# Dividimos en qué serán las características (X) y en qué será el target (Y)
X = laptops[caracteristicas_numericas + caracteristicas_categoricas].copy()
y = laptops[TARGET].astype(float).copy()


# 2. Division 70/15/15 (train/val/test)
# Acá cogemos el 15% de las filas y lo asignamos a test, dejando el 85% restante para train y val
X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

# Acá cogemos ese 85% restante y asignamos la parte del test que es el 17.6% de eso (igual a 15% del total)
# Por lo que dejamos el 70% restante para el training
X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.1765, random_state=42)

# Se muestran las dimensiones de cada split
print(f"train={X_train.shape}, val={X_val.shape}, test={X_test.shape}")


# 3 Preprocesamiento con ColumnTransformer para diferenciar entre columnas numéricas y categóricas
# Pipeline para características (columnas) numéricas
transformador_numerico = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),   # Acá reemplazamos valores nulos con la mediana de cada columna
    ("scaler", StandardScaler())                     # Acá centramos y escalamos las columnas para que tengan media 0 y desviación estándar 1
])

# Pipeline para características (columnas) categóricas
onehot_kwargs = {"handle_unknown": "ignore"}
if "min_frequency" in OneHotEncoder.__init__.__code__.co_varnames:
    onehot_kwargs["min_frequency"] = 20  # agrupa categorías raras si la versión lo permite
if "sparse_output" in OneHotEncoder.__init__.__code__.co_varnames:
    onehot_kwargs["sparse_output"] = True  # salida dispersa (aplica en sklearn recientes)
elif "sparse" in OneHotEncoder.__init__.__code__.co_varnames:
    onehot_kwargs["sparse"] = True  # compatibilidad con versiones anteriores

transformador_categorico = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(**onehot_kwargs))
])

# Combinamos ambas pipelines, aplicando cada transformador a su lista correspondiente, ignorando otras columnas (como id)
preprocesar = ColumnTransformer(
    transformers=[
        ("num", transformador_numerico, caracteristicas_numericas),
        ("cat", transformador_categorico, caracteristicas_categoricas),
    ],
    remainder="drop"  # Esto es lo que hace que ignoremos columnas que sobren
)


# 4 Pipelines de modelos 

# (A) Modelo lineal regularizado
# Acá aplicamos preprocesamiento primero y luego entrenamos un ridge regresion con los datos ya transformados 
ridge_pipe = Pipeline(steps=[
    ("prep", preprocesar),
    ("model", Ridge(alpha=1.0, random_state=42))
])

# (B) Modelo basado en árboles (menos sensible a escalado)
# Acá aplicamos preprocesamiento primero y luego entrenamos un random forest regressor con 400 árboles
rf_pipe = Pipeline(steps=[
    ("prep", preprocesar),
    ("model", RandomForestRegressor(
        n_estimators=400, max_depth=None, n_jobs=-1, random_state=42))
])

# (C) Modelo KNN
knn_pipe = Pipeline(steps=[
    ("prep", preprocesar),
    ("model", KNeighborsRegressor(
        n_neighbors=5,      # número de vecinos inicial
        weights="distance", # pondera vecinos más cercanos
        n_jobs=-1
    ))
])

# (E) Opción con PCA (si el one-hot explota dimensiones)
# Si queremos usar PCA porque el One-hot generó muchas columnas y queremos reducir dimensionalidad
USE_PCA = False  # cambiar a True si lo necesitamos
if USE_PCA:
    ridge_pca_pipe = Pipeline(steps=[
        ("prep", preprocesar),
        ("pca", PCA(n_components=0.95, svd_solver="full", random_state=42)),
        ("model", Ridge(alpha=1.0, random_state=42))
    ])


# DNN en pipeline (Keras + scikit-learn)
# --- 0) (Opcional) Asegurar salida densa tras el preprocess ---
# Si tu OneHotEncoder devolviera matriz dispersa, esto la vuelve densa antes de Keras.
to_dense = FunctionTransformer(lambda X: X.toarray() if hasattr(X, "toarray") else X)

# --- 1) Definición del modelo Keras ---
def build_dnn(
    input_dim: int,
    hidden_units=(256, 128, 64),   # 3 capas ocultas
    l2_reg=1e-4,                   # regularización L2
    dropout_rate=0.2,              # dropout entre capas
    lr=1e-3                        # learning rate
):
    model = keras.Sequential()
    model.add(layers.Input(shape=(input_dim,)))

    for h in hidden_units:
        model.add(layers.Dense(
            h,
            activation="relu",
            kernel_regularizer=regularizers.l2(l2_reg)
        ))
        model.add(layers.Dropout(dropout_rate))

    # Capa de salida para regresión
    model.add(layers.Dense(1, activation="linear"))

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="mse",
        metrics=[
            keras.metrics.MeanAbsoluteError(name="mae"),
            keras.metrics.RootMeanSquaredError(name="rmse")
        ]
    )
    return model

# --- 2) Armar el pipeline: preprocess -> (to_dense) -> KerasRegressor ---
# Nota: reutilizamos 'preprocess' y los splits X_train/X_val/X_test definidos antes.
dnn_pipe = Pipeline(steps=[
    ("prep", preprocesar),
    ("to_dense", to_dense),  # convierte sparse a densa si hace falta
    ("model", KerasRegressor(
        model=build_dnn,
        # Estos params se inyectan a build_dnn:
        input_dim=None,                 # se fijará tras 'prep' con set_params dinámico
        hidden_units=(256,128,64),
        l2_reg=1e-4,
        dropout_rate=0.2,
        lr=1e-3,
        # Hiperparámetros de entrenamiento:
        epochs=200,
        batch_size=64,
        verbose=0,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor="val_rmse", patience=15, mode="min", restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_rmse", factor=0.5, patience=7, min_lr=1e-6, mode="min", verbose=0
            )
        ],
        validation_split=0.0  # usamos nuestro X_val externo, no split interno
    ))
])

# --- 3) Ajustar input_dim automáticamente tras un .fit parcial ---
# Truco: pasamos una muestra por el 'prep' para conocer la dimensionalidad final.
# (Usa unas pocas filas para evitar costo extra)
X_sample = X_train.iloc[:10]
# Transformación sólo de 'prep' (sin to_dense) para medir n_features
prepped_sample = dnn_pipe.named_steps["prep"].fit_transform(X_train, y_train)
input_dim = prepped_sample.shape[1]

# Actualizamos el parámetro input_dim del KerasRegressor
dnn_pipe.named_steps["model"].set_params(input_dim=input_dim)

# --- 4) Entrenamiento con validación externa (X_val / y_val) ---
dnn_pipe.fit(X_train, y_train,
             model__validation_data=(dnn_pipe.named_steps["to_dense"].transform(
                 dnn_pipe.named_steps["prep"].transform(X_val)
             ), y_val))

# --- 5) Evaluación (train/val/test) ---
'''
def eval_pipe(name, pipe, Xtr, ytr, Xva, yva):
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
    pred_tr = pipe.predict(Xtr)
    pred_va = pipe.predict(Xva)
    R2_tr  = r2_score(ytr, pred_tr)
    R2_va  = r2_score(yva, pred_va)
    MAE_tr = mean_absolute_error(ytr, pred_tr)
    MAE_va = mean_absolute_error(yva, pred_va)
    RMSE_tr = mean_squared_error(ytr, pred_tr, squared=False)
    RMSE_va = mean_squared_error(yva, pred_va, squared=False)
    print(f"\n[{name}]")
    print(f"  Train -> R2={R2_tr:.3f} | MAE={MAE_tr:.1f} | RMSE={RMSE_tr:.1f}")
    print(f"  Valid -> R2={R2_va:.3f} | MAE={MAE_va:.1f} | RMSE={RMSE_va:.1f}")

eval_pipe("DNN", dnn_pipe, X_train, y_train, X_val, y_val)

# Test final
pred_test = dnn_pipe.predict(X_test)
print("\n[TEST DNN]")
print(f"  R2={r2_score(y_test, pred_test):.3f} | MAE={mean_absolute_error(y_test, pred_test):.1f} | RMSE={mean_squared_error(y_test, pred_test, squared=False):.1f}")
'''

# 5 Entrenamiento y evaluación rápida en validación 
# Función para evaluar los modelos, con el nombre, el pipeline, las características y target de entrenamiento y 
# las características y target de validación.
def evaluar_modelo(nombre, pipe, Xtr, ytr, Xva, yva):
    pipe.fit(Xtr, ytr)  # Ajustar todo el pipeline (preprocesamiento más el modelo)
    pred_tr = pipe.predict(Xtr)  # Se generan predicciones sobre el training para evaluar el modelo
    pred_va = pipe.predict(Xva)  # Se generan predicciones sobre la validación para evaluar el modelo

    # Simplemente medimos 3 métricas para determinar el desempeño del modelo
    def metricas(y_true, y_pred):
        return {
            "R2": r2_score(y_true, y_pred),  # Coeficiente de determinación donde 1 es perfecto y <0 es peor que predecir la media
            "MAE": mean_absolute_error(y_true, y_pred),  # Error absoluto medio en euros
            "RMSE": np.sqrt(mean_squared_error(y_true, y_pred))  # Raíz del error cuadrático medio, que penaliza los más a los errores grandes
        }

    # Acá evaluamos training y validación con la función metricas y retornamos un diccionario con sus 3 respectivos resultados
    m_tr = metricas(ytr, pred_tr)
    m_va = metricas(yva, pred_va)

    # Acá simplemente mostramos los resultados de la evaluación del modelo específico en entrenamiento y validación
    print(f"\n[{nombre}]")
    print(f"  Entrenamiento -> R2={m_tr['R2']:.3f} | MAE={m_tr['MAE']:.1f} | RMSE={m_tr['RMSE']:.1f}")
    print(f"  Validación -> R2={m_va['R2']:.3f} | MAE={m_va['MAE']:.1f} | RMSE={m_va['RMSE']:.1f}")

    return {
        "nombre": nombre,
        "pipeline": pipe,
        "train": m_tr,
        "valid": m_va
    }

# Acá evaluamos los modelos, el ridge, KNN y el random forest
ridge_pipe = evaluar_modelo("Ridge", ridge_pipe, X_train, y_train, X_val, y_val)
rf_pipe    = evaluar_modelo("RandomForest", rf_pipe, X_train, y_train, X_val, y_val)
knn_pipe = evaluar_modelo("KNN", knn_pipe, X_train, y_train, X_val, y_val)
dnn_pipe = evaluar_modelo("DNN", dnn_pipe, X_train, y_train, X_val, y_val)


if USE_PCA:
    ridge_pca_pipe = evaluar_modelo("Ridge+PCA", ridge_pca_pipe, X_train, y_train, X_val, y_val)


# 6 Evaluación final en test para todos los modelos y tabla comparativa

def crear_tabla_comparativa(modelos, X_test, y_test):
    """
    Crea una tabla comparativa de todos los modelos evaluados en el conjunto de test.
    """
    resultados = []
    
    for modelo_info in modelos:
        nombre = modelo_info["nombre"]
        pipeline = modelo_info["pipeline"]
        
        # Predicciones en test
        pred_test = pipeline.predict(X_test)
        
        # Calcular métricas en test
        r2 = r2_score(y_test, pred_test)
        mae = mean_absolute_error(y_test, pred_test)
        rmse = np.sqrt(mean_squared_error(y_test, pred_test))
        
        # Agregar a la lista de resultados
        resultados.append({
            'Modelo': nombre,
            'R² Score': f"{r2:.4f}",
            'MAE (€)': f"{mae:.2f}",
            'RMSE (€)': f"{rmse:.2f}"
        })
    
    # Crear DataFrame y mostrar tabla
    df_resultados = pd.DataFrame(resultados)
    
    print("\n" + "="*60)
    print("           TABLA COMPARATIVA DE MODELOS EN TEST")
    print("="*60)
    print(df_resultados.to_string(index=False))
    print("="*60)
    
    # Identificar el mejor modelo
    df_resultados['R² Score (num)'] = df_resultados['R² Score'].astype(float)
    mejor_modelo_idx = df_resultados['R² Score (num)'].idxmax()
    mejor_modelo = df_resultados.iloc[mejor_modelo_idx]['Modelo']
    mejor_r2 = df_resultados.iloc[mejor_modelo_idx]['R² Score']
    
    print(f"\n🏆 MEJOR MODELO: {mejor_modelo} (R² = {mejor_r2})")
    print("="*60)
    
    return df_resultados

# Lista de todos los modelos entrenados
modelos_entrenados = [ridge_pipe, rf_pipe]
if USE_PCA:
    modelos_entrenados.append(ridge_pca_pipe)

# Crear y mostrar la tabla comparativa
tabla_comparativa = crear_tabla_comparativa(modelos_entrenados, X_test, y_test)
