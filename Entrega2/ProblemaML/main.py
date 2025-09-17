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
transformador_categorico = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),  # Acá rellenamos los valores nulos por la cateogría más repetida
    ("onehot", OneHotEncoder(handle_unknown="ignore",      # Acá convertimos las categóricas en variables binarias, ignoramos categorías nuevas que aparezcan al predecir
                             min_frequency=20 if "min_frequency" in OneHotEncoder.__init__.__code__.co_varnames else None, # Cualquier variable con menos de 20 registros entra en una categoría "otros"
                             sparse_output=True if "sparse_output" in OneHotEncoder.__init__.__code__.co_varnames else False,  # Obtenemos matriz dispersa para ahorrar memoria
                             sparse=True if "sparse" in OneHotEncoder.__init__.__code__.co_varnames else None))
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
ridge_pipe = Pipeline(steps=[
    ("prep", preprocesar),
    ("model", Ridge(alpha=1.0, random_state=42))
])

# (B) Modelo basado en árboles (menos sensible a escalado)
rf_pipe = Pipeline(steps=[
    ("prep", preprocesar),
    ("model", RandomForestRegressor(
        n_estimators=400, max_depth=None, n_jobs=-1, random_state=42))
])

# (C) Opción con PCA (útil si el one-hot explota dimensiones)
USE_PCA = False  # cambia a True si lo necesitas
if USE_PCA:
    ridge_pca_pipe = Pipeline(steps=[
        ("prep", preprocesar),
        ("pca", PCA(n_components=0.95, svd_solver="full", random_state=42)),
        ("model", Ridge(alpha=1.0, random_state=42))
    ])


# 5 Entrenamiento y evaluación rápida en validación 
def eval_model(name, pipe, Xtr, ytr, Xva, yva):
    pipe.fit(Xtr, ytr)
    pred_tr = pipe.predict(Xtr)
    pred_va = pipe.predict(Xva)
    def metrics(y_true, y_pred):
        return {
            "R2": r2_score(y_true, y_pred),
            "MAE": mean_absolute_error(y_true, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_true, y_pred))
        }
    m_tr = metrics(ytr, pred_tr)
    m_va = metrics(yva, pred_va)
    print(f"\n[{name}]")
    print(f"  Train -> R2={m_tr['R2']:.3f} | MAE={m_tr['MAE']:.1f} | RMSE={m_tr['RMSE']:.1f}")
    print(f"  Valid -> R2={m_va['R2']:.3f} | MAE={m_va['MAE']:.1f} | RMSE={m_va['RMSE']:.1f}")
    return pipe

ridge_pipe = eval_model("Ridge", ridge_pipe, X_train, y_train, X_val, y_val)
rf_pipe    = eval_model("RandomForest", rf_pipe, X_train, y_train, X_val, y_val)

if USE_PCA:
    ridge_pca_pipe = eval_model("Ridge+PCA", ridge_pca_pipe, X_train, y_train, X_val, y_val)


# 6 Evaluación final en test (elige el mejor según Valid) 
# Ejemplo: supón que RandomForest fue mejor en validación
best_pipe = rf_pipe   # o ridge_pipe / ridge_pca_pipe
pred_test = best_pipe.predict(X_test)

print("\n[TEST FINAL]")
print(f"  R2={r2_score(y_test, pred_test):.3f} | MAE={mean_absolute_error(y_test, pred_test):.1f} | RMSE={np.sqrt(mean_squared_error(y_test, pred_test)):.1f}")
