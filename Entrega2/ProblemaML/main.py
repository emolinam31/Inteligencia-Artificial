import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor

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

# Limpiar y Preparar los Datos, para Correlaciones y para lo Siguiente
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


