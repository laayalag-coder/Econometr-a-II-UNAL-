# %% 1. Importación de paquetes ------------------------------
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import jarque_bera, probplot

# Módulos de statsmodels
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tsa.statespace.sarimax import SARIMA
from statsmodels.tsa.stattools import adfuller, kpss


# %% 2. Configuración de rutas relativas ----------------------
BASE_DIR = Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd()
DATA_DIR = BASE_DIR / "datos_trabajo"
archivo_excel = DATA_DIR / "Operaciones_aéreas_por_ruta_y_aerolínea_—_Colombia_20261015.xlsx"

print(f"Directorio raíz: {BASE_DIR}")
print(f"¿Existe el archivo Excel?: {archivo_excel.exists()}")

# %% -------------------
# PRIMERA SERIE: TRÁFICO AÉREO BOGOTÁ - MEDELLÍN (BOG-MDE)
# ----------------------

# Cargar la base de datos completa desde la carpeta de datos
pasajeros_base = pd.read_excel(archivo_excel, sheet_name="Data")

# Ver el tipo de objeto de la base de datos (Pandas.DataFrame)
print(type(pasajeros_base))

# Ver primeras y últimas observaciones de la base de datos
print(pasajeros_base.head()) # Primeras observaciones
print(pasajeros_base.tail()) # Últimas observaciones
# %%
# %% Creación del índice temporal de las series de tiempo =========================

# 1. Filtrar la ruta comercial BOG-MDE y agrupar el total mensual de pasajeros
pasajeros_bog_mde = (
    pasajeros_base[pasajeros_base['ruta_comercial'] == 'BOG-MDE']
    .groupby('fecha')['pasajeros'] #Es para agrupar por fecha y sumar el total de pasajeros en cada mes
    .sum() #Simplemente se encarga de sumar el total de pasajeros en cada mes
    .to_frame() #Nos da un DataFrame con la columna 'pasajeros' y el índice 'fecha'
)

# 2. Creación del índice temporal con frecuencia mensual (freq="MS")
fechas_pasajeros = pd.date_range(
    start="2017-01-01", #La primera fecha de la base de datos es Enero 2017
    periods=len(pasajeros_bog_mde), #Esto es para que el índice tenga la misma longitud que la base de datos
    freq="MS" #Es para indicarle que la frecuencia es mensual y que el índice debe estar en el primer día de cada mes (MS: Month Start)
)

# 3. Agregar el índice temporal a la base de datos de tráfico aéreo
pasajeros_bog_mde.index = fechas_pasajeros

# Ver el tipo de objeto de la base de datos (Pandas.DataFrame)
print(type(pasajeros_bog_mde))

# Ver primeras y últimas observaciones de la base de datos, ahora con índice temporal
print(pasajeros_bog_mde.head()) # Primeras observaciones (Enero 2017 en adelante)
print(pasajeros_bog_mde.tail()) # Últimas observaciones (Febrero 2026)
# Confirmaciones finales en consola
print(type(pasajeros_bog_mde))
print(f"\nRango muestral: {pasajeros_bog_mde.index.min().strftime('%Y-%m')} a {pasajeros_bog_mde.index.max().strftime('%Y-%m')}")
print(f"Total observaciones (T): {len(pasajeros_bog_mde)}")
print("\nPrimeras observaciones:")
print(pasajeros_bog_mde.head())
print("\nÚltimas observaciones:")
print(pasajeros_bog_mde.tail())

# %% Creación de la serie de tiempo de tráfico de pasajeros =========================

# 1. Extraer la columna de pasajeros como una serie de tiempo univariada
pasajeros_serie = pasajeros_bog_mde["pasajeros"].copy()

# 2. Asegurar la conversión a valores numéricos (limpieza de inconsistencias de formato)
pasajeros_serie = pd.to_numeric(pasajeros_serie, errors="coerce")

# 3. Eliminar posibles datos faltantes (missing values)
pasajeros_serie = pasajeros_serie.dropna()

# 4. Inspeccionar el inicio y final de la serie con su índice de fechas
print("=== Primeras observaciones ===")
print(pasajeros_serie.head())

print("\n=== Últimas observaciones ===")
print(pasajeros_serie.tail())

# 5. Confirmar que el objeto pasó de DataFrame a Pandas Series 
# (Esto porque así se aisla el vector Yt con su índice temporal, lo que facilita el paso a las funciones de autocorrelación)
print("\nTipo de objeto:")
print(type(pasajeros_serie))

# 6. Estadísticas descriptivas fundamentales
print("\n=== Estadísticas Descriptivas de la Serie (Niveles) ===")
print(pasajeros_serie.describe()) #Nos permite saber el número total de observaciones, la media, 
#la desviación estándar, el mínimo, el máximo 
# %% =========================
# Paso 1: Identificación (según Box Jenkins)
# ============================

# Gráfica de la serie de tiempo "Tráfico de pasajeros Bogotá - Medellín (BOG-MDE)"
plt.figure(figsize=(10, 5))
plt.plot(pasajeros_serie, color="navy", linewidth=1.2)
plt.title("Tráfico mensual de pasajeros Bogotá - Medellín (BOG-MDE), 2017-2025")
plt.xlabel("Fecha")
plt.ylabel("Pasajeros")
plt.grid(True, linestyle="--", alpha=0.6)
plt.show()
# %%
