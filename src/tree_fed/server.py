# server.py
# Entrenamiento federado simulado con agregación centralizada.
# - Carga los datos de test común.
# - Carga y entrena nodos a partir de CSVs (según esquema de partición).
# - Evalúa cada nodo localmente (MAE, RMSE, R²).
# - Calcula modelo global federado con agregación por media simple.
# - Guarda métricas locales y globales.
#
# Opciones:
#   --scheme       → tipo de partición: 'retailer_region', 'retailer_city' o 'region'
#   --num_nodos    → limitar el número de nodos cargados (opcional)
# Ejecutar federado sobre Retailer + City
# python -m src.server --scheme retailer_city
# Ejecutar federado sobre Region
# python -m src.server --scheme region
# Esquema clásico (Retailer + Region)
# python -m src.server --scheme retailer_region


import argparse
from src.tree_fed.client import NodoCliente, RANDOM_STATE
from src.data_loader import load_clean_adidas_data
from src.utils import preparar_X_y
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import pandas as pd
import json

# Argumentos de línea de comandos
parser = argparse.ArgumentParser()
parser.add_argument(
    "--scheme",
    type=str,
    default="retailer_region",
    choices=["retailer_region", "retailer_city", "region"],
    help="Esquema de partición: carpeta de nodos a usar"
)
parser.add_argument(
    "--num_nodos",
    type=int,
    default=None,
    help="Número de nodos a usar (por defecto todos)"
)

args = parser.parse_args()

# Prepara el test común a partir del Excel completo
df_global = load_clean_adidas_data("datos/Adidas US Sales Datasets.xlsx")
X_global, y_global = preparar_X_y(df_global)
_, X_test_global, _, y_test_global = train_test_split(
    X_global, y_global,
    test_size=0.2,
    random_state=RANDOM_STATE
)

# Descubre todos los CSV en la carpeta nodos/
if args.scheme == "retailer_city":
    BASE_DIR = Path("nodos_retailer_city")
    
elif args.scheme == "region":
    BASE_DIR = Path("nodos_region")
else:
    BASE_DIR = Path("nodos")
todos_csv = sorted(BASE_DIR.glob("*.csv"))

# Carpeta de salida según esquema
CARPETA_RESULTADOS = Path(f"resultados_{args.scheme}")
CARPETA_PREDICCIONES = Path(f"predicciones_{args.scheme}")
CARPETA_RESULTADOS.mkdir(exist_ok=True)
CARPETA_PREDICCIONES.mkdir(exist_ok=True)

# Selecciona solo los primeros N si se indica --num_nodos
if args.num_nodos:
    archivos_csv = todos_csv[: args.num_nodos]
else:
    archivos_csv = todos_csv

# Nombres limpios de nodo
nombres_nodos = [f.stem.replace("_", " ").replace("  ", " ") for f in archivos_csv]

# Carpeta de resultados locales y predicciones
Path("resultados").mkdir(exist_ok=True)
Path("predicciones_locales").mkdir(exist_ok=True)

# Carga, asigna test común, entrena y guarda resultados por nodo
clientes = []
for archivo, nombre in zip(archivos_csv, nombres_nodos):
    nodo = NodoCliente(nombre, archivo)
    nodo.asignar_test_comun(X_test_global, y_test_global)
    nodo.entrenar_modelo()

    # Guarda métricas locales y predicciones
    nodo.guardar_resultados(ruta=CARPETA_RESULTADOS)
    nodo.guardar_predicciones(ruta=CARPETA_PREDICCIONES)


    clientes.append(nodo)

# Evalúa localmente cada nodo y calcula MAE, RMSE y R²
maes, rmses, r2s = [], [], []
for nodo in clientes:
    m = nodo.evaluar_localmente()
    maes.append(m['mae'])
    rmses.append(m['rmse'])
    pred = nodo.predecir(X_test_global)
    r2s.append(r2_score(y_test_global, pred))

# Guarda las métricas locales agregadas en CSV
df_local = pd.DataFrame({
    'nodo': nombres_nodos,
    'mae': maes,
    'rmse': rmses,
    'r2': r2s
})
df_local.to_csv(CARPETA_RESULTADOS / "local_metrics.csv", index=False)
print(" Métricas locales guardadas en resultados/local_metrics.csv")

# Ponderación por tamaño de nodo
tamanios = [len(n.df) for n in clientes]
pesos = np.array(tamanios) / np.sum(tamanios)

# Apilar predicciones
preds = np.vstack([n.predecir(X_test_global) for n in clientes])

# Media ponderada
pred_global = np.average(preds, axis=0, weights=pesos)

# Guardar pesos por nodo
with open(CARPETA_RESULTADOS / "pesos_nodos.json", "w") as f:
    json.dump(dict(zip(nombres_nodos, pesos.round(5).tolist())), f, indent=2)


# Calcula métricas globales
mae_global  = mean_absolute_error(y_test_global, pred_global)
rmse_global = np.sqrt(mean_squared_error(y_test_global, pred_global))
r2_global   = r2_score(y_test_global, pred_global)

# Guarda las métricas globales en JSON
global_metrics = {
    'nodos': len(clientes),
    'mae': mae_global,
    'rmse': rmse_global,
    'r2': r2_global
}
with open(CARPETA_RESULTADOS / "global_metrics.json", "w") as f:
    json.dump(global_metrics, f, indent=2)
print(" Métricas globales guardadas en resultados/global_metrics.json")

# Muestra resumen por consola
print("\nResumen local (mean / min / max):")
print(f"MAE  → mean: {np.mean(maes):.2f}, min: {np.min(maes):.2f}, max: {np.max(maes):.2f}")
print(f"RMSE → mean: {np.mean(rmses):.2f}, min: {np.min(rmses):.2f}, max: {np.max(rmses):.2f}")
print(f"R²   → mean: {np.mean(r2s):.2f}, min: {np.min(r2s):.2f}, max: {np.max(r2s):.2f}")

print("\nEvaluación global federado:")
print(f"MAE  : {mae_global:.2f}")
print(f"RMSE : {rmse_global:.2f}")
print(f"R²   : {r2_global:.2f}")

