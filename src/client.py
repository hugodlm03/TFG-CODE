# client.py
# Este archivo define la clase NodoCliente, que representa un nodo federado en un sistema de aprendizaje federado.
# Cada nodo:
# - Carga su propio CSV con datos locales.
# - Entrena un modelo de regresión (árbol de decisión) con train/test split.
# - Evalúa el rendimiento (MAE, RMSE) localmente.
# - Guarda las métricas y predicciones en archivos para análisis federado.
#
# Ejecución recomendada desde la raíz del proyecto:
#   python -m src.client
# Alternativa temporal si falla el import:
#   $env:PYTHONPATH="."; python src/client.py

from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from src.utils import preparar_X_y
import pandas as pd
import numpy as np
import json
from pathlib import Path

RANDOM_STATE = 16062025  # Semilla fija para asegurar reproducibilidad

class NodoCliente:
    # Constructor: carga el CSV del nodo y crea el modelo de regresión
    def __init__(self, nombre_nodo: str, ruta_csv: str):                # Inicializa el nodo con su nombre y carga su CSV
        self.nombre = nombre_nodo
        self.df = pd.read_csv(ruta_csv)
        self.model = DecisionTreeRegressor(random_state=RANDOM_STATE)  # Modelo base: árbol de decisión
        self.X_test = None                                              # Se define pero se asigna desde el servidor
        self.y_test = None

    # Entrena el modelo con todos los datos locales (sin split)
    def entrenar_modelo(self):                                          
        X, y = preparar_X_y(self.df)
        self.model.fit(X, y)

    # Asigna el conjunto de test común a todos los nodos
    def asignar_test_comun(self, X_test, y_test):                      
        self.X_test = X_test
        self.y_test = y_test

    # Evalúa el modelo con el test común (MAE y RMSE)
    def evaluar_localmente(self):                                       
        pred = self.model.predict(self.X_test)
        mae = mean_absolute_error(self.y_test, pred)
        rmse = np.sqrt(mean_squared_error(self.y_test, pred))           # sqrt porque sklearn da MSE por defecto
        return {'mae': mae, 'rmse': rmse}

    # Realiza predicciones sobre un conjunto de entrada dado
    def predecir(self, X_test):                                        
        return self.model.predict(X_test)

    # Guarda las métricas locales en archivo JSON
    def guardar_resultados(self, ruta="resultados"):                   
        Path(ruta).mkdir(exist_ok=True)
        resultados = self.evaluar_localmente()
        nombre_archivo = self.nombre.replace(' ', '_').replace('/', '_') + ".json"
        with open(Path(ruta) / nombre_archivo, "w") as f:
            json.dump(resultados, f, indent=2)

    # Devuelve el modelo entrenado (por si se usa externamente)
    def obtener_modelo(self):                                          
        return self.model

    # Devuelve el test común asignado (X_test, y_test)
    def get_test_data(self):                                           
        return self.X_test, self.y_test

    # Devuelve el nombre del nodo (por claridad)
    def get_nombre(self):                                              
        return self.nombre

    # Guarda las predicciones del modelo local junto a los valores reales
    def guardar_predicciones(self, ruta="predicciones_locales"):       
        Path(ruta).mkdir(exist_ok=True)
        pred = self.model.predict(self.X_test)
        nombre_archivo = self.nombre.replace(' ', '_').replace('/', '_') + ".csv"
        df_pred = pd.DataFrame({
            'y_real': self.y_test,
            'y_pred_local': pred
        })
        df_pred.to_csv(Path(ruta) / nombre_archivo, index=False)
        print(f" Predicciones guardadas en: {Path(ruta) / nombre_archivo}")
