# compare_flower.py
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Ruta a la carpeta con métricas de clientes Flower
CARPETA_METRICAS = Path("metricas_locales")

# Recolectar métricas
datos = []
for archivo in CARPETA_METRICAS.glob("*.json"):
    with open(archivo, "r") as f:
        metricas = json.load(f)
    nodo = archivo.stem.replace("_", " ")
    datos.append({
        "nodo": nodo,
        "mae": metricas.get("mae"),
        "rmse": metricas.get("rmse"),
        "r2": metricas.get("r2")
    })

# Crear DataFrame
df = pd.DataFrame(datos)
df = df.sort_values(by="mae").reset_index(drop=True)

# Guardar CSV resumen
df.to_csv("resultados_flower_metricas.csv", index=False)
print("✓ Guardado resumen de métricas en 'resultados_flower_metricas.csv'")

# Gráfico de MAE por nodo
plt.figure(figsize=(14, 6))
plt.bar(df["nodo"], df["mae"], color="skyblue")
plt.xticks(rotation=90, fontsize=7)
plt.ylabel("MAE")
plt.title("MAE por nodo federado (Flower)")
plt.tight_layout()
plt.show()

# Gráfico de RMSE
plt.figure(figsize=(14, 6))
plt.bar(df["nodo"], df["rmse"], color="orange")
plt.xticks(rotation=90, fontsize=7)
plt.ylabel("RMSE")
plt.title("RMSE por nodo federado (Flower)")
plt.tight_layout()
plt.show()

# Gráfico de R²
plt.figure(figsize=(14, 6))
plt.bar(df["nodo"], df["r2"], color="mediumseagreen")
plt.axhline(0, color="black", linewidth=1)
plt.xticks(rotation=90, fontsize=7)
plt.ylabel("R²")
plt.title("R² por nodo federado (Flower)")
plt.tight_layout()
plt.show()
