# compare.py
# Script para visualizar y comparar el rendimiento de distintos esquemas de partición federada.
#
# Carga:
# - Métricas locales (local_metrics.csv) de cada carpeta de resultados
# - Métricas globales (global_metrics.json) correspondientes
#
# Muestra:
# - MAE por nodo para cada esquema (barras)
# - MAE global como línea discontinua por esquema
#
# Ejecutar desde raíz del proyecto:
#   python -m src.compare

import pandas as pd
import matplotlib.pyplot as plt
import json

# Diccionario: nombre visible → carpeta de resultados
esquemas = {
    "Retailer + Region": "resultados_retailer_region",
    "Retailer + City": "resultados_retailer_city",
    "Region": "resultados_region",
}

# Construimos el DataFrame unificado
df_comparativa = []

for nombre_esquema, carpeta in esquemas.items():
    try:
        # Cargar métricas locales
        df_local = pd.read_csv(f"{carpeta}/local_metrics.csv")
        df_local["esquema"] = nombre_esquema

        # Cargar métrica global
        with open(f"{carpeta}/global_metrics.json") as f:
            global_mae = json.load(f)["mae"]

        df_local["mae_global"] = global_mae
        df_comparativa.append(df_local)
    except FileNotFoundError:
        print(f"[!] No se encontró '{carpeta}', se salta.")
        continue

# Unir todo en un solo DataFrame
df = pd.concat(df_comparativa, ignore_index=True)

# Gráfico
plt.figure(figsize=(14, 6))
for esquema in df["esquema"].unique():
    subset = df[df["esquema"] == esquema]
    plt.bar(subset["nodo"], subset["mae"], label=esquema)

# Líneas horizontales para cada esquema
for esquema in df["esquema"].unique():
    valor = df[df["esquema"] == esquema]["mae_global"].iloc[0]
    plt.axhline(valor, linestyle="--", label=f"Global MAE ({esquema})")

plt.xticks(rotation=90, fontsize=6)
plt.ylabel("MAE")
plt.title("MAE local por nodo y esquema vs MAE global federado")
plt.legend()
plt.tight_layout()
plt.show()

# === GRÁFICO RESUMEN (MAE local medio vs MAE global) ===

# Calcular resumen por esquema
resumen = (
    df.groupby("esquema")
    .agg(mae_local_mean=("mae", "mean"), mae_global=("mae_global", "first"))
    .reset_index()
)

# Gráfico
plt.figure(figsize=(6, 5))
plt.bar(resumen["esquema"], resumen["mae_local_mean"], color="skyblue", label="MAE local medio")
plt.plot(resumen["esquema"], resumen["mae_global"], "k--o", label="MAE global federado")

plt.ylabel("MAE")
plt.title("Comparación de esquemas: MAE local medio vs global")
plt.legend()
plt.tight_layout()
plt.show()

# === GRÁFICO RESUMEN (RMSE y R²) ===
resumen_rmse_r2 = (
    df.groupby("esquema")
    .agg(
        rmse_local_mean=("rmse", "mean"),
        r2_local_mean=("r2", "mean"),
        rmse_global=("rmse", "first"),  # opcional si lo incluyes en el JSON
        r2_global=("r2", "first"),      # opcional si lo incluyes en el JSON
    )
    .reset_index()
)

# RMSE
plt.figure(figsize=(6, 5))
plt.bar(resumen_rmse_r2["esquema"], resumen_rmse_r2["rmse_local_mean"], color="orange", label="RMSE local medio")
plt.title("Comparación de esquemas: RMSE local medio")
plt.ylabel("RMSE")
plt.tight_layout()
plt.show()

# R²
plt.figure(figsize=(6, 5))
plt.bar(resumen_rmse_r2["esquema"], resumen_rmse_r2["r2_local_mean"], color="mediumseagreen", label="R² local medio")
plt.axhline(0, color="black", linewidth=1)  # para referencia
plt.title("Comparación de esquemas: R² local medio")
plt.ylabel("R²")
plt.tight_layout()
plt.show()

# Guardar resumen de MAE en CSV
resumen_mae = (
    df.groupby("esquema")
    .agg(mae_local_mean=("mae", "mean"), mae_global=("mae_global", "first"))
    .reset_index()
)
resumen_mae.to_csv("resultados_comparacion_mae.csv", index=False)
print("✓ Guardado resumen de MAE en 'resultados_comparacion_mae.csv'")
