#Librerías necesarias 
import pandas as pd
from pathlib import Path
import sys
# COMANDO PARA COMENZAR EL ENTORNO.\fedenv\Scripts\Activate 

# FUNCIÓN: Cargar y limpiar datos 
def load_clean_adidas_data(ruta_excel: str | Path) -> pd.DataFrame:
    """
    Carga y limpia el conjunto de datos de Adidas desde un archivo Excel.

    Parámetros:
        ruta_excel (str | Path): Ruta al archivo .xlsx.

    Retorna:
        pd.DataFrame: DataFrame limpio con tipos bien definidos.
    """

    # Leer archivo
    df = pd.read_excel(ruta_excel)

    # Eliminar primeras filas vacías y convertir fila 3 en header
    df = df.drop([0, 1, 2])
    columnas = df.loc[3]
    df = df.rename(columns=columnas.to_dict())
    df = df.drop(df.columns[0], axis=1)
    df = df.drop(index=3).reset_index(drop=True)

    # Convertir fecha y eliminar valores nulos
    df = df[pd.to_datetime(df['Invoice Date'], errors='coerce').notna()]
    df['Invoice Date'] = pd.to_datetime(df['Invoice Date'])

    # === Tipado explícito de columnas ===
    variables_categoricas = ['Retailer', 'Region', 'State', 'City', 'Product', 'Sales Method']
    variables_numericas = ['Price per Unit', 'Total Sales', 'Operating Profit', 'Operating Margin']
    variable_objetivo = 'Units Sold'

    for col in variables_categoricas:
        df[col] = df[col].astype('category')

    for col in variables_numericas:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df[variable_objetivo] = pd.to_numeric(df[variable_objetivo], errors='coerce')

    return df

# GRUPO DE FUNCIONES PARA PARTICIONAR EL DATAFRAME
# FUNCIÓN: Particionar por nodo federado (Retailer + Región)
def partition_by_retailer_region(df: pd.DataFrame) -> dict:
    """
    Divide el DataFrame en subconjuntos por combinación de Retailer y Región.

    Cada subconjunto representa un cliente (nodo) del sistema federado.

    Parámetros:
        df (pd.DataFrame): DataFrame completo de Adidas.

    Retorna:
        dict[str, pd.DataFrame]: Diccionario con claves tipo 'Retailer - Región' y DataFrames.
    """

    # Agrupar por nodos federados (cliente lógico)
    grupos = df.groupby(['Retailer', 'Region'], observed=True)  # Evita el FutureWarning

    particiones = {}

    for (retailer, region), sub_df in grupos:
        clave = f"{retailer} - {region}"
        particiones[clave] = sub_df.reset_index(drop=True)

    return particiones
# FUNCIÓN: Particionar por nodo federado (Retailer + City)
def partition_by_retailer_city(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Divide el DataFrame en subconjuntos por combinación de Retailer y City.

    Cada subconjunto representa un nodo federado lógico.
    """
    grupos = df.groupby(['Retailer', 'City'], observed=True)  # Agrupar por retailer + ciudad
    particiones = {}

    for (retailer, city), sub_df in grupos:
        clave = f"{retailer} - {city}"                        # Nombre del nodo
        particiones[clave] = sub_df.reset_index(drop=True)   # Guardar copia limpia del subgrupo

    return particiones
# FUNCIÓN: Particionar por nodo federado (REGION)
def partition_by_region(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Divide el DataFrame en subconjuntos por Region únicamente.

    Cada subconjunto representa un nodo federado lógico.
    """
    grupos = df.groupby('Region', observed=True)              # Agrupar por región
    particiones = {}

    for region, sub_df in grupos:
        clave = region                                        # Clave del nodo (región)
        particiones[clave] = sub_df.reset_index(drop=True)

    return particiones

#  FUNCIÓN: Guardar csv por nodo federado (Retailer + Región) 
def guardar_nodos_en_csv(particiones, carpeta_salida: Path | str):
    carpeta_salida = Path(carpeta_salida)
    carpeta_salida.mkdir(parents=True, exist_ok=True)

    for nombre, df_nodo in particiones.items():
        filename = f"{nombre.replace(' ', '_').replace('/', '_')}.csv"
        df_nodo.to_csv(Path(carpeta_salida) / filename, index=False)

if __name__ == "__main__":
    import sys

    # Leer esquema desde línea de comandos
    if len(sys.argv) > 1:
        scheme = sys.argv[1]
    else:
        scheme = "retailer_region"

    ruta_excel = "datos/Adidas US Sales Datasets.xlsx"
    df = load_clean_adidas_data(ruta_excel)

    # Elegir función y subcarpeta bajo nodos/
    base_dir = Path("nodos")            # carpeta raíz común
    if scheme == "retailer_city":
        particiones = partition_by_retailer_city(df)
        subdir = base_dir / "retailer_city"
    elif scheme == "region":
        particiones = partition_by_region(df)
        subdir = base_dir / "region"
    elif scheme == "retailer_region":
        particiones = partition_by_retailer_region(df)
        subdir = base_dir / "retailer_region"
    else:
        raise ValueError(f"Esquema no reconocido: {scheme}")


    # Mostrar muestra de 3 nodos
    for nombre, df_nodo in list(particiones.items())[:3]:
        print(f"\n--- {nombre} ({len(df_nodo)} registros) ---")
        print(df_nodo.head())

    # Guardar CSVs
    guardar_nodos_en_csv(particiones, subdir)
    print(f"✓ Particiones '{scheme}' guardadas en '{subdir}/'")

