# -----------------------------------------------------------------------------
# Configuración del proyecto
# -----------------------------------------------------------------------------

from pathlib import Path
from typing import Tuple

import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split

from src.data_loader import load_clean_adidas_data # De aqui obtenemos el DataFrame limpio.
from src.utils import preparar_X_y  # Función para separar X e y del DataFrame          

# Semilla para reproducibilidad en todo el proyecto
RANDOM_STATE = 16062025       

# -----------------------------------------------------------------------------
# Funciones auxiliares para convertir el CSV/Excel de Adidas en DMatrices
# -----------------------------------------------------------------------------

def train_test_split_dmatrix(
    df: pd.DataFrame,
    *,
    test_fraction: float = 0.2,
    random_state: int = RANDOM_STATE,
) -> Tuple[xgb.DMatrix, xgb.DMatrix, int, int]:
    """Divide *df* y devuelve dos DMatrices + sus tamaños.

    Parameters
    ----------
    df : pd.DataFrame
        Datos limpios de un nodo federado.
    test_fraction : float, default=0.2
        Proporción para el *hold‑out* de validación local.
    random_state : int
        Seed para la aleatoriedad reproducible.

    Returns
    -------
    train_dmatrix : xgb.DMatrix
    valid_dmatrix : xgb.DMatrix
    num_train : int
    num_valid : int
    """
    X, y = preparar_X_y(df)
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=test_fraction, random_state=random_state
    )

    train_dmatrix = xgb.DMatrix(X_train, label=y_train)
    valid_dmatrix = xgb.DMatrix(X_valid, label=y_valid)

    return train_dmatrix, valid_dmatrix, len(y_train), len(y_valid)


def dataframe_to_dmatrix(df: pd.DataFrame) -> xgb.DMatrix:
    """Convierte el *DataFrame* completo en un `xgb.DMatrix`.
    
    Esto lo hago, porque en el ejemplo del tutorial de XGBoost
    se usa un `xgb.DMatrix` ya que tiene ventajas como carga más rápida
    y optimización de memoria.
    
    """
    X, y = preparar_X_y(df)
    return xgb.DMatrix(X, label=y)


# Script rápido de prueba
if __name__ == "__main__":
    # Sólo para pruebas locales: python -m src.task_adidas
    datos_path = Path(__file__).resolve().parent.parent / "datos" / "Adidas US Sales Datasets.xlsx"
    df_full = load_clean_adidas_data(datos_path)

    dtrain, dvalid, n_tr, n_val = train_test_split_dmatrix(df_full)
    print("Train shape:", dtrain.num_row(), "Valid shape:", dvalid.num_row())
