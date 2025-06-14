# -----------------------------------------------------------------------------
# Configuración del proyecto
# -----------------------------------------------------------------------------
from __future__ import annotations
from pathlib import Path
import argparse
import json
import pickle

import flwr as fl
import numpy as np
import xgboost as xgb
from flwr.common import Code, EvaluateRes, FitRes, Parameters, Status
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
from flwr.client import Client  # typed alias
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.task_adidas import train_test_split_dmatrix, dataframe_to_dmatrix
from src.utils import preparar_X_y

# -----------------------------------------------------------------------------
# Documentación del cliente
# -----------------------------------------------------------------------------

"""Cliente Flower para XGBoost‑Bagging en modo *regresión*.

Este cliente entrena un modelo XGBoost en modo *bagging* para la tarea de
regresión, usando un CSV local pre‑particionado por nodo federado.

El cliente sigue el esquema de entrenamiento federado:
* Cada cliente representa un nodo federado (ej. Retailer + Ciudad).
* Cada nodo carga su propio CSV (pre‑particionado en `nodos/.../*.csv`).
* El CSV se transforma en `xgb.DMatrix` usando los *helpers* de ``src.task_adidas``.
* Objetivo: ``reg:squarederror`` y métrica ``rmse``.
* Estrategia: en la **primera ronda** entrena `num_local_round` árboles desde cero;
  en rondas posteriores **carga el modelo global** y añade la misma
  cantidad de árboles.
* Devuelve al servidor **solo** los nuevos árboles (`bst.save_raw("json")`).
"""

# -----------------------------------------------------------------------------
# Config global
# -----------------------------------------------------------------------------
RANDOM_STATE = 16062025
NUM_LOCAL_ROUND = 1  # número de árboles que añade cada cliente por ronda

PARAMS = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "eta": 0.1,
    "max_depth": 8,
    "nthread": 16,
    "subsample": 1.0,
    "tree_method": "hist",  # usa CPU hist; cámbialo a gpu_hist si tienes CUDA.
}

# carpeta donde guardamos métricas locales → útiles para compare_xgb.py
# Carpeta base → metricas/xgb_bagging/*.json
METRIC_DIR = Path("metricas") / "xgb_bagging"
METRIC_DIR.mkdir(parents=True, exist_ok=True) # Asegura que la carpeta existe
METRIC_DIR.mkdir(exist_ok=True) # Crea la carpeta si no existe

# -----------------------------------------------------------------------------
# Flower Client implementation
# -----------------------------------------------------------------------------
class XgbClient(fl.client.Client):
    """Cliente que entrena/actualiza un modelo XGBoost y reporta métricas."""

    def __init__(self, csv_path: Path, node_name: str):
        self.node_name = node_name
        self.csv_path = csv_path

        # Cargar datos
        import pandas as pd

        df = pd.read_csv(csv_path)
        (
            self.dtrain,
            self.dvalid,
            self.num_train,
            self.num_val,
        ) = train_test_split_dmatrix(df)

    # --------------------------- fit --------------------------- #
    def fit(self, ins: fl.common.FitIns) -> FitRes:
        """Entrena desde cero o continúa el modelo global y devuelve
        **solo** los nuevos árboles en formato ndarray(uint8)."""
        global_round = int(ins.config.get("global_round", 1))

        if global_round == 1:
            # 1) entrenamiento inicial
            bst = xgb.train(
                PARAMS,
                self.dtrain,
                num_boost_round=NUM_LOCAL_ROUND,
                evals=[(self.dvalid, "valid"), (self.dtrain, "train")],
                verbose_eval=False,
            )
        else:
            # 1) cargamos el modelo global recibido
            bst = xgb.Booster(params=PARAMS)
            global_arrs = parameters_to_ndarrays(ins.parameters)
            if global_arrs:
                bst.load_model(bytearray(global_arrs[0].tobytes()))
            # 2) añadimos árboles locales (bagging)
            bst = self._local_boost(bst)

        # 3) serializamos los NUEVOS árboles → bytes → ndarray(uint8)
        json_bytes = bst.save_raw("json")           # bytes
        local_model_arr = np.frombuffer(json_bytes, dtype=np.uint8)

        return FitRes(
            status=Status(code=Code.OK, message="OK"),
            parameters=ndarrays_to_parameters([local_model_arr]),
            num_examples=self.num_train,
            metrics={},
        )

    # ------------------------- evaluate ------------------------ #
    def evaluate(self, ins: fl.common.EvaluateIns) -> EvaluateRes:
        """Evalúa el modelo global en la validación local y reporta RMSE, MAE, R²."""
        # 1) reconstruimos el Booster con los bytes recibidos
        bst = xgb.Booster(params=PARAMS)
        global_arrs = parameters_to_ndarrays(ins.parameters)
        if global_arrs:
            bst.load_model(bytearray(global_arrs[0].tobytes()))

        # 2) cálculo de métricas
        eval_str = bst.eval_set(
            [(self.dvalid, "valid")],
            iteration=bst.num_boosted_rounds() - 1,
        )  # ej. "[valid]\trmse:0.52341"
        rmse_val = float(eval_str.split(":")[1])

        preds = bst.predict(self.dvalid)
        y_true = self.dvalid.get_label()
        mae = mean_absolute_error(y_true, preds)
        r2 = r2_score(y_true, preds)

        # 3) guardado opcional de métricas
        METRIC_DIR.mkdir(exist_ok=True)
        with open(METRIC_DIR / f"{self.node_name.replace(' ', '_')}.json", "w") as f:
            json.dump({"rmse": rmse_val, "mae": mae, "r2": r2}, f, indent=2)

        return EvaluateRes(
            status=Status(code=Code.OK, message="OK"),
            loss=rmse_val,
            num_examples=self.num_val,
            metrics={"RMSE": round(rmse_val, 4), "MAE": mae, "R2": r2},
        )

    # ---------------------- helpers internos -------------------- #
    def _local_boost(self, bst_input: xgb.Booster) -> xgb.Booster:
        """Añade `NUM_LOCAL_ROUND` árboles nuevos al *Booster*
        y devuelve **solo** esos árboles (bagging)."""
        for _ in range(NUM_LOCAL_ROUND):
            bst_input.update(self.dtrain, bst_input.num_boosted_rounds())

        start = bst_input.num_boosted_rounds() - NUM_LOCAL_ROUND
        end = bst_input.num_boosted_rounds()
        return bst_input[start:end]

# -----------------------------------------------------------------------------
# CLI entry point
# -----------------------------------------------------------------------------

def main() -> None:  # pragma: no cover
    parser = argparse.ArgumentParser("Cliente XGBoost‑Bagging (regresión)")
    parser.add_argument("--csv", type=Path, required=True, help="Ruta al CSV local del nodo")
    parser.add_argument("--name", type=str, required=True, help="Nombre del nodo (para métricas)")
    parser.add_argument("--server", type=str, default="127.0.0.1:8087", help="Dirección servidor Flower")
    args = parser.parse_args()

    client: Client = XgbClient(args.csv, args.name)
    fl.client.start_client(server_address=args.server, client=client)


if __name__ == "__main__":  # pragma: no cover
    main()
