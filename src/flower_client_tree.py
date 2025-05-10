import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
import flwr as fl
import pickle  
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.utils import preparar_X_y

RANDOM_STATE = 16062025  # Mismo seed del resto del proyecto

# Guarda un JSON con las métricas de este cliente para auditoría posterior.
def save_local_metrics(node_name: str, metrics: dict, output_dir: str = "metricas_locales") -> None:
    Path(output_dir).mkdir(exist_ok=True)
    fname = node_name.replace(" ", "_").replace("/", "_") + ".json"
    with open(Path(output_dir) / fname, "w") as f:
        json.dump(metrics, f, indent=2)

# Cliente Flower basado en DecisionTreeRegressor (un único árbol).
class TreeClient(fl.client.NumPyClient):
    def __init__(self, csv_path: Path, node_name: str):
        self.node_name = node_name
        self.df = pd.read_csv(csv_path)
        self.X, self.y = preparar_X_y(self.df)
        self.model = DecisionTreeRegressor(random_state=RANDOM_STATE)

    #  Métodos requeridos por NumPyClient 
    def get_parameters(self, config):  # noqa: D401
        """Devuelve la lista de parámetros del modelo.

        Para modelos de sklearn árboles de decisión no hay *weights* numéricos
        intercambiables de forma sencilla; devolvemos lista vacía para que el
        servidor ignore la agregación en FedAvg (será puramente *majority vote*
        de métricas, no de pesos).
        """
        return []
    # Entrena el árbol localmente y devuelve (params, n_samples, info).
    def fit(self, parameters, config):  # noqa: D401
        self.model.fit(self.X, self.y)
        # No enviamos parámetros => lista vacía
        return [], len(self.y), {}
    # Evalúa el modelo local usando el test común que envía el servidor.
    def evaluate(self, parameters, config):
        """Evalúa con el test que llegue desde el servidor. Si no llega, avisa."""
        # Asegura que el modelo está entrenado por si el server llama antes de .fit()
        if not hasattr(self.model, "tree_"):
            self.model.fit(self.X, self.y)

        # ➋ recibir bytes y des-serializar
        X_bytes = config.get("X_test")
        y_bytes = config.get("y_test")
        if X_bytes is None or y_bytes is None:
            raise ValueError("El servidor no envió X_test / y_test")
        X_test = pickle.loads(X_bytes)
        y_test = pickle.loads(y_bytes)

        # ➌ predicciones y métricas
        preds = self.model.predict(X_test)
        mae  = mean_absolute_error(y_test, preds)
        mse  = mean_squared_error(y_test, preds)
        rmse = np.sqrt(mse)
        r2   = r2_score(y_test, preds)

        metrics = {"mae": mae, "rmse": rmse, "r2": r2}
        save_local_metrics(self.node_name, metrics)

        return float(mae), len(y_test), metrics


#  Entry point CLI para el cliente Flower.
def main() -> None:
    parser = argparse.ArgumentParser(description="Cliente Flower (<árbol de decisión>)")
    parser.add_argument("--csv", type=Path, required=True, help="Ruta al CSV local del nodo")
    parser.add_argument("--name", type=str, required=True, help="Nombre del nodo (para métricas)")
    parser.add_argument("--server", type=str, default="127.0.0.1:8080", help="Dirección del servidor Flower")

    args = parser.parse_args()

    client = TreeClient(args.csv, args.name)
    # Arrancar cliente y bloquear hasta fin de la simulación
    fl.client.start_client(server_address=args.server, client=client.to_client())


if __name__ == "__main__":
    main()
