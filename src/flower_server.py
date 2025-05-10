import argparse
import logging
from pathlib import Path
from typing import Optional
import pickle

import flwr as fl
from sklearn.model_selection import train_test_split

from src.data_loader import load_clean_adidas_data
from src.utils import preparar_X_y

# ----------------------------------------------------------------------------------
# Configuración y datos globales ----------------------------------------------------
# ----------------------------------------------------------------------------------
RANDOM_STATE = 16062025
DATA_PATH = Path("datos/Adidas US Sales Datasets.xlsx")

# Carga y prepara el conjunto de test común (≈20 % del total)
_df = load_clean_adidas_data(DATA_PATH)
X_full, y_full = preparar_X_y(_df)
_, X_test_glb, _, y_test_glb = train_test_split(
    X_full, y_full, test_size=0.2, random_state=RANDOM_STATE
)
# Serializa para enviar por la red
X_TEST_BYTES = pickle.dumps(X_test_glb)
Y_TEST_BYTES = pickle.dumps(y_test_glb)

# ----------------------------------------------------------------------------------
# Estrategia personalizada que adjunta el test a EvaluateIns ------------------------
# ----------------------------------------------------------------------------------

class FedAvgWithGlobalTest(fl.server.strategy.FedAvg):


    def configure_evaluate(self, server_round, parameters, client_manager):
        eval_cfg = super().configure_evaluate(
            server_round, parameters, client_manager
        )
        # eval_cfg → List[Tuple[ClientProxy, EvaluateIns]]
        for _, evaluate_ins in eval_cfg:            # <- desempaquetamos
            evaluate_ins.config["X_test"] = X_TEST_BYTES
            evaluate_ins.config["y_test"] = Y_TEST_BYTES
        return eval_cfg

# ----------------------------------------------------------------------------------
# Función para arrancar el servidor -------------------------------------------------
# ----------------------------------------------------------------------------------

def start_server(address: str, num_rounds: int = 1, log_level: str = "INFO") -> None:
    logging.basicConfig(level=getattr(logging, log_level))

    base_dir = Path(__file__).resolve().parent      # carpeta del script
    csv_files = list((base_dir / "nodos").glob("*.csv"))
    total      = len(csv_files)                    

    strategy = FedAvgWithGlobalTest(
        min_available_clients=28,# <- número de nodos disponibles
        min_fit_clients=28,         # <- número de nodos para entrenar
        min_evaluate_clients=28,     # <- número de nodos para evaluar
    )


    fl.server.start_server(
        server_address = address,
        config  = fl.server.ServerConfig(num_rounds=num_rounds, round_timeout=300),
        strategy = strategy,
    )


# ----------------------------------------------------------------------------------
# CLI helper ------------------------------------------------------------------------
# ----------------------------------------------------------------------------------

def _cli(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        prog="flower_server.py",
        description="Servidor Flower (FedAvg + test global) para TFG",
    )
    parser.add_argument("--server", default="127.0.0.1:8085", help="IP:PUERTO para escuchar")
    parser.add_argument("--rounds", type=int, default=1, help="Nº de rondas (default 1)")
    parser.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Nivel de logging"
    )
    args = parser.parse_args(argv)
    start_server(address=args.server, num_rounds=args.rounds, log_level=args.log_level)


if __name__ == "__main__":
    _cli()
