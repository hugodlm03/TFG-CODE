from __future__ import annotations

"""Servidor Flower para XGBoost-Bagging (regresión).

* Arranca la estrategia `FedXgbBagging` de Flower.
* Agrega árboles concatenando boosters (bagging) y promedia la **RMSE** que
  devuelven los clientes.
* CLI flags:
    --server   IP:PUERTO donde escuchar (default 127.0.0.1:8087)
    --rounds   Nº de rondas federadas (default 5)
    --csv-dir  Carpeta con los CSV de los nodos (default nodos/retailer_region)
"""

from pathlib import Path
import argparse
from typing import Dict, List, Tuple

import flwr as fl
from flwr.common import EvaluateRes, FitRes, Parameters, Scalar
from flwr.server.client_proxy import ClientProxy
from flwr.common.logger import log
import logging

# -----------------------------------------------------------------------------
# Estrategia personalizada (adaptada a regresión → RMSE)
# -----------------------------------------------------------------------------

try:
    from flwr.server.strategy.fedxgb_bagging import FedXgbBagging
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "Tu versión de Flower no incluye FedXgbBagging. Actualiza → pip install -U flwr>=1.18"
    ) from e


def evaluate_metrics_aggregation(eval_metrics: List[Tuple[int, Dict[str, Scalar]]]):
    """Pondera RMSE por nº de ejemplos."""
    total = sum(num for num, _ in eval_metrics)
    rmse = sum(float(m["RMSE"]) * num for num, m in eval_metrics) / total
    return {"RMSE": rmse}


def config_func(rnd: int) -> Dict[str, str]:
    return {"global_round": str(rnd)}


# -----------------------------------------------------------------------------
# Función principal
# -----------------------------------------------------------------------------

def start_server(address: str, rounds: int, csv_dir: Path) -> None:
    csv_files = list(csv_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No se encontraron CSV en {csv_dir}. Genera las particiones primero.")

    total = len(csv_files)
    logging.info(f"Iniciando servidor con {total} clientes esperados…")

    strategy = FedXgbBagging(
        fraction_fit=1.0,              # todos los clientes entrenan en cada ronda
        fraction_evaluate=1.0,         # todos evalúan
        min_available_clients = total,
        min_fit_clients       = total,
        min_evaluate_clients  = total,
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation,
        on_evaluate_config_fn=config_func,
        on_fit_config_fn=config_func,
        initial_parameters=None,
    )

    fl.server.start_server(
        server_address=address,
        config=fl.server.ServerConfig(num_rounds=rounds, round_timeout=600),
        strategy=strategy,
    )


# -----------------------------------------------------------------------------
# CLI helper
# -----------------------------------------------------------------------------

def _cli() -> None:  # pragma: no cover
    parser = argparse.ArgumentParser("XGBoost-Bagging Server (regresión)")
    parser.add_argument("--server", default="127.0.0.1:8087", help="IP:PUERTO donde escuchar")
    parser.add_argument("--rounds", type=int, default=5, help="Nº de rondas federadas")
    parser.add_argument(
        "--csv-dir",
        default=str(Path("nodos") / "retailer_region"),
        help="Directorio con los CSV de los nodos",
    )
    args = parser.parse_args()

    start_server(address=args.server, rounds=args.rounds, csv_dir=Path(args.csv_dir))


if __name__ == "__main__":  # pragma: no cover
    _cli()