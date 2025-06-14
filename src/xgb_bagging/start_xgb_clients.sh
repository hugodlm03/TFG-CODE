#!/usr/bin/env bash
# ------------------------------------------------------------------
# Lanza en background todos los clientes XGBoost-Bagging encontrados
# en el directorio de CSV que se le pase como argumento.
#
# Uso:
#   ./start_xgb_clients.sh [CARPETA_CSV] [IP:PUERTO]
#   ./start_xgb_clients.sh nodos/retailer_region 127.0.0.1:8087
# ------------------------------------------------------------------

CSV_DIR="${1:-nodos/retailer_region}"   # por defecto retailer_region
SERVER="${2:-127.0.0.1:8087}"           # por defecto localhost

mkdir -p logs                    # guardaremos logs aquí

shopt -s nullglob                # evita bucle si no hay CSV
for csv in "$CSV_DIR"/*.csv; do
  name=$(basename "$csv" .csv)
  echo "➤ Lanzando cliente $name"
  python -m src.xgb_bagging.client \
         --csv  "$csv" \
         --name "$name" \
         --server "$SERVER" \
         > "logs/${name}.out" 2>&1 &
done
echo " Todos los clientes arrancados (tail -f logs/*.out para ver la salida)"
