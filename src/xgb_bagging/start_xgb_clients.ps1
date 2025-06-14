#!/usr/bin/env bash
CSV_DIR="${1:-nodos/retailer_region}"     # usa 1er arg o ruta por defecto
SERVER="${2:-127.0.0.1:8087}"

for csv in "$CSV_DIR"/*.csv; do
  name=$(basename "$csv" .csv)
  python -m src.xgb_bagging.client \
         --csv "$csv" \
         --name "$name" \
         --server "$SERVER" &
done

wait       # opcional: bloquea hasta que acaben
