#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(pwd)"
SERV_DIR="${ROOT_DIR}/services"

docker rm -f qdrant >/dev/null 2>&1 || true

if [[ -f "${SERV_DIR}/weaviate/docker-compose.yml" ]]; then
  docker compose -f "${SERV_DIR}/weaviate/docker-compose.yml" down || true
fi

if [[ -f "${SERV_DIR}/milvus/docker-compose.yml" ]]; then
  docker compose -f "${SERV_DIR}/milvus/docker-compose.yml" down || true
fi

echo "All baselines stopped."
