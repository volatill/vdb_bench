#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(pwd)"
SERV_DIR="${ROOT_DIR}/services"

docker rm -f elasticsearch >/dev/null 2>&1 || true
docker rm -f pgvector >/dev/null 2>&1 || true
docker rm -f qdrant >/dev/null 2>&1 || true

if [[ -f "${SERV_DIR}/weaviate/docker-compose.yml" ]]; then
  docker compose -f "${SERV_DIR}/weaviate/docker-compose.yml" down || true
fi

if [[ -f "${SERV_DIR}/milvus/docker-compose.yml" ]]; then
  docker compose -f "${SERV_DIR}/milvus/docker-compose.yml" down || true
fi

rm -f "${SERV_DIR}/baseline.env"

echo "All local baselines stopped."
