#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(pwd)"
SERV_DIR="${ROOT_DIR}/services"
mkdir -p "${SERV_DIR}"

echo "[1/3] Start Milvus standalone (docker compose)"
MILVUS_VER="${MILVUS_VER:-v2.3.21}"
mkdir -p "${SERV_DIR}/milvus"
if [[ ! -f "${SERV_DIR}/milvus/docker-compose.yml" ]]; then
  wget -q "https://github.com/milvus-io/milvus/releases/download/${MILVUS_VER}/milvus-standalone-docker-compose.yml" \
    -O "${SERV_DIR}/milvus/docker-compose.yml"
fi
docker compose -f "${SERV_DIR}/milvus/docker-compose.yml" up -d

echo "[2/3] Start Qdrant"
docker rm -f qdrant >/dev/null 2>&1 || true
mkdir -p "${SERV_DIR}/qdrant_storage"
docker run -d --name qdrant --restart unless-stopped \
  -p 6333:6333 -p 6334:6334 \
  -v "${SERV_DIR}/qdrant_storage:/qdrant/storage" \
  qdrant/qdrant:latest

echo "[3/3] Start Weaviate"
mkdir -p "${SERV_DIR}/weaviate"
cat > "${SERV_DIR}/weaviate/docker-compose.yml" <<'YAML'
version: "3.4"
services:
  weaviate:
    image: cr.weaviate.io/semitechnologies/weaviate:latest
    restart: unless-stopped
    ports:
      - "8080:8080"
      - "50051:50051"
    environment:
      QUERY_DEFAULTS_LIMIT: "25"
      AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED: "true"
      PERSISTENCE_DATA_PATH: "/var/lib/weaviate"
      DEFAULT_VECTORIZER_MODULE: "none"
      CLUSTER_HOSTNAME: "node1"
YAML
docker compose -f "${SERV_DIR}/weaviate/docker-compose.yml" up -d

echo "All baselines started."
echo "Milvus:   localhost:19530"
echo "Qdrant:   localhost:6333"
echo "Weaviate: localhost:8080"
