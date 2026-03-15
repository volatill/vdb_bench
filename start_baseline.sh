#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(pwd)"
SERV_DIR="${ROOT_DIR}/services"
mkdir -p "${SERV_DIR}"

POSTGRES_USER="${POSTGRES_USER:-postgres}"
POSTGRES_PASSWORD="${POSTGRES_PASSWORD:-postgres}"
POSTGRES_DB="${POSTGRES_DB:-vectordb}"
PGVECTOR_PORT="${PGVECTOR_PORT:-5432}"
ELASTICSEARCH_PORT="${ELASTICSEARCH_PORT:-9200}"
BASELINE_ENV_FILE="${SERV_DIR}/baseline.env"

isPortInUse() {
  local port="$1"
  (echo >"/dev/tcp/127.0.0.1/${port}") >/dev/null 2>&1
}

choosePgVectorPort() {
  local requested="$1"
  local candidates=()
  candidates+=("$requested" "15432" "25432")

  local p
  for p in "${candidates[@]}"; do
    if ! isPortInUse "$p"; then
      echo "$p"
      return 0
    fi
  done

  echo "ERROR: could not find an available host port for pgvector among: ${candidates[*]}" >&2
  exit 1
}

echo "[1/5] Start Milvus standalone (docker compose)"
MILVUS_VER="${MILVUS_VER:-v2.3.21}"
mkdir -p "${SERV_DIR}/milvus"
if [[ ! -f "${SERV_DIR}/milvus/docker-compose.yml" ]]; then
  wget -q "https://github.com/milvus-io/milvus/releases/download/${MILVUS_VER}/milvus-standalone-docker-compose.yml" \
    -O "${SERV_DIR}/milvus/docker-compose.yml"
fi
docker compose -f "${SERV_DIR}/milvus/docker-compose.yml" up -d

echo "[2/5] Start Qdrant"
docker rm -f qdrant >/dev/null 2>&1 || true
mkdir -p "${SERV_DIR}/qdrant_storage"
docker run -d --name qdrant --restart unless-stopped \
  -p 6333:6333 -p 6334:6334 \
  -v "${SERV_DIR}/qdrant_storage:/qdrant/storage" \
  qdrant/qdrant:latest

echo "[3/5] Start Weaviate"
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

echo "[4/5] Start pgvector (Postgres + pgvector extension)"
docker rm -f pgvector >/dev/null 2>&1 || true
mkdir -p "${SERV_DIR}/pgvector_data"
PGVECTOR_EFFECTIVE_PORT="$(choosePgVectorPort "$PGVECTOR_PORT")"
if [[ "$PGVECTOR_EFFECTIVE_PORT" != "$PGVECTOR_PORT" ]]; then
  echo "[WARN] PGVECTOR_PORT=${PGVECTOR_PORT} is unavailable, fallback to ${PGVECTOR_EFFECTIVE_PORT}"
fi
docker run -d --name pgvector --restart unless-stopped \
  -e POSTGRES_USER="${POSTGRES_USER}" \
  -e POSTGRES_PASSWORD="${POSTGRES_PASSWORD}" \
  -e POSTGRES_DB="${POSTGRES_DB}" \
  -p "${PGVECTOR_EFFECTIVE_PORT}:5432" \
  -v "${SERV_DIR}/pgvector_data:/var/lib/postgresql/data" \
  pgvector/pgvector:pg16

echo "[5/5] Start Elasticsearch"
docker rm -f elasticsearch >/dev/null 2>&1 || true
mkdir -p "${SERV_DIR}/elasticsearch_data"
docker run -d --name elasticsearch --restart unless-stopped \
  -e discovery.type=single-node \
  -e xpack.security.enabled=true \
  -e xpack.security.http.ssl.enabled=false \
  -e ELASTIC_PASSWORD=elastic \
  -e ES_JAVA_OPTS="-Xms1g -Xmx1g" \
  -p "${ELASTICSEARCH_PORT}:9200" \
  -v "${SERV_DIR}/elasticsearch_data:/usr/share/elasticsearch/data" \
  docker.elastic.co/elasticsearch/elasticsearch:8.14.3

cat > "${BASELINE_ENV_FILE}" <<EOF
export PGVECTOR_HOST=localhost
export PGVECTOR_PORT=${PGVECTOR_EFFECTIVE_PORT}
export PGVECTOR_DB_NAME=${POSTGRES_DB}
export PGVECTOR_USER_NAME=${POSTGRES_USER}
export PGVECTOR_PASSWORD=${POSTGRES_PASSWORD}
export ELASTICSEARCH_SCHEME=http
export ELASTICSEARCH_HOST=localhost
export ELASTICSEARCH_PORT=${ELASTICSEARCH_PORT}
export ELASTICSEARCH_USER=elastic
export ELASTICSEARCH_PASSWORD=elastic
EOF

echo "All local baselines started."
echo "Milvus:         localhost:19530"
echo "Qdrant:         localhost:6333"
echo "Weaviate:       localhost:8080"
echo "pgvector:       localhost:${PGVECTOR_EFFECTIVE_PORT} (user=${POSTGRES_USER}, db=${POSTGRES_DB})"
echo "Elasticsearch:  localhost:${ELASTICSEARCH_PORT} (user=elastic, password=elastic)"
echo "Pinecone:       managed service, set PINECONE_API_KEY / PINECONE_INDEX_NAME before running benchmark"
echo "Wrote baseline env: ${BASELINE_ENV_FILE}"
echo "Use it via: source ${BASELINE_ENV_FILE}"
