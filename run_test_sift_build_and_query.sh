#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SIFT100K_DIR="${SIFT100K_DIR:-$ROOT/datasets/custom/sift100k}"
SIFT1M_DIR="${SIFT1M_DIR:-$ROOT/datasets/custom/sift1M}"

OUT_YML="${OUT_YML:-$ROOT/bench_runs/bench_custom_sift_build_then_query.yml}"

MILVUS_URI="${MILVUS_URI:-http://localhost:19530}"
MILVUS_PASSWORD="${MILVUS_PASSWORD:-}"

QDRANT_URL="${QDRANT_URL:-http://localhost:6333}"

WEAVIATE_URL="${WEAVIATE_URL:-http://localhost:8080}"
WEAVIATE_NO_AUTH="${WEAVIATE_NO_AUTH:-true}"

DOCKER_VOLUME_DIRECTORY="${DOCKER_VOLUME_DIRECTORY:-/var/tmp}"

LSMVEC_DB_ROOT_100K="${LSMVEC_DB_ROOT_100K:-$DOCKER_VOLUME_DIRECTORY/lsmvec/sift100k}"
LSMVEC_DB_ROOT_1M="${LSMVEC_DB_ROOT_1M:-$DOCKER_VOLUME_DIRECTORY/lsmvec/sift1m}"

K="${K:-10}"
NUM_CONCURRENCY="${NUM_CONCURRENCY:-1}"

HNSW_M="${HNSW_M:-32}"
HNSW_EF_CONSTRUCTION="${HNSW_EF_CONSTRUCTION:-200}"
MILVUS_EF_SEARCH="${MILVUS_EF_SEARCH:-50}"
WEAVIATE_EF="${WEAVIATE_EF:-10}"
QDRANT_EF_CONSTRUCT="${QDRANT_EF_CONSTRUCT:-200}"

REPEATS="${REPEATS:-1}"

# docker 采样配置
ENABLE_DOCKER_MEM_SAMPLING="${ENABLE_DOCKER_MEM_SAMPLING:-true}"
DOCKER_STATS_INTERVAL_SEC="${DOCKER_STATS_INTERVAL_SEC:-1}"
DOCKER_STATS_OUT_DIR="${DOCKER_STATS_OUT_DIR:-$ROOT/bench_runs/docker_stats}"
# 只采样这些 compose project
DOCKER_STATS_PROJECTS="${DOCKER_STATS_PROJECTS:-milvus,weaviate}"
# 额外按名字匹配这些容器（逗号分隔）
DOCKER_STATS_NAME_FILTERS="${DOCKER_STATS_NAME_FILTERS:-qdrant}"

#   ./run_bench_with_docker_mem.sh all
#   ./run_bench_with_docker_mem.sh baselines
#   ./run_bench_with_docker_mem.sh lsmvec
TARGET="${1:-all}"

verifyDatasetDir() {
  local dir="$1"
  if [[ ! -d "$dir" ]]; then
    echo "ERROR: dataset dir not found: $dir" >&2
    exit 1
  fi

  if [[ ! -f "$dir/train.parquet" ]]; then
    shopt -s nullglob
    local shards=("$dir"/train-*-of-*.parquet)
    shopt -u nullglob
    if [[ ${#shards[@]} -eq 0 ]]; then
      echo "ERROR: missing train.parquet or train-*-of-*.parquet in $dir" >&2
      exit 1
    fi
  fi

  [[ -f "$dir/test.parquet" ]] || { echo "ERROR: missing test.parquet in $dir" >&2; exit 1; }
  [[ -f "$dir/neighbors.parquet" ]] || { echo "ERROR: missing neighbors.parquet in $dir" >&2; exit 1; }
}

trainFileCount() {
  local dir="$1"
  if [[ -f "$dir/train.parquet" ]]; then
    echo 1
    return
  fi
  shopt -s nullglob
  local shards=("$dir"/train-*-of-*.parquet)
  shopt -u nullglob
  echo "${#shards[@]}"
}

writeYaml() {
  local sift100kCount sift1mCount
  sift100kCount="$(trainFileCount "$SIFT100K_DIR")"
  sift1mCount="$(trainFileCount "$SIFT1M_DIR")"

  mkdir -p "$(dirname "$OUT_YML")"

  cat > "$OUT_YML" <<YML
# Auto-generated. Do not edit by hand.
YML

  if [[ "$TARGET" == "all" || "$TARGET" == "baselines" ]]; then
    cat >> "$OUT_YML" <<YML

milvushnsw:
  - db_label: milvus_sift100k
    uri: "$MILVUS_URI"
    password: "$MILVUS_PASSWORD"
    case_type: PerformanceCustomDataset
    custom_case_name: PerfSIFT128D100K
    custom_dataset_name: SIFT
    custom_dataset_dir: "$SIFT100K_DIR"
    custom_dataset_size: 100000
    custom_dataset_dim: 128
    custom_dataset_metric_type: L2
    custom_dataset_file_count: $sift100kCount
    custom_dataset_use_shuffled: false
    custom_dataset_with_gt: true
    k: $K
    num_concurrency: $NUM_CONCURRENCY
    m: $HNSW_M
    ef_construction: $HNSW_EF_CONSTRUCTION
    ef_search: $MILVUS_EF_SEARCH
    drop_old: true
    load: true

  - db_label: milvus_sift1m
    uri: "$MILVUS_URI"
    password: "$MILVUS_PASSWORD"
    case_type: PerformanceCustomDataset
    custom_case_name: PerfSIFT128D1M
    custom_dataset_name: SIFT
    custom_dataset_dir: "$SIFT1M_DIR"
    custom_dataset_size: 1000000
    custom_dataset_dim: 128
    custom_dataset_metric_type: L2
    custom_dataset_file_count: $sift1mCount
    custom_dataset_use_shuffled: false
    custom_dataset_with_gt: true
    k: $K
    num_concurrency: $NUM_CONCURRENCY
    m: $HNSW_M
    ef_construction: $HNSW_EF_CONSTRUCTION
    ef_search: $MILVUS_EF_SEARCH
    drop_old: true
    load: true

qdrantlocal:
  - db_label: qdrant_sift100k
    url: "$QDRANT_URL"
    case_type: PerformanceCustomDataset
    custom_case_name: PerfSIFT128D100K
    custom_dataset_name: SIFT
    custom_dataset_dir: "$SIFT100K_DIR"
    custom_dataset_size: 100000
    custom_dataset_dim: 128
    custom_dataset_metric_type: L2
    custom_dataset_file_count: $sift100kCount
    custom_dataset_use_shuffled: false
    custom_dataset_with_gt: true
    k: $K
    num_concurrency: $NUM_CONCURRENCY
    m: $HNSW_M
    ef_construct: $QDRANT_EF_CONSTRUCT
    drop_old: true
    load: true

  - db_label: qdrant_sift1m
    url: "$QDRANT_URL"
    case_type: PerformanceCustomDataset
    custom_case_name: PerfSIFT128D1M
    custom_dataset_name: SIFT
    custom_dataset_dir: "$SIFT1M_DIR"
    custom_dataset_size: 1000000
    custom_dataset_dim: 128
    custom_dataset_metric_type: L2
    custom_dataset_file_count: $sift1mCount
    custom_dataset_use_shuffled: false
    custom_dataset_with_gt: true
    k: $K
    num_concurrency: $NUM_CONCURRENCY
    m: $HNSW_M
    ef_construct: $QDRANT_EF_CONSTRUCT
    drop_old: true
    load: true

weaviate:
  - db_label: weaviate_sift100k
    url: "$WEAVIATE_URL"
    case_type: PerformanceCustomDataset
    custom_case_name: PerfSIFT128D100K
    custom_dataset_name: SIFT
    custom_dataset_dir: "$SIFT100K_DIR"
    custom_dataset_size: 100000
    custom_dataset_dim: 128
    custom_dataset_metric_type: L2
    custom_dataset_file_count: $sift100kCount
    custom_dataset_use_shuffled: false
    custom_dataset_with_gt: true
    k: $K
    num_concurrency: $NUM_CONCURRENCY
    m: $HNSW_M
    ef_construction: $HNSW_EF_CONSTRUCTION
    ef: $WEAVIATE_EF
    drop_old: true
    load: true
    no_auth: true
    api_key: ""

  - db_label: weaviate_sift1m
    url: "$WEAVIATE_URL"
    case_type: PerformanceCustomDataset
    custom_case_name: PerfSIFT128D1M
    custom_dataset_name: SIFT
    custom_dataset_dir: "$SIFT1M_DIR"
    custom_dataset_size: 1000000
    custom_dataset_dim: 128
    custom_dataset_metric_type: L2
    custom_dataset_file_count: $sift1mCount
    custom_dataset_use_shuffled: false
    custom_dataset_with_gt: true
    k: $K
    num_concurrency: $NUM_CONCURRENCY
    m: $HNSW_M
    ef_construction: $HNSW_EF_CONSTRUCTION
    ef: $WEAVIATE_EF
    drop_old: true
    load: true
    no_auth: true
    api_key: ""
YML
  fi

  if [[ "$TARGET" == "all" || "$TARGET" == "lsmvec" ]]; then
    cat >> "$OUT_YML" <<YML

lsmvec:
  - db_label: lsmvec_sift100k
    db_root: "$LSMVEC_DB_ROOT_100K"
    case_type: PerformanceCustomDataset
    custom_case_name: PerfSIFT128D100K
    custom_dataset_name: SIFT
    custom_dataset_dir: "$SIFT100K_DIR"
    custom_dataset_size: 100000
    custom_dataset_dim: 128
    custom_dataset_metric_type: L2
    custom_dataset_file_count: $sift100kCount
    custom_dataset_use_shuffled: false
    custom_dataset_with_gt: true
    k: $K
    num_concurrency: $NUM_CONCURRENCY
    drop_old: true
    load: true

  - db_label: lsmvec_sift1m
    db_root: "$LSMVEC_DB_ROOT_1M"
    case_type: PerformanceCustomDataset
    custom_case_name: PerfSIFT128D1M
    custom_dataset_name: SIFT
    custom_dataset_dir: "$SIFT1M_DIR"
    custom_dataset_size: 1000000
    custom_dataset_dim: 128
    custom_dataset_metric_type: L2
    custom_dataset_file_count: $sift1mCount
    custom_dataset_use_shuffled: false
    custom_dataset_with_gt: true
    k: $K
    num_concurrency: $NUM_CONCURRENCY
    drop_old: true
    load: true
YML
  fi

  echo "Batch config written to: $OUT_YML"
}

# 返回三列: container_id \t container_name \t compose_project
listTargetContainers() {
  local projectsCsv="$1"
  local nameFiltersCsv="$2"

  {
    # 按 project label 抓 milvus 和 weaviate
    IFS=',' read -r -a projects <<< "$projectsCsv"
    for p in "${projects[@]}"; do
      p="$(echo "$p" | xargs)"
      [[ -z "$p" ]] && continue
      docker ps \
        --filter "label=com.docker.compose.project=$p" \
        --format '{{.ID}}\t{{.Names}}\t{{.Label "com.docker.compose.project"}}' || true
    done

    # 额外按名字抓 qdrant（它没有 compose label）
    IFS=',' read -r -a names <<< "$nameFiltersCsv"
    for n in "${names[@]}"; do
      n="$(echo "$n" | xargs)"
      [[ -z "$n" ]] && continue
      docker ps \
        --filter "name=$n" \
        --format '{{.ID}}\t{{.Names}}\t{{.Label "com.docker.compose.project"}}' || true
    done
  } | awk 'NF>=2 {print $1 "\t" $2 "\t" $3}' | sort -u
}

SAMPLER_PID=""

startDockerMemSampler() {
  local outFile="$1"
  local intervalSec="$2"
  local projectsCsv="$3"
  local nameFiltersCsv="$4"

  mkdir -p "$(dirname "$outFile")"
  echo -e "ts_utc\tcontainer_name\tcompose_project\tmem_usage\tmem_perc\tcpu_perc" > "$outFile"

  (
    while true; do
      ts="$(date -u '+%Y-%m-%dT%H:%M:%S.%3NZ')"

      containerList="$(listTargetContainers "$projectsCsv" "$nameFiltersCsv" || true)"
      if [[ -z "$containerList" ]]; then
        sleep "$intervalSec"
        continue
      fi

      ids="$(echo "$containerList" | awk '{print $1}' | tr '\n' ' ')"
      if [[ -z "$ids" ]]; then
        sleep "$intervalSec"
        continue
      fi

      stats="$(docker stats --no-stream \
        --format '{{.Container}}\t{{.MemUsage}}\t{{.MemPerc}}\t{{.CPUPerc}}' \
        $ids 2>/dev/null || true)"

      if [[ -n "$stats" ]]; then
        awk -v ts="$ts" -F'\t' '
          NR==FNR {name[$1]=$2; proj[$1]=$3; next}
          {
            id=$1
            if (id in name) {
              print ts "\t" name[id] "\t" proj[id] "\t" $2 "\t" $3 "\t" $4
            }
          }
        ' <(echo "$containerList") <(echo "$stats") >> "$outFile"
      fi

      sleep "$intervalSec"
    done
  ) >/dev/null 2>&1 &

  SAMPLER_PID=$!
}

stopDockerMemSampler() {
  local pid="${1:-}"
  if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
    kill "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
  fi
}

main() {
  verifyDatasetDir "$SIFT100K_DIR"
  verifyDatasetDir "$SIFT1M_DIR"

  if [[ "$TARGET" != "all" && "$TARGET" != "baselines" && "$TARGET" != "lsmvec" ]]; then
    echo "Usage: $0 [all|baselines|lsmvec]" >&2
    exit 1
  fi

  writeYaml

  for i in $(seq 1 "$REPEATS"); do
    echo "Run $i / $REPEATS (TARGET=$TARGET)"

    local runTs outLog statsFile samplerPid
    runTs="$(date -u '+%Y%m%dT%H%M%SZ')"

    outLog="$ROOT/bench_runs/run_${runTs}_target_${TARGET}.log"
    statsFile="$DOCKER_STATS_OUT_DIR/docker_mem_${runTs}_target_${TARGET}.tsv"
    samplerPid=""

    if [[ "$ENABLE_DOCKER_MEM_SAMPLING" == "true" ]]; then
      echo "Docker mem sampling enabled, interval=${DOCKER_STATS_INTERVAL_SEC}s"
      echo "Projects: $DOCKER_STATS_PROJECTS"
      echo "Name filters: $DOCKER_STATS_NAME_FILTERS"
      echo "Stats file: $statsFile"
      startDockerMemSampler "$statsFile" "$DOCKER_STATS_INTERVAL_SEC" "$DOCKER_STATS_PROJECTS" "$DOCKER_STATS_NAME_FILTERS"
      samplerPid="$SAMPLER_PID"
      trap 'stopDockerMemSampler "${samplerPid:-}"; exit 1' INT TERM
      trap 'stopDockerMemSampler "${samplerPid:-}"' EXIT
    fi

    set +e
    vectordbbench batchcli --batch-config-file "$OUT_YML" 2>&1 | tee "$outLog"
    exitCode=${PIPESTATUS[0]}
    set -e

    if [[ "$ENABLE_DOCKER_MEM_SAMPLING" == "true" ]]; then
      stopDockerMemSampler "$samplerPid"
      echo "Docker mem sampling stopped, file written: $statsFile"
    fi

    if [[ $exitCode -ne 0 ]]; then
      echo "ERROR: vectordbbench exited with code $exitCode, log: $outLog" >&2
      exit $exitCode
    fi
  done
}

main "$@"
