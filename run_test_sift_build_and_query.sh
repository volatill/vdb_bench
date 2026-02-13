#!/usr/bin/env bash
set -Eeuo pipefail

onErr() {
  local rc=$?
  echo "[ERROR] rc=${rc} line=${BASH_LINENO[0]} cmd=${BASH_COMMAND}" >&2
}
onExit() {
  local rc=$?
  echo "[EXIT] rc=${rc}" >&2
}
trap onErr ERR
trap onExit EXIT

requireCmd() {
  local cmd="$1"
  command -v "$cmd" >/dev/null 2>&1 || {
    echo "ERROR: missing command: $cmd" >&2
    exit 127
  }
}

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

ENABLE_DOCKER_MEM_SAMPLING="${ENABLE_DOCKER_MEM_SAMPLING:-true}"
DOCKER_STATS_INTERVAL_SEC="${DOCKER_STATS_INTERVAL_SEC:-1}"
DOCKER_STATS_OUT_DIR="${DOCKER_STATS_OUT_DIR:-$ROOT/bench_runs/docker_stats}"
DOCKER_STATS_PROJECTS="${DOCKER_STATS_PROJECTS:-milvus,weaviate}"
DOCKER_STATS_NAME_FILTERS="${DOCKER_STATS_NAME_FILTERS:-qdrant}"

ENABLE_HOST_MEM_SAMPLING="${ENABLE_HOST_MEM_SAMPLING:-true}"
HOST_STATS_INTERVAL_SEC="${HOST_STATS_INTERVAL_SEC:-1}"
HOST_STATS_OUT_DIR="${HOST_STATS_OUT_DIR:-$ROOT/bench_runs/host_stats}"

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

getServerMemSummaryMiB() {
  awk '
    $1=="MemTotal:" {t=$2}
    $1=="MemAvailable:" {a=$2}
    END {
      if (t=="" || a=="") {
        printf "0.00\t0.00\t0.00\n"
        exit
      }
      used_kib = t - a
      printf "%.2f\t%.2f\t%.2f\n", used_kib/1024.0, t/1024.0, a/1024.0
    }
  ' /proc/meminfo 2>/dev/null || echo -e "0.00\t0.00\t0.00"
}


printSystemMemBefore() {
  local label="$1"
  local used total avail
  IFS=$'\t' read -r used total avail < <(getServerMemSummaryMiB)
  echo "system memory before running ${label} is ${used} MiB (used), total ${total} MiB, available ${avail} MiB"
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

listTargetContainers() {
  local projectsCsv="$1"
  local nameFiltersCsv="$2"

  {
    IFS=',' read -r -a projects <<< "$projectsCsv"
    for p in "${projects[@]}"; do
      p="$(echo "$p" | xargs)"
      [[ -z "$p" ]] && continue
      docker ps \
        --filter "label=com.docker.compose.project=$p" \
        --format '{{.ID}}\t{{.Names}}\t{{.Label "com.docker.compose.project"}}' || true
    done

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

DOCKER_SAMPLER_PID=""
dockerSamplerLoop() {
  local outFile="$1"
  local intervalSec="$2"
  local projectsCsv="$3"
  local nameFiltersCsv="$4"

  while true; do
    local ts containerList ids stats
    local used total avail serverMemUsed

    ts="$(date -u '+%Y-%m-%dT%H:%M:%S.%3NZ')"
    IFS=$'\t' read -r used total avail < <(getServerMemSummaryMiB)
    serverMemUsed="$used"

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
      awk -v ts="$ts" -v serverMemUsed="$serverMemUsed" -F'\t' '
        NR==FNR {name[$1]=$2; proj[$1]=$3; next}
        {
          id=$1
          if (id in name) {
            print ts "\t" name[id] "\t" proj[id] "\t" $2 "\t" $3 "\t" $4 "\t" serverMemUsed
          }
        }
      ' <(echo "$containerList") <(echo "$stats") >> "$outFile"
    fi

    sleep "$intervalSec"
  done
}

startDockerMemSampler() {
  local outFile="$1"
  local intervalSec="$2"
  local projectsCsv="$3"
  local nameFiltersCsv="$4"

  mkdir -p "$(dirname "$outFile")"
  echo -e "ts_utc\tcontainer_name\tcompose_project\tmem_usage\tmem_perc\tcpu_perc\tserver_mem_used_mib" > "$outFile"

  dockerSamplerLoop "$outFile" "$intervalSec" "$projectsCsv" "$nameFiltersCsv" >/dev/null 2>&1 &
  DOCKER_SAMPLER_PID=$!
}

stopDockerMemSampler() {
  local pid="${1:-}"
  if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
    kill "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
  fi
}

getProcessTreePids() {
  local rootPid="$1"
  if [[ -z "$rootPid" ]] || ! kill -0 "$rootPid" 2>/dev/null; then
    return 0
  fi

  declare -A visited
  local -a queue all
  queue=("$rootPid")
  all=("$rootPid")
  visited["$rootPid"]=1

  while [[ ${#queue[@]} -gt 0 ]]; do
    local pid="${queue[0]}"
    queue=("${queue[@]:1}")

    local children
    children="$(ps --no-headers -o pid= --ppid "$pid" 2>/dev/null | awk '{print $1}' || true)"
    [[ -z "$children" ]] && continue

    local c
    while read -r c; do
      [[ -z "$c" ]] && continue
      if [[ -z "${visited[$c]+x}" ]]; then
        visited["$c"]=1
        all+=("$c")
        queue+=("$c")
      fi
    done <<< "$children"
  done

  printf "%s\n" "${all[@]}"
}

HOST_SAMPLER_PID=""
hostSamplerLoop() {
  local outFile="$1"
  local intervalSec="$2"
  local rootPid="$3"

  while kill -0 "$rootPid" 2>/dev/null; do
    local ts rssSumKib rssSumMib num
    local used total avail serverMemUsed
    local -a pids

    ts="$(date -u '+%Y-%m-%dT%H:%M:%S.%3NZ')"
    IFS=$'\t' read -r used total avail < <(getServerMemSummaryMiB)
    serverMemUsed="$used"

    mapfile -t pids < <(getProcessTreePids "$rootPid" || true)
    num="${#pids[@]}"

    if [[ "$num" -eq 0 ]]; then
      echo -e "${ts}\t${rootPid}\t0\t0\t0\t${serverMemUsed}" >> "$outFile"
      sleep "$intervalSec"
      continue
    fi

    rssSumKib="$(
      ps -o rss= -p "$(IFS=','; echo "${pids[*]}")" 2>/dev/null \
        | awk '{s+=$1} END{printf "%.0f", s+0}'
    )"
    rssSumMib="$(awk -v x="$rssSumKib" 'BEGIN{printf "%.2f", x/1024.0}')"

    echo -e "${ts}\t${rootPid}\t${num}\t${rssSumKib}\t${rssSumMib}\t${serverMemUsed}" >> "$outFile"
    sleep "$intervalSec"
  done
}

startHostMemSampler() {
  local outFile="$1"
  local intervalSec="$2"
  local rootPid="$3"

  mkdir -p "$(dirname "$outFile")"
  echo -e "ts_utc\troot_pid\tnum_procs\trss_kib\trss_mib\tserver_mem_used_mib" > "$outFile"

  hostSamplerLoop "$outFile" "$intervalSec" "$rootPid" >/dev/null 2>&1 &
  HOST_SAMPLER_PID=$!
}

stopHostMemSampler() {
  local pid="${1:-}"
  if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
    kill "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
  fi
}

main() {
  requireCmd awk
  requireCmd ps
  requireCmd tee
  requireCmd vectordbbench

  if [[ "$ENABLE_DOCKER_MEM_SAMPLING" == "true" ]]; then
    requireCmd docker
  fi

  mkdir -p "$ROOT/bench_runs" "$DOCKER_STATS_OUT_DIR" "$HOST_STATS_OUT_DIR"

  verifyDatasetDir "$SIFT100K_DIR"
  verifyDatasetDir "$SIFT1M_DIR"

  if [[ "$TARGET" != "all" && "$TARGET" != "baselines" && "$TARGET" != "lsmvec" ]]; then
    echo "Usage: $0 [all|baselines|lsmvec]" >&2
    exit 1
  fi

  writeYaml

  for i in $(seq 1 "$REPEATS"); do
    echo "Run $i / $REPEATS (TARGET=$TARGET)"

    local runTs outLog dockerStatsFile hostStatsFile
    local dockerSamplerPid hostSamplerPid benchPid exitCode

    runTs="$(date -u '+%Y%m%dT%H%M%SZ')"
    outLog="$ROOT/bench_runs/run_${runTs}_target_${TARGET}.log"
    dockerStatsFile="$DOCKER_STATS_OUT_DIR/docker_mem_${runTs}_target_${TARGET}.tsv"
    hostStatsFile="$HOST_STATS_OUT_DIR/host_mem_${runTs}_target_${TARGET}.tsv"

    dockerSamplerPid=""
    hostSamplerPid=""

    if [[ "$ENABLE_DOCKER_MEM_SAMPLING" == "true" ]]; then
      printSystemMemBefore "docker mem sampler"
      echo "Docker mem sampling enabled, interval=${DOCKER_STATS_INTERVAL_SEC}s"
      echo "Projects: $DOCKER_STATS_PROJECTS"
      echo "Name filters: $DOCKER_STATS_NAME_FILTERS"
      echo "Docker stats file: $dockerStatsFile"
      startDockerMemSampler "$dockerStatsFile" "$DOCKER_STATS_INTERVAL_SEC" "$DOCKER_STATS_PROJECTS" "$DOCKER_STATS_NAME_FILTERS"
      dockerSamplerPid="$DOCKER_SAMPLER_PID"
    fi

    printSystemMemBefore "vectordbbench batchcli"
    echo "Run log: $outLog"

    # Run vectordbbench in background so we can sample its process tree.
    # Use process substitution for tee so we keep the real PID.
    stdbuf -oL -eL vectordbbench batchcli --batch-config-file "$OUT_YML" > >(tee "$outLog") 2>&1 &
    benchPid=$!

    if [[ "$ENABLE_HOST_MEM_SAMPLING" == "true" ]]; then
      printSystemMemBefore "host mem sampler (process tree of pid ${benchPid})"
      echo "Host mem sampling enabled, interval=${HOST_STATS_INTERVAL_SEC}s"
      echo "Host stats file: $hostStatsFile"
      echo "Host root pid: $benchPid"
      startHostMemSampler "$hostStatsFile" "$HOST_STATS_INTERVAL_SEC" "$benchPid"
      hostSamplerPid="$HOST_SAMPLER_PID"
    fi

    wait "$benchPid"
    exitCode=$?

    stopHostMemSampler "$hostSamplerPid"
    stopDockerMemSampler "$dockerSamplerPid"

    [[ "$ENABLE_DOCKER_MEM_SAMPLING" == "true" ]] && echo "Docker stats: $dockerStatsFile"
    [[ "$ENABLE_HOST_MEM_SAMPLING" == "true" ]] && echo "Host stats: $hostStatsFile"

    if [[ $exitCode -ne 0 ]]; then
      echo "ERROR: vectordbbench exited with code $exitCode" >&2
      exit $exitCode
    fi
  done
}

main "$@"
