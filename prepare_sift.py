#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Prepare SIFT datasets (100k / 1M / 10M / 100M) for VectorDBBench custom dataset.

Output directory layout (one folder per size):
  outRoot/
    sift100k/
      train.parquet OR train-00-of-XX.parquet ...
      test.parquet
      neighbors.parquet
    sift1M/
    sift10M/
    sift100M/

VDBBench custom dataset requirements:
- train*.parquet: columns: id (incrementing int), emb (array of float32)
- test.parquet:   columns: id (incrementing int), emb (array of float32)
- neighbors.parquet: columns: id (query id), neighbors_id (array of int)
- train split naming: train-[index]-of-[file_count].parquet, 0-indexed
See VDBBench README.  (format details referenced by the user)
"""

# python prepare_sift.py   --out-root ./datasets/custom   --sizes 100k,1M   --num-queries 1000   --topk 100   --shard-rows 1000000   --chunk-rows 200000

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except Exception as e:
    print("Missing dependency: pyarrow. Please install: pip install pyarrow", file=sys.stderr)
    raise

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


BIGANN_FTP_ROOT = "ftp://ftp.irisa.fr/local/texmex/corpus/"
BIGANN_BASE_GZ = "bigann_base.bvecs.gz"
BIGANN_QUERY_GZ = "bigann_query.bvecs.gz"
BIGANN_GND_TAR_GZ = "bigann_gnd.tar.gz"

DIM = 128
RECORD_BYTES = 4 + DIM  # bvecs record: int32 dim + DIM bytes

DEFAULT_SIZES = ["100k", "1M", "10M", "100M"]
SIZE_TO_N = {
    "100k": 100_000,
    "1M": 1_000_000,
    "10M": 10_000_000,
    "100M": 100_000_000,
}


@dataclass
class PrepareArgs:
    rawDir: Path
    outRoot: Path
    sizes: List[str]
    topK: int
    numQueries: int
    shardRows: int
    chunkRows: int
    force: bool
    skipDownload: bool


def runCmd(cmd: list[str], streamOutput: bool = False) -> None:
    if streamOutput:
        proc = subprocess.run(cmd)
        if proc.returncode != 0:
            raise RuntimeError(f"Command failed: {' '.join(cmd)}")
        return

    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            "Command failed:\n"
            f"  cmd: {' '.join(cmd)}\n"
            f"  stdout:\n{proc.stdout}\n"
            f"  stderr:\n{proc.stderr}\n"
        )


def ensureTool(toolName: str) -> str:
    toolPath = shutil.which(toolName)
    if toolPath is None:
        raise RuntimeError(f"Required tool not found: {toolName}. Please install it.")
    return toolPath


def downloadFile(url: str, outPath: Path) -> None:
    outPath.parent.mkdir(parents=True, exist_ok=True)
    if outPath.exists() and outPath.stat().st_size > 0:
        return

    # Prefer wget for ftp resume
    wgetPath = shutil.which("wget")
    curlPath = shutil.which("curl")

    if wgetPath is not None:
        runCmd([wgetPath, "-c", url, "-O", str(outPath)], streamOutput=True)
        return
    if curlPath is not None:
        runCmd([curlPath, "-L", "-C", "-", url, "-o", str(outPath)], streamOutput=True)
        return

    raise RuntimeError("Need wget or curl to download (ftp/https).")


def extractTarGz(tarGzPath: Path, outDir: Path) -> None:
    # tar -xzf bigann_gnd.tar.gz -C outDir
    ensureTool("tar")
    outDir.mkdir(parents=True, exist_ok=True)
    marker = outDir / ".extracted_ok"
    if marker.exists():
        return
    runCmd(["tar", "-xzf", str(tarGzPath), "-C", str(outDir)])
    marker.write_text("ok\n", encoding="utf-8")


def extractBvecsPrefixFromGz(gzPath: Path, outPath: Path, numVecs: int) -> None:
    """
    Stream decompress and keep only the first numVecs records.
    This avoids decompressing the full 1B file.
    """
    ensureTool("gzip")
    ensureTool("head")
    ensureTool("bash")

    outPath.parent.mkdir(parents=True, exist_ok=True)
    expectedBytes = numVecs * RECORD_BYTES
    if outPath.exists() and outPath.stat().st_size == expectedBytes:
        return

    if outPath.exists():
        outPath.unlink()

    # gzip -dc file.gz | head -c BYTES > outPath
    bashCmd = f"set -o pipefail; gzip -dc '{gzPath}' | head -c {expectedBytes} > '{outPath}'"
    runCmd(["bash", "-lc", bashCmd])

    actualBytes = outPath.stat().st_size
    if actualBytes != expectedBytes:
        raise RuntimeError(f"Prefix extraction size mismatch: expected={expectedBytes}, actual={actualBytes}")


def readBvecsChunkAsFloat32(bvecsPath: Path, startRow: int, numRows: int) -> np.ndarray:
    """
    Read [startRow, startRow+numRows) from a (possibly prefix) bvecs file.
    Returns float32 array shape (numRows, DIM).
    """
    fileBytes = bvecsPath.stat().st_size
    totalRows = fileBytes // RECORD_BYTES
    if startRow < 0 or startRow >= totalRows:
        raise ValueError("startRow out of range")
    endRow = min(startRow + numRows, totalRows)
    numRows = endRow - startRow

    mm = np.memmap(bvecsPath, dtype=np.uint8, mode="r")
    offsetBytes = startRow * RECORD_BYTES
    sliceBytes = numRows * RECORD_BYTES
    view = mm[offsetBytes : offsetBytes + sliceBytes].reshape(numRows, RECORD_BYTES)

    # Optional sanity check on the first row dim
    dimBytes = view[0, :4].tobytes()
    dimVal = int(np.frombuffer(dimBytes, dtype=np.int32)[0])
    if dimVal != DIM:
        raise RuntimeError(f"Unexpected dim in bvecs: {dimVal} (expected {DIM})")

    vectorsU8 = view[:, 4:]
    vectorsF32 = vectorsU8.astype(np.float32, copy=False)
    # copy=False still returns float32 view only if already float32. It will copy here, which is ok per chunk.
    vectorsF32 = vectorsU8.astype(np.float32, copy=True)
    return vectorsF32


def readIvecsTopK(ivecsPath: Path, topK: int, numQueries: Optional[int] = None) -> np.ndarray:
    """
    ivecs format: int32 k, then k int32, repeated.
    Returns int32 array shape (nq, topK).
    """
    arr = np.fromfile(ivecsPath, dtype=np.int32)
    if arr.size == 0:
        raise RuntimeError(f"Empty ivecs: {ivecsPath}")

    k = int(arr[0])
    if k < topK:
        raise RuntimeError(f"ivecs k={k} < topK={topK}: {ivecsPath}")

    rowWidth = 1 + k
    if arr.size % rowWidth != 0:
        raise RuntimeError(f"ivecs size not divisible by row width: {ivecsPath}")

    nq = arr.size // rowWidth
    mat = arr.reshape(nq, rowWidth)[:, 1 : 1 + topK]

    if numQueries is not None:
        mat = mat[:numQueries, :]
    return mat


def writeVectorsParquet(outPath: Path, startId: int, vectorsF32: np.ndarray) -> None:
    """
    vectorsF32: shape (n, DIM), float32
    Writes columns: id(int64), emb(fixed_size_list<float32, DIM>)
    """
    n = vectorsF32.shape[0]
    ids = pa.array(np.arange(startId, startId + n, dtype=np.int64), type=pa.int64())

    flat = pa.array(vectorsF32.reshape(-1), type=pa.float32())
    emb = pa.FixedSizeListArray.from_arrays(flat, DIM)

    table = pa.Table.from_arrays([ids, emb], names=["id", "emb"])
    pq.write_table(table, outPath, compression="zstd", use_dictionary=True)


def writeNeighborsParquet(outPath: Path, neighbors: np.ndarray) -> None:
    """
    neighbors: shape (nq, topK), int32 or int64
    Writes columns: id(int64), neighbors_id(fixed_size_list<int64, topK>)
    Query id is 0..nq-1
    """
    nq, topK = neighbors.shape
    ids = pa.array(np.arange(0, nq, dtype=np.int64), type=pa.int64())

    neigh64 = neighbors.astype(np.int64, copy=False)
    flat = pa.array(neigh64.reshape(-1), type=pa.int64())
    neighArr = pa.FixedSizeListArray.from_arrays(flat, topK)

    table = pa.Table.from_arrays([ids, neighArr], names=["id", "neighbors_id"])
    pq.write_table(table, outPath, compression="zstd", use_dictionary=True)


def maybeComputeGtWithFaiss(
    baseBvecsPath: Path,
    queryBvecsPath: Path,
    baseN: int,
    numQueries: int,
    topK: int,
) -> np.ndarray:
    """
    Compute exact ground truth for smaller sizes.
    Uses faiss if installed.
    """
    try:
        import faiss  # type: ignore
    except Exception as e:
        raise RuntimeError("Need faiss to compute GT. Install: pip install faiss-cpu") from e

    print(f"Computing exact GT with faiss: baseN={baseN}, queries={numQueries}, topK={topK}")

    xb = readBvecsChunkAsFloat32(baseBvecsPath, 0, baseN)
    xq = readBvecsChunkAsFloat32(queryBvecsPath, 0, numQueries)

    index = faiss.IndexFlatL2(DIM)
    index.add(xb)
    _, I = index.search(xq, topK)
    return I.astype(np.int32, copy=False)


def prepareOneSize(args: PrepareArgs, sizeTag: str, baseGzPath: Path, queryBvecsPath: Path, gndDir: Path) -> None:
    if sizeTag not in SIZE_TO_N:
        raise ValueError(f"Unsupported size: {sizeTag}")
    baseN = SIZE_TO_N[sizeTag]

    outDir = args.outRoot / f"sift{sizeTag}"
    if outDir.exists() and args.force:
        shutil.rmtree(outDir)
    outDir.mkdir(parents=True, exist_ok=True)

    # 1) Extract base prefix bvecs for this size
    basePrefixPath = args.rawDir / f"bigann_base_{sizeTag}.bvecs"
    extractBvecsPrefixFromGz(baseGzPath, basePrefixPath, baseN)

    # 2) Write train parquet (possibly sharded)
    shardRows = args.shardRows
    fileCount = (baseN + shardRows - 1) // shardRows
    width = max(2, len(str(fileCount)))

    def trainFileName(index: int) -> str:
        if fileCount == 1:
            return "train.parquet"
        return f"train-{index:0{width}d}-of-{fileCount:0{width}d}.parquet"

    iterator = range(0, baseN, shardRows)
    if tqdm is not None:
        iterator = tqdm(iterator, desc=f"[{sizeTag}] writing train shards", unit="vec")

    shardIndex = 0
    for startRow in iterator:
        count = min(shardRows, baseN - startRow)
        outPath = outDir / trainFileName(shardIndex)
        if outPath.exists() and not args.force:
            shardIndex += 1
            continue

        # write in smaller chunks to reduce peak memory
        remaining = count
        offset = startRow
        tmpParts: List[Path] = []
        partIndex = 0
        while remaining > 0:
            chunk = min(args.chunkRows, remaining)
            vectors = readBvecsChunkAsFloat32(basePrefixPath, offset, chunk)
            tmpPath = outDir / f".tmp_{trainFileName(shardIndex)}_part{partIndex}.parquet"
            writeVectorsParquet(tmpPath, startId=offset, vectorsF32=vectors)
            tmpParts.append(tmpPath)
            offset += chunk
            remaining -= chunk
            partIndex += 1

        # If only one part, rename it. If multiple, concatenate by rewriting once (simple and safe).
        if len(tmpParts) == 1:
            tmpParts[0].replace(outPath)
        else:
            tables = [pq.read_table(p) for p in tmpParts]
            tableAll = pa.concat_tables(tables, promote=True)
            pq.write_table(tableAll, outPath, compression="zstd", use_dictionary=True)
            for p in tmpParts:
                p.unlink(missing_ok=True)

        shardIndex += 1

    # 3) Write test.parquet
    testPath = outDir / "test.parquet"
    if not testPath.exists() or args.force:
        q = readBvecsChunkAsFloat32(queryBvecsPath, 0, args.numQueries)
        writeVectorsParquet(testPath, startId=0, vectorsF32=q)

    # 4) Write neighbors.parquet
    neighborsPath = outDir / "neighbors.parquet"
    if neighborsPath.exists() and not args.force:
        return

    # Prefer precomputed GT from gnd (idx_1M.ivecs, idx_10M.ivecs, idx_100M.ivecs)
    gtCandidates = []
    if sizeTag in ["1M", "10M", "100M"]:
        gtCandidates.append(gndDir / "gnd" / f"idx_{sizeTag}.ivecs")
        gtCandidates.append(gndDir / f"idx_{sizeTag}.ivecs")  # some tar layouts differ

    gtPath = next((p for p in gtCandidates if p.exists()), None)

    if gtPath is not None:
        gt = readIvecsTopK(gtPath, topK=args.topK, numQueries=args.numQueries)
    else:
        # For 100k (and as fallback), compute exact GT
        # Warning: do not try compute for huge baseN unless you really want to
        if baseN > 2_000_000:
            raise RuntimeError(
                f"No precomputed GT found for {sizeTag}, and baseN={baseN} is too large for exact GT. "
                "Either provide idx_*.ivecs, or lower size."
            )
        gt = maybeComputeGtWithFaiss(
            baseBvecsPath=basePrefixPath,
            queryBvecsPath=queryBvecsPath,
            baseN=baseN,
            numQueries=args.numQueries,
            topK=args.topK,
        )

    writeNeighborsParquet(neighborsPath, gt)

    # Small info file for convenience
    info = {
        "dataset": f"sift{sizeTag}",
        "dim": DIM,
        "metric": "L2",
        "base_size": baseN,
        "query_size": args.numQueries,
        "topk": args.topK,
        "train_file_count": fileCount,
        "train_shard_rows": shardRows,
    }
    (outDir / "info.json").write_text(str(info) + "\n", encoding="utf-8")


def parseArgs() -> PrepareArgs:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=str, default="./raw_sift_bigann", help="raw download and extracted files")
    parser.add_argument("--out-root", type=str, default="./datasets/custom", help="output root directory")
    parser.add_argument(
        "--sizes",
        type=str,
        default=",".join(DEFAULT_SIZES),
        help="comma separated: 100k,1M,10M,100M",
    )
    parser.add_argument("--topk", type=int, default=100, help="topK for neighbors.parquet")
    parser.add_argument("--num-queries", type=int, default=1000, help="how many queries to keep in test/gt")
    parser.add_argument(
        "--shard-rows",
        type=int,
        default=1_000_000,
        help="rows per train shard parquet (train-[i]-of-[n].parquet)",
    )
    parser.add_argument(
        "--chunk-rows",
        type=int,
        default=200_000,
        help="rows per in-memory chunk when writing parquet to control peak memory",
    )
    parser.add_argument("--force", action="store_true", help="overwrite existing output")
    parser.add_argument("--skip-download", action="store_true", help="assume raw files already exist")
    ns = parser.parse_args()

    sizes = [s.strip() for s in ns.sizes.split(",") if s.strip()]
    for s in sizes:
        if s not in SIZE_TO_N:
            raise ValueError(f"Unknown size tag: {s}")

    return PrepareArgs(
        rawDir=Path(ns.raw_dir).resolve(),
        outRoot=Path(ns.out_root).resolve(),
        sizes=sizes,
        topK=int(ns.topk),
        numQueries=int(ns.num_queries),
        shardRows=int(ns.shard_rows),
        chunkRows=int(ns.chunk_rows),
        force=bool(ns.force),
        skipDownload=bool(ns.skip_download),
    )


def main() -> None:
    args = parseArgs()
    args.rawDir.mkdir(parents=True, exist_ok=True)
    args.outRoot.mkdir(parents=True, exist_ok=True)

    baseGzPath = args.rawDir / BIGANN_BASE_GZ
    queryGzPath = args.rawDir / BIGANN_QUERY_GZ
    gndTarGzPath = args.rawDir / BIGANN_GND_TAR_GZ

    if not args.skipDownload:
        print("Downloading BIGANN files (resume supported)...")
        downloadFile(BIGANN_FTP_ROOT + BIGANN_BASE_GZ, baseGzPath)
        downloadFile(BIGANN_FTP_ROOT + BIGANN_QUERY_GZ, queryGzPath)
        downloadFile(BIGANN_FTP_ROOT + BIGANN_GND_TAR_GZ, gndTarGzPath)

    # Extract query.bvecs
    ensureTool("gzip")
    queryBvecsPath = args.rawDir / "bigann_query.bvecs"
    if (not queryBvecsPath.exists()) or args.force:
        if queryBvecsPath.exists():
            queryBvecsPath.unlink()
        runCmd(["bash", "-lc", f"gzip -dc '{queryGzPath}' > '{queryBvecsPath}'"])

    # Extract gnd tar
    gndDir = args.rawDir / "gnd_extracted"
    extractTarGz(gndTarGzPath, gndDir)

    print(f"Raw dir: {args.rawDir}")
    print(f"Out root: {args.outRoot}")
    print(f"Sizes: {args.sizes}")
    print(f"Queries kept: {args.numQueries}")
    print(f"TopK: {args.topK}")
    print(f"Train shard rows: {args.shardRows}, chunk rows: {args.chunkRows}")

    for sizeTag in args.sizes:
        print(f"\n=== Preparing {sizeTag} ===")
        prepareOneSize(args, sizeTag, baseGzPath, queryBvecsPath, gndDir)

    print("\nDone.")
    print("Next step in VDBBench: use this folder as custom dataset dir, and set file_count to train_file_count in info.json.")


if __name__ == "__main__":
    main()
