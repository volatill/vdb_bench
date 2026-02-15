#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import bisect
import csv
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from typing import List, Optional, Tuple


ANSI_ESCAPE_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")

LOG_TS_RE = re.compile(r"^(\d{4}-\d{2}-\d{2}) (\d{2}:\d{2}:\d{2}),(\d{3})")
START_CASE_RE = re.compile(r"\[1/1\]\s+start case:.*?'name':\s*'([^']+)'.*?'db':\s*'([^']+)'")
FINISH_CASE_RE = re.compile(r"\[1/1\]\s+finish case:.*?'name':\s*'([^']+)'.*?'db':\s*'([^']+)'")
METRIC_RE = re.compile(r"Performance case got result: Metric\([^)]*load_duration=([0-9.]+),\s*qps=([0-9.]+),")
HOST_STATS_PATH_RE = re.compile(r"^Host stats file:\s*(\S+)\s*$")

RUN_TS_FROM_HOST_RE = re.compile(r"host_mem_(\d{8}T\d{6}Z)_target_")
RUN_TS_FROM_LOG_RE = re.compile(r"run_(\d{8}T\d{6}Z)_target_")


@dataclass
class CaseRecord:
    method_: str
    dataset_: str
    start_utc_: datetime
    end_utc_: datetime
    build_time_s_: Optional[float] = None
    qps_: Optional[float] = None
    baseline_mem_used_mib_: Optional[float] = None
    peak_mem_used_mib_: Optional[float] = None
    peak_delta_mib_: Optional[float] = None


def stripAnsi(text: str) -> str:
    return ANSI_ESCAPE_RE.sub("", text)


def parseLogTimestampToUtc(line: str, localTz: ZoneInfo) -> Optional[datetime]:
    match = LOG_TS_RE.match(line)
    if match is None:
        return None
    datePart, timePart, msPart = match.group(1), match.group(2), match.group(3)
    dtLocal = datetime.strptime(f"{datePart} {timePart}.{msPart}", "%Y-%m-%d %H:%M:%S.%f").replace(tzinfo=localTz)
    return dtLocal.astimezone(timezone.utc)


def mapDbToMethod(dbStr: str) -> str:
    # dbStr examples:
    # - "Milvus-milvus_sift100k"
    # - "QdrantLocal"
    # - "LsmVec-lsmvec_sift1m"
    if dbStr.startswith("Milvus"):
        return "milvushnsw"
    if dbStr.startswith("QdrantLocal"):
        return "qdrantlocal"
    if dbStr.startswith("Weaviate"):
        return "weaviate"
    if dbStr.startswith("LsmVec"):
        return "lsmvec"
    return dbStr.lower()


def mapCaseNameToDataset(caseName: str) -> str:
    # caseName examples: PerfSIFT128D100K, PerfSIFT128D1M
    if "100K" in caseName:
        return "sift100k"
    if "1M" in caseName:
        return "sift1m"
    return caseName


def parseHostStatsPathFromLog(logPath: str) -> Optional[str]:
    with open(logPath, "r", encoding="utf-8", errors="replace") as f:
        for rawLine in f:
            line = stripAnsi(rawLine).rstrip("\n")
            match = HOST_STATS_PATH_RE.match(line)
            if match:
                return match.group(1)
    return None


def parseRunTs(logPath: str, hostStatsPath: Optional[str]) -> Optional[str]:
    if hostStatsPath:
        base = os.path.basename(hostStatsPath)
        m = RUN_TS_FROM_HOST_RE.search(base)
        if m:
            return m.group(1)

    baseLog = os.path.basename(logPath)
    m2 = RUN_TS_FROM_LOG_RE.search(baseLog)
    if m2:
        return m2.group(1)

    return None


def parseHostStats(hostStatsPath: str) -> Tuple[List[datetime], List[float]]:
    timesUtc: List[datetime] = []
    memUsed: List[float] = []

    with open(hostStatsPath, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if "ts_utc" not in reader.fieldnames or "server_mem_used_mib" not in reader.fieldnames:
            raise RuntimeError(f"Host stats header missing required columns: {reader.fieldnames}")

        for row in reader:
            tsRaw = row["ts_utc"].strip()
            memRaw = row["server_mem_used_mib"].strip()
            if not tsRaw or not memRaw:
                continue

            # ts is like 2026-02-12T12:02:41.520Z
            tsIso = tsRaw.replace("Z", "+00:00")
            ts = datetime.fromisoformat(tsIso).astimezone(timezone.utc)
            try:
                mem = float(memRaw)
            except ValueError:
                continue

            timesUtc.append(ts)
            memUsed.append(mem)

    if not timesUtc:
        raise RuntimeError(f"No rows parsed from host stats: {hostStatsPath}")

    return timesUtc, memUsed


def getValueAtOrBefore(timesUtc: List[datetime], values: List[float], targetUtc: datetime) -> float:
    idx = bisect.bisect_right(timesUtc, targetUtc) - 1
    if idx < 0:
        return values[0]
    return values[idx]


def getMaxBetween(timesUtc: List[datetime], values: List[float], startUtc: datetime, endUtc: datetime) -> float:
    left = bisect.bisect_left(timesUtc, startUtc)
    right = bisect.bisect_right(timesUtc, endUtc)
    if left >= right:
        # If no samples fall inside, fall back to nearest at/before start
        return getValueAtOrBefore(timesUtc, values, startUtc)
    return max(values[left:right])


def parseCasesFromLog(logPath: str, localTz: ZoneInfo) -> List[CaseRecord]:
    cases: List[CaseRecord] = []
    currentCase: Optional[CaseRecord] = None

    with open(logPath, "r", encoding="utf-8", errors="replace") as f:
        for rawLine in f:
            line = stripAnsi(rawLine).rstrip("\n")
            tsUtc = parseLogTimestampToUtc(line, localTz)

            if tsUtc is None:
                continue

            startMatch = START_CASE_RE.search(line)
            if startMatch:
                caseName = startMatch.group(1)
                dbStr = startMatch.group(2)
                method = mapDbToMethod(dbStr)
                dataset = mapCaseNameToDataset(caseName)

                currentCase = CaseRecord(
                    method_=method,
                    dataset_=dataset,
                    start_utc_=tsUtc,
                    end_utc_=tsUtc,  # placeholder, will update on finish
                )
                continue

            metricMatch = METRIC_RE.search(line)
            if metricMatch and currentCase is not None:
                try:
                    currentCase.build_time_s_ = float(metricMatch.group(1))
                    currentCase.qps_ = float(metricMatch.group(2))
                except ValueError:
                    pass
                continue

            finishMatch = FINISH_CASE_RE.search(line)
            if finishMatch and currentCase is not None:
                # Finish the current case window
                currentCase.end_utc_ = tsUtc
                cases.append(currentCase)
                currentCase = None
                continue

    return cases


def writeOutputTsv(outPath: str, cases: List[CaseRecord]) -> None:
    os.makedirs(os.path.dirname(outPath), exist_ok=True)
    with open(outPath, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["method", "dataset", "build time", "qps", "peak memory"])
        for c in cases:
            buildTime = "" if c.build_time_s_ is None else f"{c.build_time_s_:.4f}"
            qps = "" if c.qps_ is None else f"{c.qps_:.4f}"
            peakMem = "" if c.peak_delta_mib_ is None else f"{c.peak_delta_mib_:.2f}"
            writer.writerow([c.method_, c.dataset_, buildTime, qps, peakMem])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract per-method peak system memory delta + build time + qps from vdbbench log and host_stats."
    )
    parser.add_argument("--log", required=True, help="Path to run_*.log")
    parser.add_argument("--host-stats", default="", help="Path to host_mem_*.tsv (optional, can be auto-detected from log)")
    parser.add_argument("--out", default="", help="Output TSV path (optional)")
    parser.add_argument("--local-tz", default="Asia/Singapore", help="Timezone for log timestamps (default: Asia/Singapore)")

    args = parser.parse_args()
    logPath = args.log
    hostStatsPath = args.host_stats.strip()

    localTz = ZoneInfo(args.local_tz)

    if not hostStatsPath:
        hostStatsPath = parseHostStatsPathFromLog(logPath) or ""
        if not hostStatsPath:
            raise SystemExit("Cannot find host stats path in log. Please pass --host-stats explicitly.")

    runTs = parseRunTs(logPath, hostStatsPath)
    if not runTs:
        raise SystemExit("Cannot infer runTs. Please pass --out explicitly.")

    if args.out.strip():
        outPath = args.out.strip()
    else:
        benchRunsDir = os.path.dirname(logPath)
        outPath = os.path.join(benchRunsDir, f"{runTs}_key_metric_table.tsv")

    timesUtc, memUsed = parseHostStats(hostStatsPath)
    cases = parseCasesFromLog(logPath, localTz)

    if not cases:
        raise SystemExit("No cases found in log (no '[1/1] start case' / 'finish case' pairs).")

    # Compute baseline/peak/delta for each case
    for c in cases:
        baseline = getValueAtOrBefore(timesUtc, memUsed, c.start_utc_)
        peak = getMaxBetween(timesUtc, memUsed, c.start_utc_, c.end_utc_)
        c.baseline_mem_used_mib_ = baseline
        c.peak_mem_used_mib_ = peak
        c.peak_delta_mib_ = peak - baseline

        buildTimeStr = "N/A" if c.build_time_s_ is None else f"{c.build_time_s_:.4f}s"
        qpsStr = "N/A" if c.qps_ is None else f"{c.qps_:.4f}"
        print(
            f"{c.method_}\t{c.dataset_}\tbuild_time={buildTimeStr}\tqps={qpsStr}\t"
            f"baseline={baseline:.2f}MiB\tpeak={peak:.2f}MiB\tdelta={c.peak_delta_mib_:.2f}MiB"
        )

    writeOutputTsv(outPath, cases)
    print(f"\nWrote: {outPath}")


if __name__ == "__main__":
    main()
