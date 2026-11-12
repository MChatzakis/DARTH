#!/usr/bin/env python3
"""Run and independently audit per-query Quake results for DARTH's IVF protocol.

Uses only the Python standard library. Native Quake performs all searches;
ground truth is used here and by the runner solely to score returned IDs.
"""

import argparse
import os
import csv
import hashlib
import json
import math
from pathlib import Path
import struct
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
DATASETS = {
    "SIFT100M": {"nlist": 10000, "nprobe": 150, "dimension": 128, "query_size": "10K"},
    "DEEP100M": {"nlist": 10000, "nprobe": 150, "dimension": 96, "query_size": "10K"},
    "GLOVE100": {"nlist": 1000, "nprobe": 100, "dimension": 100, "query_size": "10K"},
    "GIST1M": {"nlist": 1000, "nprobe": 200, "dimension": 960, "query_size": "1K"},
    "T2I100M": {"nlist": 20000, "nprobe": 500, "dimension": 200, "query_size": "10K"},
}
TARGETS = [0.80, 0.85, 0.90, 0.95, 0.99]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_vecs_prefix(path, count, kind, width=None):
    """Read only selected rows; validate headers and hash the exact input prefix."""
    digest = hashlib.sha256()
    records = []
    with path.open("rb") as source:
        header = source.read(4)
        require(len(header) == 4, f"Missing dimension: {path}")
        dimension = struct.unpack("<i", header)[0]
        require(0 < dimension <= 1000000, f"Invalid dimension: {path}")
        row_bytes = 4 * (dimension + 1)
        size = path.stat().st_size
        require(size % row_bytes == 0 and size // row_bytes >= count, f"Invalid/short vecs file: {path}")
        width = dimension if width is None else width
        require(0 < width <= dimension, f"Requested width exceeds vecs dimension: {path}")
        source.seek(0)
        for qid in range(count):
            row = source.read(row_bytes)
            require(len(row) == row_bytes and struct.unpack_from("<i", row)[0] == dimension,
                    f"Invalid vecs row {qid}: {path}")
            digest.update(row)
            values = struct.unpack_from(f"<{width}{kind}", row, 4)
            require(all(math.isfinite(v) for v in values), f"Non-finite query/GT row {qid}: {path}")
            records.append(values)
    return records, dimension, digest.hexdigest()


def file_signature(path):
    stat = path.stat()
    require(path.is_file() and stat.st_size > 0, f"Missing/empty input: {path}")
    return {"path": str(path.resolve()), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def input_info(args, dataset):
    conf = DATASETS[dataset]
    source = args.dataset_directory / dataset
    query = source / f"query.{conf['query_size']}.fvecs"
    gt_path = source / f"query.groundtruth.{conf['query_size']}.k1000.ivecs"
    index = args.index_directory / dataset / f"{dataset}.nlist{conf['nlist']}.index"
    signatures = {name: file_signature(path) for name, path in
                  [("index", index), ("queries", query), ("ground_truth", gt_path)]}
    _, dimension, query_hash = read_vecs_prefix(query, args.query_count, "f")
    require(dimension == conf["dimension"], f"{dataset}: expected dimension {conf['dimension']}, found {dimension}")
    truth, _, gt_hash = read_vecs_prefix(gt_path, args.query_count, "i", args.k)
    require(all(len(set(row)) == args.k and min(row) >= 0 for row in truth), f"{dataset}: invalid top-k ground truth")
    signatures["queries"]["prefix_sha256"] = query_hash
    signatures["ground_truth"]["prefix_sha256"] = gt_hash
    return index, query, gt_path, truth, signatures


def result_name(dataset, target, k, count):
    conf = DATASETS[dataset]
    return f"nlist{conf['nlist']}_nprobe{conf['nprobe']}_qs{count}_tr{target:.2f}.csv"


def audit_csv(path, truth, target, k, conf, metadata):
    """Recompute recall and work from returned IDs/list IDs, independently of C++."""
    with path.open(newline="") as source:
        reader = csv.DictReader(source)
        required = {"qid", "dists", "elaps_ms", "r", "clusters_searched", "hits", "k", "target_recall",
                    "nlist", "nprobe", "coarse_ms", "scan_ms", "neighbor_ids", "scanned_list_ids"}
        require(required.issubset(reader.fieldnames or []), f"Missing CSV columns: {path}")
        rows = list(reader)
    require(len(rows) == len(truth), f"Wrong query count: {path}")
    total_hits = under_target = total_dists = 0
    for qid, row in enumerate(rows):
        require(int(row["qid"]) == qid, f"Missing/reordered/duplicate qid: {path}")
        require(int(row["k"]) == k and int(row["nlist"]) == conf["nlist"] and int(row["nprobe"]) == conf["nprobe"],
                f"Wrong search configuration: {path}")
        require(float(row["target_recall"]) == target, f"Wrong recall target: {path}")
        ids = [int(v) for v in row["neighbor_ids"].split(";")]
        valid_ids = [v for v in ids if v != -1]
        require(len(ids) == k and len(set(valid_ids)) == len(valid_ids) and
                all(0 <= v < metadata["ntotal"] for v in valid_ids), f"Invalid returned IDs: {path}, qid={qid}")
        hits = len(set(valid_ids).intersection(truth[qid]))
        require(int(row["hits"]) == hits and math.isclose(float(row["r"]), hits / k, rel_tol=0, abs_tol=1e-12),
                f"Recall does not match returned IDs/ground truth: {path}, qid={qid}")
        lists = [int(v) for v in row["scanned_list_ids"].split(";")]
        require(0 < len(lists) <= conf["nprobe"] and len(set(lists)) == len(lists) and
                all(0 <= p < conf["nlist"] for p in lists), f"Invalid scanned lists: {path}, qid={qid}")
        require(int(row["clusters_searched"]) == len(lists), f"Invalid cluster count: {path}, qid={qid}")
        distances = sum(metadata["partition_sizes"][p] for p in lists)
        require(int(row["dists"]) == distances, f"Invalid distance count: {path}, qid={qid}")
        timings = [float(row[name]) for name in ("coarse_ms", "scan_ms", "elaps_ms")]
        require(all(math.isfinite(t) and t >= 0 for t in timings) and
                math.isclose(timings[0] + timings[1], timings[2], rel_tol=1e-9, abs_tol=1e-9),
                f"Invalid timings: {path}, qid={qid}")
        total_hits += hits
        under_target += hits / k < target
        total_dists += distances
    return {"query_count": len(rows), "mean_recall": total_hits / (len(rows) * k),
            "rqut": under_target / len(rows), "mean_dists": total_dists / len(rows), "sha256": sha256_file(path)}


def audit_dataset(args, dataset, truth, directory):
    conf = DATASETS[dataset]
    metadata = json.loads((directory / "index_metadata.json").read_text())
    upstream_commit = (ROOT / "UPSTREAM_COMMIT").read_text().strip()
    require(metadata["upstream_commit"] == upstream_commit, "Wrong Quake version")
    require(metadata["nlist"] == conf["nlist"] and metadata["nprobe"] == conf["nprobe"] and
            metadata["dimension"] == conf["dimension"] and metadata["metric"] == "l2", "Wrong index metadata")
    for flag in ["maintenance", "track_hits", "use_numa", "use_gpu", "use_auncel", "use_spann"]:
        require(metadata[flag] is False, f"Unexpected enabled feature: {flag}")
    require(metadata["num_workers"] == 0 and metadata["search_threads"] == 1 and
            metadata["static_index_checks_passed"] is True, "Static single-query protocol was not followed")
    sizes = metadata["partition_sizes"]
    require(len(sizes) == conf["nlist"] and all(isinstance(v, int) and v >= 0 for v in sizes) and
            sum(sizes) == metadata["ntotal"], "Invalid partition sizes")
    reports = {}
    for target in args.target_recalls:
        name = result_name(dataset, target, args.k, args.query_count)
        reports[name] = audit_csv(directory / name, truth, target, args.k, conf, metadata)
    return reports


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def run_logged(command, log):
    with log.open("a") as output:
        with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1) as process:
            try:
                for line in process.stdout:
                    print(line, end="", flush=True)
                    output.write(line)
                    output.flush()
                require(process.wait() == 0, f"Native evaluator failed; see {log}")
            except BaseException:
                process.terminate()
                process.wait()
                raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--k", type=int, default=50)
    parser.add_argument("--target-recalls", nargs="+", type=float, default=TARGETS)
    parser.add_argument("--query-count", type=int, default=1000)
    parser.add_argument("--dataset-directory", type=Path, default=Path(os.environ.get("DATASET_DIRECTORY", os.environ.get("DARTH_DATASET_DIR", ROOT.parent.parent / "data"))))
    parser.add_argument("--index-directory", type=Path, default=Path(os.environ.get("DARTH_IVF_INDEX_DIR", ROOT.parent.parent / "indexes" / "ivf")))
    parser.add_argument("--output-directory", type=Path, default=ROOT.parent.parent / "experiments/ivf-results/quake")
    parser.add_argument("--executable", type=Path, default=ROOT / "build/darth_quake_eval")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    require(args.k > 0 and args.query_count > 0, "k and query count must be positive")
    require(len(set(args.datasets)) == len(args.datasets), "Duplicate datasets")
    require(all(math.isfinite(t) and 0 < t <= 1 for t in args.target_recalls), "Targets must be in (0, 1]")
    require(len({f"{t:.2f}" for t in args.target_recalls}) == len(args.target_recalls), "Targets have colliding filenames")
    # Fail on any missing dataset BEFORE loading a large index or running searches.
    inputs = {dataset: input_info(args, dataset) for dataset in args.datasets}
    if args.preflight_only:
        for dataset in args.datasets:
            print(f"Preflight passed: {dataset}, k={args.k}, queries={args.query_count}, {DATASETS[dataset]}")
        return
    executable = args.executable.resolve()
    executable_hash = sha256_file(executable)
    for dataset, (index, query, gt, truth, signatures) in inputs.items():
        directory = args.output_directory / dataset / f"k{args.k}"
        expected = {"dataset": dataset, "k": args.k, "query_count": args.query_count,
                    "target_recalls": args.target_recalls, "inputs": signatures,
                    "executable_sha256": executable_hash, "evaluator_sha256": sha256_file(Path(__file__)),
                    "protocol": "static_exact_ivf_partitions_native_quake_aps_v1"}
        manifest = directory / "run_manifest.json"
        if args.resume or args.validate_only:
            try:
                saved = json.loads(manifest.read_text())
                require(saved["configuration"] == expected, "Configuration/input/build changed")
                reports = audit_dataset(args, dataset, truth, directory)
                require(reports == saved["results"], "CSV contents changed since completion")
                require(sha256_file(directory / "index_metadata.json") == saved["index_metadata_sha256"], "Index metadata changed")
                print(f"Validated {'existing' if args.resume else 'saved'} results: {dataset}", flush=True)
                continue
            except (ValueError, OSError, KeyError, TypeError) as error:
                if args.validate_only:
                    raise ValueError(f"{dataset}: {error}") from error
                print(f"Running {dataset}: no complete matching results ({error})", flush=True)
        directory.mkdir(parents=True, exist_ok=True)
        # An interrupted rerun must never leave a previous completion marker.
        if manifest.exists():
            manifest.unlink()
        conf = DATASETS[dataset]
        command = [str(executable), "--index", str(index), "--queries", str(query), "--ground-truth", str(gt),
                   "--output-directory", str(directory), "--dataset", dataset, "--nlist", str(conf["nlist"]),
                   "--nprobe", str(conf["nprobe"]), "--k", str(args.k), "--query-count", str(args.query_count),
                   "--target-recalls", ",".join(str(t) for t in args.target_recalls)]
        start = time.time()
        run_logged(command, directory / "evaluation.log")
        reports = audit_dataset(args, dataset, truth, directory)
        atomic_json(manifest, {"configuration": expected, "results": reports,
                              "index_metadata_sha256": sha256_file(directory / "index_metadata.json"),
                              "elapsed_s": time.time() - start, "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        print(f"Validated {len(reports)} recall targets for {dataset}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, KeyError, TypeError) as error:
        sys.exit(f"Quake evaluation: {error}")
