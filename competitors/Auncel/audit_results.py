#!/usr/bin/env python3
"""Independently audit Auncel per-query CSVs against ground truth.

Recomputes every query's hits and recall from its returned neighbor ids and
the top-k ground-truth ids, and checks row order, id validity, the nprobe
budget, and the averages in each preparation report. Exits nonzero on any
mismatch.
"""

import argparse
import csv
import sys

import numpy as np

HEADER = ["qid", "dists", "elaps_ms", "r", "clusters_searched", "hits", "neighbor_ids"]


def read_ivecs(path, rows, cols):
    raw = np.memmap(path, dtype=np.int32, mode="r")
    dim = int(raw[0])
    table = raw.reshape(-1, dim + 1)
    if table.shape[0] < rows or dim < cols:
        sys.exit(f"{path} is too small for {rows} rows x {cols} columns")
    return np.asarray(table[:rows, 1:cols + 1])


def audit(path, truth, k, queries, nprobe):
    errors, recalls = [], []
    report_path = path[:-len(".csv")] + "_preparation.txt"
    with open(report_path) as handle:
        report = dict(line.strip().split("=", 1) for line in handle if "=" in line and not line.startswith("tuning "))
    ntotal = int(report["ntotal"])
    with open(path, newline="") as handle:
        reader = csv.reader(handle)
        if next(reader) != HEADER:
            return [f"{path}: unexpected header"], None
        rows = list(reader)
    if len(rows) != queries:
        return [f"{path}: {len(rows)} rows, expected {queries}"], None
    for q, (qid, dists, elaps_ms, r, lists, hits, ids) in enumerate(rows):
        ids = [int(x) for x in ids.split(";")] if ids else []
        problems = []
        if int(qid) != q:
            problems.append(f"qid {qid}")
        found = [i for i in ids if i >= 0]
        if len(ids) != k or len(set(found)) != len(found):
            problems.append(f"{len(ids)} ids, duplicates among found ids")
        if any(i >= ntotal for i in found):
            problems.append("id out of range")
        true_hits = len(set(found) & set(truth[q].tolist()))
        if true_hits != int(hits) or abs(true_hits / k - float(r)) > 1e-9:
            problems.append(f"hits/r {hits}/{r}, recomputed {true_hits}/{true_hits / k}")
        if not 0 < int(lists) <= nprobe or int(dists) <= 0 or float(elaps_ms) <= 0:
            problems.append(f"lists {lists} (budget {nprobe}), dists {dists}, time {elaps_ms}")
        if problems:
            errors.append(f"{path} row {q}: " + "; ".join(problems))
        recalls.append(true_hits / k)
    if abs(float(report["avg_recall"]) - float(np.mean(recalls))) > 1e-6:
        errors.append(f"{report_path}: avg_recall {report['avg_recall']} != recomputed {np.mean(recalls)}")
    return errors, np.array(recalls)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt", required=True, help="test-query ground-truth .ivecs")
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--queries", type=int, required=True)
    parser.add_argument("--nprobe", type=int, required=True, help="list budget")
    parser.add_argument("csvs", nargs="+")
    args = parser.parse_args()

    truth = read_ivecs(args.gt, args.queries, args.k)
    failed = False
    for path in args.csvs:
        errors, recalls = audit(path, truth, args.k, args.queries, args.nprobe)
        for error in errors[:10]:
            print(error)
        if errors:
            failed = True
            print(f"FAIL {path}: {len(errors)} problems")
        else:
            print(f"ok {path.rsplit('/', 1)[-1]}: recomputed avg recall {recalls.mean():.4f}, "
                  f"min {recalls.min():.2f}, zero-recall queries {int((recalls == 0).sum())}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
