#!/usr/bin/env python3
"""Exact inner-product ground truth for a prefix of a query file.

T2I10M lacks the innprod.*.groundtruth files that DARTH+ used, so the sweep
recomputes them with FAISS IndexFlatIP (exact search). Outputs follow DARTH's
layout: <prefix>.ivecs holds neighbor ids, <prefix>.fvecs their inner products,
best first. With --verify, the result is compared with an existing DARTH
ground-truth file instead of being written.
"""

import argparse
import os
import sys

import faiss
import numpy as np


def read_vecs(path, dtype, rows=None):
    raw = np.memmap(path, dtype=np.int32, mode="r")
    dim = int(raw[0])
    if dim <= 0 or raw.size % (dim + 1) != 0:
        sys.exit(f"invalid vecs file: {path}")
    table = raw.reshape(-1, dim + 1)
    if rows is not None:
        if rows > table.shape[0]:
            sys.exit(f"{path} has {table.shape[0]} rows; {rows} needed")
        table = table[:rows]
    return table[:, 1:].view(dtype)


def write_vecs(path, matrix):
    rows, dim = matrix.shape
    table = np.empty((rows, dim + 1), dtype=np.int32)
    table[:, 0] = dim
    table[:, 1:] = matrix.view(np.int32)
    temporary = path + ".tmp"
    table.tofile(temporary)
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--queries", required=True)
    parser.add_argument("--count", type=int, required=True, help="first N queries")
    parser.add_argument("--k", type=int, default=1000)
    parser.add_argument("--out-prefix", help="write <prefix>.ivecs and <prefix>.fvecs")
    parser.add_argument("--verify", help="existing ground-truth .ivecs to compare with")
    parser.add_argument("--verify-k", type=int, default=50)
    args = parser.parse_args()
    if (args.out_prefix is None) == (args.verify is None):
        parser.error("pass exactly one of --out-prefix and --verify")

    base = read_vecs(args.base, np.float32)
    queries = np.ascontiguousarray(read_vecs(args.queries, np.float32, args.count))
    if base.shape[1] != queries.shape[1]:
        sys.exit("base and query dimensions differ")

    index = faiss.IndexFlatIP(base.shape[1])
    for start in range(0, base.shape[0], 1_000_000):
        index.add(np.ascontiguousarray(base[start:start + 1_000_000]))
    similarities, ids = index.search(queries, args.k)
    if (ids < 0).any():
        sys.exit("FAISS returned fewer than k neighbors")

    if args.verify:
        reference = read_vecs(args.verify, np.int32, args.count)[:, :args.verify_k]
        ours = ids[:, :args.verify_k]
        overlap = np.array([len(np.intersect1d(a, b)) for a, b in zip(ours, reference)])
        identical = int((overlap == args.verify_k).sum())
        print(f"top-{args.verify_k} sets identical for {identical}/{args.count} queries; "
              f"mean overlap {overlap.mean() / args.verify_k:.6f}")
        sys.exit(0 if identical == args.count else 1)

    os.makedirs(os.path.dirname(os.path.abspath(args.out_prefix)), exist_ok=True)
    write_vecs(args.out_prefix + ".fvecs", similarities.astype(np.float32))
    write_vecs(args.out_prefix + ".ivecs", ids.astype(np.int32))
    print(f"wrote {args.out_prefix}.ivecs/.fvecs: {args.count} queries, k={args.k}")


if __name__ == "__main__":
    main()
