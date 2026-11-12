#!/usr/bin/env python3
"""Prepare exact DARTH+ query splits for the ConANN IVF comparison.

The output query order is fixed and recorded in split.ini:

  1. first 8,333 DARTH+ training (learn) queries: ConANN calibration
  2. next 1,667 DARTH+ training (learn) queries: ConANN tuning
  3. first 1,000 DARTH+ testing (query) queries: evaluation

Ground truth is truncated from DARTH's top-1000 files to the same
``k = 10, 25, 50, 75, 100`` sweep used by DARTH+ by default.
Only the required prefixes are read, so the multi-gigabyte learn ground-truth
files are never loaded fully into memory.
"""

import argparse
import os

import numpy as np
from split_metadata import write_split


DARTH_DATA_DIR = os.environ.get("DATASET_DIRECTORY", os.environ.get("DARTH_DATASET_DIR", os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data")))
CONANN_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

DEFAULT_CALIBRATION_COUNT = 8333
DEFAULT_TUNING_COUNT = 1667
DEFAULT_TEST_COUNT = 1000
DEFAULT_K_VALUES = (10, 25, 50, 75, 100)

# ConANN/DARTH result name -> source directory and exact DARTH files.
DATASETS = {
    "DEEP100M": dict(
        darth_dir="DEEP100M",
        base="base.100M.fvecs",
        train=("learn", "1M"),
        test=("query", "10K"),
    ),
    "T2I100M": dict(
        darth_dir="T2I100M",
        base="base.100M.fvecs",
        train=("learn", "1M"),
        test=("query", "10K"),
    ),
    "SIFT100M": dict(
        darth_dir="SIFT100M",
        base="base.100M.fvecs",
        train=("learn", "1M"),
        test=("query", "10K"),
    ),
    "GIST1M": dict(
        darth_dir="GIST1M",
        base="base.1M.fvecs",
        train=("learn", "100K"),
        test=("query", "1K"),
    ),
    "GLOVE100": dict(
        darth_dir="GLOVE100",
        base="base.1183514.fvecs",
        train=("learn", "100K"),
        test=("query", "10K"),
    ),
}


def query_path(src_dir, prefix, size_label):
    return os.path.join(src_dir, "{}.{}.fvecs".format(prefix, size_label))


def gt_paths(src_dir, prefix, size_label):
    stem = "{}.groundtruth.{}.k1000".format(prefix, size_label)
    return os.path.join(src_dir, stem + ".ivecs"), os.path.join(src_dir, stem + ".fvecs")


def vecs_shape(path):
    with open(path, "rb") as source:
        dim_raw = source.read(4)
    if len(dim_raw) != 4:
        raise ValueError("{}: missing vector dimension".format(path))
    dim = int(np.frombuffer(dim_raw, dtype=np.int32)[0])
    row_bytes = (dim + 1) * 4
    size = os.path.getsize(path)
    if dim <= 0 or size % row_bytes:
        raise ValueError("{}: invalid vecs file size/dimension".format(path))
    return size // row_bytes, dim


def append_prefix(src_path, output, count, expected_dim=None):
    available, dim = vecs_shape(src_path)
    if count > available:
        raise ValueError("{}: requested {} rows, only {} available".format(src_path, count, available))
    if expected_dim is not None and dim != expected_dim:
        raise ValueError("dimension mismatch: {} has {}, expected {}".format(src_path, dim, expected_dim))

    remaining = count * (dim + 1) * 4
    with open(src_path, "rb") as source:
        while remaining:
            chunk = source.read(min(remaining, 8 * 1024 * 1024))
            if not chunk:
                raise IOError("unexpected EOF in {}".format(src_path))
            output.write(chunk)
            remaining -= len(chunk)
    return dim


def write_query_pool(train_path, test_path, train_count, test_count, dst_path):
    tmp_path = dst_path + ".tmp"
    with open(tmp_path, "wb") as output:
        dim = append_prefix(train_path, output, train_count)
        append_prefix(test_path, output, test_count, expected_dim=dim)
    os.replace(tmp_path, dst_path)
    print(">> wrote {} ({} training + {} testing rows)".format(dst_path, train_count, test_count))


def truncate_prefix(src_path, new_k, count):
    available, stored_k = vecs_shape(src_path)
    if count > available:
        raise ValueError("{}: requested {} rows, only {} available".format(src_path, count, available))
    if new_k > stored_k:
        raise ValueError("{}: requested k={} > stored k={}".format(src_path, new_k, stored_k))

    raw = np.memmap(src_path, dtype=np.int32, mode="r", shape=(available, stored_k + 1))
    output = np.empty((count, new_k + 1), dtype=np.int32)
    output[:, 0] = new_k
    output[:, 1:] = raw[:count, 1:new_k + 1]
    return output


def write_ground_truth(train_path, test_path, k, train_count, test_count, dst_path):
    train = truncate_prefix(train_path, k, train_count)
    test = truncate_prefix(test_path, k, test_count)
    tmp_path = dst_path + ".tmp"
    with open(tmp_path, "wb") as output:
        train.tofile(output)
        test.tofile(output)
    os.replace(tmp_path, dst_path)
    print(">> wrote {} ({} rows x k={})".format(dst_path, train_count + test_count, k))


def symlink(src, dst):
    if os.path.lexists(dst):
        if os.path.islink(dst) and os.path.realpath(dst) == os.path.realpath(src):
            return
        os.remove(dst)
    os.symlink(src, dst)
    print(">> symlinked {} -> {}".format(dst, src))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    k_group = parser.add_mutually_exclusive_group()
    k_group.add_argument("--k", type=int, help="prepare one k value (legacy shortcut)")
    k_group.add_argument("--k-values", nargs="+", type=int, help="k values to prepare")
    parser.add_argument("--calibration-count", type=int, default=DEFAULT_CALIBRATION_COUNT)
    parser.add_argument("--tuning-count", type=int, default=DEFAULT_TUNING_COUNT)
    parser.add_argument("--test-count", type=int, default=DEFAULT_TEST_COUNT)
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=list(DATASETS))
    parser.add_argument("--darth-data-dir", default=DARTH_DATA_DIR)
    args = parser.parse_args()

    counts = (args.calibration_count, args.tuning_count, args.test_count)
    if any(count <= 0 for count in counts):
        parser.error("all split counts must be positive")
    k_values = args.k_values or ([args.k] if args.k is not None else list(DEFAULT_K_VALUES))
    if any(k <= 0 for k in k_values):
        parser.error("all k values must be positive")
    k_values = sorted(set(k_values))
    train_count = args.calibration_count + args.tuning_count

    for name in args.datasets:
        info = DATASETS[name]
        src_dir = os.path.join(args.darth_data_dir, info["darth_dir"])
        dst_dir = os.path.join(CONANN_DATA_DIR, name)
        os.makedirs(dst_dir, exist_ok=True)

        train_query = query_path(src_dir, *info["train"])
        test_query = query_path(src_dir, *info["test"])
        train_ids, train_distances = gt_paths(src_dir, *info["train"])
        test_ids, test_distances = gt_paths(src_dir, *info["test"])

        print("\n=== {} (DARTH directory: {}) ===".format(name, info["darth_dir"]))
        symlink(os.path.join(src_dir, info["base"]), os.path.join(dst_dir, "db.fvecs"))
        write_query_pool(
            train_query,
            test_query,
            train_count,
            args.test_count,
            os.path.join(dst_dir, "queries.fvecs"),
        )
        for k in k_values:
            write_ground_truth(
                train_ids,
                test_ids,
                k,
                train_count,
                args.test_count,
                os.path.join(dst_dir, "indices-{}.fvecs".format(k)),
            )
            write_ground_truth(
                train_distances,
                test_distances,
                k,
                train_count,
                args.test_count,
                os.path.join(dst_dir, "distances-{}.fvecs".format(k)),
            )

        metadata = {
            "dataset": name,
            "darth_dataset": info["darth_dir"],
            "k_values": k_values,
            "calibration_queries": args.calibration_count,
            "tuning_queries": args.tuning_count,
            "testing_queries": args.test_count,
            "total_queries": sum(counts),
            "query_order": ["calibration", "tuning", "testing"],
            "training_query_source": train_query,
            "testing_query_source": test_query,
        }
        metadata_path = os.path.join(dst_dir, "split.ini")
        write_split(metadata_path, metadata)
        print(">> wrote {}".format(metadata_path))


if __name__ == "__main__":
    main()
