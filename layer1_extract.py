"""
Layer 1 — Feature Extraction (Dask, embarrassingly parallel)
Each song file is processed independently; workers write local HDF5 shards.
Master merges shards via merge_shards.py.
"""
import os
import sys
import h5py
import numpy as np
from pathlib import Path
from typing import List, Tuple

import dask
import dask.bag as db
from dask.distributed import Client, LocalCluster


# ---------------------------------------------------------------------------
# Feature extraction (single song)
# ---------------------------------------------------------------------------

def extract_one(fpath: str) -> Tuple[str, np.ndarray]:
    """Read one MSD HDF5 file and return (song_id, feature_vector)."""
    with h5py.File(fpath, "r") as f:
        meta = f["metadata/songs"][0]
        ana  = f["analysis/songs"][0]

        song_id    = meta["song_id"].decode()
        tempo      = float(ana["tempo"])
        loudness   = float(ana["loudness"])
        duration   = float(ana["duration"])
        key        = float(ana["key"])
        mode       = float(ana["mode"])
        time_sig   = float(ana["time_signature"])

        # Segment-level aggregates (mean / std of pitch & timbre)
        pitches = f["analysis/segments_pitches"][:]   # (T, 12)
        timbre  = f["analysis/segments_timbre"][:]    # (T, 12)
        pitch_mean, pitch_std = pitches.mean(0), pitches.std(0)
        timbre_mean, timbre_std = timbre.mean(0), timbre.std(0)

    feat = np.concatenate([
        [tempo, loudness, duration, key, mode, time_sig],
        pitch_mean, pitch_std,
        timbre_mean, timbre_std,
    ]).astype(np.float32)   # dim = 6 + 48 = 54

    return song_id, feat


# ---------------------------------------------------------------------------
# Worker: process a partition and write a local shard
# ---------------------------------------------------------------------------

def process_partition(paths: List[str], shard_path: str) -> str:
    """Process a list of file paths and write results to a shard HDF5."""
    results = []
    for p in paths:
        try:
            results.append(extract_one(p))
        except Exception as e:
            print(f"[WARN] skipping {p}: {e}", file=sys.stderr)

    if not results:
        return shard_path

    ids, feats = zip(*results)
    with h5py.File(shard_path, "w") as f:
        f.create_dataset("song_ids", data=np.array(ids, dtype="S18"))
        f.create_dataset("features", data=np.array(feats), compression="gzip")

    return shard_path


# ---------------------------------------------------------------------------
# Master: scan, sort, dispatch
# ---------------------------------------------------------------------------

def scan_files(root: str) -> List[str]:
    """Recursively find all MSD .h5 files, sorted largest-first (straggler avoidance)."""
    paths = list(Path(root).rglob("*.h5"))
    paths.sort(key=lambda p: p.stat().st_size, reverse=True)
    return [str(p) for p in paths]


def run(msd_root: str, out_dir: str, n_workers: int, partition_size: int = 5000):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    shard_dir = Path(out_dir) / "shards"
    shard_dir.mkdir(exist_ok=True)

    paths = scan_files(msd_root)
    print(f"Found {len(paths)} song files")

    # Build Dask bag; each partition → one worker task
    bag = db.from_sequence(paths, partition_size=partition_size)

    def worker(partition, block_id=None):
        shard_path = str(shard_dir / f"shard_{block_id[0]:05d}.h5")
        return process_partition(list(partition), shard_path)

    with LocalCluster(n_workers=n_workers, threads_per_worker=1) as cluster, \
         Client(cluster) as client:
        futures = bag.map_partitions(worker).compute()

    print(f"Wrote {len(futures)} shards to {shard_dir}")
    return list(futures)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--msd-root", required=True)
    p.add_argument("--out-dir",  default="data/layer1")
    p.add_argument("--n-workers", type=int, default=32)
    p.add_argument("--partition-size", type=int, default=5000)
    args = p.parse_args()
    run(args.msd_root, args.out_dir, args.n_workers, args.partition_size)
