"""
Layer 1 Reducer — merge per-worker HDF5 shards into features.h5
Run as a single process after all Dask workers complete.
"""
import argparse
from pathlib import Path
from utils.hdf5_io import merge_shards


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-dir",  required=True,  help="Directory containing shard_*.h5")
    parser.add_argument("--output",     default="data/features.h5")
    args = parser.parse_args()

    shards = sorted(Path(args.shard_dir).glob("shard_*.h5"))
    if not shards:
        raise FileNotFoundError(f"No shards found in {args.shard_dir}")

    print(f"Merging {len(shards)} shards → {args.output}")
    merge_shards([str(s) for s in shards], args.output)
    print("Done.")


if __name__ == "__main__":
    main()
