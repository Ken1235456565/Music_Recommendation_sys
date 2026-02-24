"""Shared HDF5 read/write utilities for inter-layer data exchange."""
import h5py
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional


def write_features(path: str, song_ids: List[str], features: np.ndarray) -> None:
    """Write song content features to HDF5.
    
    Args:
        path: Output HDF5 file path
        song_ids: List of song ID strings
        features: Array of shape (N, feature_dim)
    """
    with h5py.File(path, "w") as f:
        f.create_dataset("song_ids", data=np.array(song_ids, dtype="S18"))
        f.create_dataset("features", data=features, compression="gzip", chunks=True)
        f.attrs["n_songs"] = len(song_ids)
        f.attrs["feature_dim"] = features.shape[1]


def read_features_by_id(path: str, song_ids: List[str]) -> Dict[str, np.ndarray]:
    """Random access: read feature vectors for given song IDs."""
    result = {}
    with h5py.File(path, "r") as f:
        stored_ids = f["song_ids"][:].astype(str)
        id_to_idx = {sid: i for i, sid in enumerate(stored_ids)}
        features = f["features"]
        for sid in song_ids:
            if sid in id_to_idx:
                result[sid] = features[id_to_idx[sid]]
    return result


def write_als_vectors(path: str,
                      user_ids: List[int], user_vectors: np.ndarray,
                      song_ids: List[str], song_vectors: np.ndarray) -> None:
    """Write ALS latent vectors to HDF5."""
    with h5py.File(path, "w") as f:
        ug = f.create_group("users")
        ug.create_dataset("ids", data=np.array(user_ids, dtype=np.int32))
        ug.create_dataset("vectors", data=user_vectors, compression="gzip", chunks=True)

        sg = f.create_group("songs")
        sg.create_dataset("ids", data=np.array(song_ids, dtype="S18"))
        sg.create_dataset("vectors", data=song_vectors, compression="gzip", chunks=True)

        f.attrs["n_factors"] = user_vectors.shape[1]


def read_all_vectors(path: str):
    """Load all vectors into memory (for /dev/shm preloading)."""
    with h5py.File(path, "r") as f:
        user_ids = f["users/ids"][:]
        user_vecs = f["users/vectors"][:]
        song_ids = f["songs/ids"][:].astype(str)
        song_vecs = f["songs/vectors"][:]
    return (user_ids, user_vecs), (song_ids, song_vecs)


def merge_shards(shard_paths: List[str], output_path: str) -> None:
    """Reducer: merge per-worker HDF5 shards into a single features.h5."""
    all_ids, all_feats = [], []
    for sp in shard_paths:
        with h5py.File(sp, "r") as f:
            all_ids.append(f["song_ids"][:])
            all_feats.append(f["features"][:])

    song_ids = np.concatenate(all_ids)
    features = np.concatenate(all_feats, axis=0)
    write_features(output_path, song_ids.astype(str).tolist(), features)
    print(f"Merged {len(shard_paths)} shards → {output_path} ({len(song_ids)} songs)")
