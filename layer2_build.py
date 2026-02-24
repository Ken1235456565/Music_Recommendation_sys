"""
Layer 2 — Build CSR interaction matrix from MSD user-song play counts.
Output: data/interaction_csr.npz  (scipy sparse, CSR format)
Also dumps user_id / song_id index maps for downstream use.
"""
import argparse
import pickle
import numpy as np
import scipy.sparse as sp
from pathlib import Path


def load_triplets(filepath: str):
    """
    Load (user_id_str, song_id_str, play_count) triplets.
    MSD taste profile: tab-separated text file.
    """
    users, songs, counts = [], [], []
    with open(filepath) as fh:
        for line in fh:
            u, s, c = line.strip().split("\t")
            users.append(u)
            songs.append(s)
            counts.append(int(c))
    return users, songs, counts


def build_csr(triplet_file: str, out_dir: str):
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    print("Loading triplets …")
    users, songs, counts = load_triplets(triplet_file)

    # Build integer index maps
    unique_users = sorted(set(users))
    unique_songs = sorted(set(songs))
    user2idx = {u: i for i, u in enumerate(unique_users)}
    song2idx = {s: i for i, s in enumerate(unique_songs)}

    row = np.array([user2idx[u] for u in users], dtype=np.int32)
    col = np.array([song2idx[s] for s in songs],  dtype=np.int32)
    data = np.array(counts, dtype=np.float32)

    n_users, n_songs = len(unique_users), len(unique_songs)
    print(f"Matrix shape: {n_users} × {n_songs}, nnz={len(data)}")

    csr = sp.csr_matrix((data, (row, col)), shape=(n_users, n_songs))

    # Persist
    sp.save_npz(f"{out_dir}/interaction_csr.npz", csr)
    with open(f"{out_dir}/user2idx.pkl", "wb") as f:
        pickle.dump(user2idx, f)
    with open(f"{out_dir}/song2idx.pkl", "wb") as f:
        pickle.dump(song2idx, f)

    print(f"Saved to {out_dir}/")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--triplets", required=True, help="MSD taste profile triplets file")
    p.add_argument("--out-dir",  default="data/layer2")
    args = p.parse_args()
    build_csr(args.triplets, args.out_dir)
