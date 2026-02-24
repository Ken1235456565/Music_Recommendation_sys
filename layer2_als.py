"""
Layer 2 — Distributed ALS via mpi4py
Each MPI rank holds a row-partition of the user matrix (CSR).
Step A: update user vectors  → AllGather item vectors
Step B: update item vectors  → CSR→CSC transpose (≈ AllToAll) + AllGather user vectors

Run:
    mpirun -n 8 python distributed_als.py --data-dir data/layer2 --out data/als_vectors.h5
"""
import argparse
import pickle
import time
import numpy as np
import scipy.sparse as sp
from mpi4py import MPI

from utils.hdf5_io import write_als_vectors


# ---------------------------------------------------------------------------
# ALS core (single-rank, operates on local row partition)
# ---------------------------------------------------------------------------

def als_update_users(Rlocal: sp.csr_matrix, Y: np.ndarray,
                     lambda_: float) -> np.ndarray:
    """Update local user vectors given fixed item matrix Y."""
    n_factors = Y.shape[1]
    YtY = Y.T @ Y + lambda_ * np.eye(n_factors)
    X = np.zeros((Rlocal.shape[0], n_factors), dtype=np.float32)
    for i in range(Rlocal.shape[0]):
        row = Rlocal.getrow(i)
        idx = row.indices
        if len(idx) == 0:
            continue
        Yi = Y[idx]
        ri = row.data
        A = YtY + (lambda_ * (len(idx) - 1)) * np.eye(n_factors) + Yi.T @ Yi
        b = Yi.T @ ri
        X[i] = np.linalg.solve(A, b)
    return X


def als_update_items(Rlocal_csc: sp.csc_matrix, X: np.ndarray,
                     lambda_: float) -> np.ndarray:
    """Update local item vectors given fixed user matrix X."""
    n_factors = X.shape[1]
    XtX = X.T @ X + lambda_ * np.eye(n_factors)
    Y = np.zeros((Rlocal_csc.shape[1], n_factors), dtype=np.float32)
    for j in range(Rlocal_csc.shape[1]):
        col = Rlocal_csc.getcol(j)
        idx = col.indices
        if len(idx) == 0:
            continue
        Xi = X[idx]
        rj = col.data
        A = XtX + (lambda_ * (len(idx) - 1)) * np.eye(n_factors) + Xi.T @ Xi
        b = Xi.T @ rj
        Y[j] = np.linalg.solve(A, b)
    return Y


# ---------------------------------------------------------------------------
# Distributed driver
# ---------------------------------------------------------------------------

def run_als(data_dir: str, out_path: str,
            n_factors: int = 128, n_iters: int = 15,
            lambda_: float = 0.01):

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # ---- Load data on rank 0, scatter row partitions ----
    if rank == 0:
        csr_full = sp.load_npz(f"{data_dir}/interaction_csr.npz")
        with open(f"{data_dir}/user2idx.pkl", "rb") as f:
            user2idx = pickle.load(f)
        with open(f"{data_dir}/song2idx.pkl", "rb") as f:
            song2idx = pickle.load(f)
        n_users, n_songs = csr_full.shape
        rows_per_rank = [n_users // size + (1 if i < n_users % size else 0)
                         for i in range(size)]
        splits = np.cumsum([0] + rows_per_rank)
        partitions = [csr_full[splits[i]:splits[i+1]] for i in range(size)]
        print(f"ALS: {n_users} users × {n_songs} songs, "
              f"{n_factors} factors, {size} ranks")
    else:
        partitions = None
        n_songs = None
        user2idx = song2idx = None

    Rlocal: sp.csr_matrix = comm.scatter(partitions, root=0)
    n_songs = comm.bcast(n_songs, root=0)
    user2idx = comm.bcast(user2idx, root=0)
    song2idx = comm.bcast(song2idx, root=0)

    # ---- Initialize latent matrices ----
    rng = np.random.default_rng(42 + rank)
    X_local = rng.standard_normal((Rlocal.shape[0], n_factors)).astype(np.float32) * 0.01
    Y = rng.standard_normal((n_songs, n_factors)).astype(np.float32) * 0.01

    # ---- ALS iterations ----
    for it in range(n_iters):
        t0 = time.perf_counter()

        # Step A: update local user vectors (item vectors Y broadcast)
        Y = comm.bcast(Y, root=0)          # AllGather equivalent for read-only
        X_local = als_update_users(Rlocal, Y, lambda_)
        t_a = time.perf_counter() - t0

        # Step B: update item vectors
        # Transpose local CSR → CSC (the communication bottleneck step)
        t_transpose = time.perf_counter()
        Rlocal_csc = Rlocal.T.tocsc()     # local transpose only
        t_transpose = time.perf_counter() - t_transpose

        # Gather all user vectors to update items
        X_all_parts = comm.allgather(X_local)
        X_all = np.vstack(X_all_parts)     # global user matrix (AllGather)
        Y = als_update_items(Rlocal_csc, X_all, lambda_)
        # Reduce item vectors across ranks (each rank computed same columns — average)
        comm.Allreduce(MPI.IN_PLACE, Y, op=MPI.SUM)
        Y /= size

        t_iter = time.perf_counter() - t0
        if rank == 0:
            print(f"[iter {it+1:3d}] total={t_iter:.2f}s  "
                  f"user-update={t_a:.2f}s  transpose={t_transpose:.4f}s")

    # ---- Gather results & save ----
    X_all_parts = comm.gather(X_local, root=0)
    if rank == 0:
        X_final = np.vstack(X_all_parts)
        user_ids = list(user2idx.keys())
        song_ids = list(song2idx.keys())
        write_als_vectors(out_path,
                          list(range(len(user_ids))), X_final,
                          song_ids, Y)
        print(f"ALS complete. Vectors saved to {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir",  default="data/layer2")
    p.add_argument("--out",       default="data/als_vectors.h5")
    p.add_argument("--n-factors", type=int, default=128)
    p.add_argument("--n-iters",   type=int, default=15)
    p.add_argument("--lambda",    type=float, default=0.01, dest="lambda_")
    args = p.parse_args()
    run_als(args.data_dir, args.out, args.n_factors, args.n_iters, args.lambda_)
