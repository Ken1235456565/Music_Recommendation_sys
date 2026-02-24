"""
Layer 3 — DDP Training Entry Point
Preloads HDF5 feature vectors into /dev/shm before spawning workers.

Launch (Slurm):  torchrun --nproc_per_node=$GPUS_PER_NODE train_ddp.py [args]
"""
import os
import sys
import argparse
import time
import pickle
import shutil
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from layer3.model   import TransformerRecommender
from layer3.dataset import SongSequenceDataset, BucketSampler, collate_fn
from utils.hdf5_io  import read_all_vectors


# ---------------------------------------------------------------------------
# /dev/shm preloading (rank 0 only, others wait)
# ---------------------------------------------------------------------------

SHM_DIR = "/dev/shm/hpc_music"


def preload_to_shm(features_h5: str, als_h5: str) -> None:
    """Copy HDF5 feature arrays into shared memory as .npy files."""
    os.makedirs(SHM_DIR, exist_ok=True)
    (user_ids, user_vecs), (song_ids, song_vecs) = read_all_vectors(als_h5)
    np.save(f"{SHM_DIR}/user_ids.npy",  user_ids)
    np.save(f"{SHM_DIR}/user_vecs.npy", user_vecs)
    np.save(f"{SHM_DIR}/song_ids.npy",  song_ids.astype("U18"))
    np.save(f"{SHM_DIR}/song_vecs.npy", song_vecs)

    # Layer 1 content features
    import h5py
    with h5py.File(features_h5, "r") as f:
        c_ids  = f["song_ids"][:].astype(str)
        c_feats = f["features"][:]
    np.save(f"{SHM_DIR}/content_ids.npy",  c_ids.astype("U18"))
    np.save(f"{SHM_DIR}/content_vecs.npy", c_feats)
    print(f"[rank 0] Preloaded to {SHM_DIR}")


def load_from_shm():
    """Each GPU rank loads vectors from /dev/shm (no disk I/O)."""
    user_ids   = np.load(f"{SHM_DIR}/user_ids.npy")
    user_vecs  = np.load(f"{SHM_DIR}/user_vecs.npy")
    song_ids   = np.load(f"{SHM_DIR}/song_ids.npy")
    song_vecs  = np.load(f"{SHM_DIR}/song_vecs.npy")
    c_ids      = np.load(f"{SHM_DIR}/content_ids.npy")
    c_vecs     = np.load(f"{SHM_DIR}/content_vecs.npy")
    return (user_ids, user_vecs, song_ids, song_vecs, c_ids, c_vecs)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args):
    dist.init_process_group("nccl")
    rank       = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    device     = torch.device(f"cuda:{local_rank}")

    # ---- Preload shared memory on rank 0, barrier for others ----
    if rank == 0:
        preload_to_shm(args.features_h5, args.als_h5)
    dist.barrier()

    user_ids, user_vecs, song_ids, song_vecs, c_ids, c_vecs = load_from_shm()
    n_songs = len(song_ids)

    user_idx  = {u: i for i, u in enumerate(user_ids)}
    song_idx  = {s: i for i, s in enumerate(song_ids)}
    c_idx     = {s: i for i, s in enumerate(c_ids)}

    # ---- Dataset (loaded from pickle of user histories) ----
    with open(args.histories, "rb") as f:
        user_histories = pickle.load(f)   # {user_idx: [song_idx, ...]}

    dataset = SongSequenceDataset(user_histories, max_len=args.max_seq)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    loader  = DataLoader(dataset, batch_size=args.batch_size,
                         sampler=sampler, collate_fn=collate_fn,
                         num_workers=4, pin_memory=True)

    # ---- Model ----
    model = TransformerRecommender(
        n_songs=n_songs,
        als_dim=user_vecs.shape[1],
        content_dim=c_vecs.shape[1],
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
    ).to(device)
    model = DDP(model, device_ids=[local_rank])

    # Pre-move lookup tables to GPU tensors
    user_vecs_t = torch.tensor(user_vecs, dtype=torch.float32, device=device)
    song_vecs_t = torch.tensor(c_vecs,   dtype=torch.float32, device=device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        model.train()
        total_loss = 0.0
        t0 = time.perf_counter()

        for batch_idx, (seq, target, lengths) in enumerate(loader):
            seq, target, lengths = seq.to(device), target.to(device), lengths.to(device)

            # Lookup ALS / content vectors for last song in each history (on GPU)
            last_song_ids = seq[torch.arange(len(seq)), (lengths - 1).clamp(min=0)]
            als_v     = song_vecs_t[last_song_ids]      # (B, als_dim)  — placeholder
            content_v = song_vecs_t[last_song_ids]      # (B, content_dim)

            optimizer.zero_grad()
            logits = model(seq, als_v, content_v, lengths)
            loss   = criterion(logits, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        elapsed = time.perf_counter() - t0
        if rank == 0:
            avg = total_loss / max(len(loader), 1)
            print(f"[epoch {epoch+1}] loss={avg:.4f}  time={elapsed:.1f}s")

        # Checkpoint on rank 0
        if rank == 0 and (epoch + 1) % args.save_every == 0:
            ckpt = f"{args.out_dir}/ckpt_epoch{epoch+1}.pt"
            torch.save(model.module.state_dict(), ckpt)
            print(f"  Saved {ckpt}")

    dist.destroy_process_group()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--features-h5",  required=True)
    p.add_argument("--als-h5",       required=True)
    p.add_argument("--histories",    required=True, help="Pickle: {user_idx: [song_idx,...]}")
    p.add_argument("--out-dir",      default="data/checkpoints")
    p.add_argument("--epochs",       type=int,   default=10)
    p.add_argument("--batch-size",   type=int,   default=256)
    p.add_argument("--lr",           type=float, default=1e-4)
    p.add_argument("--d-model",      type=int,   default=256)
    p.add_argument("--n-heads",      type=int,   default=8)
    p.add_argument("--n-layers",     type=int,   default=4)
    p.add_argument("--max-seq",      type=int,   default=200)
    p.add_argument("--save-every",   type=int,   default=2)
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    train(args)
