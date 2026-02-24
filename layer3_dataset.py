"""
Layer 3 — Sequence Dataset + Bucket Sampler
User listening history → next-song prediction (language-model framing).
"""
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler
from typing import List, Dict, Tuple, Iterator


PAD_ID = 0   # reserved padding token


class SongSequenceDataset(Dataset):
    """
    Each sample: (history_token_ids, target_token_id)
    history is a variable-length prefix of the user's ordered play events.
    """

    def __init__(self,
                 user_histories: Dict[int, List[int]],
                 max_len: int = 200):
        """
        Args:
            user_histories: {user_idx: [song_idx, song_idx, ...]} (ordered by time)
            max_len: hard cap on history length
        """
        self.samples: List[Tuple[List[int], int]] = []
        for uid, hist in user_histories.items():
            hist = hist[-max_len - 1:]          # keep tail
            for t in range(1, len(hist)):
                self.samples.append((hist[:t], hist[t]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        hist, target = self.samples[idx]
        return torch.tensor(hist, dtype=torch.long), torch.tensor(target, dtype=torch.long)


def collate_fn(batch):
    """Pad histories within a batch to the longest sequence in that batch."""
    seqs, targets = zip(*batch)
    max_len = max(s.size(0) for s in seqs)
    padded = torch.full((len(seqs), max_len), PAD_ID, dtype=torch.long)
    for i, s in enumerate(seqs):
        padded[i, :s.size(0)] = s
    targets = torch.stack(targets)
    lengths = torch.tensor([s.size(0) for s in seqs], dtype=torch.long)
    return padded, targets, lengths


class BucketSampler(Sampler):
    """
    Group samples by sequence length bucket so each batch has similar-length
    sequences → minimal padding waste → better GPU utilization under DDP.

    DDP note: each rank calls this sampler independently.  To keep length
    distributions balanced, shuffle within buckets using a shared seed.
    """

    def __init__(self, dataset: SongSequenceDataset,
                 batch_size: int,
                 bucket_width: int = 10,
                 shuffle: bool = True,
                 seed: int = 0):
        self.batch_size  = batch_size
        self.shuffle     = shuffle
        self.seed        = seed

        # Group indices by length bucket
        from collections import defaultdict
        buckets: Dict[int, List[int]] = defaultdict(list)
        for i, (hist, _) in enumerate(dataset.samples):
            bucket = len(hist) // bucket_width
            buckets[bucket].append(i)
        self.buckets = list(buckets.values())

    def __iter__(self) -> Iterator[int]:
        rng = np.random.default_rng(self.seed)
        order = []
        for bucket in self.buckets:
            idxs = np.array(bucket)
            if self.shuffle:
                rng.shuffle(idxs)
            # Yield in batch-sized chunks so each batch stays within bucket
            for start in range(0, len(idxs), self.batch_size):
                order.extend(idxs[start:start + self.batch_size].tolist())
        return iter(order)

    def __len__(self) -> int:
        return sum(len(b) for b in self.buckets)
