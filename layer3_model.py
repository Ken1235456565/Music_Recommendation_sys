"""
Layer 3 — Transformer Recommendation Model
Fuses collaborative (ALS latent vectors) + content (Layer 1 features)
embeddings with cold-start dynamic weighting.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ColdStartFusion(nn.Module):
    """
    Fuse ALS latent vector and content feature vector per song.
    Weight α is computed on-GPU from interaction count to avoid CPU→GPU transfer.
    α → 1  (content-heavy) when history is short (cold start)
    α → 0  (ALS-heavy)    when history is long
    """
    def __init__(self, als_dim: int, content_dim: int, out_dim: int):
        super().__init__()
        self.als_proj     = nn.Linear(als_dim,     out_dim)
        self.content_proj = nn.Linear(content_dim, out_dim)

    def forward(self,
                als_vec: torch.Tensor,       # (B, als_dim)
                content_vec: torch.Tensor,   # (B, content_dim)
                history_len: torch.Tensor    # (B,)  int, number of past interactions
               ) -> torch.Tensor:
        # α ∈ (0,1): shorter history → higher content weight
        alpha = torch.exp(-history_len.float() / 50.0).unsqueeze(-1)  # (B,1)
        fused = alpha * self.content_proj(content_vec) \
              + (1 - alpha) * self.als_proj(als_vec)   # (B, out_dim)
        return fused


class TransformerRecommender(nn.Module):
    """
    Sequence-to-next-item transformer.
    Input:  padded token sequence  (B, T)
    Output: logits over song vocab (B, n_songs)
    """
    def __init__(self,
                 n_songs: int,
                 als_dim: int      = 128,
                 content_dim: int  = 54,
                 d_model: int      = 256,
                 n_heads: int      = 8,
                 n_layers: int     = 4,
                 d_ff: int         = 1024,
                 dropout: float    = 0.1,
                 max_seq: int      = 200):
        super().__init__()
        self.d_model = d_model

        # Token embedding (song ID → dense vector)
        self.token_emb = nn.Embedding(n_songs + 1, d_model, padding_idx=0)
        self.pos_emb   = nn.Embedding(max_seq, d_model)

        # Cold-start fusion
        self.fusion = ColdStartFusion(als_dim, content_dim, d_model)

        # Transformer encoder
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_ff, dropout=dropout,
            batch_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        self.norm   = nn.LayerNorm(d_model)
        self.output = nn.Linear(d_model, n_songs)

    def forward(self,
                seq: torch.Tensor,           # (B, T)  padded token ids
                als_vec: torch.Tensor,        # (B, als_dim)
                content_vec: torch.Tensor,    # (B, content_dim)
                history_len: torch.Tensor,    # (B,)
                ) -> torch.Tensor:
        B, T = seq.shape
        positions = torch.arange(T, device=seq.device).unsqueeze(0)  # (1,T)

        # Sequence embedding
        x = self.token_emb(seq) + self.pos_emb(positions)            # (B,T,d)

        # Fuse cold-start info into the CLS-like first position
        fused = self.fusion(als_vec, content_vec, history_len)        # (B,d)
        x[:, 0] += fused

        # Causal mask: token at position i attends only to ≤ i
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=seq.device)

        # Padding mask
        pad_mask = (seq == 0)   # (B,T)  True where padded

        x = self.transformer(x, mask=mask, src_key_padding_mask=pad_mask)
        x = self.norm(x)

        # Use last non-padded position for prediction
        # history_len gives actual length; index is len-1
        idx = (history_len - 1).clamp(min=0)                          # (B,)
        last = x[torch.arange(B, device=x.device), idx]               # (B,d)
        return self.output(last)                                        # (B,n_songs)
