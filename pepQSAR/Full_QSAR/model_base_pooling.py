import math
import torch
import torch.nn as nn
from typing import Optional

class PositionalEncoding(nn.Module):
    """Learnable positional encoding (same interface as standard PositionalEncoding)."""

    def __init__(self, d_model: int, max_len: int = 50, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pos_embedding = nn.Embedding(max_len, d_model)

        # Weight init can use PyTorch defaults or be set explicitly:
        nn.init.normal_(self.pos_embedding.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch, seq_len, d_model]
        """
        bsz, seq_len, _ = x.size()
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(bsz, -1)
        x = x + self.pos_embedding(positions)  # [B, L, d_model]
        return self.dropout(x)


class TransformerRegressor(nn.Module):
    """
    Multi-task regression model based on Transformer (attention pooling):
    - Input: [batch, seq_len, input_dim]
    - Architecture: Linear projection -> positional encoding -> TransformerEncoder -> attention pooling -> multi-task heads
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 6,
        dropout: float = 0.1,
        output_dim: int = 1,
        max_len: int = 50,
    ):
        super().__init__()

        self.d_model = d_model
        self.projection = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_len, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
            bias=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.layer_norm = nn.LayerNorm(d_model)

        # Attention pooling: score each position, learn weights, then weighted sum
        self.attn_pool = nn.Linear(d_model, 1)

        # Multi-task heads: one linear layer per task
        self.heads = nn.ModuleList([nn.Linear(d_model, 1) for _ in range(output_dim)])
        # # Shared output layer: one linear layer for all task outputs
        # self.fc = nn.Linear(d_model, output_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)
        nn.init.xavier_uniform_(self.attn_pool.weight)
        nn.init.zeros_(self.attn_pool.bias)
        # nn.init.xavier_uniform_(self.fc.weight)
        # nn.init.zeros_(self.fc.bias)
        for head in self.heads:
            nn.init.xavier_uniform_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(
        self,
        src: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        src : [batch, seq_len, input_dim]
        src_key_padding_mask : [batch, seq_len], True marks positions to mask (padding)

        Returns
        -------
        out : [batch, output_dim], multi-task regression output
        """
        # 1. Linear projection to d_model
        x = self.projection(src)  # [B, L, d_model]
        x = x * math.sqrt(self.d_model)

        # 2. Add positional encoding
        x = self.pos_encoder(x)  # [B, L, d_model]

        # 3. Transformer encoding
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)  # [B, L, d_model]
        x = self.layer_norm(x)

        # 4. Attention pooling
        if src_key_padding_mask is not None:
            # valid_mask: True marks valid positions
            valid_mask = ~src_key_padding_mask  # [B, L]
        else:
            valid_mask = torch.ones(x.size()[:2], dtype=torch.bool, device=x.device)

        # 4.1 Score each position [B, L]
        scores = self.attn_pool(x).squeeze(-1)
        # Set padding scores to a very small value so they are excluded from softmax
        scores = scores.masked_fill(~valid_mask, -1e9)

        # 4.2 Normalize to weights [B, L]
        alpha = torch.softmax(scores, dim=1)

        # 4.3 Weighted sum to get global representation [B, d_model]
        x_pooled = (x * alpha.unsqueeze(-1)).sum(dim=1)

        
        # 5. Multi-task head output [B, output_dim]
        outs = [head(x_pooled).squeeze(-1) for head in self.heads]
        out = torch.stack(outs, dim=-1)
        # # 5. Shared output layer -> [B, output_dim]
        # out = self.fc(x_pooled)
        return out

