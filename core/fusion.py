"""Bidirectional Cross-Attention Fusion.

Implements Section 4.3 of DESIGN.md:
- Inputs: spatial tokens S [B, 49, 256], spectral tokens P [B, 49, 256]
- Block 1 (spatial queries spectral): Q=S W_q, K=P W_k, V=P W_v -> A_sp = softmax(Q K^T / sqrt(d)) V
- Block 2 (spectral queries spatial): Q=P W'_q, K=S W'_k, V=S W'_v -> A_pe = softmax(Q K^T / sqrt(d)) S
- Residual and LayerNorm on each block output: LayerNorm(S + A_sp), LayerNorm(P + A_pe)
- Pooling: Mean over 49 tokens of each stream, then sum -> fused [B, 256]
- Attention weights returned alongside fused vector for explainability
"""

from typing import Tuple, Optional
import math
import torch
import torch.nn as nn


class CrossAttentionBlock(nn.Module):
    """Single-head or multi-head cross-attention block with residual and LayerNorm."""

    def __init__(
        self,
        dim: int = 256,
        num_heads: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        if self.head_dim * num_heads != dim:
            raise ValueError(f"dim {dim} must be divisible by num_heads {num_heads}")

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)
        self.scale = 1.0 / math.sqrt(self.head_dim)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute cross-attention.
        
        Args:
            query: Query tensor [B, N_q, dim]
            key: Key tensor [B, N_k, dim]
            value: Value tensor [B, N_k, dim]
            
        Returns:
            Tuple of:
            - Attended normalized output with residual [B, N_q, dim]
            - Attention weights [B, num_heads, N_q, N_k] or [B, N_q, N_k] if num_heads == 1
        """
        b, n_q, _ = query.shape
        _, n_k, _ = key.shape

        # Projections and reshape for multi-head
        # [B, N, H, D_h] -> [B, H, N, D_h]
        q = self.q_proj(query).view(b, n_q, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(b, n_k, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(b, n_k, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, H, N_q, N_k]
        attn_weights = torch.softmax(scores, dim=-1)
        dropped_weights = self.dropout(attn_weights)

        # Attention output: [B, H, N_q, D_h] -> [B, N_q, H * D_h] = [B, N_q, dim]
        context = torch.matmul(dropped_weights, v)
        context = context.transpose(1, 2).contiguous().view(b, n_q, self.dim)
        context = self.out_proj(context)

        # Residual connection + LayerNorm
        out = self.norm(query + context)

        # Return weights (squeeze head dimension if single head)
        if self.num_heads == 1:
            attn_weights_ret = attn_weights.squeeze(1)  # [B, N_q, N_k]
        else:
            attn_weights_ret = attn_weights

        return out, attn_weights_ret


class CrossAttentionFusion(nn.Module):
    """Bidirectional Cross-Attention Fusion module.
    
    Spatial tokens attend to spectral tokens, and spectral tokens attend to
    spatial tokens. Attended representations are pooled and summed.
    """

    def __init__(
        self,
        dim: int = 256,
        num_heads: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads

        # Block 1: Spatial queries Spectral (Q=Spatial, K=Spectral, V=Spectral)
        self.spatial_queries_spectral = CrossAttentionBlock(
            dim=dim, num_heads=num_heads, dropout=dropout
        )

        # Block 2: Spectral queries Spatial (Q=Spectral, K=Spatial, V=Spatial)
        self.spectral_queries_spatial = CrossAttentionBlock(
            dim=dim, num_heads=num_heads, dropout=dropout
        )

    def forward(
        self,
        spatial_tokens: torch.Tensor,
        spectral_tokens: torch.Tensor,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Bidirectionally fuse spatial and spectral tokens.
        
        Args:
            spatial_tokens: [B, 49, 256]
            spectral_tokens: [B, 49, 256]
            
        Returns:
            Tuple of:
            - Fused vector [B, 256]
            - Tuple of attention weights:
              (spatial_to_spectral_weights [B, 49, 49], spectral_to_spatial_weights [B, 49, 49])
        """
        # Block 1: Spatial attends to Spectral
        attended_spatial, sp_attn_weights = self.spatial_queries_spectral(
            query=spatial_tokens,
            key=spectral_tokens,
            value=spectral_tokens,
        )

        # Block 2: Spectral attends to Spatial
        attended_spectral, pe_attn_weights = self.spectral_queries_spatial(
            query=spectral_tokens,
            key=spatial_tokens,
            value=spatial_tokens,
        )

        # Pooling: Mean over the 49 tokens of each stream, then sum -> [B, 256]
        pool_spatial = attended_spatial.mean(dim=1)   # [B, 256]
        pool_spectral = attended_spectral.mean(dim=1) # [B, 256]
        fused = pool_spatial + pool_spectral          # [B, 256]

        return fused, (sp_attn_weights, pe_attn_weights)
