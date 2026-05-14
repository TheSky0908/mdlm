"""
Neural discriminator h_φ(x_t, t) for hard-constrained discrete diffusion.

Theory (from FHS derivation):
    h_t(x) = P(X_0 ∈ C | X_t = x)

Training loss (BCE, correcting the typo in the original note – log(1-h) not 1-log h):
    L(φ) = -E_{x0~data, t~U[0,T], x_t~q(·|x0)} [
        1{x0∈C} log h_φ(x_t, t) + (1-1{x0∈C}) log(1 - h_φ(x_t, t))
    ]

During FHS sampling the modified token distribution at each masked position i is:
    P(a | x, τ, i) ∝ μ^i(a | x, t) · h_t(x^{-i}⊕a)

where x^{-i}⊕a denotes x with position i replaced by token a, and
μ^i(a|x,t) is the diffusion model's predicted probability for token a.
"""
from __future__ import annotations

import math
import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Architecture
# ---------------------------------------------------------------------------

class _SinusoidalTimeEmbed(nn.Module):
    """Maps scalar t ∈ [0,1] → dense embedding via sinusoidal encoding + MLP."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, t: Tensor) -> Tensor:
        # t: [B]
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device) / half
        )
        args = t[:, None] * freqs[None]  # [B, half]
        emb = torch.cat([args.sin(), args.cos()], dim=-1)  # [B, dim]
        return self.mlp(emb)


class HDiscriminator(nn.Module):
    """
    Small transformer classifier: h_φ(x_t, t) → (0, 1).

    Architecture:
        token_embed + pos_embed + time_embed → [CLS | seq] → TransformerEncoder
        → CLS token → MLP head → sigmoid
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
        max_length: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_size = hidden_size

        self.token_embed = nn.Embedding(vocab_size, hidden_size)
        self.pos_embed = nn.Embedding(max_length + 1, hidden_size)  # +1 for CLS
        self.time_embed = _SinusoidalTimeEmbed(hidden_size)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        nn.init.normal_(self.cls_token, std=0.02)

        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, x_t: Tensor, t: Tensor) -> Tensor:
        """
        Args:
            x_t: [B, L] integer token ids
            t:   [B]    time values in [0, 1]
        Returns:
            h:   [B]    probability in (0, 1)
        """
        B, L = x_t.shape
        pos = torch.arange(L, device=x_t.device)

        tok_emb = self.token_embed(x_t)          # [B, L, H]
        pos_emb = self.pos_embed(pos)[None]       # [1, L, H]
        time_emb = self.time_embed(t)[:, None]    # [B, 1, H]

        seq = tok_emb + pos_emb + time_emb        # [B, L, H]

        cls = self.cls_token.expand(B, -1, -1)    # [B, 1, H]
        h = torch.cat([cls, seq], dim=1)           # [B, L+1, H]

        h = self.transformer(h)
        cls_out = h[:, 0]                          # [B, H]

        logit = self.head(cls_out).squeeze(-1)     # [B]
        return torch.sigmoid(logit)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def discriminator_loss(h: Tensor, labels: Tensor) -> Tensor:
    """
    Binary cross-entropy loss.
        L = -E[ y log h + (1-y) log(1-h) ]
    Args:
        h:      [B] discriminator output in (0,1)
        labels: [B] float 1.0 if x0 ∈ C else 0.0
    """
    return F.binary_cross_entropy(h, labels.float(), reduction="mean")


def train_discriminator(
    discriminator: HDiscriminator,
    diffusion,          # Diffusion model – used only for q_xt
    constraint,         # BaseConstraint
    train_dataloader,
    config,
    device: torch.device,
    save_path: Optional[str] = None,
) -> HDiscriminator:
    """
    Train h_φ with BCE loss.

    Training distribution:
        x0  ~ dataset
        t   ~ Uniform[eps, 1]         (eps from config.training.sampling_eps)
        x_t ~ q(·|x0) = absorbing mask with prob t
        y   = 1{x0 ∈ C}
    """
    lr = getattr(config.discriminator, "lr", 3e-4)
    wd = getattr(config.discriminator, "weight_decay", 0.0)
    max_steps = getattr(config.discriminator, "max_steps", 10_000)
    log_every = getattr(config.discriminator, "log_every", 100)
    eps = config.training.sampling_eps

    optimizer = torch.optim.AdamW(
        discriminator.parameters(), lr=lr, weight_decay=wd
    )

    discriminator.to(device).train()
    step = 0
    running_loss = 0.0

    for batch in train_dataloader:
        if isinstance(batch, dict):
            x0 = batch["input_ids"].to(device)
        else:
            x0 = batch[0].to(device)

        B = x0.shape[0]

        # Sample t ~ Uniform[eps, 1]
        t = torch.rand(B, device=device) * (1 - eps) + eps  # [B]

        # Forward diffuse: mask each token independently with prob t
        move_chance = t[:, None]                             # [B, 1]
        xt = diffusion.q_xt(x0, move_chance)                # [B, L]

        # Constraint labels
        labels = constraint.check(x0).float()               # [B]

        h = discriminator(xt, t)
        loss = discriminator_loss(h, labels)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(discriminator.parameters(), 1.0)
        optimizer.step()

        running_loss += loss.item()
        step += 1

        if step % log_every == 0:
            avg = running_loss / log_every
            running_loss = 0.0
            sat_frac = labels.mean().item()
            print(
                f"[Discriminator] step={step:6d}  loss={avg:.4f}"
                f"  constraint_rate={sat_frac:.3f}"
            )

        if step >= max_steps:
            break

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(discriminator.state_dict(), save_path)
        print(f"[Discriminator] Saved to {save_path}")

    return discriminator


def load_discriminator(path: str, config, vocab_size: int) -> HDiscriminator:
    disc = HDiscriminator(
        vocab_size=vocab_size,
        hidden_size=getattr(config.discriminator, "hidden_size", 256),
        num_layers=getattr(config.discriminator, "num_layers", 4),
        num_heads=getattr(config.discriminator, "num_heads", 4),
        max_length=getattr(config.discriminator, "max_length", 1024),
        dropout=getattr(config.discriminator, "dropout", 0.1),
    )
    disc.load_state_dict(torch.load(path, map_location="cpu"))
    return disc
