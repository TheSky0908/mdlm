"""
Hard-constrained FHS sampling for masked discrete diffusion (Case 2).

Theory recap (from the derivation notes):
    Let C be a constraint set.  Define the cost c(x) = 0 if x∈C, +∞ otherwise.

    h_t(x) = P(X_0 ∈ C | X_t = x)          ← discriminator target

    The constrained rate matrix (Doob h-transform):
        Q_t^λ(x,y) = Q_t(x,y) · h_t(y) / h_t(x)

    In the FHS setting the next-token distribution becomes:
        P(a | x, τ, i) ∝ μ^i(a|x,t) · h_t(x^{-i}⊕a)

    where μ^i(a|x,t) is the diffusion model's predicted token probability
    and x^{-i}⊕a is x with position i replaced by token a.

Two constraint backends are supported:
  1. Analytical (PrefixConstraint, BannedTokensConstraint): additive -inf mask
     applied to log p_x0.  Exact and fast – no extra neural network needed.
  2. Neural discriminator h_φ: for each masked position, top-K candidates are
     scored by h_φ(x^{-i}⊕a, t) and used to reweight the model distribution.
"""
from __future__ import annotations

import itertools
import time
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from diffusion import _sample_categorical   # reuse existing helper


# ---------------------------------------------------------------------------
# Core reweighting step
# ---------------------------------------------------------------------------

def apply_constraint_to_p_x0(
    p_x0: Tensor,
    x: Tensor,
    t_scalar: Tensor,
    mask_index: int,
    constraint,
    discriminator: Optional[nn.Module],
    topk: int = 50,
) -> Tensor:
    """
    Reweight p_x0 to encode the hard constraint h_t(x^{-i}⊕a).

    Args:
        p_x0:       [B, L, V] probabilities from the diffusion model
        x:          [B, L]    current noisy token ids
        t_scalar:   [B]       time values in [0, 1]
        mask_index: int       id of the [MASK] token
        constraint:           BaseConstraint with .logits_mask(), or None
        discriminator:        HDiscriminator, or None
        topk:       int       number of candidates to evaluate per position
                              when using the neural discriminator

    Returns:
        [B, L, V] reweighted probability tensor (sums to 1 over V)
    """
    # --- (a) Analytical constraint: additive -inf mask in log-space ----------
    if constraint is not None and hasattr(constraint, "logits_mask"):
        log_p = p_x0.clamp(min=1e-30).log()
        log_p = log_p + constraint.logits_mask(x, log_p)
        # Safe re-normalisation (some rows might be all -inf at the mask token)
        lse = torch.logsumexp(log_p, dim=-1, keepdim=True)
        log_p = log_p - lse
        p_x0 = log_p.exp()

    # --- (b) Neural discriminator: h_φ(x^{-i}⊕a, t) reweighting ------------
    if discriminator is not None:
        B, L, V = p_x0.shape
        p_x0 = p_x0.clone()

        for b in range(B):
            masked_pos = (x[b] == mask_index).nonzero(as_tuple=False).squeeze(1)
            if masked_pos.numel() == 0:
                continue

            for pos in masked_pos.tolist():
                k = min(topk, V)
                top_probs, top_ids = p_x0[b, pos].topk(k)  # [K]

                # Create K sequences each with a different token at `pos`
                x_cands = x[b].unsqueeze(0).expand(k, -1).clone()  # [K, L]
                x_cands[:, pos] = top_ids
                t_cands = t_scalar[b:b+1].expand(k)               # [K]

                with torch.no_grad():
                    h_scores = discriminator(x_cands, t_cands)     # [K] ∈ (0,1)

                # P(a|x,τ,i) ∝ μ^i(a|x,t) · h_t(x^{-i}⊕a)
                weighted = top_probs * (h_scores + 1e-8)
                weighted = weighted / (weighted.sum() + 1e-30)

                new_row = torch.zeros(V, device=p_x0.device)
                new_row.scatter_(0, top_ids, weighted)
                p_x0[b, pos] = new_row

    return p_x0


# ---------------------------------------------------------------------------
# Single denoising step
# ---------------------------------------------------------------------------

def ddpm_caching_step_constrained(
    diffusion,
    x: Tensor,
    t: Tensor,
    dt: float,
    constraint,
    discriminator: Optional[nn.Module],
    p_x0_cache: Optional[Tensor],
    topk: int = 50,
):
    """
    One constrained DDPM-cache denoising step.

    Returns (p_x0, x_next) mirroring the signature of _ddpm_caching_update.
    """
    assert diffusion.config.noise.type == "loglinear"
    sigma_t, _ = diffusion.noise(t)
    t_1d = t.squeeze(-1) if t.ndim > 1 else t   # [B]
    assert t_1d.ndim == 1

    move_chance_t = t_1d[:, None, None]
    move_chance_s = (t_1d - dt)[:, None, None]

    if p_x0_cache is None:
        p_x0 = diffusion.forward(x, sigma_t).exp()   # [B, L, V]
    else:
        p_x0 = p_x0_cache

    # Reweight by constraint / discriminator
    p_x0 = apply_constraint_to_p_x0(
        p_x0, x, t_1d,
        mask_index=diffusion.mask_index,
        constraint=constraint,
        discriminator=discriminator,
        topk=topk,
    )

    # Standard DDPM-cache transition q(x_s | x_t, x_0)
    q_xs = p_x0 * (move_chance_t - move_chance_s)
    q_xs[:, :, diffusion.mask_index] = move_chance_s[:, :, 0]
    x_next_candidate = _sample_categorical(q_xs)

    copy_flag = (x != diffusion.mask_index).to(x.dtype)
    x_next = copy_flag * x + (1 - copy_flag) * x_next_candidate

    return p_x0, x_next


# ---------------------------------------------------------------------------
# Full constrained sampling loop
# ---------------------------------------------------------------------------

@torch.no_grad()
def sample_fhs_constrained(
    diffusion,
    constraint,
    discriminator: Optional[nn.Module],
    num_steps: Optional[int] = None,
    eps: float = 1e-5,
    topk: int = 50,
) -> Tensor:
    """
    Full constrained FHS sampling from all-mask initial condition.

    Each denoising step applies the h_t reweighting so that the reverse
    process gradually concentrates mass on sequences in C.

    Args:
        diffusion:      Diffusion LightningModule (loaded, in eval mode)
        constraint:     BaseConstraint or None (analytical fast path)
        discriminator:  HDiscriminator or None
        num_steps:      number of denoising steps (default: config value)
        eps:            minimum time (avoids t=0 numerical issues)
        topk:           top-K vocab candidates when using discriminator

    Returns:
        x: [B, L] sampled token ids
    """
    batch_size = diffusion.config.loader.eval_batch_size
    if num_steps is None:
        num_steps = diffusion.config.sampling.steps

    x = diffusion._sample_prior(batch_size, diffusion.config.model.length).to(
        diffusion.device
    )
    timesteps = torch.linspace(1, eps, num_steps + 1, device=diffusion.device)
    dt = (1 - eps) / num_steps
    p_x0_cache = None

    for i in range(num_steps):
        t = timesteps[i] * torch.ones(x.shape[0], 1, device=diffusion.device)
        p_x0_cache, x_next = ddpm_caching_step_constrained(
            diffusion, x, t, dt,
            constraint=constraint,
            discriminator=discriminator,
            p_x0_cache=p_x0_cache,
            topk=topk,
        )
        if not torch.allclose(x_next, x):
            p_x0_cache = None
        x = x_next

    # Final argmax denoising with constraint enforced
    if diffusion.config.sampling.noise_removal:
        t_final = timesteps[-1] * torch.ones(x.shape[0], 1, device=diffusion.device)
        sigma_final = diffusion.noise(t_final)[0]
        p_x0_final = diffusion.forward(x, sigma_final).exp()
        p_x0_final = apply_constraint_to_p_x0(
            p_x0_final,
            x,
            timesteps[-1].expand(x.shape[0]),
            mask_index=diffusion.mask_index,
            constraint=constraint,
            discriminator=discriminator,
            topk=topk,
        )
        x = p_x0_final.argmax(dim=-1)

    return x


# ---------------------------------------------------------------------------
# Public entry-point (wraps EMA store/restore around sampling)
# ---------------------------------------------------------------------------

def restore_and_sample_constrained(
    diffusion,
    num_steps: int,
    constraint=None,
    discriminator: Optional[nn.Module] = None,
    eps: float = 1e-5,
    topk: int = 50,
) -> Tensor:
    """
    Load EMA weights, run constrained FHS sampling, restore weights.

    This mirrors diffusion.restore_model_and_sample() but with constraint.
    """
    if diffusion.ema:
        diffusion.ema.store(
            itertools.chain(
                diffusion.backbone.parameters(), diffusion.noise.parameters()
            )
        )
        diffusion.ema.copy_to(
            itertools.chain(
                diffusion.backbone.parameters(), diffusion.noise.parameters()
            )
        )
    diffusion.backbone.eval()
    diffusion.noise.eval()

    t0 = time.time()
    samples = sample_fhs_constrained(
        diffusion,
        constraint=constraint,
        discriminator=discriminator,
        num_steps=num_steps,
        eps=eps,
        topk=topk,
    )
    print(f"[FHS-Constrained] elapsed={time.time()-t0:.2f}s  "
          f"constraint_sat={constraint.check(samples).float().mean().item():.3f}"
          if constraint is not None else
          f"[FHS-Constrained] elapsed={time.time()-t0:.2f}s")

    if diffusion.ema:
        diffusion.ema.restore(
            itertools.chain(
                diffusion.backbone.parameters(), diffusion.noise.parameters()
            )
        )
    diffusion.backbone.train()
    diffusion.noise.train()
    return samples
