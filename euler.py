"""Euler sampler for absorbing-state discrete diffusion.

This module implements the reverse-generator Euler update

    P(x_{t-dt}=y | x_t=x) ~= I + dt * R_rev_t(x, y),

specialized to the absorbing mask graph used by ``diffusion.py``.
"""
from __future__ import annotations

import itertools
import os
import time

import hydra
import hydra.utils
import lightning as L
import torch
from torch import Tensor
from tqdm.auto import tqdm

import constraints as constraint_lib
import dataloader
import discriminator as disc_lib
import main as main_lib
import utils
from diffusion import _sample_categorical


def _as_vector(x: Tensor) -> Tensor:
    if x.ndim > 1:
        return x.squeeze(-1)
    return x


@torch.no_grad()
def euler_update(diffusion, x: Tensor, t: Tensor, dt: float) -> Tensor:
    """Take one reverse Euler step for the absorbing mask process.

    For a masked token, the off-diagonal reverse rates are

        R_rev(mask, a) = d sigma / dt * s_theta(mask, a),  a != mask.

    Euler turns those rates into probabilities with ``dt * R_rev`` and
    assigns the remaining probability mass to staying masked. Unmasked
    positions are absorbing in the reverse sampler and are copied through.
    """
    sigma_t, dsigma_dt = diffusion.noise(t)
    sigma_t = _as_vector(sigma_t)
    dsigma_dt = _as_vector(dsigma_dt)

    score = diffusion.get_score(x, sigma_t)
    jump_probs = score * (dt * dsigma_dt)[:, None, None]
    jump_probs[:, :, diffusion.mask_index] = 0

    total_jump_prob = jump_probs.sum(dim=-1)
    stay_prob = (1 - total_jump_prob).clamp_min(0)
    probs = jump_probs.clone()
    probs[:, :, diffusion.mask_index] = stay_prob

    sampled = _sample_categorical(probs)
    copy_flag = (x != diffusion.mask_index).to(x.dtype)
    return copy_flag * x + (1 - copy_flag) * sampled


@torch.no_grad()
def _apply_discriminator_ratio(
    diffusion,
    x: Tensor,
    t: Tensor,
    jump_probs: Tensor,
    discriminator,
    topk: int,
    eps: float = 1e-8,
) -> Tensor:
    if discriminator is None:
        return jump_probs

    B, L, V = jump_probs.shape
    t_vec = _as_vector(t)
    h_x = discriminator(x, t_vec).clamp_min(eps)
    constrained = torch.zeros_like(jump_probs)

    masked = x == diffusion.mask_index
    k = max(1, min(topk, V - 1))
    batch_ids = torch.arange(B, device=x.device)

    for pos in range(L):
        active = masked[:, pos]
        if not active.any():
            continue

        active_ids = batch_ids[active]
        rates = jump_probs[active_ids, pos]
        top_rates, top_tokens = rates.topk(k, dim=-1)

        x_cands = x[active_ids].repeat_interleave(k, dim=0)
        cand_tokens = top_tokens.reshape(-1)
        x_cands[:, pos] = cand_tokens
        t_cands = t_vec[active_ids].repeat_interleave(k)

        h_tilde = discriminator(x_cands, t_cands).reshape(-1, k)
        ratios = h_tilde / h_x[active_ids, None]
        weighted_rates = top_rates * ratios

        constrained[active_ids[:, None], pos, top_tokens] = weighted_rates

    constrained[:, :, diffusion.mask_index] = 0
    return constrained


@torch.no_grad()
def constrained_euler_update(
    diffusion,
    x: Tensor,
    t: Tensor,
    dt: float,
    discriminator=None,
    topk: int = 50,
) -> Tensor:
    """Take one constrained reverse Euler step.

    The constrained reverse generator is

        R_new(x, xtilde) = R_rev(x, xtilde) * h(xtilde, t) / h(x, t).
    """
    sigma_t, dsigma_dt = diffusion.noise(t)
    sigma_t = _as_vector(sigma_t)
    dsigma_dt = _as_vector(dsigma_dt)

    score = diffusion.get_score(x, sigma_t)
    jump_probs = score * (dt * dsigma_dt)[:, None, None]
    jump_probs[:, :, diffusion.mask_index] = 0

    jump_probs = _apply_discriminator_ratio(
        diffusion=diffusion,
        x=x,
        t=t,
        jump_probs=jump_probs,
        discriminator=discriminator,
        topk=topk,
    )

    total_jump_prob = jump_probs.sum(dim=-1)
    stay_prob = (1 - total_jump_prob).clamp_min(0)
    probs = jump_probs.clone()
    probs[:, :, diffusion.mask_index] = stay_prob

    sampled = _sample_categorical(probs)
    copy_flag = (x != diffusion.mask_index).to(x.dtype)
    return copy_flag * x + (1 - copy_flag) * sampled


@torch.no_grad()
def sample_euler(diffusion, num_steps: int | None = None, eps: float = 1e-5) -> Tensor:
    """Generate unconditional samples with reverse-generator Euler sampling."""
    batch_size = diffusion.config.loader.eval_batch_size
    length = diffusion.config.model.length
    device = diffusion.device

    if num_steps is None:
        num_steps = diffusion.config.sampling.steps

    x = diffusion._sample_prior(batch_size, length).to(device)
    timesteps = torch.linspace(1, eps, num_steps + 1, device=device)
    dt = (1 - eps) / num_steps

    for i in tqdm(range(num_steps), desc="Euler sampling", unit="step"):
        t = timesteps[i] * torch.ones(x.shape[0], 1, device=device)
        x = euler_update(diffusion, x, t, dt)

    if diffusion.config.sampling.noise_removal:
        t = timesteps[-1] * torch.ones(x.shape[0], 1, device=device)
        unet_conditioning = diffusion.noise(t)[0]
        x = diffusion.forward(x, unet_conditioning).argmax(dim=-1)

    return x


@torch.no_grad()
def sample_constrained_euler(
    diffusion,
    discriminator=None,
    topk: int = 50,
    num_steps: int | None = None,
    eps: float = 1e-5,
) -> Tensor:
    """Generate samples with the h-transform constrained Euler sampler."""
    batch_size = diffusion.config.loader.eval_batch_size
    length = diffusion.config.model.length
    device = diffusion.device

    if num_steps is None:
        num_steps = diffusion.config.sampling.steps

    x = diffusion._sample_prior(batch_size, length).to(device)
    timesteps = torch.linspace(1, eps, num_steps + 1, device=device)
    dt = (1 - eps) / num_steps

    for i in tqdm(range(num_steps), desc="Constrained Euler sampling", unit="step"):
        t = timesteps[i] * torch.ones(x.shape[0], 1, device=device)
        x = constrained_euler_update(
            diffusion=diffusion,
            x=x,
            t=t,
            dt=dt,
            discriminator=discriminator,
            topk=topk,
        )

    if diffusion.config.sampling.noise_removal:
        t = timesteps[-1] * torch.ones(x.shape[0], 1, device=device)
        unet_conditioning = diffusion.noise(t)[0]
        x = diffusion.forward(x, unet_conditioning).argmax(dim=-1)

    return x


def restore_and_sample_euler(
    diffusion,
    num_steps: int | None = None,
    eps: float = 1e-5,
) -> Tensor:
    """Run Euler sampling with the model's EMA weights, mirroring diffusion.py."""
    if diffusion.ema:
        diffusion.ema.store(
            itertools.chain(diffusion.backbone.parameters(), diffusion.noise.parameters())
        )
        diffusion.ema.copy_to(
            itertools.chain(diffusion.backbone.parameters(), diffusion.noise.parameters())
        )

    diffusion.backbone.eval()
    diffusion.noise.eval()

    start_time = time.time()
    samples = sample_euler(diffusion, num_steps=num_steps, eps=eps)
    print("Elapsed time: ", time.time() - start_time)

    if diffusion.ema:
        diffusion.ema.restore(
            itertools.chain(diffusion.backbone.parameters(), diffusion.noise.parameters())
        )

    diffusion.backbone.train()
    diffusion.noise.train()
    return samples


def restore_and_sample_constrained_euler(
    diffusion,
    discriminator=None,
    topk: int = 50,
    num_steps: int | None = None,
    eps: float = 1e-5,
) -> Tensor:
    """Run constrained Euler sampling with EMA weights."""
    if diffusion.ema:
        diffusion.ema.store(
            itertools.chain(diffusion.backbone.parameters(), diffusion.noise.parameters())
        )
        diffusion.ema.copy_to(
            itertools.chain(diffusion.backbone.parameters(), diffusion.noise.parameters())
        )

    diffusion.backbone.eval()
    diffusion.noise.eval()
    if discriminator is not None:
        discriminator.eval()

    start_time = time.time()
    samples = sample_constrained_euler(
        diffusion=diffusion,
        discriminator=discriminator,
        topk=topk,
        num_steps=num_steps,
        eps=eps,
    )
    print("Elapsed time: ", time.time() - start_time)

    if diffusion.ema:
        diffusion.ema.restore(
            itertools.chain(diffusion.backbone.parameters(), diffusion.noise.parameters())
        )

    diffusion.backbone.train()
    diffusion.noise.train()
    return samples


def _load_optional_discriminator(config, model, logger):
    disc_path = getattr(getattr(config, "discriminator", object()), "load_path", "")
    if disc_path == "":
        logger.info("No discriminator.load_path provided; running unconditional Euler.")
        return None

    if not os.path.isabs(disc_path):
        disc_path = os.path.join(hydra.utils.get_original_cwd(), disc_path)
    if not os.path.exists(disc_path):
        logger.warning(f"Discriminator file not found: {disc_path}.")
        return None

    disc = disc_lib.load_discriminator(disc_path, config, model.vocab_size)
    disc = disc.to(model.device).eval()
    logger.info(f"Loaded discriminator from {disc_path}")
    return disc


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(config):
    """CLI entry point for unconditional Euler sampling."""
    L.seed_everything(config.seed)
    logger = utils.get_logger(__name__)
    logger.info("Generating samples with Euler sampler.")

    tokenizer = dataloader.get_tokenizer(config)
    model = main_lib._load_from_checkpoint(config=config, tokenizer=tokenizer)
    if config.eval.disable_ema:
        logger.info("Disabling EMA.")
        model.ema = None

    constrained = getattr(getattr(config, "euler", object()), "constrained", False)
    constraint = None
    disc = None
    topk = getattr(getattr(config, "discriminator", object()), "topk", 50)
    if constrained:
        constraint = constraint_lib.build_constraint(config, tokenizer, model.mask_index)
        disc = _load_optional_discriminator(config, model, logger)
        if disc is not None:
            logger.info("Using discriminator-guided constrained Euler.")

    model.gen_ppl_metric.reset()
    text_samples = []
    for _ in range(config.sampling.num_sample_batches):
        if constrained and disc is not None:
            samples = restore_and_sample_constrained_euler(
                diffusion=model,
                discriminator=disc,
                topk=topk,
                num_steps=config.sampling.steps,
            )
        else:
            samples = restore_and_sample_euler(model, num_steps=config.sampling.steps)
        texts = model.tokenizer.batch_decode(samples)
        text_samples.extend(texts)
        model.compute_generative_perplexity(texts)

    print("Text samples:", text_samples)
    print("Generative perplexity:", model.gen_ppl_metric.compute())

    if constrained and constraint is not None:
        ids = model.tokenizer(
            text_samples,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=config.model.length,
        ).input_ids
        sat = constraint.check(ids).float().mean().item()
        print(f"Constraint satisfaction rate: {sat:.3f}")


if __name__ == "__main__":
    main()
