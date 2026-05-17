"""
Hard-constrained FHS (First Hitting Sampler) for masked discrete diffusion.

Theory (Case 2 from the notes):
    h_t(x) = P(X_0 ∈ C | X_t = x)   ← 判别器目标

FHS constrained 的三步流程（与原始 FHS 完全对齐，仅第三步加 h 矫正）：

  Step 1  更新时间
            u  ~ Uniform[0,1]^{1/num_masked}   (order statistic trick)
            s  = u * t

  Step 2  选 unmask 位置
            selected_index ~ Uniform(masked positions in x)

  Step 3  预测 token（加 h 矫正）
            原始 FHS:  P(a | x, s, i) ∝ μ^i(a | x, s)
            Constrained:  P(a | x, s, i) ∝ μ^i(a | x, s) · h_s(x^{-i}⊕a)

其中 μ^i(a|x,s) = p_x0[b, selected_index, a]（扩散模型的输出）。
h_s(x^{-i}⊕a) 通过以下两种方式之一计算：
  - 解析约束（PrefixConstraint / BannedTokensConstraint）：直接对 token 分布置零
  - 神经判别器 h_φ：对 top-K 候选 token 逐一评估 h_φ(x^{-i}⊕a, s)
"""
from __future__ import annotations

import itertools
import time
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from tqdm.auto import tqdm

from diffusion import _sample_categorical


# ---------------------------------------------------------------------------
# Step 3 的 h 矫正
# ---------------------------------------------------------------------------

def _apply_h_to_position(
    q_xs: Tensor,           # [B, V]  当前位置的 token 分布（已去掉 mask token）
    x: Tensor,              # [B, L]  当前含噪序列
    selected_index: Tensor, # [B]     本轮要 unmask 的位置
    t: Tensor,              # [B]     当前时间
    constraint,
    discriminator: Optional[nn.Module],
    topk: int,
) -> Tensor:
    """
    对选中位置的 token 分布应用约束矫正：
        P(a | x, s, i) ∝ q_xs[a] · h_s(x^{-i}⊕a)

    返回归一化后的 [B, V] 分布。
    """
    B, V = q_xs.shape

    # --- 解析约束：直接对 q_xs 置零 ---
    if constraint is not None and hasattr(constraint, "logits_mask"):
        # 为每个 batch 项构造一个"只有选中位置有效"的假 p_x0，
        # 借用 logits_mask 接口取出该位置的 mask 向量。
        # logits_mask 返回 [B, L, V]，我们只取 selected_index 那行。
        dummy_log_p = q_xs.log().unsqueeze(1).expand(B, x.shape[1], V)  # [B, L, V]
        additive_mask = constraint.logits_mask(x, dummy_log_p)           # [B, L, V]

        # 取每个 batch 对应选中位置的那行
        pos_mask = additive_mask[
            torch.arange(B, device=x.device), selected_index
        ]  # [B, V]

        log_q = q_xs.clamp(min=1e-30).log() + pos_mask
        # 重新归一化（有些 token 被置为 -inf）
        log_q = log_q - torch.logsumexp(log_q, dim=-1, keepdim=True)
        q_xs = log_q.exp()

    # --- 神经判别器：top-K 候选的 h 矫正 ---
    if discriminator is not None:
        q_xs = q_xs.clone()
        for b in range(B):
            k = min(topk, V)
            top_probs, top_ids = q_xs[b].topk(k)   # [K]

            # 构造 K 条序列，每条在 selected_index[b] 位置换成不同 token
            pos = selected_index[b].item()
            x_cands = x[b].unsqueeze(0).expand(k, -1).clone()  # [K, L]
            x_cands[:, pos] = top_ids
            t_cands = t[b:b+1].expand(k)                        # [K]

            with torch.no_grad():
                h_scores = discriminator(x_cands, t_cands)      # [K] ∈ (0,1)

            weighted = top_probs * (h_scores + 1e-8)
            weighted = weighted / (weighted.sum() + 1e-30)

            new_row = torch.zeros(V, device=q_xs.device)
            new_row.scatter_(0, top_ids, weighted)
            q_xs[b] = new_row

    return q_xs


# ---------------------------------------------------------------------------
# 完整 FHS constrained 采样循环
# ---------------------------------------------------------------------------

@torch.no_grad()
def sample_fhs_constrained(
    diffusion,
    constraint,
    discriminator: Optional[nn.Module],
    topk: int = 50,
    prefix_ids: Optional[list] = None,
) -> Tensor:
    """
    FHS constrained 采样，与原始 restore_model_and_sample_first_hitting 流程完全对齐。

    每轮迭代：
      1. 更新时间  t → s
      2. 选 unmask 位置  selected_index
      3. 用扩散模型预测 token 分布，乘以 h 矫正项后采样
    """
    batch_size = diffusion.config.loader.eval_batch_size
    L = diffusion.config.model.length
    device = diffusion.device

    x = diffusion._sample_prior(batch_size, L).to(device)   # 全 MASK 初始化
    B = x.shape[0]

    # 预填 prefix：直接写入前 k 个位置，FHS 只 unmask 剩余位置
    prefix_len = 0
    if prefix_ids:
        prefix_len = len(prefix_ids)
        prefix_tensor = torch.tensor(prefix_ids, device=device, dtype=x.dtype)
        x[:, :prefix_len] = prefix_tensor.unsqueeze(0)      # 广播到整个 batch

    L_unmask = L - prefix_len
    t = torch.ones(B, device=device)                         # t=1 对应全 MASK

    for i in tqdm(range(L_unmask), desc="FHS sampling", unit="step"):
        num_masked = L_unmask - i

        # ---- Step 1: 更新时间 ----
        # u ~ max of num_masked uniform draws = U^{1/num_masked}
        u = torch.float_power(torch.rand_like(t), 1.0 / num_masked)
        s = (u * t).to(t.dtype)
        sigma_s, _ = diffusion.noise(s)

        # ---- Step 2: 选 unmask 位置 ----
        coord = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)  # [B, L]
        masked_indices = torch.masked_select(
            coord, x == diffusion.mask_index
        ).reshape(B, num_masked)                                            # [B, num_masked]

        rand_pos = torch.randint(high=num_masked, size=(B,), device=device)
        selected_index = masked_indices[torch.arange(B, device=device), rand_pos]  # [B]

        # ---- Step 3: 预测 token 分布 + h 矫正 ----
        p_x0 = diffusion.forward(x, sigma_s).exp()          # [B, L, V]
        p_x0[:, :, diffusion.mask_index] = 0
        p_x0 = p_x0 / p_x0.sum(dim=-1, keepdim=True)

        # 取出选中位置的 token 分布  μ^i(a|x,s)
        q_xs = p_x0[torch.arange(B, device=device), selected_index]  # [B, V]

        # 乘以 h_s(x^{-i}⊕a)
        q_xs = _apply_h_to_position(
            q_xs, x, selected_index, s,
            constraint=constraint,
            discriminator=discriminator,
            topk=topk,
        )

        # 采样并写回
        token = _sample_categorical(q_xs)                    # [B]
        x[torch.arange(B, device=device), selected_index] = token

        t = s

    return x


# ---------------------------------------------------------------------------
# 公共入口（带 EMA store/restore）
# ---------------------------------------------------------------------------

def restore_and_sample_constrained(
    diffusion,
    constraint=None,
    discriminator: Optional[nn.Module] = None,
    topk: int = 50,
    prefix_ids: Optional[list] = None,
) -> Tensor:
    """加载 EMA 权重 → 约束 FHS 采样 → 还原权重。"""
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
        topk=topk,
        prefix_ids=prefix_ids,
    )
    elapsed = time.time() - t0

    if constraint is not None:
        sat = constraint.check(samples).float().mean().item()
        print(f"[FHS-Constrained] elapsed={elapsed:.2f}s  constraint_sat={sat:.3f}")
    else:
        print(f"[FHS-Constrained] elapsed={elapsed:.2f}s")

    if diffusion.ema:
        diffusion.ema.restore(
            itertools.chain(
                diffusion.backbone.parameters(), diffusion.noise.parameters()
            )
        )
    diffusion.backbone.train()
    diffusion.noise.train()
    return samples
