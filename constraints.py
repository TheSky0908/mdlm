"""
Hard constraint definitions for constrained discrete diffusion.

Each constraint defines a set C ⊆ vocab^L.  During FHS sampling the
constraint is enforced by zeroing out disallowed tokens in p_x0 at every
denoising step (analytical case) or by reweighting with a learned h_φ
(neural case).
"""
from __future__ import annotations

from typing import List, Optional

import torch
from torch import Tensor


class BaseConstraint:
    def check(self, x: Tensor) -> Tensor:
        """Return bool tensor [B]: True iff sequence satisfies constraint."""
        raise NotImplementedError

    def logits_mask(self, x: Tensor, p_x0: Tensor) -> Tensor:
        """
        Additive log-mask applied to p_x0 before softmax.

        Args:
            x:     [B, L] current noisy token ids
            p_x0:  [B, L, V] log-probabilities from diffusion model

        Returns:
            mask:  [B, L, V] – 0 where allowed, -inf where forbidden
        """
        raise NotImplementedError


class PrefixConstraint(BaseConstraint):
    """
    Hard prefix constraint: positions 0..k-1 must match given token ids.

    During denoising, the model's p_x0 at each prefix position is
    zeroed for all tokens except the required one, so those positions
    always unmask to the correct token.
    """

    def __init__(self, prefix_token_ids: List[int]):
        self.prefix = prefix_token_ids

    def check(self, x: Tensor) -> Tensor:
        k = len(self.prefix)
        pref = torch.tensor(self.prefix, device=x.device, dtype=x.dtype)
        return (x[:, :k] == pref[None, :]).all(dim=1)

    def logits_mask(self, x: Tensor, p_x0: Tensor) -> Tensor:
        B, L, V = p_x0.shape
        mask = torch.zeros(B, L, V, device=p_x0.device)
        for pos, tok in enumerate(self.prefix):
            if pos >= L:
                break
            neg_inf = torch.full((B, V), float("-inf"), device=p_x0.device)
            neg_inf[:, tok] = 0.0
            mask[:, pos, :] = neg_inf
        return mask


class BannedTokensConstraint(BaseConstraint):
    """
    Constraint: sequence must not contain any of the banned token ids.

    During denoising, banned tokens are zeroed in p_x0 so the model
    never predicts them.
    """

    def __init__(self, banned_ids: List[int]):
        self.banned_ids = banned_ids

    def check(self, x: Tensor) -> Tensor:
        banned = torch.tensor(self.banned_ids, device=x.device)
        has_banned = (x.unsqueeze(-1) == banned).any(dim=-1).any(dim=-1)
        return ~has_banned

    def logits_mask(self, x: Tensor, p_x0: Tensor) -> Tensor:
        mask = torch.zeros_like(p_x0)
        for bid in self.banned_ids:
            mask[:, :, bid] = float("-inf")
        return mask


class ToxicityConstraint(BaseConstraint):
    """
    Soft-turned-hard toxicity constraint: C = {x : toxicity(x) < threshold}.

    Uses a pre-trained classifier (default: unitary/toxic-bert) as the oracle
    to label clean sequences.  No logits_mask — toxicity cannot be enforced
    analytically, so this constraint must be paired with a neural discriminator
    h_φ during FHS sampling.

    The classifier is only needed at training time (to generate BCE labels for
    h_φ).  At sampling time only h_φ is queried.
    """

    TOXIC_LABEL = "toxic"

    def __init__(
        self,
        diffusion_tokenizer,
        classifier_name: str = "unitary/toxic-bert",
        threshold: float = 0.5,
        device: str = "cpu",
    ):
        import transformers

        self.diffusion_tokenizer = diffusion_tokenizer
        self.threshold = threshold

        # Pipeline runs on its own device; keep it on CPU by default to avoid
        # competing with the diffusion model for GPU memory.
        self.pipeline = transformers.pipeline(
            "text-classification",
            model=classifier_name,
            device=device,
            truncation=True,
            max_length=512,
        )

    @torch.no_grad()
    def check(self, x: Tensor) -> Tensor:
        """
        Decode token ids → text, run toxicity classifier, return bool [B].
        True  = non-toxic (satisfies constraint).
        False = toxic    (violates constraint).
        """
        texts = self.diffusion_tokenizer.batch_decode(
            x, skip_special_tokens=True
        )
        results = self.pipeline(texts)
        satisfies = []
        for r in results:
            label = r["label"].lower()
            score = r["score"]
            # pipeline returns the winning label with its probability
            tox_score = score if label == self.TOXIC_LABEL else 1.0 - score
            satisfies.append(tox_score < self.threshold)
        return torch.tensor(satisfies, dtype=torch.bool, device=x.device)


class ComposedConstraint(BaseConstraint):
    """AND composition of multiple constraints."""

    def __init__(self, constraints: List[BaseConstraint]):
        self.constraints = constraints

    def check(self, x: Tensor) -> Tensor:
        result = torch.ones(x.shape[0], dtype=torch.bool, device=x.device)
        for c in self.constraints:
            result &= c.check(x)
        return result

    def logits_mask(self, x: Tensor, p_x0: Tensor) -> Tensor:
        mask = torch.zeros_like(p_x0)
        for c in self.constraints:
            mask = mask + c.logits_mask(x, p_x0)
        return mask


def build_constraint(config, tokenizer, mask_index: int) -> Optional[BaseConstraint]:
    """Instantiate a constraint from the Hydra config."""
    ctype = getattr(config, "constraint", None)
    if ctype is None:
        return None
    ctype = config.constraint.type

    if ctype == "none":
        return None

    elif ctype == "prefix":
        text = config.constraint.prefix_text
        ids = tokenizer.encode(text, add_special_tokens=False)
        return PrefixConstraint(ids)

    elif ctype == "banned_words":
        words = config.constraint.banned_words
        banned_ids: List[int] = []
        for w in words:
            banned_ids.extend(tokenizer.encode(w, add_special_tokens=False))
        return BannedTokensConstraint(banned_ids)

    elif ctype == "toxicity":
        return ToxicityConstraint(
            diffusion_tokenizer=tokenizer,
            classifier_name=getattr(
                config.constraint, "classifier_name", "unitary/toxic-bert"
            ),
            threshold=getattr(config.constraint, "threshold", 0.5),
            device=getattr(config.constraint, "classifier_device", "cpu"),
        )

    elif ctype == "composed":
        subs = []
        for sub_cfg in config.constraint.constraints:
            # Recursively build each sub-constraint by creating a temporary
            # namespace that looks like a top-level config.
            import omegaconf
            tmp = omegaconf.OmegaConf.create({"constraint": sub_cfg})
            subs.append(build_constraint(tmp, tokenizer, mask_index))
        return ComposedConstraint(subs)

    else:
        raise ValueError(f"Unknown constraint type: {ctype!r}")
