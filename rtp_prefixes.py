"""Utilities for selecting prefixes from RealToxicityPrompts."""
from __future__ import annotations

from typing import Any


def _cfg_get(config: Any, key: str, default=None):
    if config is None:
        return default
    return getattr(config, key, default)


def _nested_get(item: dict, path: str, default=None):
    value = item
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            return default
        value = value[part]
    return value


def load_prefixes(config, tokenizer, model_length: int, logger):
    """Return manual prefix or RealToxicityPrompts prefixes if enabled."""
    rtp_cfg = getattr(config, "rtp", None)
    if not _cfg_get(rtp_cfg, "enabled", False):
        prefix_text = getattr(config.sampling, "prefix_text", "")
        return [prefix_text] if prefix_text else [None]

    import datasets

    dataset_name = _cfg_get(rtp_cfg, "dataset_name", "allenai/real-toxicity-prompts")
    split = _cfg_get(rtp_cfg, "split", "train")
    num_prefixes = int(_cfg_get(rtp_cfg, "num_prefixes", 10))
    max_scan = int(_cfg_get(rtp_cfg, "max_scan", 5000))
    toxicity_threshold = float(_cfg_get(rtp_cfg, "toxicity_threshold", 0.5))
    max_prompt_ppl = _cfg_get(rtp_cfg, "max_prompt_ppl", None)
    seed = int(_cfg_get(rtp_cfg, "seed", config.seed))

    dataset = datasets.load_dataset(dataset_name, split=split)
    dataset = dataset.shuffle(seed=seed)

    prefixes = []
    saw_ppl_field = False
    for row in dataset.select(range(min(max_scan, len(dataset)))):
        prompt = row.get("prompt", {})
        text = prompt.get("text", "")
        toxicity = prompt.get("toxicity", None)
        if not text or toxicity is None or toxicity <= toxicity_threshold:
            continue

        ppl = (
            prompt.get("ppl", None)
            or prompt.get("perplexity", None)
            or _nested_get(row, "prompt_ppl", None)
            or _nested_get(row, "prompt_perplexity", None)
        )
        if ppl is not None:
            saw_ppl_field = True
        if max_prompt_ppl is not None and ppl is not None and float(ppl) >= float(max_prompt_ppl):
            continue

        token_ids = tokenizer.encode(text, add_special_tokens=False)
        if len(token_ids) >= model_length:
            continue

        prefixes.append(text)
        if len(prefixes) >= num_prefixes:
            break

    if max_prompt_ppl is not None and not saw_ppl_field:
        logger.warning(
            "RealToxicityPrompts rows did not expose a prompt PPL field; "
            "using toxicity filtering only."
        )
    if not prefixes:
        raise ValueError(
            "No RealToxicityPrompts prefixes matched the filters. "
            "Try increasing rtp.max_scan or lowering rtp.toxicity_threshold."
        )

    logger.info(
        "Loaded %d RealToxicityPrompts prefixes with prompt toxicity > %.3f.",
        len(prefixes),
        toxicity_threshold,
    )
    return prefixes
