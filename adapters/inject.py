# Copyright (c) 2024, E2GAN-for-StyleGAN2-ADA contributors.
# SPDX-License-Identifier: MIT
#
# LoRA injection, extraction, and loading utilities for StyleGAN2-ADA.
# Operates on the generator module tree without modifying any backbone code.

"""Inject, extract, and load LoRA adapters on a StyleGAN2-ADA generator."""

import copy
import re
from collections import OrderedDict

import torch

from adapters.lora_layers import LoRALinear


# ---------------------------------------------------------------------------
# Target pattern registry
# ---------------------------------------------------------------------------

# Maps user-facing target names to regex patterns over module paths.
# In MVP only 'affine' is supported.  The patterns match the module names
# produced by Generator.named_modules() in StyleGAN2-ADA.
TARGET_PATTERNS = {
    'affine': re.compile(
        r'^synthesis\.b\d+\.'       # SynthesisBlock at some resolution
        r'(?:conv[01]|torgb)\.'     # SynthesisLayer or ToRGBLayer
        r'affine$'                  # the FullyConnectedLayer inside it
    ),
}

# Supported targets for MVP.
SUPPORTED_TARGETS_MVP = frozenset(['affine'])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def inject_lora(G, rank=4, alpha=1.0, targets='affine'):
    """Inject LoRA adapters into selected generator layers.

    Walks ``G.named_modules()``, identifies layers whose full name matches
    the target pattern, and replaces each with a ``LoRALinear`` wrapper.
    All non-LoRA parameters on G are frozen after injection.

    Args:
        G: A ``Generator`` (or ``SynthesisNetwork``) instance.
        rank: Low-rank decomposition rank for every adapter.
        alpha: LoRA scaling factor (effective scaling = alpha/rank).
        targets: Which layers to target.  For MVP, only ``'affine'`` is
            supported.  This selects the style-modulation affine transforms
            inside every ``SynthesisLayer`` and ``ToRGBLayer``.

    Returns:
        dict: Metadata about the injection, including:
            - ``rank``, ``alpha``, ``targets``: echo of input args
            - ``injected_layers``: list of full module paths that were wrapped
            - ``total_params``: total parameter count after injection
            - ``trainable_params``: number of trainable (LoRA) parameters
    """
    # --- validate targets ---------------------------------------------------
    if targets not in SUPPORTED_TARGETS_MVP:
        raise ValueError(
            f"Unsupported LoRA target '{targets}' for MVP.  "
            f"Supported: {sorted(SUPPORTED_TARGETS_MVP)}"
        )
    pattern = TARGET_PATTERNS[targets]

    # --- collect matching modules -------------------------------------------
    # We need (parent_module, attr_name, child_module, full_path) tuples.
    matches = []
    for full_name, module in G.named_modules():
        if pattern.match(full_name):
            # Split 'a.b.c.affine' → parent_path='a.b.c', attr='affine'
            parts = full_name.rsplit('.', 1)
            if len(parts) == 2:
                parent_path, attr_name = parts
            else:
                # Top-level (unlikely for affine targets, but handle it)
                parent_path, attr_name = '', parts[0]
            parent = _get_module_by_path(G, parent_path) if parent_path else G
            matches.append((parent, attr_name, module, full_name))

    if not matches:
        raise RuntimeError(
            f"No modules matched target pattern '{targets}'.  "
            f"Is G a standard StyleGAN2-ADA Generator?"
        )

    # --- inject LoRA wrappers -----------------------------------------------
    injected_layers = []
    for parent, attr_name, orig_module, full_name in matches:
        lora_wrapper = LoRALinear(orig_module, rank=rank, alpha=alpha)
        setattr(parent, attr_name, lora_wrapper)
        injected_layers.append(full_name)

    # --- freeze all non-LoRA parameters on G --------------------------------
    for p in G.parameters():
        if not getattr(p, '_is_lora', False):
            p.requires_grad = False

    # --- compute statistics -------------------------------------------------
    total_params = sum(p.numel() for p in G.parameters())
    trainable_params = sum(p.numel() for p in G.parameters() if p.requires_grad)

    metadata = {
        'rank': rank,
        'alpha': alpha,
        'targets': targets,
        'injected_layers': injected_layers,
        'total_params': total_params,
        'trainable_params': trainable_params,
    }
    return metadata


def extract_lora_state_dict(G):
    """Extract adapter-only weights and metadata from an injected generator.

    Returns a dict suitable for ``torch.save``:

    .. code-block:: python

        {
            'metadata': { 'rank': …, 'alpha': …, 'targets': …, 'injected_layers': […] },
            'state_dict': OrderedDict({ '<layer>.lora_A': tensor, '<layer>.lora_B': tensor, … })
        }

    Args:
        G: Generator with LoRA adapters already injected.

    Returns:
        dict: Checkpoint dict with 'metadata' and 'state_dict' keys.
    """
    state_dict = OrderedDict()
    injected_layers = []
    rank = None
    alpha = None

    for name, module in G.named_modules():
        if isinstance(module, LoRALinear):
            injected_layers.append(name)
            state_dict[f'{name}.lora_A'] = module.lora_A.detach().cpu().clone()
            state_dict[f'{name}.lora_B'] = module.lora_B.detach().cpu().clone()
            # Record rank/alpha from first adapter (all should be the same
            # in MVP since we use uniform rank).
            if rank is None:
                rank = module.rank
                alpha = module.alpha

    if not injected_layers:
        raise RuntimeError(
            'No LoRALinear modules found on G.  '
            'Was inject_lora() called before extract_lora_state_dict()?'
        )

    # Determine the target pattern that was used.
    # In MVP this is always 'affine', but we detect it for forward-compat.
    targets = _infer_targets(injected_layers)

    metadata = {
        'rank': rank,
        'alpha': alpha,
        'targets': targets,
        'injected_layers': injected_layers,
    }

    return {'metadata': metadata, 'state_dict': state_dict}


def load_lora_state_dict(G, ckpt):
    """Load LoRA adapter weights onto a generator.

    This function:
    1. Reads metadata from the checkpoint to determine injection config.
    2. Calls ``inject_lora`` on ``G`` to create the adapter structure.
    3. Loads the saved adapter weights into the newly created LoRA layers.

    Args:
        G: A *fresh* (non-injected) Generator instance with the same
           architecture as the one that produced the checkpoint.
        ckpt: Checkpoint dict as produced by ``extract_lora_state_dict``,
            or the path string to a ``.pt`` file.

    Returns:
        dict: The injection metadata (same as ``inject_lora`` returns).
    """
    # Handle file path input.
    if isinstance(ckpt, str):
        ckpt = torch.load(ckpt, map_location='cpu')

    meta = ckpt['metadata']
    adapter_sd = ckpt['state_dict']

    # Inject the LoRA structure first.
    inject_meta = inject_lora(
        G,
        rank=meta['rank'],
        alpha=meta['alpha'],
        targets=meta['targets'],
    )

    # Load adapter weights into the injected modules.
    # We use strict=False because only LoRA params are in adapter_sd.
    missing, unexpected = [], []
    for key, value in adapter_sd.items():
        # key looks like 'synthesis.b4.conv1.affine.lora_A'
        # We need to set it on the correct submodule.
        parts = key.rsplit('.', 1)
        if len(parts) != 2:
            unexpected.append(key)
            continue
        module_path, param_name = parts
        try:
            module = _get_module_by_path(G, module_path)
        except (AttributeError, KeyError):
            unexpected.append(key)
            continue
        param = getattr(module, param_name, None)
        if param is None or not isinstance(param, torch.nn.Parameter):
            unexpected.append(key)
            continue
        param.data.copy_(value.to(param.device))

    if unexpected:
        raise RuntimeError(
            f'Unexpected keys when loading LoRA state dict: {unexpected}'
        )

    return inject_meta


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _get_module_by_path(model, path):
    """Get a submodule by dot-separated path (e.g. 'synthesis.b4.conv1')."""
    parts = path.split('.')
    module = model
    for part in parts:
        module = getattr(module, part)
    return module


def _infer_targets(injected_layers):
    """Infer the target pattern name from a list of injected layer paths."""
    for target_name, pattern in TARGET_PATTERNS.items():
        if all(pattern.match(name) for name in injected_layers):
            return target_name
    return 'affine'  # fallback


# ---------------------------------------------------------------------------
