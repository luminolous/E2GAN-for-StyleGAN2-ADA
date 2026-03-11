#!/usr/bin/env python3
# Copyright (c) 2024, E2GAN-for-StyleGAN2-ADA contributors.
# SPDX-License-Identifier: MIT
#
# CPU-only smoke test for the E2GAN-inspired LoRA extension.
# Validates injection, parameter freezing, forward-pass shape,
# zero-init identity start, and adapter checkpoint round-trip.
#
# Usage:
#   python tests/test_lora_smoke.py
#
# No GPU, dataset, or pretrained weights required.

"""Automated smoke test for LoRA adapter modules and injection utilities."""

import os
import sys
import tempfile

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Ensure the repo root is on sys.path so that imports work when running
# this script directly (e.g. `python tests/test_lora_smoke.py`).
# ---------------------------------------------------------------------------
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from adapters.lora_layers import LoRALinear
from adapters.inject import inject_lora, extract_lora_state_dict, load_lora_state_dict

# We also need the StyleGAN2-ADA network definitions.
import dnnlib
from training.networks import Generator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_tiny_generator(seed=42):
    """Build a minimal Generator on CPU with deterministic weights."""
    torch.manual_seed(seed)
    G = Generator(
        z_dim=32,
        c_dim=0,
        w_dim=32,
        img_resolution=16,
        img_channels=3,
        mapping_kwargs={'num_layers': 2},
    ).cpu().eval()
    return G


def _run_forward(G, seed=123):
    """Run a deterministic forward pass and return the output tensor."""
    torch.manual_seed(seed)
    z = torch.randn(1, G.z_dim)
    c = torch.zeros(1, G.c_dim)
    with torch.no_grad():
        out = G(z, c, truncation_psi=1.0, noise_mode='const')
    return out


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

passed = 0
failed = 0


def _check(name, condition, detail=''):
    global passed, failed
    if condition:
        print(f'  [PASS] {name}')
        passed += 1
    else:
        print(f'  [FAIL] {name}  {detail}')
        failed += 1


def test_default_mode():
    """Test 1: default mode (no LoRA) — forward pass produces correct shape."""
    print('\n--- Test 1: Default mode (no LoRA) ---')
    G = _build_tiny_generator()
    out = _run_forward(G)
    _check('Output shape is [1, 3, 16, 16]', list(out.shape) == [1, 3, 16, 16],
           f'got {list(out.shape)}')


def test_injection():
    """Test 2: LoRA injection — backbone frozen, LoRA params trainable."""
    print('\n--- Test 2: LoRA injection ---')
    G = _build_tiny_generator()
    meta = inject_lora(G, rank=2, alpha=1.0, targets='affine')

    _check('Injected layers > 0', len(meta['injected_layers']) > 0,
           f'got {len(meta["injected_layers"])}')
    _check('Trainable < total', meta['trainable_params'] < meta['total_params'],
           f'trainable={meta["trainable_params"]}, total={meta["total_params"]}')

    # Verify backbone is frozen.
    non_lora_grads = [
        name for name, p in G.named_parameters()
        if p.requires_grad and not getattr(p, '_is_lora', False)
    ]
    _check('All non-LoRA params are frozen', len(non_lora_grads) == 0,
           f'{len(non_lora_grads)} non-LoRA params still require grad')

    # Verify LoRA params require grad.
    lora_params = [
        name for name, p in G.named_parameters()
        if getattr(p, '_is_lora', False)
    ]
    lora_grads = [
        name for name, p in G.named_parameters()
        if getattr(p, '_is_lora', False) and p.requires_grad
    ]
    _check('LoRA params exist', len(lora_params) > 0)
    _check('All LoRA params require grad', len(lora_grads) == len(lora_params),
           f'{len(lora_grads)}/{len(lora_params)}')


def test_forward_post_injection():
    """Test 3: forward pass post-injection — same shape, ≈ same output (zero init)."""
    print('\n--- Test 3: Forward pass post-injection ---')
    G = _build_tiny_generator()
    out_before = _run_forward(G)

    inject_lora(G, rank=2, alpha=1.0, targets='affine')
    out_after = _run_forward(G)

    _check('Output shape unchanged', list(out_after.shape) == list(out_before.shape))

    # Because B is zero-initialized, ΔW = 0, so output should be identical.
    diff = (out_after - out_before).abs().max().item()
    _check('Output ≈ pre-injection (zero-init delta)', diff < 1e-4,
           f'max diff = {diff:.6e}')


def test_adapter_round_trip():
    """Test 4: adapter save/load round-trip with same base weights."""
    print('\n--- Test 4: Adapter round-trip ---')

    # Build G1, inject, extract checkpoint.
    G1 = _build_tiny_generator(seed=42)
    inject_lora(G1, rank=2, alpha=1.0, targets='affine')

    # Manually set some non-zero LoRA weights to test actual data transfer.
    for name, p in G1.named_parameters():
        if getattr(p, '_is_lora', False):
            p.data.fill_(0.01)

    out_g1 = _run_forward(G1)
    ckpt = extract_lora_state_dict(G1)

    # Save to temp file.
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        tmp_path = f.name
        torch.save(ckpt, f)

    try:
        # Build G2 with SAME seed, load adapter.
        G2 = _build_tiny_generator(seed=42)
        load_lora_state_dict(G2, tmp_path)
        out_g2 = _run_forward(G2)

        diff = (out_g1 - out_g2).abs().max().item()
        _check('Round-trip output matches', diff < 1e-5,
               f'max diff = {diff:.6e}')

        # Verify metadata round-trip.
        ckpt_loaded = torch.load(tmp_path, map_location='cpu')
        _check('Metadata rank preserved', ckpt_loaded['metadata']['rank'] == 2)
        _check('Metadata alpha preserved', ckpt_loaded['metadata']['alpha'] == 1.0)
        _check('Metadata targets preserved', ckpt_loaded['metadata']['targets'] == 'affine')
        _check('Injected layers preserved',
               len(ckpt_loaded['metadata']['injected_layers']) > 0)
    finally:
        os.unlink(tmp_path)


def test_lora_linear_standalone():
    """Test 5: LoRALinear module in isolation."""
    print('\n--- Test 5: LoRALinear standalone ---')

    # Create a simple FC layer to wrap.
    from training.networks import FullyConnectedLayer
    fc = FullyConnectedLayer(in_features=32, out_features=64).cpu()

    lora = LoRALinear(fc, rank=4, alpha=2.0)
    _check('LoRA rank', lora.rank == 4)
    _check('LoRA alpha', lora.alpha == 2.0)
    _check('LoRA scaling', abs(lora.scaling - 0.5) < 1e-6, f'got {lora.scaling}')
    _check('lora_A shape', list(lora.lora_A.shape) == [4, 32])
    _check('lora_B shape', list(lora.lora_B.shape) == [64, 4])
    _check('lora_B is zero-init', lora.lora_B.abs().max().item() == 0.0)

    # Forward pass.
    x = torch.randn(2, 32)
    with torch.no_grad():
        base_out = fc(x)
        lora_out = lora(x)
    diff = (base_out - lora_out).abs().max().item()
    _check('Zero-init → output matches base', diff < 1e-5, f'max diff = {diff:.6e}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global passed, failed
    print('=' * 60)
    print('E2GAN-LoRA Smoke Test for StyleGAN2-ADA')
    print('=' * 60)

    test_default_mode()
    test_injection()
    test_forward_post_injection()
    test_adapter_round_trip()
    test_lora_linear_standalone()

    print('\n' + '=' * 60)
    total = passed + failed
    print(f'Results: {passed}/{total} passed, {failed}/{total} failed')
    if failed > 0:
        print('STATUS: FAIL')
        sys.exit(1)
    else:
        print('STATUS: PASS')
        sys.exit(0)


if __name__ == '__main__':
    main()
