#!/usr/bin/env python3
# Copyright (c) 2024, E2GAN-for-StyleGAN2-ADA contributors.
# SPDX-License-Identifier: MIT
#
# Dry-run LoRA injection: prints total, trainable, and frozen parameter
# counts along with the names of all injected adapter layers.

"""Inspect trainable parameters after LoRA injection (dry-run, no training)."""

import argparse
import sys
import os

import torch
import dnnlib
import legacy

from adapters.inject import inject_lora


def main():
    parser = argparse.ArgumentParser(
        description='Inspect trainable parameters after LoRA injection.'
    )
    parser.add_argument('--network', required=True,
                        help='Path or URL to a base network pickle (.pkl)')
    parser.add_argument('--lora-rank', type=int, default=4,
                        help='LoRA rank (default: 4)')
    parser.add_argument('--lora-alpha', type=float, default=1.0,
                        help='LoRA alpha (default: 1.0)')
    args = parser.parse_args()

    # Load base network.
    print(f'Loading network from "{args.network}"...')
    with dnnlib.util.open_url(args.network) as f:
        data = legacy.load_network_pkl(f)
    G = data['G_ema'].cpu().eval()

    # Show pre-injection stats.
    total_before = sum(p.numel() for p in G.parameters())
    print(f'\n--- Before LoRA injection ---')
    print(f'Total parameters: {total_before:,}')

    # Inject LoRA.
    meta = inject_lora(G, rank=args.lora_rank, alpha=args.lora_alpha, targets='affine')

    # Show post-injection stats.
    print(f'\n--- After LoRA injection (rank={args.lora_rank}, alpha={args.lora_alpha}) ---')
    print(f'Total parameters:     {meta["total_params"]:,}')
    print(f'Trainable parameters: {meta["trainable_params"]:,}')
    frozen = meta['total_params'] - meta['trainable_params']
    print(f'Frozen parameters:    {frozen:,}')
    pct = 100.0 * meta['trainable_params'] / max(meta['total_params'], 1)
    print(f'Trainable percentage: {pct:.2f}%')

    print(f'\n--- Injected adapter layers ({len(meta["injected_layers"])}) ---')
    for name in meta['injected_layers']:
        # Find the LoRA module and print its shape.
        parts = name.split('.')
        mod = G
        for part in parts:
            mod = getattr(mod, part)
        a_shape = tuple(mod.lora_A.shape)
        b_shape = tuple(mod.lora_B.shape)
        print(f'  {name}  A={a_shape}  B={b_shape}')

    print('\nDone.')


if __name__ == '__main__':
    main()
