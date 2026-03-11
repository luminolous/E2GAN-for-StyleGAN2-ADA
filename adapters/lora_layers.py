# Copyright (c) 2024, E2GAN-for-StyleGAN2-ADA contributors.
# SPDX-License-Identifier: MIT
#
# Low-rank adapter (LoRA) layer modules for StyleGAN2-ADA.
# Inspired by E2GAN's selective parameter-efficient fine-tuning approach.
#
# These modules wrap existing FullyConnectedLayer instances with a trainable
# low-rank decomposition, allowing fine-tuning with very few parameters.

"""LoRA layer modules for parameter-efficient fine-tuning."""

import math
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------


class LoRALinear(nn.Module):
    """Low-rank adapter that wraps an existing fully-connected layer.

    Given a frozen original layer with weight W ∈ R^{out×in}, this module
    adds a trainable low-rank delta:

        output = orig_layer(x) + (x @ A^T) @ B^T * (alpha / rank)

    where A ∈ R^{rank×in_features} and B ∈ R^{out_features×rank}.

    Initialization:
        - A is initialized with Kaiming uniform so the adapter has a
          reasonable starting scale.
        - B is initialized to zeros so the adapter starts as an identity
          mapping (ΔW = 0 at init).

    Args:
        orig_layer: The original FullyConnectedLayer (or any nn.Module with
            a .weight attribute of shape [out_features, in_features]).
            Its parameters will be frozen.
        rank: Rank of the low-rank decomposition. Lower = fewer params.
        alpha: Scaling factor. The effective scaling is alpha/rank.
    """

    def __init__(self, orig_layer, rank=4, alpha=1.0):
        super().__init__()

        # Store the original layer as a submodule (its params stay frozen).
        self.orig_layer = orig_layer

        # Read dimensions from the original weight.
        # FullyConnectedLayer stores weight as [out_features, in_features].
        weight = orig_layer.weight
        out_features, in_features = weight.shape

        # Validate rank.
        if rank < 1:
            raise ValueError(f'LoRA rank must be >= 1, got {rank}')
        if rank > min(in_features, out_features):
            raise ValueError(
                f'LoRA rank ({rank}) must not exceed min(in={in_features}, '
                f'out={out_features})'
            )

        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # Low-rank factors.
        # A: down-projection  (rank × in_features)
        # B: up-projection    (out_features × rank)
        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

        # Kaiming uniform init for A (like nn.Linear default).
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        # Freeze original layer parameters.
        for p in self.orig_layer.parameters():
            p.requires_grad = False

        # Tag LoRA parameters for easy identification in the training loop.
        self.lora_A._is_lora = True  # noqa: SLF001
        self.lora_B._is_lora = True  # noqa: SLF001

    def forward(self, x):
        """Forward pass: base output + low-rank delta.

        The original layer's forward (including its activation function,
        bias, gain, etc.) runs unmodified. The LoRA delta is a pure additive
        term applied to the *pre-activation* linear transform. Because
        FullyConnectedLayer may apply a non-linear activation internally,
        we compute the delta separately and add it to the full output.

        Note: For affine layers in StyleGAN2-ADA's SynthesisLayer, the
        activation is 'linear' (identity), so additive composition is exact.
        """
        # Original forward (frozen weights, may include bias + activation).
        base_out = self.orig_layer(x)

        # Low-rank delta: x @ A^T @ B^T * scaling.
        # Shape: x is [..., in_features] → delta is [..., out_features].
        lora_out = x.to(self.lora_A.dtype)
        lora_out = lora_out @ self.lora_A.T  # [..., rank]
        lora_out = lora_out @ self.lora_B.T  # [..., out_features]
        lora_out = lora_out * self.scaling

        return base_out + lora_out.to(base_out.dtype)

    def merge(self):
        """Bake the LoRA delta into the original layer's weight (in-place).

        After merging, the forward pass through orig_layer alone produces the
        adapted output. Useful for export / inference without LoRA overhead.
        This is a Phase-2 feature but the method is included for completeness.
        """
        with torch.no_grad():
            # ΔW = B @ A * scaling, shape [out, in]
            delta = (self.lora_B @ self.lora_A) * self.scaling
            # FullyConnectedLayer applies weight_gain in forward, so we need
            # to account for it: effective W = weight * weight_gain.
            # We want: (weight + delta/weight_gain) * weight_gain = W + delta*weight_gain ... no.
            # Actually FullyConnectedLayer.forward does: w = self.weight * self.weight_gain
            # then x @ w.T.  Our lora adds x @ A.T @ B.T * scaling after the
            # original layer's full computation.  To merge, we need
            # (weight + delta') * weight_gain == weight * weight_gain + delta
            # → delta' = delta / weight_gain.
            weight_gain = getattr(self.orig_layer, 'weight_gain', 1.0)
            self.orig_layer.weight.add_(delta.to(self.orig_layer.weight.dtype) / weight_gain)

    def unmerge(self):
        """Reverse a previous merge (subtract the LoRA delta from weight)."""
        with torch.no_grad():
            delta = (self.lora_B @ self.lora_A) * self.scaling
            weight_gain = getattr(self.orig_layer, 'weight_gain', 1.0)
            self.orig_layer.weight.sub_(delta.to(self.orig_layer.weight.dtype) / weight_gain)

    def extra_repr(self):
        weight = self.orig_layer.weight
        out_f, in_f = weight.shape
        return (
            f'in_features={in_f}, out_features={out_f}, '
            f'rank={self.rank}, alpha={self.alpha}, scaling={self.scaling:.4f}'
        )


# ---------------------------------------------------------------------------
