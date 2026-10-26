"""Original SEAGLMI 4x6 pool: stock forward/indices, fixed-order backward.

Execution repair for PyTorch 2.1.2 CUDA, not a network modification.
The CPU-ordered gradient accumulation is not a claim to recover a unique
bit pattern from the original nondeterministic CUDA atomic accumulation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

COUNTS = {"forward": 0, "backward": 0, "heights": {}}
ORIGINAL_MODULE_FORWARD = nn.AdaptiveMaxPool2d.forward


class FixedOrderAdaptiveMax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        if x.ndim != 4 or x.shape[1] != 8 or x.shape[-1] != 48:
            raise RuntimeError("Only original SEA N x 8 x H x 48 pooling is supported")
        if not 1 <= x.shape[-2] <= 28 or x.dtype != torch.float32:
            raise RuntimeError("Unexpected original SEA pooling height/dtype")
        y, indices = F.adaptive_max_pool2d(x, (4, 6), return_indices=True)
        ctx.save_for_backward(indices)
        ctx.input_shape = tuple(x.shape)
        COUNTS["forward"] += 1
        h = str(x.shape[-2])
        COUNTS["heights"][h] = COUNTS["heights"].get(h, 0) + 1
        return y

    @staticmethod
    def backward(ctx, grad_output):
        (indices,) = ctx.saved_tensors
        n, c, h, w = ctx.input_shape
        grad = grad_output.reshape(n, c, 24)
        ix = indices.reshape(n, c, 24)
        flat_position = torch.arange(h*w, device=indices.device).reshape(1, 1, -1)
        gi = grad_output.new_zeros(n, c, h*w)
        zero = grad_output.new_zeros(())
        # Each iteration owns every destination element. No competing writes,
        # atomic scatter or unspecified reduction order is used.
        for k in range(24):
            gi.add_(torch.where(ix[:, :, k:k+1] == flat_position,
                                grad[:, :, k:k+1], zero))
        COUNTS["backward"] += 1
        return gi.reshape(ctx.input_shape)


def module_forward(module, x):
    if tuple(module.output_size) != (4, 6) or module.return_indices:
        raise RuntimeError("Unexpected pooling module in isolated SEA execution")
    return FixedOrderAdaptiveMax.apply(x)


def install():
    if nn.AdaptiveMaxPool2d.forward is not ORIGINAL_MODULE_FORWARD:
        raise RuntimeError("Another pooling patch is already installed")
    nn.AdaptiveMaxPool2d.forward = module_forward


def restore():
    nn.AdaptiveMaxPool2d.forward = ORIGINAL_MODULE_FORWARD
