# reference: https://github.com/SqueezeAILab/KVQuant/blob/main/quant/kvquant/simquant_module_quantizer.py#L364
import numpy as np
import torch
def sim_quantize(X): # bits=8, per_channel=True, qchannel=0, include_sparse=False, sparsity_threshold=0.999, cap_outliers=False
    bits=8
    per_channel=True
    qchannel=0
    include_sparse=False
    sparsity_threshold=0.999
    cap_outliers=False
    if include_sparse:
        threshold = 1 - ((1 - sparsity_threshold) / 2)
    else:
        threshold = 1  # Use full range for min-max quantization

    if per_channel:
        min_vals = X.min(dim=qchannel, keepdim=True)[0]
        max_vals = X.max(dim=qchannel, keepdim=True)[0]
    else:
        min_vals = X.min().expand_as(X)
        max_vals = X.max().expand_as(X)

    range_vals = max_vals - min_vals
    range_vals[range_vals == 0] = 1  # Prevent division by zero

    # Calculate scale and zero point for each channel
    scale = ((2 ** bits - 1) / range_vals).to(X.dtype)
    zero_point = (-min_vals * scale).round().to(X.dtype)

    # Apply quantization
    X_quant = torch.clamp((X * scale + zero_point).round(), 0, 2 ** bits - 1)

    # Dequantization
    X_dequant = (X_quant - zero_point) / scale

    # Cap outliers if requested
    if cap_outliers:
        percentile_low = np.percentile(X.cpu().numpy(), (1 - threshold) * 100)
        percentile_high = np.percentile(X.cpu().numpy(), threshold * 100)
        X_quant = torch.clamp(X_quant, percentile_low, percentile_high)
        X_dequant = (X_quant - zero_point) / scale  # Recalculate dequantized values

    return X_quant.to(torch.int8), X_dequant