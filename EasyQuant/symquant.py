import torch
def sym_quantize_8bit(X):
    """
    Follows:
    https://github.com/microsoft/DeepSpeed/blob/4c15ad9f8d51a1950842c69bbbc9d93c73afbcfc/deepspeed/compression/utils.py#L62
    """
    max_val = torch.max(torch.abs(X))
    scale = max_val / (255 / 2)

    quantized = torch.round(X / scale).clamp(-128, 127)

    # Dequantize
    X_dequant = quantized.float() * scale

    return quantized.to(torch.int8), X_dequant