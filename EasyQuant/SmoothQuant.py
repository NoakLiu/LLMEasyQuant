# https://github.com/mit-han-lab/smoothquant
class SmoothQuantMatrix:
    def __init__(self, alpha=0.5):
        self.alpha = alpha

    def smooth_quant_apply(self, X):
        """
        Applies smoothing directly to an input matrix X based on activity scales and returns both quantized and dequantized matrices.

        Args:
            X (torch.Tensor): The input matrix to be smoothed.
            act_scales (torch.Tensor): Activity scales that determine how the input matrix is smoothed.

        Returns:
            torch.Tensor: The smoothed (quantized) matrix.
            torch.Tensor: The dequantized matrix.
        """
        # assert X.dim() == 2, "X must be a 2D matrix"
        # assert X.size(1) == act_scales.numel(), "Mismatch between the number of columns in X and the length of act_scales"

        act_scales = torch.rand(X.shape[-1])

        device, dtype = X.device, X.dtype
        act_scales = act_scales.to(device=device, dtype=dtype)

        # Calculate the weight scales along the columns (features) of X
        weight_scales = X.abs().max(dim=0)[0].clamp(min=1e-5)

        # Compute the scales to adjust the matrix X
        scales = (
            (act_scales.pow(self.alpha) / weight_scales.pow(1 - self.alpha))
            .clamp(min=1e-5)
            .to(device)
            .to(dtype)
        )

        # Apply smoothing to X by element-wise multiplication with scales
        smoothed_X = X * scales

        # Dequantize the matrix by dividing the smoothed matrix by the scales
        dequantized_X = smoothed_X / scales

        return smoothed_X, dequantized_X