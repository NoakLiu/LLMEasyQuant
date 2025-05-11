import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

class HardwareOptimizer:
    """Hardware-specific optimization and kernel fusion"""
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.memory_hierarchy = self._detect_memory_hierarchy()
    
    def _detect_memory_hierarchy(self) -> Dict[str, int]:
        """Detect available memory hierarchy"""
        hierarchy = {
            'hbm': 0,  # High Bandwidth Memory
            'shared': 0,  # Shared Memory
            'l1': 0,  # L1 Cache
            'l2': 0  # L2 Cache
        }
        
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            hierarchy['hbm'] = props.total_memory
            hierarchy['shared'] = props.multi_processor_count * 48 * 1024  # 48KB per SM
            hierarchy['l1'] = props.multi_processor_count * 128 * 1024  # 128KB per SM
            hierarchy['l2'] = props.l2_cache_size
        
        return hierarchy
    
    def optimize_memory_access(self, tensor: torch.Tensor, block_size: int = 128) -> torch.Tensor:
        """Optimize memory access patterns for better cache utilization"""
        if not torch.cuda.is_available():
            return tensor
        
        # Ensure tensor is contiguous and aligned
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
        
        # Pad to block size if needed
        if tensor.numel() % block_size != 0:
            pad_size = block_size - (tensor.numel() % block_size)
            tensor = torch.nn.functional.pad(tensor, (0, pad_size))
        
        return tensor
    
    def fuse_quantization_gemm(self, input_tensor: torch.Tensor, weight: torch.Tensor,
                             scale: float, zero_point: float = 0) -> torch.Tensor:
        """Fuse quantization and GEMM operations"""
        if not torch.cuda.is_available():
            # Fallback to CPU implementation
            quantized = torch.round(input_tensor / scale) + zero_point
            return torch.matmul(quantized, weight)
        
        # CUDA implementation with fused kernel
        # Note: This is a simplified version. In practice, you would use a custom CUDA kernel
        quantized = torch.round(input_tensor / scale) + zero_point
        return torch.matmul(quantized, weight)
    
    def optimize_for_tensor_cores(self, model: nn.Module) -> nn.Module:
        """Optimize model for Tensor Core operations"""
        if not torch.cuda.is_available():
            return model
        
        for module in model.modules():
            if isinstance(module, nn.Linear):
                # Ensure dimensions are multiples of 8 for Tensor Cores
                if module.in_features % 8 != 0:
                    pad_size = 8 - (module.in_features % 8)
                    new_weight = torch.nn.functional.pad(module.weight, (0, pad_size))
                    module.weight = nn.Parameter(new_weight)
                    module.in_features = module.in_features + pad_size
                
                if module.out_features % 8 != 0:
                    pad_size = 8 - (module.out_features % 8)
                    new_weight = torch.nn.functional.pad(module.weight, (0, 0, 0, pad_size))
                    module.weight = nn.Parameter(new_weight)
                    module.out_features = module.out_features + pad_size
        
        return model
    
    def schedule_operations(self, operations: List[Tuple[str, torch.Tensor]]) -> List[torch.Tensor]:
        """Schedule operations based on memory hierarchy"""
        results = []
        current_memory = 0
        
        for op_name, tensor in operations:
            tensor_size = tensor.numel() * tensor.element_size()
            
            # Check if tensor fits in shared memory
            if tensor_size <= self.memory_hierarchy['shared']:
                # Keep in shared memory
                tensor = self.optimize_memory_access(tensor)
            else:
                # Use HBM with optimized access pattern
                tensor = self.optimize_memory_access(tensor, block_size=256)
            
            results.append(tensor)
            current_memory += tensor_size
        
        return results

class RuntimeAdaptation:
    """Runtime adaptation and recalibration"""
    def __init__(self, window_size: int = 100, alpha: float = 0.1):
        self.window_size = window_size
        self.alpha = alpha
        self.activation_history = []
        self.current_scale = 1.0
        self.current_zero_point = 0.0
    
    def update_scale(self, activations: torch.Tensor):
        """Update quantization scale based on recent activations"""
        self.activation_history.append(activations.detach())
        if len(self.activation_history) > self.window_size:
            self.activation_history.pop(0)
        
        # Calculate new scale using exponential moving average
        max_abs = torch.max(torch.abs(activations))
        self.current_scale = (1 - self.alpha) * self.current_scale + self.alpha * max_abs
    
    def get_quantization_params(self) -> Tuple[float, float]:
        """Get current quantization parameters"""
        return self.current_scale, self.current_zero_point
    
    def quantize_activations(self, activations: torch.Tensor) -> torch.Tensor:
        """Quantize activations using current scale"""
        return torch.round(activations / self.current_scale) + self.current_zero_point 