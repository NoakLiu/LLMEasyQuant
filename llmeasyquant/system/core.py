import torch
import torch.distributed as dist
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import psutil
import GPUtil
import time
from concurrent.futures import ThreadPoolExecutor
import logging

class AlgorithmBackend:
    """Algorithm Backend Layer containing quantization strategies"""
    def __init__(self):
        self.quantization_methods = {
            'absmax': self.absmax_quantize,
            'zeropoint': self.zeropoint_quantize,
            'smoothquant': self.smooth_quantize,
            'simquant': self.sim_quantize
        }
    
    def absmax_quantize(self, tensor: torch.Tensor, bits: int = 8) -> Tuple[torch.Tensor, float]:
        """AbsMax quantization implementation"""
        scale = torch.max(torch.abs(tensor)) / (2 ** (bits - 1) - 1)
        quantized = torch.round(tensor / scale)
        return quantized, scale
    
    def zeropoint_quantize(self, tensor: torch.Tensor, bits: int = 8) -> Tuple[torch.Tensor, float, float]:
        """Zero-point quantization implementation"""
        scale = (torch.max(tensor) - torch.min(tensor)) / (2 ** bits - 1)
        zero_point = torch.round(-torch.min(tensor) / scale)
        quantized = torch.round(tensor / scale) + zero_point
        return quantized, scale, zero_point
    
    def smooth_quantize(self, tensor: torch.Tensor, alpha: float = 0.5, bits: int = 8) -> Tuple[torch.Tensor, float]:
        """SmoothQuant implementation"""
        scale = torch.pow(torch.abs(tensor), alpha).mean() / (2 ** (bits - 1) - 1)
        quantized = torch.round(tensor / scale)
        return quantized, scale
    
    def sim_quantize(self, tensor: torch.Tensor, bits: int = 8) -> Tuple[torch.Tensor, float]:
        """SimQuant implementation for KV caches"""
        scale = torch.std(tensor) / (2 ** (bits - 1) - 1)
        quantized = torch.round(tensor / scale)
        return quantized, scale

class ExecutionRuntime:
    """Execution Runtime Layer for dispatching quantization"""
    def __init__(self, backend: AlgorithmBackend):
        self.backend = backend
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def quantize_module(self, module: nn.Module, method: str, bits: int = 8) -> nn.Module:
        """Quantize a single module"""
        if method not in self.backend.quantization_methods:
            raise ValueError(f"Unknown quantization method: {method}")
        
        quantize_func = self.backend.quantization_methods[method]
        
        for name, param in module.named_parameters():
            if param.requires_grad:
                quantized, *scales = quantize_func(param.data, bits)
                param.data = quantized
                # Store scales for dequantization
                setattr(module, f"{name}_scale", scales[0])
                if len(scales) > 1:
                    setattr(module, f"{name}_zero_point", scales[1])
        
        return module
    
    def quantize_model(self, model: nn.Module, method: str, bits: int = 8) -> nn.Module:
        """Quantize entire model"""
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                self.quantize_module(module, method, bits)
        return model

class DistributedController:
    """Distributed Controller Layer for multi-GPU support"""
    def __init__(self, runtime: ExecutionRuntime):
        self.runtime = runtime
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.rank = dist.get_rank() if dist.is_initialized() else 0
    
    def broadcast_scales(self, scales: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Broadcast quantization scales across GPUs"""
        if self.world_size == 1:
            return scales
        
        for key in scales:
            dist.broadcast(scales[key], src=0)
        return scales
    
    def parallel_quantize(self, model: nn.Module, method: str, bits: int = 8) -> nn.Module:
        """Parallel quantization across GPUs"""
        if self.world_size == 1:
            return self.runtime.quantize_model(model, method, bits)
        
        # Split model parameters across GPUs
        params_per_gpu = len(list(model.parameters())) // self.world_size
        start_idx = self.rank * params_per_gpu
        end_idx = start_idx + params_per_gpu if self.rank < self.world_size - 1 else len(list(model.parameters()))
        
        # Quantize local parameters
        local_scales = {}
        for i, (name, param) in enumerate(model.named_parameters()):
            if start_idx <= i < end_idx:
                quantized, *scales = self.runtime.backend.quantization_methods[method](param.data, bits)
                param.data = quantized
                local_scales[name] = scales[0]
        
        # Synchronize scales
        global_scales = self.broadcast_scales(local_scales)
        
        # Apply scales to model
        for name, scale in global_scales.items():
            setattr(model, f"{name}_scale", scale)
        
        return model

class SystemMonitor:
    """System resource monitoring"""
    def __init__(self):
        self.cpu_usage = []
        self.memory_usage = []
        self.gpu_usage = []
        self.start_time = None
    
    def start_monitoring(self):
        """Start monitoring system resources"""
        self.start_time = time.time()
        self.record_metrics()
    
    def record_metrics(self):
        """Record current system metrics"""
        self.cpu_usage.append(psutil.cpu_percent())
        self.memory_usage.append(psutil.virtual_memory().percent)
        
        if torch.cuda.is_available():
            gpu = GPUtil.getGPUs()[0]
            self.gpu_usage.append(gpu.load * 100)
    
    def get_summary_stats(self) -> Dict[str, float]:
        """Get summary statistics of recorded metrics"""
        return {
            "cpu_avg": np.mean(self.cpu_usage) if self.cpu_usage else 0,
            "memory_avg": np.mean(self.memory_usage) if self.memory_usage else 0,
            "gpu_util_avg": np.mean(self.gpu_usage) if self.gpu_usage else 0,
            "duration": time.time() - self.start_time if self.start_time else 0
        }

class ModelSystemMetrics:
    """Model-specific system metrics"""
    def __init__(self, model: nn.Module):
        self.model = model
        self.inference_times = []
        self.memory_usage = []
    
    def measure_inference(self, input_tensor: torch.Tensor) -> float:
        """Measure inference time and memory usage"""
        torch.cuda.synchronize()
        start_time = time.time()
        start_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        
        with torch.no_grad():
            _ = self.model(input_tensor)
        
        torch.cuda.synchronize()
        end_time = time.time()
        end_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        
        self.inference_times.append(end_time - start_time)
        self.memory_usage.append(end_memory - start_memory)
        
        return self.inference_times[-1]
    
    def get_metrics(self) -> Dict[str, float]:
        """Get summary of model metrics"""
        return {
            "avg_inference_time": np.mean(self.inference_times),
            "std_inference_time": np.std(self.inference_times),
            "avg_memory_usage": np.mean(self.memory_usage),
            "max_memory_usage": np.max(self.memory_usage)
        } 