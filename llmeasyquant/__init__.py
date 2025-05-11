"""
LLMEasyQuant: A Scalable Quantization Framework for LLM Inference
"""

from .system.core import AlgorithmBackend, ExecutionRuntime, DistributedController
from .system.hardware import HardwareOptimizer, RuntimeAdaptation
from .system.export import ONNXExporter

__version__ = "0.1.0" 