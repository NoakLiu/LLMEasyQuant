"""
Algorithm Layer: Quantization algorithms and optimization methods

This module provides various quantization algorithms for LLM models:

1. Basic Quantization Methods:
   - absmax: Absolute maximum quantization
   - zeropoint: Zero-point quantization
   - smoothquant: Smooth quantization with scaling factors

2. Advanced Quantization Methods:
   - simquant: Similarity-based quantization
   - symquant: Symmetric quantization
   - zeroquant: Zero-aware quantization
   - awq: Activation-aware weight quantization
   - billm: Bi-level LLM quantization
   - qlora: Quantized LoRA

3. Knowledge Distillation:
   - Teacher-student distillation
   - Layer-wise distillation
   - Attention distillation

Each algorithm is implemented with:
- Clear mathematical formulation
- Step-by-step quantization process
- Performance metrics
- Usage examples
"""

from .basic import AbsMaxQuantizer, ZeroPointQuantizer, SmoothQuantizer
from .advanced import SimQuantizer, SymQuantizer, ZeroQuantizer, AWQQuantizer, BiLLMQuantizer, QLoraQuantizer
from .distillation import TeacherStudentDistillation, LayerWiseDistillation, AttentionDistillation

__all__ = [
    # Basic Quantization
    'AbsMaxQuantizer',
    'ZeroPointQuantizer',
    'SmoothQuantizer',
    
    # Advanced Quantization
    'SimQuantizer',
    'SymQuantizer',
    'ZeroQuantizer',
    'AWQQuantizer',
    'BiLLMQuantizer',
    'QLoraQuantizer',
    
    # Knowledge Distillation
    'TeacherStudentDistillation',
    'LayerWiseDistillation',
    'AttentionDistillation'
] 