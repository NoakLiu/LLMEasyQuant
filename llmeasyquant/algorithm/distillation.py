"""
Knowledge distillation algorithms implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

class TeacherStudentDistillation:
    """Teacher-student knowledge distillation"""
    
    def __init__(self, temperature: float = 2.0, alpha: float = 0.5):
        self.temperature = temperature
        self.alpha = alpha
        
    def distill(self, teacher: nn.Module, student: nn.Module, 
                inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Perform knowledge distillation"""
        with torch.no_grad():
            teacher_logits = teacher(inputs)
        
        student_logits = student(inputs)
        
        # Calculate distillation loss
        distillation_loss = F.kl_div(
            F.log_softmax(student_logits / self.temperature, dim=-1),
            F.softmax(teacher_logits / self.temperature, dim=-1),
            reduction='batchmean'
        ) * (self.temperature ** 2)
        
        # Calculate student loss
        student_loss = F.cross_entropy(student_logits, teacher_logits.argmax(dim=-1))
        
        # Combine losses
        total_loss = self.alpha * distillation_loss + (1 - self.alpha) * student_loss
        
        return total_loss, student_logits

class LayerWiseDistillation:
    """Layer-wise knowledge distillation"""
    
    def __init__(self, layer_mapping: Dict[str, str], temperature: float = 2.0):
        self.layer_mapping = layer_mapping
        self.temperature = temperature
        
    def distill(self, teacher: nn.Module, student: nn.Module, 
                inputs: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Perform layer-wise distillation"""
        teacher_outputs = {}
        student_outputs = {}
        
        # Get teacher outputs
        def get_teacher_output(name):
            def hook(module, input, output):
                teacher_outputs[name] = output
            return hook
        
        # Register hooks
        hooks = []
        for name, module in teacher.named_modules():
            if name in self.layer_mapping:
                hooks.append(module.register_forward_hook(get_teacher_output(name)))
        
        # Forward pass
        with torch.no_grad():
            teacher(inputs)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        # Get student outputs
        def get_student_output(name):
            def hook(module, input, output):
                student_outputs[name] = output
            return hook
        
        hooks = []
        for name, module in student.named_modules():
            if name in self.layer_mapping.values():
                hooks.append(module.register_forward_hook(get_student_output(name)))
        
        student(inputs)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        # Calculate layer-wise losses
        layer_losses = {}
        total_loss = 0
        for t_name, s_name in self.layer_mapping.items():
            t_output = teacher_outputs[t_name]
            s_output = student_outputs[s_name]
            
            # Calculate MSE loss for each layer
            layer_loss = F.mse_loss(s_output, t_output)
            layer_losses[f"{t_name}->{s_name}"] = layer_loss
            total_loss += layer_loss
        
        return total_loss, layer_losses

class AttentionDistillation:
    """Attention-based knowledge distillation"""
    
    def __init__(self, temperature: float = 2.0, alpha: float = 0.5):
        self.temperature = temperature
        self.alpha = alpha
        
    def distill(self, teacher: nn.Module, student: nn.Module, 
                inputs: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Perform attention distillation"""
        teacher_attentions = {}
        student_attentions = {}
        
        # Get teacher attention maps
        def get_teacher_attention(name):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    teacher_attentions[name] = output[1]  # Attention weights
            return hook
        
        # Register hooks for attention layers
        hooks = []
        for name, module in teacher.named_modules():
            if hasattr(module, 'attention'):
                hooks.append(module.attention.register_forward_hook(get_teacher_attention(name)))
        
        # Forward pass
        with torch.no_grad():
            teacher(inputs)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        # Get student attention maps
        def get_student_attention(name):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    student_attentions[name] = output[1]  # Attention weights
            return hook
        
        hooks = []
        for name, module in student.named_modules():
            if hasattr(module, 'attention'):
                hooks.append(module.attention.register_forward_hook(get_student_attention(name)))
        
        student(inputs)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        # Calculate attention distillation loss
        attention_losses = {}
        total_loss = 0
        for name in teacher_attentions:
            if name in student_attentions:
                t_attn = teacher_attentions[name]
                s_attn = student_attentions[name]
                
                # Calculate KL divergence for attention maps
                attn_loss = F.kl_div(
                    F.log_softmax(s_attn / self.temperature, dim=-1),
                    F.softmax(t_attn / self.temperature, dim=-1),
                    reduction='batchmean'
                ) * (self.temperature ** 2)
                
                attention_losses[name] = attn_loss
                total_loss += attn_loss
        
        return total_loss, attention_losses 