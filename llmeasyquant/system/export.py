import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union
import onnx
import onnxruntime
import numpy as np

class ONNXExporter:
    """ONNX export and serialization support"""
    def __init__(self, model: nn.Module):
        self.model = model
        self.quantization_params = {}
    
    def export_quantized_model(self, input_shape: Tuple[int, ...], output_path: str,
                             quantization_params: Dict[str, Tuple[float, float]]):
        """Export quantized model to ONNX format"""
        self.quantization_params = quantization_params
        
        # Create dummy input
        dummy_input = torch.randn(input_shape)
        
        # Export model
        torch.onnx.export(
            self.model,
            dummy_input,
            output_path,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            },
            opset_version=13,  # Use opset 13 for quantization support
            do_constant_folding=True
        )
        
        # Add quantization parameters to ONNX model
        self._add_quantization_params(output_path)
    
    def _add_quantization_params(self, model_path: str):
        """Add quantization parameters to ONNX model"""
        model = onnx.load(model_path)
        
        # Add quantization parameters as metadata
        for name, (scale, zero_point) in self.quantization_params.items():
            # Add scale
            scale_node = onnx.helper.make_node(
                'Constant',
                inputs=[],
                outputs=[f'{name}_scale'],
                value=onnx.helper.make_tensor(
                    name=f'{name}_scale',
                    data_type=onnx.TensorProto.FLOAT,
                    dims=[1],
                    vals=[scale]
                )
            )
            
            # Add zero point
            zp_node = onnx.helper.make_node(
                'Constant',
                inputs=[],
                outputs=[f'{name}_zero_point'],
                value=onnx.helper.make_tensor(
                    name=f'{name}_zero_point',
                    data_type=onnx.TensorProto.INT8,
                    dims=[1],
                    vals=[int(zero_point)]
                )
            )
            
            model.graph.node.extend([scale_node, zp_node])
        
        # Save modified model
        onnx.save(model, model_path)
    
    def verify_onnx_model(self, model_path: str, input_shape: Tuple[int, ...]) -> bool:
        """Verify exported ONNX model"""
        try:
            # Load ONNX model
            model = onnx.load(model_path)
            onnx.checker.check_model(model)
            
            # Test inference
            ort_session = onnxruntime.InferenceSession(model_path)
            dummy_input = np.random.randn(*input_shape).astype(np.float32)
            ort_inputs = {ort_session.get_inputs()[0].name: dummy_input}
            ort_outputs = ort_session.run(None, ort_inputs)
            
            return True
        except Exception as e:
            print(f"ONNX model verification failed: {str(e)}")
            return False
    
    def optimize_onnx_model(self, model_path: str, output_path: str):
        """Optimize ONNX model for inference"""
        # Load model
        model = onnx.load(model_path)
        
        # Optimize model
        optimized_model = onnx.optimizer.optimize(model)
        
        # Save optimized model
        onnx.save(optimized_model, output_path)
    
    def get_quantization_info(self, model_path: str) -> Dict[str, Dict[str, float]]:
        """Get quantization information from ONNX model"""
        model = onnx.load(model_path)
        quant_info = {}
        
        for node in model.graph.node:
            if node.op_type == 'Constant':
                if node.output[0].endswith('_scale'):
                    name = node.output[0].replace('_scale', '')
                    scale = node.attribute[0].t.float_data[0]
                    if name not in quant_info:
                        quant_info[name] = {}
                    quant_info[name]['scale'] = scale
                elif node.output[0].endswith('_zero_point'):
                    name = node.output[0].replace('_zero_point', '')
                    zero_point = node.attribute[0].t.int32_data[0]
                    if name not in quant_info:
                        quant_info[name] = {}
                    quant_info[name]['zero_point'] = zero_point
        
        return quant_info 