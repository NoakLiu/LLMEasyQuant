# LLMEasyQuant

LLMEasyQuant is a package developed for Easy Quantization Deployment for LLM applications. Nowadays, packages like TensorRT and Quanto have many underlying structures and self-invoking internal functions, which are not conducive to developers' personalized development and learning for deployment. LLMEasyQuant is developed to tackle this problem.

Author: Dong Liu, Kaiser Pister

### Deployment Methods:
#### Define the model
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig


# Set device to CPU for now
device = 'cpu'
# device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load model and tokenizer
model_id = 'gpt2'  # 137m F32 params
# model_id = 'facebook/opt-1.3b' # 1.3b f16 params
# model_id = 'mistralai/Mistral-7B-v0.1'  # 7.24b bf16 params, auth required
# model_id = 'meta-llama/Llama-2-7b-hf' # auth required

model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_id)

model_int8 = AutoModelForCausalLM.from_pretrained(model_id,
                                                  device_map='auto',
                                                  quantization_config=BitsAndBytesConfig(
                                                      load_in_8bit=True)
                                                  )
model_int8.name_or_path += "_int8"
```

#### mode deployment
absmax
```python
absq = Quantizer(model, tokenizer, absmax_quantize)
quantizers.append(absq)
```

zeropoint
```python
zpq = Quantizer(model, tokenizer, zeropoint_quantize)
quantizers.append(zpq)
```

smoothquant
```python
smooth_quant = SmoothQuantMatrix(alpha=0.5)
smoothq = Quantizer (model, tokenizer, smooth_quant.smooth_quant_apply)
quantizers.append(smoothq)
```

simquant
```python
simq = Quantizer(model, tokenizer, sim_quantize )
quantizers.append(simq)
```

simquant, zeroquant and knowledge distllation of both each
```python
symq = Quantizer(model, tokenizer, sym_quantize_8bit)
zeroq = Quantizer(model, tokenizer, sym_quantize_8bit, zeroquant_func)
quantizers.extend([symq, zeroq])
```

AWQ
```python
awq = Quantizer(model, tokenizer, awq_quantize )
quantizers.append(simq)
```

BiLLM
```python
billmq = Quantizer(model, tokenizer, billm_quantize )
quantizers.append(simq)
```

QLora
```python
qloraq = Quantizer(model, tokenizer, qlora_quantize )
quantizers.append(simq)
```

#### model computation
```python
[q.quantize() for q in quantizers]
```

#### visualization
```python
dist_plot([model, model_int8] + [q.quant for q in quantizers])
```

#### model comparision
```python
generated = compare_generation([model, model_int8] + [q.quant for q in quantizers], tokenizer, max_length=200, temperature=0.8)
```

#### perplexity analysis
```python
ppls = compare_ppl([model, model_int8] + [q.quant for q in quantizers], tokenizer, list(generated.values()))
```


### Results:

<!-- ![](results/quant_weights_distribution.jpeg) -->
<!-- ![](results/perfermance_comparision.jpeg){width=500px} -->
<p align="center">
  <img src="figures/quant_weights_distribution.jpeg" alt="Quant Weights Distribution" width="700">
</p>
<p align="center">
  <img src="figures/perfermance_comparision.jpeg" alt="Performance Comparison" width="500">
</p>
<!-- ![](results/ppl_analysis.jpeg) -->
<p align="center">
  <img src="figures/ppl_analysis.jpeg" alt="PPL Analysis" width="500">
</p>


### Conclusion:
In the research, we develop LLMEasyQuant, it is a package aiming to for easy quantization deployment which is user-friendly and easy to be deployed when computational resouces is limited.

### Deployment Simplicity Comparison Table

| Feature/Package            | TensorRT                               | Quanto                                  | LLMEasyQuant                           |
|----------------------------|----------------------------------------|-----------------------------------------|----------------------------------------|
| **Hardware Requirements**  | GPU                                   | GPU                                    | **CPU** and GPU                         |
| **Deployment Steps**       | Complex setup with CUDA dependencies  | Complex setup with multiple dependencies | Streamlined, minimal setup, includes AWQ, BiLLM, QLora |
| **Quantization Methods**   | Limited to specific optimizations      | Limited to specific optimizations       | Variety of methods with simple interface, includes AWQ, BiLLM, QLora |
| **Supported Methods**      | TensorRT-specific methods             | Quanto-specific methods                 | Absmax, Zeropoint, SmoothQuant, SimQuant, SymQuant, ZeroQuant, AWQ, BiLLM, QLora |
| **Integration Process**    | Requires integration with NVIDIA stack | Requires integration with specific frameworks | Clear integration with `transformers` |
| **Visualization Tools**    | External tools needed                 | External tools needed                   | Built-in visualization functions       |
| **Performance Analysis**   | External tools needed                 | External tools needed                   | Built-in performance analysis functions |

### Summary of LLMEasyQuant Advantages

1. **Hardware Flexibility**: Supports both CPU and GPU, providing flexibility for developers with different hardware resources.
2. **Simplified Deployment**: Requires minimal setup steps, making it user-friendly and accessible.
3. **Comprehensive Quantization Methods**: Offers a wide range of quantization methods, including AWQ, BiLLM, and QLora, with easy-to-use interfaces.
4. **Built-in Visualization and Analysis**: Includes tools for visualizing and comparing model performance, simplifying the evaluation process.


### Citation
If you find LLMEasyQuant useful or relevant to your project and research, please kindly cite our paper:

```
@article{liu2024llmeasyquanteasyuse,
      title={LLMEasyQuant -- An Easy to Use Toolkit for LLM Quantization}, 
      author={Dong Liu and Meng Jiang and Kaiser Pister},
      year={2024},
      eprint={2406.19657},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2406.19657}, 
}
```

