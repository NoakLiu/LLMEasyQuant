import logging
import math
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from typing import Callable

import matplotlib.pyplot as plt
import torch
import transformers
from tqdm.auto import tqdm

class Quantizer:
    def __init__(self,
                 model,
                 tokenizer,
                 quantize_func: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
                 custom_quant_fn: Callable[
                                      [transformers.PreTrainedModel,
                                       transformers.PreTrainedTokenizerBase,
                                       Callable], transformers.PreTrainedModel] | None = None,
                 device="cpu"
                 ):
        """
        :type quantize_func: function with input: 2D fp Tensor; output: 2D int quantized Tensor + dequantize fp input
        :type custom_quant: optional function to override quantization,
                                 with input: [model, tokenizer, quantize_func]; output: quantized model
        """
        self.device = device

        self.model: transformers.PreTrainedModel = model.to(self.device)
        self.tokenizer: transformers.PreTrainedTokenizerBase = tokenizer

        self.quantize_func = quantize_func
        self.custom_quant_fn = custom_quant_fn
        self.quant_name = quantize_func.__name__
        if self.custom_quant_fn is not None:
            self.quant_name += f"_{custom_quant_fn.__name__}"

        self.quant = None

    def sample_weights(self) -> torch.Tensor:
        """
        :return: Extract weights of the first layer
        """
        if isinstance(self.model, transformers.GPT2LMHeadModel):
            return self.model.transformer.h[0].attn.c_attn.weight.data
        if isinstance(self.model, transformers.OPTForCausalLM):
            return self.model.model.decoder.layers[0].self_attn.k_proj.weight.data

    def sample_quant(self, plot=True) -> tuple[torch.Tensor, torch.Tensor]:
        """
        run quantization with sample weighs
        :param plot: whether to graph the tensor distribution, requires lovely-tensors library
        :return: quantized and dequantized tensors
        """
        weights = self.sample_weights()
        print(f"Original weights: {weights}")

        quant_weights, dequantized = self.quantize_func(weights)
        print(f"Quantized ({self.quant_name}) weights: {quant_weights}")

        if plot:
            if "lovely_tensors" not in sys.modules:
                logging.error("import lovely-tensors library first before graphing")
            else:
                plots = [
                    weights,
                    quant_weights
                ]

                fig, axs = plt.subplots(len(plots), figsize=(10, 3), constrained_layout=True)
                for i, p in enumerate(plots):
                    p.plt(ax=axs[i])
                plt.show()

        return quant_weights, dequantized

    def quantize(self) -> transformers.PreTrainedModel:
        if self.custom_quant_fn is None:

            # Create model to quantize
            self.quant = deepcopy(self.model)

            # Quantize all model weights
            print("Quantizing...")

            # multithread parallel quantization
            with ThreadPoolExecutor(max_workers=math.floor(os.cpu_count() * 0.8)) as executor:
                # Map the `quantize_func` over all model parameters
                results: list[tuple[torch.Tensor, torch.Tensor]] = list(
                    tqdm(executor.map(
                        self.quantize_func, self.quant.parameters()),
                        total=len(list(self.quant.parameters()))
                    )
                )

            # update all model weights
            for param, result in zip(self.quant.parameters(), results):
                param.data = result[1]

            self.quant.name_or_path += f"_{self.quant_name}"

        else:
            self.quant = self.custom_quant_fn(self.model, self.tokenizer, self.quantize_func)
            self.quant.name_or_path += f"_{self.quant_name}"


        return self.quant
