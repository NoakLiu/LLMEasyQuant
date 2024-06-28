from copy import deepcopy
from torch import optim, nn
import torch
import transformers
# layer-by-layer zeroquant by knowledge distillation
def zeroquant_func(model, tokenizer, quantization_fn=sym_quantize_8bit):
    quant = deepcopy(model)
    input_x = tokenizer.encode("You also have a dream", return_tensors='pt').to(model.device)

    with torch.no_grad():
        teacher_outputs = model(input_x, output_hidden_states=True).hidden_states

    if isinstance(quant, transformers.GPT2LMHeadModel):
        quant_layers = quant.transformer.h
    elif isinstance(quant, transformers.OPTForCausalLM):
        quant_layers = quant.model.decoder.layers
    else:
        raise ValueError("Unsupported model type")

    layer_num = len(quant_layers)

    for layer_idx in range(1, layer_num):
        for i, param in enumerate(quant.parameters()):
            param.requires_grad = (i == layer_idx)  # freeze other layers

        for param in quant_layers[layer_idx].parameters():
            param.data = quantization_fn(param)[1]

        quantized_output = quant_layers[layer_idx](teacher_outputs[layer_idx - 1])[0]

        loss_fn = nn.MSELoss()
        loss = loss_fn(teacher_outputs[layer_idx], quantized_output)

        # update loss
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, quant.parameters()), lr=0.001)
        optimizer.zero_grad()
        loss.requires_grad = True  ## ?? temporary error fixes, conflicts with layer freeze code above

        loss.backward()
        optimizer.step()

        print(f"Loss at layer {layer_idx}: {loss:.4f}")
    return quant