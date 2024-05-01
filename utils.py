import numpy as np
import torch
import transformers
from matplotlib import pyplot as plt, ticker


# Functions modified from:
# https://github.com/mlabonne/llm-course/tree/main?tab=readme-ov-file#quantization

def dist_plot(models: list[transformers.PreTrainedModel]):
    """
    plot weight distribution histograms, for comparisons between models
    :param models: assume the first model is the base model
    """
    assert len(models) >= 2, "At least 2 models is required to plot"

    model_weights = []
    for model in models:
        weights = [param.data.clone() for param in model.parameters()]

        # Flatten weight tensors
        model_weights.append((model,
                              np.concatenate(
                                  [t.cpu().detach().numpy().flatten() for t in weights])
                              ))

    plt.style.use('ggplot')
    colors = [color['color'] for color in plt.rcParams['axes.prop_cycle']]

    # Create a figure to plot
    fig, axs = plt.subplots(len(model_weights) - 1, figsize=(10, 3 * len(model_weights)), sharex=True)

    original_model, original_weights = model_weights[0]
    for i, (model, weights) in enumerate(model_weights):
        if i == 0:
            continue

        if len(model_weights) == 2:
            ax = axs
        else:
            ax = axs[i - 1]

        ax.hist(original_weights,
                bins=200, range=(-2, 2),
                color=colors[0], alpha=0.5, label=F'Original {original_model.name_or_path} weights',
                )
        ax.hist(weights,
                bins=200, range=(-2, 2),
                color=colors[1:][i % (len(colors) - 1)], alpha=0.5, label=f'{model.name_or_path} weights')

        # Add legend
        ax.legend()

        # Add title and labels
        ax.set_title(
            f'Comparison of Original ({original_model.name_or_path}) and Quantized ({model.name_or_path}) Weights',
            fontsize=16)
        ax.set_xlabel('Weights', fontsize=14)
        ax.set_ylabel('Count', fontsize=14)

        # Configure tick formatter for better readability
        ax.yaxis.set_major_formatter(ticker.EngFormatter())

    # Setting font size globally
    plt.rc('font', size=12)

    plt.tight_layout()
    plt.show()


def calculate_perplexity(model, tokenizer, text) -> float:
    # Encode the text
    encodings = tokenizer(text, return_tensors='pt').to(model.device)

    # Define input_ids and target_ids
    input_ids = encodings.input_ids
    target_ids = input_ids.clone()

    with torch.no_grad():
        outputs = model(input_ids, labels=target_ids)

    # Loss calculation
    neg_log_likelihood = outputs.loss

    # Perplexity calculation
    ppl = torch.exp(neg_log_likelihood)

    return ppl.item()


def compare_ppl(models: list, tokenizer, texts: list, verbose=True) -> dict[str, float]:
    """
    :return: dict of model name + ppl score
    """
    ppls = {models[i].name_or_path: calculate_perplexity(models[i], tokenizer, texts[i])
            for i in range(len(models))}

    if verbose:
        print(f"\n".join([f"{name} ppl: {ppl:.2f}" for name, ppl in ppls.items()]))

    return ppls


def generate_text(model, tokenizer, prompt="I have a dream", max_length=50) -> str:
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(model.device)

    output = model.generate(inputs=input_ids,
                            max_length=max_length,
                            do_sample=True,
                            top_k=30,
                            pad_token_id=tokenizer.eos_token_id,
                            attention_mask=input_ids.new_ones(input_ids.shape))

    return tokenizer.decode(output[0], skip_special_tokens=True)


def compare_generation(models: list, tokenizer, prompt="I have a dream", max_length=50, verbose=True) \
        -> dict[str, float]:
    """
    :return: dict of model name + generated text
    """
    texts = {model.name_or_path: generate_text(model, tokenizer, prompt, max_length)
             for model in models}

    if verbose:
        print(f"\n{'-' * 50}\n".join([f"{name}:\n{text}" for name, text in texts.items()]))

    return texts
