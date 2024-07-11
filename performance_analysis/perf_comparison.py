import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt
import numpy as np

# List of model names
models = ["gpt2", "gpt2-medium", "gpt2-large"]

# Dictionary to store results
results = {
    "model": [],
    "size_mb": [],
    "inference_time_ms": [],
    "perplexity": []
}

# Simple test data
test_data = [
    "Hello, my name is",
    "The weather today is"
]

def calculate_perplexity(model, tokenizer, texts):
    total_log_prob = 0
    total_length = 0
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
            log_likelihood = outputs.loss.item() * inputs["input_ids"].size(1)
            total_log_prob += log_likelihood
            total_length += inputs["input_ids"].size(1)
    return torch.exp(total_log_prob / total_length)

# Evaluate each model
for model_name in models:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Model size
    size_mb = sum(p.numel() for p in model.parameters()) * 4 / (1024 ** 2)
    
    # Inference speed
    start_time = time.time()
    for text in test_data:
        inputs = tokenizer.encode(text, return_tensors="pt")
        _ = model.generate(inputs, max_length=50)
    inference_time_ms = (time.time() - start_time) * 1000 / len(test_data)
    
    # Perplexity
    perplexity = calculate_perplexity(model, tokenizer, test_data).item()
    
    # Record results
    results["model"].append(model_name)
    results["size_mb"].append(size_mb)
    results["inference_time_ms"].append(inference_time_ms)
    results["perplexity"].append(perplexity)

# Convert to Numpy arrays
size_mb = np.array(results["size_mb"])
inference_time_ms = np.array(results["inference_time_ms"])
perplexity = np.array(results["perplexity"])

# Visualization
fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('Model')
ax1.set_ylabel('Size (MB)', color=color)
ax1.bar(results["model"], size_mb, color=color, alpha=0.6)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('Inference Time (ms) and Perplexity', color=color)
ax2.plot(results["model"], inference_time_ms, color='blue', marker='o', label='Inference Time (ms)')
ax2.plot(results["model"], perplexity, color='green', marker='x', linestyle='--', label='Perplexity')
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()
fig.legend(loc="upper left", bbox_to_anchor=(0.1, 0.9))

plt.title('Model Performance Comparison')
plt.show()
