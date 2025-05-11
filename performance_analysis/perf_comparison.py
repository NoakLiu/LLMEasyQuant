import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt
import numpy as np
from system_design import SystemMonitor, ModelSystemMetrics

# List of model names to test
models = [
    "gpt2",  # 117M
    "gpt2-medium",  # 345M
    "gpt2-large",  # 774M
    "facebook/opt-1.3b",  # 1.3B
    "meta-llama/Llama-2-7b-hf",  # 7B
    "Qwen/Qwen-7B",  # 7B
    "mistralai/Mistral-7B-v0.1"  # 7B
]

# Dictionary to store results
results = {
    "model": [],
    "size_mb": [],
    "inference_time_ms": [],
    "perplexity": [],
    "cpu_usage": [],
    "memory_usage": [],
    "gpu_usage": []
}

# Test data with varying lengths
test_data = [
    "Hello, my name is",
    "The weather today is",
    "In the field of artificial intelligence",
    "The quantum computing revolution",
    "As large language models continue to evolve"
]

def calculate_perplexity(model, tokenizer, texts):
    total_log_prob = 0
    total_length = 0
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
            log_likelihood = outputs.loss.item() * inputs["input_ids"].size(1)
            total_log_prob += log_likelihood
            total_length += inputs["input_ids"].size(1)
    return torch.exp(total_log_prob / total_length)

def evaluate_model(model_name, system_monitor):
    print(f"\nEvaluating {model_name}...")
    
    # Initialize model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    if torch.cuda.is_available():
        model = model.cuda()
    
    # Model size
    size_mb = sum(p.numel() for p in model.parameters()) * 4 / (1024 ** 2)
    
    # System monitoring
    system_monitor.start_monitoring()
    
    # Inference speed
    start_time = time.time()
    for text in test_data:
        inputs = tokenizer.encode(text, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = inputs.cuda()
        _ = model.generate(inputs, max_length=50)
        system_monitor.record_metrics()
    
    inference_time_ms = (time.time() - start_time) * 1000 / len(test_data)
    
    # Perplexity
    perplexity = calculate_perplexity(model, tokenizer, test_data).item()
    
    # Get system metrics
    system_metrics = system_monitor.get_summary_stats()
    
    return {
        "model": model_name,
        "size_mb": size_mb,
        "inference_time_ms": inference_time_ms,
        "perplexity": perplexity,
        "cpu_usage": system_metrics["cpu_avg"],
        "memory_usage": system_metrics["memory_avg"],
        "gpu_usage": system_metrics["gpu_util_avg"]
    }

def plot_results(results):
    # Create subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Model size vs Inference time
    ax1.scatter(results["size_mb"], results["inference_time_ms"], alpha=0.6)
    for i, model in enumerate(results["model"]):
        ax1.annotate(model.split("/")[-1], 
                    (results["size_mb"][i], results["inference_time_ms"][i]))
    ax1.set_xlabel('Model Size (MB)')
    ax1.set_ylabel('Inference Time (ms)')
    ax1.set_title('Model Size vs Inference Time')
    
    # Perplexity vs Model size
    ax2.scatter(results["size_mb"], results["perplexity"], alpha=0.6)
    for i, model in enumerate(results["model"]):
        ax2.annotate(model.split("/")[-1], 
                    (results["size_mb"][i], results["perplexity"][i]))
    ax2.set_xlabel('Model Size (MB)')
    ax2.set_ylabel('Perplexity')
    ax2.set_title('Model Size vs Perplexity')
    
    # Resource usage
    x = range(len(results["model"]))
    width = 0.25
    
    ax3.bar([i - width for i in x], results["cpu_usage"], width, label='CPU Usage (%)')
    ax3.bar(x, results["memory_usage"], width, label='Memory Usage (%)')
    ax3.bar([i + width for i in x], results["gpu_usage"], width, label='GPU Usage (%)')
    ax3.set_xticks(x)
    ax3.set_xticklabels([m.split("/")[-1] for m in results["model"]], rotation=45)
    ax3.set_ylabel('Usage (%)')
    ax3.set_title('Resource Usage by Model')
    ax3.legend()
    
    # Performance metrics
    metrics = np.array([results["inference_time_ms"], results["perplexity"]])
    metrics = (metrics - metrics.min(axis=1, keepdims=True)) / (metrics.max(axis=1, keepdims=True) - metrics.min(axis=1, keepdims=True))
    
    ax4.plot(x, metrics[0], 'o-', label='Normalized Inference Time')
    ax4.plot(x, metrics[1], 's-', label='Normalized Perplexity')
    ax4.set_xticks(x)
    ax4.set_xticklabels([m.split("/")[-1] for m in results["model"]], rotation=45)
    ax4.set_ylabel('Normalized Score')
    ax4.set_title('Normalized Performance Metrics')
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig('model_comparison.png')
    plt.show()

def main():
    system_monitor = SystemMonitor()
    
    # Evaluate each model
    for model_name in models:
        try:
            result = evaluate_model(model_name, system_monitor)
            for key in results:
                results[key].append(result[key])
        except Exception as e:
            print(f"Error evaluating {model_name}: {str(e)}")
    
    # Plot results
    plot_results(results)
    
    # Print summary
    print("\nModel Evaluation Summary:")
    print("-" * 80)
    print(f"{'Model':<30} {'Size (MB)':<12} {'Time (ms)':<12} {'Perplexity':<12} {'CPU %':<8} {'Memory %':<8} {'GPU %':<8}")
    print("-" * 80)
    for i in range(len(results["model"])):
        print(f"{results['model'][i].split('/')[-1]:<30} "
              f"{results['size_mb'][i]:<12.2f} "
              f"{results['inference_time_ms'][i]:<12.2f} "
              f"{results['perplexity'][i]:<12.2f} "
              f"{results['cpu_usage'][i]:<8.2f} "
              f"{results['memory_usage'][i]:<8.2f} "
              f"{results['gpu_usage'][i]:<8.2f}")

if __name__ == "__main__":
    main()
