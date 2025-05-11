import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from system_design import SystemMonitor, ModelSystemMetrics
import json
import os
import time

class ModelArchitecture:
    def __init__(self, name, model_id, size_mb, architecture_type):
        self.name = name
        self.model_id = model_id
        self.size_mb = size_mb
        self.architecture_type = architecture_type

# Define model architectures to compare
architectures = [
    ModelArchitecture("GPT-2", "gpt2", 117, "Decoder-only"),
    ModelArchitecture("GPT-2 Medium", "gpt2-medium", 345, "Decoder-only"),
    ModelArchitecture("GPT-2 Large", "gpt2-large", 774, "Decoder-only"),
    ModelArchitecture("OPT-1.3B", "facebook/opt-1.3b", 1300, "Decoder-only"),
    ModelArchitecture("LLaMA-2-7B", "meta-llama/Llama-2-7b-hf", 7000, "Decoder-only"),
    ModelArchitecture("Qwen-7B", "Qwen/Qwen-7B", 7000, "Decoder-only"),
    ModelArchitecture("Mistral-7B", "mistralai/Mistral-7B-v0.1", 7000, "Decoder-only")
]

class ModelComparison:
    def __init__(self):
        self.system_monitor = SystemMonitor()
        self.results = {
            "model": [],
            "architecture": [],
            "size_mb": [],
            "inference_time_ms": [],
            "perplexity": [],
            "cpu_usage": [],
            "memory_usage": [],
            "gpu_usage": []
        }
        
        # Test data with varying lengths and complexity
        self.test_data = [
            "Hello, my name is",
            "The weather today is",
            "In the field of artificial intelligence",
            "The quantum computing revolution",
            "As large language models continue to evolve",
            "The impact of climate change on global ecosystems",
            "Recent advances in quantum computing and their implications",
            "The role of artificial intelligence in modern healthcare systems"
        ]

    def calculate_perplexity(self, model, tokenizer, texts):
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

    def evaluate_model(self, arch):
        print(f"\nEvaluating {arch.name}...")
        
        try:
            # Initialize model and tokenizer
            tokenizer = AutoTokenizer.from_pretrained(arch.model_id)
            model = AutoModelForCausalLM.from_pretrained(arch.model_id)
            if torch.cuda.is_available():
                model = model.cuda()
            
            # System monitoring
            self.system_monitor.start_monitoring()
            
            # Inference speed
            start_time = time.time()
            for text in self.test_data:
                inputs = tokenizer.encode(text, return_tensors="pt")
                if torch.cuda.is_available():
                    inputs = inputs.cuda()
                _ = model.generate(inputs, max_length=50)
                self.system_monitor.record_metrics()
            
            inference_time_ms = (time.time() - start_time) * 1000 / len(self.test_data)
            
            # Perplexity
            perplexity = self.calculate_perplexity(model, tokenizer, self.test_data).item()
            
            # Get system metrics
            system_metrics = self.system_monitor.get_summary_stats()
            
            return {
                "model": arch.name,
                "architecture": arch.architecture_type,
                "size_mb": arch.size_mb,
                "inference_time_ms": inference_time_ms,
                "perplexity": perplexity,
                "cpu_usage": system_metrics["cpu_avg"],
                "memory_usage": system_metrics["memory_avg"],
                "gpu_usage": system_metrics["gpu_util_avg"]
            }
        except Exception as e:
            print(f"Error evaluating {arch.name}: {str(e)}")
            return None

    def plot_architecture_comparison(self):
        # Create subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Group by architecture type
        arch_types = set(self.results["architecture"])
        colors = plt.cm.Set3(np.linspace(0, 1, len(arch_types)))
        color_map = dict(zip(arch_types, colors))
        
        # Model size vs Inference time
        for arch_type in arch_types:
            mask = [a == arch_type for a in self.results["architecture"]]
            ax1.scatter(
                np.array(self.results["size_mb"])[mask],
                np.array(self.results["inference_time_ms"])[mask],
                c=[color_map[arch_type]],
                label=arch_type,
                alpha=0.6
            )
        
        ax1.set_xlabel('Model Size (MB)')
        ax1.set_ylabel('Inference Time (ms)')
        ax1.set_title('Model Size vs Inference Time by Architecture')
        ax1.legend()
        
        # Perplexity vs Model size
        for arch_type in arch_types:
            mask = [a == arch_type for a in self.results["architecture"]]
            ax2.scatter(
                np.array(self.results["size_mb"])[mask],
                np.array(self.results["perplexity"])[mask],
                c=[color_map[arch_type]],
                label=arch_type,
                alpha=0.6
            )
        
        ax2.set_xlabel('Model Size (MB)')
        ax2.set_ylabel('Perplexity')
        ax2.set_title('Model Size vs Perplexity by Architecture')
        ax2.legend()
        
        # Resource usage by architecture
        x = range(len(arch_types))
        width = 0.25
        
        for i, arch_type in enumerate(arch_types):
            mask = [a == arch_type for a in self.results["architecture"]]
            cpu_avg = np.mean(np.array(self.results["cpu_usage"])[mask])
            mem_avg = np.mean(np.array(self.results["memory_usage"])[mask])
            gpu_avg = np.mean(np.array(self.results["gpu_usage"])[mask])
            
            ax3.bar(i - width, cpu_avg, width, color='blue', alpha=0.6)
            ax3.bar(i, mem_avg, width, color='green', alpha=0.6)
            ax3.bar(i + width, gpu_avg, width, color='red', alpha=0.6)
        
        ax3.set_xticks(x)
        ax3.set_xticklabels(arch_types, rotation=45)
        ax3.set_ylabel('Average Usage (%)')
        ax3.set_title('Resource Usage by Architecture Type')
        ax3.legend(['CPU', 'Memory', 'GPU'])
        
        # Performance metrics by architecture
        for arch_type in arch_types:
            mask = [a == arch_type for a in self.results["architecture"]]
            metrics = np.array([
                np.array(self.results["inference_time_ms"])[mask],
                np.array(self.results["perplexity"])[mask]
            ])
            metrics = (metrics - metrics.min(axis=1, keepdims=True)) / (metrics.max(axis=1, keepdims=True) - metrics.min(axis=1, keepdims=True))
            
            ax4.plot(range(len(metrics[0])), metrics[0], 'o-', label=f'{arch_type} - Time')
            ax4.plot(range(len(metrics[1])), metrics[1], 's-', label=f'{arch_type} - Perplexity')
        
        ax4.set_xlabel('Model Index')
        ax4.set_ylabel('Normalized Score')
        ax4.set_title('Normalized Performance Metrics by Architecture')
        ax4.legend()
        
        plt.tight_layout()
        plt.savefig('architecture_comparison.png')
        plt.show()

    def save_results(self, filename='model_comparison_results.json'):
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=4)

    def run_comparison(self):
        for arch in architectures:
            result = self.evaluate_model(arch)
            if result:
                for key in self.results:
                    self.results[key].append(result[key])
        
        # Plot results
        self.plot_architecture_comparison()
        
        # Save results
        self.save_results()
        
        # Print summary
        print("\nModel Architecture Comparison Summary:")
        print("-" * 100)
        print(f"{'Model':<20} {'Architecture':<15} {'Size (MB)':<12} {'Time (ms)':<12} {'Perplexity':<12} {'CPU %':<8} {'Memory %':<8} {'GPU %':<8}")
        print("-" * 100)
        for i in range(len(self.results["model"])):
            print(f"{self.results['model'][i]:<20} "
                  f"{self.results['architecture'][i]:<15} "
                  f"{self.results['size_mb'][i]:<12.2f} "
                  f"{self.results['inference_time_ms'][i]:<12.2f} "
                  f"{self.results['perplexity'][i]:<12.2f} "
                  f"{self.results['cpu_usage'][i]:<8.2f} "
                  f"{self.results['memory_usage'][i]:<8.2f} "
                  f"{self.results['gpu_usage'][i]:<8.2f}")

if __name__ == "__main__":
    comparison = ModelComparison()
    comparison.run_comparison() 