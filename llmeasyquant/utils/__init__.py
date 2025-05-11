"""
Utility functions for LLMEasyQuant

This module provides various utility functions for:
1. Performance Analysis
   - Model evaluation metrics
   - Speed and memory profiling
   - Accuracy measurements

2. Visualization
   - Quantization results plotting
   - Performance comparison charts
   - Memory usage visualization

3. Data Processing
   - Dataset loading and preprocessing
   - Tokenization utilities
   - Batch processing helpers

4. Model Management
   - Model loading and saving
   - Checkpoint management
   - Configuration handling

5. System Utilities
   - Hardware detection
   - Memory management
   - Logging and monitoring
"""

from .metrics import (
    calculate_accuracy,
    measure_inference_time,
    profile_memory_usage,
    compute_compression_ratio
)

from .visualization import (
    plot_quantization_results,
    plot_performance_comparison,
    plot_memory_usage,
    create_quantization_heatmap
)

from .data import (
    load_dataset,
    preprocess_text,
    create_dataloader,
    batch_process
)

from .model import (
    load_model,
    save_model,
    load_checkpoint,
    save_checkpoint,
    load_config
)

from .system import (
    get_hardware_info,
    monitor_memory,
    setup_logging,
    get_system_metrics
)

__all__ = [
    # Metrics
    'calculate_accuracy',
    'measure_inference_time',
    'profile_memory_usage',
    'compute_compression_ratio',
    
    # Visualization
    'plot_quantization_results',
    'plot_performance_comparison',
    'plot_memory_usage',
    'create_quantization_heatmap',
    
    # Data Processing
    'load_dataset',
    'preprocess_text',
    'create_dataloader',
    'batch_process',
    
    # Model Management
    'load_model',
    'save_model',
    'load_checkpoint',
    'save_checkpoint',
    'load_config',
    
    # System Utilities
    'get_hardware_info',
    'monitor_memory',
    'setup_logging',
    'get_system_metrics'
] 