# FewLearn

A Python module for few-shot learning with pretrained models, enabling efficient model evaluation and comparison through prototypical networks.

## Overview

FewLearn provides tools for comparing and evaluating multiple pretrained models in parallel using few-shot learning techniques like Prototypical Networks. The framework allows you to:

- Efficiently evaluate multiple backbone models in parallel
- Analyze model performance across various metrics
- Visualize embeddings, confusion matrices, and model comparisons
- Export models for deployment

## Key Components

- **MINDS**: Main framework for coordinating few-shot learning evaluations
- **Prototypical Networks**: Implementation of the few-shot learning algorithm
- **Backbones**: Support for various pre-trained model architectures 
- **Evaluation**: Protocols and metrics for comparing model performance
- **Visualization**: Tools for visualizing embeddings and results

## Installation

```bash
# Basic installation
pip install fewlearn

# With optional dependencies
pip install fewlearn[dev,easyfsl,demo]
```

## Quick Start

```python
import torch
from torchvision.datasets import Omniglot
from torchvision import transforms
from fewlearn import MINDS, PrototypicalNetworks, Evaluator, EpisodicProtocol

# 1. Initialize the MINDS framework
minds = MINDS()

# 2. Add different backbone models to evaluate
minds.add_model("resnet18", PrototypicalNetworks(backbone="resnet18"))
minds.add_model("mobilenet_v2", PrototypicalNetworks(backbone="mobilenet_v2"))
minds.add_model("efficientnet_b0", PrototypicalNetworks(backbone="efficientnet_b0"))

# 3. Prepare dataset
transform = transforms.Compose([
    transforms.Resize((84, 84)),
    transforms.Grayscale(num_output_channels=3),  # Convert grayscale to RGB for ImageNet normalization
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

dataset = Omniglot(root='./data', download=True, transform=transform)

# 4. Create an evaluation protocol
protocol = EpisodicProtocol(n_way=5, n_shot=5,n_query=15, episodes=100)

# 5. Create an evaluator
evaluator = Evaluator(
    protocol=protocol,
    metrics=["accuracy", "f1"],
    parallel=True  # Enable parallel evaluation
)

# 6. Run the evaluation
results = evaluator.evaluate(
    models={name: model for name, model in minds.models.items()},
    dataset=dataset
)

# 7. Get a summary of the results
summary = evaluator.summary()
print(summary)

# 8. Get the best model
best_model_name, best_model = minds.get_best_model(results)
print(f"Best model: {best_model_name}")

# 9. Export the best model for deployment
export_path = minds.export_model(best_model_name, format="onnx")
print(f"Model exported to: {export_path}")
```

## Advanced Features

### Custom Backbone Models

```python
from torch import nn
from fewlearn.backbones import register_backbone

# Define a custom backbone
def my_custom_backbone(pretrained=True):
    # Create your custom model here
    model = nn.Sequential(
        # ...layers
    )
    return model

# Register the backbone
register_backbone("my_custom_model", my_custom_backbone)

# Use it in a few-shot model
model = PrototypicalNetworks(backbone="my_custom_model")
```

### Visualization

```python
from fewlearn.visualization import (
    plot_prototype_embeddings,
    plot_confusion_matrix,
    plot_performance_comparison
)

# Plot model performance comparison
fig = plot_performance_comparison(results)
fig.savefig("model_comparison.png")

# Plot embedding space visualizations
fig = plot_prototype_embeddings(
    support_embeddings,
    support_labels,
    query_embeddings,
    query_labels
)
fig.savefig("embeddings.png")
```

## Development

### Setting Up Development Environment

```bash
# Clone the repository
git clone https://github.com/AdityaSharma2485/fewlearn.git
cd fewlearn

# Install development dependencies
pip install -e ".[dev]"
```

### Running Tests

FewLearn comes with a suite of unit tests to ensure the functionality works as expected:

```bash
# Run all tests
pytest tests/

# Run specific test modules
pytest tests/test_minds.py
pytest tests/test_prototypical.py
```

## Demo Application

A Streamlit-based demo application is available in the `demoapp.py` file. To run it:

```bash
# Install demo dependencies
pip install -e ".[demo]"

# Run the demo app
streamlit run demoapp.py
```

The demo allows you to:
- Compare different backbone models
- Test on the Omniglot dataset
- Upload and test your own custom datasets
- Visualize model performance and predictions

## Contributing

Contributions to FewLearn are welcome! To contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests to ensure they pass (`pytest tests/`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

Please ensure your code follows the project's style guidelines and includes appropriate tests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- PyTorch team for the excellent deep learning framework
- Authors of the paper "Prototypical Networks for Few-shot Learning" for the foundational algorithm
- Contributors and users of the FewLearn library

