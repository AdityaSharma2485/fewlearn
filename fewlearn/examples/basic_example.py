"""
A basic example of using the fewlearn module for few-shot learning.

This example demonstrates how to:
1. Initialize the MINDS framework
2. Add different backbone models
3. Evaluate them in parallel on the Omniglot dataset
4. Get the best performing model
"""

import torch
import matplotlib.pyplot as plt
from torchvision.datasets import Omniglot
from torchvision import transforms
import os

from fewlearn import MINDS, PrototypicalNetworks, EpisodicProtocol, Evaluator
from fewlearn.visualization import plot_performance_comparison, plot_prototype_embeddings

# Set random seed for reproducibility
torch.manual_seed(42)

def main():
    print("Initializing FewLearn demo...")
    
    # 1. Initialize the MINDS framework
    minds = MINDS()
    
    # 2. Add different backbone models to evaluate
    print("Adding models with different backbones...")
    minds.add_model("resnet18", PrototypicalNetworks(backbone="resnet18"))
    minds.add_model("mobilenet_v2", PrototypicalNetworks(backbone="mobilenet_v2"))
    minds.add_model("efficientnet_b0", PrototypicalNetworks(backbone="efficientnet_b0"))
    
    # 3. Prepare dataset
    print("Preparing Omniglot dataset...")
    os.makedirs('./data', exist_ok=True)
    
    # Fix: Convert grayscale to RGB BEFORE converting to tensor
    transform = transforms.Compose([
        transforms.Resize((84, 84)),
        transforms.Grayscale(num_output_channels=3),  # Convert grayscale to 3 channels FIRST
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    # Use Omniglot dataset for few-shot learning
    dataset = Omniglot(root='./data', download=True, transform=transform)
    
    # 4. Create an evaluation protocol
    protocol = EpisodicProtocol(n_way=5, n_shot=1, n_query=15, episodes=50)
    
    # 5. Create an evaluator
    evaluator = Evaluator(
        protocol=protocol,
        metrics=["accuracy", "f1"],
        parallel=True  # Enable parallel evaluation
    )
    
    # 6. Run the evaluation with a progress callback
    print("Evaluating models in parallel...")
    
    # Define a simple progress callback
    def progress_callback(progress):
        print(f"Evaluation progress: {progress:.1%}", end="\r")
    
    results = evaluator.evaluate(
        models=minds.models,
        dataset=dataset,
        progress_callback=progress_callback
    )
    
    print("\nEvaluation complete!")
    
    # 7. Get a summary of the results
    summary = evaluator.summary()
    print("\nResults summary:")
    print(summary)
    
    # 8. Get the best model
    best_model_name, best_model = minds.get_best_model(results)
    print(f"\nBest model: {best_model_name}")
    
    # 9. Create a visualization folder
    os.makedirs('./visualizations', exist_ok=True)
    
    # 10. Plot model performance comparison
    print("Creating performance comparison visualization...")
    fig = plot_performance_comparison(results)
    fig.savefig("./visualizations/model_comparison.png")
    
    # 11. Export the best model for deployment
    print("Exporting the best model...")
    export_path = minds.export_model(best_model_name, format="pt")
    print(f"Model exported to: {export_path}")
    
    print("Demo completed successfully!")

if __name__ == "__main__":
    main() 
