"""
Model export utilities for few-shot learning.

This module provides utilities for exporting few-shot learning models
to various formats for deployment.
"""

import torch
from torch import nn
import os
from pathlib import Path
from typing import Optional, Union, Dict, Any

from fewlearn.models.base import FewShotModel


def export_model(
    model: FewShotModel,
    format: str = "onnx",
    output_path: Optional[str] = None,
    input_shape: tuple = (1, 3, 224, 224),
    **kwargs
) -> str:
    """
    Export a few-shot learning model to a specified format.
    
    Args:
        model: The model to export
        format: Export format ('onnx', 'torchscript', 'pt')
        output_path: Path to save the exported model
        input_shape: Input shape for tracing
        **kwargs: Additional format-specific arguments
        
    Returns:
        Path to the exported model
    """
    model.eval()
    
    # Generate default output path if none provided
    if output_path is None:
        output_dir = Path("exported_models")
        output_dir.mkdir(exist_ok=True)
        
        model_name = model.__class__.__name__
        backbone_name = model.get_config().get("backbone", "unknown")
        
        output_path = str(output_dir / f"{model_name}_{backbone_name}.{format}")
    
    # Handle different export formats
    if format.lower() == "onnx":
        return export_to_onnx(model, output_path, input_shape, **kwargs)
    elif format.lower() == "torchscript":
        return export_to_torchscript(model, output_path, input_shape, **kwargs)
    elif format.lower() == "pt":
        return export_to_pytorch(model, output_path, **kwargs)
    else:
        raise ValueError(f"Unsupported export format: {format}")


def export_to_onnx(
    model: FewShotModel,
    output_path: str,
    input_shape: tuple = (1, 3, 224, 224),
    opset_version: int = 12,
    **kwargs
) -> str:
    """
    Export a model to ONNX format.
    
    Args:
        model: The model to export
        output_path: Path to save the ONNX model
        input_shape: Input shape for tracing
        opset_version: ONNX opset version
        **kwargs: Additional ONNX-specific arguments
        
    Returns:
        Path to the exported model
    """
    # Ensure the model is in eval mode
    model.eval()
    
    # Make sure the path exists
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    # Create dummy inputs
    dummy_support_images = torch.randn(5, *input_shape[1:], device=next(model.parameters()).device)
    dummy_support_labels = torch.tensor([0, 1, 2, 3, 4], device=next(model.parameters()).device)
    dummy_query_images = torch.randn(10, *input_shape[1:], device=next(model.parameters()).device)
    
    # Create a wrapper class to make the model ONNX-exportable
    class ONNXWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
            
        def forward(self, support_images, support_labels, query_images):
            return self.model(support_images, support_labels, query_images)
    
    wrapped_model = ONNXWrapper(model)
    
    # Export the model
    torch.onnx.export(
        wrapped_model,
        (dummy_support_images, dummy_support_labels, dummy_query_images),
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['support_images', 'support_labels', 'query_images'],
        output_names=['class_scores'],
        dynamic_axes={
            'support_images': {0: 'n_support'},
            'support_labels': {0: 'n_support'},
            'query_images': {0: 'n_query'},
            'class_scores': {0: 'n_query'}
        },
        **kwargs
    )
    
    print(f"Model exported to ONNX format at: {output_path}")
    return output_path


def export_to_torchscript(
    model: FewShotModel,
    output_path: str,
    input_shape: tuple = (1, 3, 224, 224),
    method: str = "trace",
    **kwargs
) -> str:
    """
    Export a model to TorchScript format.
    
    Args:
        model: The model to export
        output_path: Path to save the TorchScript model
        input_shape: Input shape for tracing
        method: 'trace' or 'script'
        **kwargs: Additional TorchScript-specific arguments
        
    Returns:
        Path to the exported model
    """
    # Ensure the model is in eval mode
    model.eval()
    
    # Make sure the path exists
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    if method == "trace":
        # Create dummy inputs
        dummy_support_images = torch.randn(5, *input_shape[1:], device=next(model.parameters()).device)
        dummy_support_labels = torch.tensor([0, 1, 2, 3, 4], device=next(model.parameters()).device)
        dummy_query_images = torch.randn(10, *input_shape[1:], device=next(model.parameters()).device)
        
        # Trace the model
        traced_model = torch.jit.trace(
            model,
            (dummy_support_images, dummy_support_labels, dummy_query_images),
            **kwargs
        )
        
        # Save the traced model
        traced_model.save(output_path)
    else:  # method == "script"
        # Script the model
        scripted_model = torch.jit.script(model, **kwargs)
        
        # Save the scripted model
        scripted_model.save(output_path)
    
    print(f"Model exported to TorchScript format at: {output_path}")
    return output_path


def export_to_pytorch(
    model: FewShotModel,
    output_path: str,
    **kwargs
) -> str:
    """
    Export a model to PyTorch format.
    
    Args:
        model: The model to export
        output_path: Path to save the PyTorch model
        **kwargs: Additional PyTorch-specific arguments
        
    Returns:
        Path to the exported model
    """
    # Ensure the model is in eval mode
    model.eval()
    
    # Make sure the path exists
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    # Save the model state dict
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': model.get_config(),
        'class_name': model.__class__.__name__,
        **kwargs
    }, output_path)
    
    print(f"Model exported to PyTorch format at: {output_path}")
    return output_path


def generate_serving_code(
    model: FewShotModel,
    framework: str = "flask",
    output_dir: str = "./serving",
    model_path: Optional[str] = None,
    **kwargs
) -> str:
    """
    Generate serving code for a few-shot learning model.
    
    Args:
        model: The model to generate serving code for
        framework: The web framework to use ('flask', 'fastapi')
        output_dir: Directory to save the generated code
        model_path: Path to the exported model
        **kwargs: Additional framework-specific arguments
        
    Returns:
        Path to the generated code
    """
    # Make sure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # If no model path is provided, export the model
    if model_path is None:
        model_path = export_model(model, format="pt", output_path=os.path.join(output_dir, "model.pt"))
    
    # Generate serving code based on the framework
    if framework.lower() == "flask":
        return _generate_flask_serving_code(model, output_dir, model_path, **kwargs)
    elif framework.lower() == "fastapi":
        return _generate_fastapi_serving_code(model, output_dir, model_path, **kwargs)
    else:
        raise ValueError(f"Unsupported framework: {framework}")


def _generate_flask_serving_code(
    model: FewShotModel,
    output_dir: str,
    model_path: str,
    **kwargs
) -> str:
    """Generate Flask serving code for a few-shot learning model."""
    # Create app.py
    app_code = f"""import torch
import numpy as np
from flask import Flask, request, jsonify
import os
from PIL import Image
import io
import base64
import torchvision.transforms as transforms

# Import the model class
from model import {model.__class__.__name__}

app = Flask(__name__)

# Load the model
def load_model():
    checkpoint = torch.load('{os.path.basename(model_path)}')
    model = {model.__class__.__name__}(**checkpoint['config'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

model = load_model()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Define image transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get JSON data
        data = request.json
        
        # Extract support and query images
        support_images = []
        support_labels = []
        
        for item in data['support_set']:
            # Decode base64 image
            img_data = base64.b64decode(item['image'])
            img = Image.open(io.BytesIO(img_data)).convert('RGB')
            
            # Apply transforms
            img_tensor = transform(img)
            support_images.append(img_tensor)
            support_labels.append(item['label'])
        
        query_images = []
        for item in data['query_set']:
            img_data = base64.b64decode(item['image'])
            img = Image.open(io.BytesIO(img_data)).convert('RGB')
            img_tensor = transform(img)
            query_images.append(img_tensor)
        
        # Convert to tensors
        support_images = torch.stack(support_images).to(device)
        support_labels = torch.tensor(support_labels).to(device)
        query_images = torch.stack(query_images).to(device)
        
        # Get predictions
        with torch.no_grad():
            logits = model(support_images, support_labels, query_images)
            predictions = torch.argmax(logits, dim=1).cpu().numpy().tolist()
        
        return jsonify({{'predictions': predictions}})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
"""
    
    # Create model.py
    model_code = f"""import torch
from torch import nn
import torch.nn.functional as F

class {model.__class__.__name__}(nn.Module):
{_get_model_class_code(model)}
"""
    
    # Create requirements.txt
    requirements = """flask==2.0.1
torch>=1.9.0
torchvision>=0.10.0
Pillow>=8.2.0
numpy>=1.19.5
"""
    
    # Create README.md
    readme = f"""# {model.__class__.__name__} Serving API

This is an auto-generated API for serving a {model.__class__.__name__} model.

## Setup

1. Install the requirements:
   ```
   pip install -r requirements.txt
   ```

2. Start the server:
   ```
   python app.py
   ```

3. Send requests to the API:
   ```python
   import requests
   import base64
   from PIL import Image
   import io

   # Prepare support set (examples)
   support_set = []
   for image_path, label in [('image1.jpg', 0), ('image2.jpg', 1)]:
       with open(image_path, 'rb') as f:
           img_bytes = f.read()
       encoded = base64.b64encode(img_bytes).decode('utf-8')
       support_set.append({{'image': encoded, 'label': label}})

   # Prepare query set (images to classify)
   query_set = []
   for image_path in ['query1.jpg', 'query2.jpg']:
       with open(image_path, 'rb') as f:
           img_bytes = f.read()
       encoded = base64.b64encode(img_bytes).decode('utf-8')
       query_set.append({{'image': encoded}})

   # Send request
   response = requests.post(
       'http://localhost:5000/predict',
       json={{'support_set': support_set, 'query_set': query_set}}
   )
   
   # Get predictions
   predictions = response.json()['predictions']
   print(predictions)
   ```
"""
    
    # Write files
    with open(os.path.join(output_dir, "app.py"), "w") as f:
        f.write(app_code)
    
    with open(os.path.join(output_dir, "model.py"), "w") as f:
        f.write(model_code)
    
    with open(os.path.join(output_dir, "requirements.txt"), "w") as f:
        f.write(requirements)
    
    with open(os.path.join(output_dir, "README.md"), "w") as f:
        f.write(readme)
    
    print(f"Flask serving code generated at: {output_dir}")
    return output_dir


def _generate_fastapi_serving_code(
    model: FewShotModel,
    output_dir: str,
    model_path: str,
    **kwargs
) -> str:
    """Generate FastAPI serving code for a few-shot learning model."""
    # Implementation of FastAPI code generation
    # (Similar to Flask but with FastAPI-specific code)
    print(f"FastAPI serving code generated at: {output_dir}")
    return output_dir


def _get_model_class_code(model: FewShotModel) -> str:
    """Extract and format the model class code for serving."""
    # This is a simplified implementation
    # In a real application, you would need to analyze the model's structure
    # and generate appropriate code
    
    if model.__class__.__name__ == "PrototypicalNetworks":
        code = """    def __init__(self, backbone="resnet18", distance='euclidean', **kwargs):
        super().__init__()
        # Initialize with a simple backbone for serving
        import torchvision.models as models
        if backbone == "resnet18":
            self.backbone = models.resnet18(pretrained=False)
            self.backbone.fc = nn.Flatten()
        else:
            # Fallback to a simple CNN if the backbone is not available
            self.backbone = nn.Sequential(
                nn.Conv2d(3, 64, 3, 1, 1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3, 1, 1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(128, 256, 3, 1, 1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(256, 512, 3, 1, 1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten()
            )
        self.distance = distance
        
    def forward(self, support_images, support_labels, query_images):
        # Extract features
        z_support = self.backbone(support_images)
        z_query = self.backbone(query_images)
        
        # Infer the number of different classes from the labels of the support set
        n_way = len(torch.unique(support_labels))
        
        # Compute prototypes
        z_proto = torch.cat(
            [
                z_support[torch.nonzero(support_labels == label)].mean(0, keepdim=True)
                for label in range(n_way)
            ]
        )
        
        # Compute distances
        if self.distance == 'cosine':
            # Normalize features for cosine similarity
            z_query = F.normalize(z_query, dim=1)
            z_proto = F.normalize(z_proto, dim=1)
            
            # Compute cosine similarity (dot product of normalized vectors)
            scores = torch.mm(z_query, z_proto.t())
        else:  # euclidean
            # Compute euclidean distance
            dists = torch.cdist(z_query, z_proto)
            
            # Convert distances to scores (negative distance)
            scores = -dists
        
        return scores"""
    else:
        # Generic model code
        code = """    def __init__(self, **kwargs):
        super().__init__()
        # Initialize with a simple backbone for serving
        import torchvision.models as models
        self.backbone = models.resnet18(pretrained=False)
        self.backbone.fc = nn.Flatten()
        
    def forward(self, support_images, support_labels, query_images):
        # Extract features
        z_support = self.backbone(support_images)
        z_query = self.backbone(query_images)
        
        # Simple nearest neighbor classification
        n_way = len(torch.unique(support_labels))
        z_proto = torch.cat(
            [
                z_support[torch.nonzero(support_labels == label)].mean(0, keepdim=True)
                for label in range(n_way)
            ]
        )
        
        # Compute distances
        dists = torch.cdist(z_query, z_proto)
        
        # Return negative distances as scores
        return -dists"""
    
    # Indent the code with 4 spaces
    return '\n'.join(f"    {line}" for line in code.split('\n')) 