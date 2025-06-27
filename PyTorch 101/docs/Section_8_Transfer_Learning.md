
# Section 8: Transfer Learning in PyTorch

Transfer learning is a powerful technique in machine learning that leverages a pre-trained model (trained on a large dataset) to adapt it to a new, related task with a smaller dataset. In PyTorch, transfer learning is often used for image classification, object detection, and NLP tasks where models are fine-tuned or adapted to new problems, saving time and computational resources.

Let’s dive deeper into the key aspects of transfer learning in PyTorch, how to fine-tune models, and how to use pre-trained models effectively.

## 8.1 What is Transfer Learning?
Transfer learning involves taking a model that has been pre-trained on a large dataset (such as ImageNet for image tasks) and using that model’s learned features to solve a different but related problem.

There are two primary strategies for transfer learning:

1. **Feature Extraction**: Freeze the pre-trained model's weights and use it as a feature extractor. You only train a new classifier layer on top of the frozen model.
2. **Fine-Tuning**: Unfreeze some or all layers of the pre-trained model and continue training it on the new task, updating the weights based on the new data.

## 8.2 Why Use Transfer Learning?

Transfer learning is useful when:

- **Data Scarcity**: You have a small dataset, but the task is related to a well-established domain (e.g., image recognition). Pre-trained models already have useful features that can generalize well to new tasks.
- **Reduced Training Time**: Instead of training a model from scratch, which can be time-consuming and resource-intensive, transfer learning leverages the learned weights of a pre-trained model, drastically reducing training time.
- **Improved Performance**: Transfer learning often results in better performance on small datasets compared to training from scratch.

## 8.3 Pre-trained Models in PyTorch
PyTorch provides several pre-trained models for tasks like image classification, object detection, and natural language processing through the `torchvision.models` and `transformers` libraries.

Some commonly used pre-trained models available in `torchvision` include:

- **ResNet**: A deep residual network that is widely used for image classification.
- **VGG**: A convolutional neural network known for its simplicity and depth.
- **Inception**: A deep convolutional architecture optimized for image classification.
- **MobileNet**: A lightweight convolutional model optimized for mobile and embedded vision applications.

These models are pre-trained on the **ImageNet** dataset, a large dataset of over 1.2 million images classified into 1000 categories.

### Example: Loading a Pre-trained ResNet Model
```python
import torch
import torchvision.models as models

# Load a pre-trained ResNet model
model = models.resnet18(pretrained=True)

# Set the model to evaluation mode
model.eval()
```
- **`pretrained=True`** loads the model with the weights trained on the ImageNet dataset.
- **`eval()`** sets the model to evaluation mode (important when using models for inference).

## 8.4 Feature Extraction vs. Fine-Tuning

### 1. **Feature Extraction**
In this approach, you freeze the weights of the pre-trained model (i.e., the convolutional layers) so that their parameters are not updated during training. You replace the final classification layer with a new layer corresponding to your specific task, and only train this new layer.

Example:
```python
import torch.nn as nn

# Load pre-trained ResNet18 model
model = models.resnet18(pretrained=True)

# Freeze all layers (disable gradient updates for all parameters)
for param in model.parameters():
    param.requires_grad = False

# Replace the final fully connected layer with a new layer (for 10-class classification)
num_features = model.fc.in_features  # Get the number of input features to the last layer
model.fc = nn.Linear(num_features, 10)  # Replace it with a new layer
```

Here:

- **Freezing layers**: You freeze the pre-trained layers by setting param.`requires_grad = False`. This means that these layers won’t be updated during training.
- **Replacing the final layer**: You replace the final layer (`model.fc`) with a new fully connected layer for your task (e.g., a 10-class classification problem).

This approach is useful when you have a small dataset and don’t want to risk overfitting by updating the large number of parameters in the pre-trained model.


### 2. **Fine-Tuning**
In fine-tuning, you unfreeze the pre-trained model and train some or all of its layers on the new dataset. Typically, you will fine-tune the later layers of the model, as these contain more task-specific information.

Example:
```python
# Load pre-trained ResNet18 model
model = models.resnet18(pretrained=True)

# Freeze all layers
for param in model.parameters():
    param.requires_grad = False

# Unfreeze the last layer
for param in model.fc.parameters():
    param.requires_grad = True

# Replace the final fully connected layer for your task
model.fc = nn.Linear(model.fc.in_features, 10)
```

In this approach:

- You freeze most of the layers and only fine-tune the last few layers (e.g., the fully connected layer).
- Fine-tuning is more computationally expensive than feature extraction but can yield better results if the new task is significantly different from the original task.

## 8.5 Using Pre-trained Models for Image Classification
Here’s an end-to-end example of using transfer learning with a pre-trained ResNet model for image classification on a custom dataset:

### Step 1: Load a Pre-trained Model
```python
import torch
import torchvision.models as models

# Load pre-trained ResNet18 model
model = models.resnet18(pretrained=True)

# Replace the final layer for 10-class classification
num_features = model.fc.in_features
model.fc = torch.nn.Linear(num_features, 10)
```

### Step 2: Freeze Layers (for Feature Extraction)
```python
for param in model.parameters():
    param.requires_grad = False  # Freeze all layers except the last one

# Unfreeze the final layer to train it
for param in model.fc.parameters():
    param.requires_grad = True
```

### Step 3: Prepare the Data
```python
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Define data transformations
transform = transforms.Compose([
    transforms.Resize(224),  # ResNet expects 224x224 images
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load custom dataset
train_dataset = datasets.ImageFolder(root='path/to/train_data', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
```

### Step 4: Define Loss Function and Optimizer
```python
import torch.optim as optim

# Define loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()  # For classification
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)  # Only optimize the final layer
```

### Step 5: Training the Model
```python
model = model.to(device)  # Move model to GPU

# Training loop
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)  # Move data to GPU

        optimizer.zero_grad()  # Zero the gradients
        outputs = model(inputs)  # Forward pass
        loss = criterion(outputs, labels)  # Compute loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update weights

        running_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader)}")
```

### Step 6: Evaluating the Model
After training, you can evaluate the model on a validation or test set:
```python
model.eval()  # Set model to evaluation mode

correct = 0
total = 0
with torch.no_grad():  # Disable gradient computation for evaluation
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy: {100 * correct / total}%')
```

## 8.6 Using Pre-trained Models for NLP Tasks

Transfer learning is also common in natural language processing (NLP). Pre-trained models like BERT, GPT, and T5 can be fine-tuned for various downstream NLP tasks such as text classification, sentiment analysis, and question answering.

You can use the Hugging Face Transformers library for easy access to pre-trained models for NLP tasks.

Example of loading a pre-trained BERT model:
```python
from transformers import BertTokenizer, BertForSequenceClassification

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Tokenize input
inputs = tokenizer("Example sentence for classification", return_tensors="pt")

# Forward pass
outputs = model(**inputs)
```

The Hugging Face Transformers library allows for easy fine-tuning of pre-trained NLP models for specific tasks with minimal changes.

## Conclusion for Section 8
Transfer learning in PyTorch is a highly effective technique for leveraging pre-trained models on new tasks, especially when you have limited data or computational resources. PyTorch’s `torchvision.models` library provides easy access to pre-trained models for vision tasks, while Hugging Face’s Transformers library provides models for NLP. You can either use these models for **feature extraction** or **fine-tuning** based on the complexity of your new task. This approach allows you to save time, improve performance, and reduce the need for extensive model training from scratch.
