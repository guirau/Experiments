
# Section 3: Neural Networks (`nn.Module`) in PyTorch

In PyTorch, neural networks are built using the `torch.nn` module, which provides a high-level abstraction for defining and managing layers, loss functions, and more. A neural network in PyTorch is typically defined as a class that inherits from `nn.Module`. Let’s dive deeper into how to build and use neural networks in PyTorch.

## 3.1 What is `nn.Module`?
The `nn.Module` is the base class for all neural network layers and models in PyTorch. Every layer of a neural network is a subclass of `nn.Module`. When you define a neural network, you inherit from this class and implement the layers and the forward pass.

Key points:
- It provides infrastructure for parameter handling and backpropagation.
- It organizes all the parameters (weights, biases) and submodules (layers) neatly.
- The `forward()` method defines the forward pass of the model.

## 3.2 Defining a Simple Neural Network
To define a neural network, you typically follow these steps:
1. Subclass `nn.Module`.
2. Define the layers of the network in the `__init__()` method.
3. Implement the forward pass in the `forward()` method.

Example:
```python
import torch
import torch.nn as nn

# Define a simple neural network
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(10, 50)  # Fully connected layer: 10 input features -> 50 output features
        self.fc2 = nn.Linear(50, 1)   # Fully connected layer: 50 input features -> 1 output feature

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # Apply ReLU activation function after the first layer
        x = self.fc2(x)              # Output layer
        return x

# Instantiate the model
model = SimpleNN()
```

In this example:

- `__init__()` defines the layers (two fully connected layers: fc1 and fc2).
- `forward()` specifies how the input moves through the layers and applies activation functions.
- **ReLU** (Rectified Linear Unit) is applied after the first layer to introduce non-linearity.

## 3.3 Layer Types
PyTorch provides a wide variety of layers for constructing different types of neural networks. Here are some of the most commonly used layer types:

1. **Linear (Fully Connected) Layers**: These layers perform a matrix multiplication of the input with the weights and add a bias term.
   ```python
   nn.Linear(in_features, out_features)
   ```

2. **Convolutional Layers**: These layers are used in convolutional neural networks (CNNs), commonly used for image processing.
   ```python
   nn.Conv2d(in_channels, out_channels, kernel_size)
   ```

3. **Recurrent Layers**: Layers like `nn.RNN`, `nn.LSTM`, and `nn.GRU` are used in recurrent neural networks (RNNs), commonly applied in sequence data (e.g., time series or natural language processing).
   ```python
   nn.LSTM(input_size, hidden_size, num_layers)
   ```

4. **Dropout Layers**: These layers randomly drop some neurons during training to prevent overfitting.
   ```python
   nn.Dropout(p=0.5)  # p is the probability of dropping a neuron
   ```

5. **Batch Normalization**: Normalizes the output of a previous layer, which can help with the stability of the network.
   ```python
   nn.BatchNorm1d(num_features)
   ```

## 3.4 Activations
Activation functions introduce non-linearity into the network, allowing it to learn more complex functions. Common activation functions include:
- **ReLU**: `torch.relu(x)` or `nn.ReLU()`
- **Sigmoid**: `torch.sigmoid(x)` or `nn.Sigmoid()`
- **Tanh**: `torch.tanh(x)` or `nn.Tanh()`

In practice, you usually define these activations within the `forward()` method, like this:
```python
def forward(self, x):
    x = torch.relu(self.fc1(x))
    x = torch.sigmoid(self.fc2(x))  # Use Sigmoid in the output layer for binary classification
    return x
```

## 3.5 Forward Pass
The `forward()` method in a `nn.Module` defines the forward pass of the network. When you pass input data to the model, PyTorch internally calls this method to compute the output.

Example:
```python
input_data = torch.randn(32, 10)  # Batch of 32 examples, each with 10 features
output = model(input_data)        # Forward pass through the model
```
In this case, the forward pass will compute the output by passing the input through the layers in the order defined in `forward()`.

## 3.6 Backward Pass
Once you define the forward pass and compute a loss function, you can compute the gradients for the model parameters using `.backward()`.

Example:
```python
# Define loss function and optimizer
loss_fn = nn.MSELoss()  # Mean squared error loss
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Forward pass: compute the output
output = model(input_data)

# Compute loss
loss = loss_fn(output, target)

# Backward pass: compute gradients
loss.backward()

# Update model parameters using the optimizer
optimizer.step()
```

In this example:

- The forward pass computes the output.
- The loss function compares the output to the target and computes the loss.
- The `backward()` method calculates the gradients of the loss with respect to the model parameters.
- The optimizer updates the parameters based on the computed gradients.

## 3.7 Parameter Management
All layers defined in a `nn.Module` automatically register their parameters (like weights and biases), making it easy to access and manage them.

- **Accessing parameters**:
  ```python
  for name, param in model.named_parameters():
      print(name, param.size())
  ```
  This prints the name and size of each parameter in the model.

- **Initializing parameters manually**:
  You can initialize the parameters manually using PyTorch’s in-built functions or custom methods.
  ```python
  nn.init.xavier_uniform_(model.fc1.weight)  # Xavier initialization for fc1's weights
  ```

## 3.8 Submodules
One powerful feature of `nn.Module` is that models can contain other models (submodules), making it easy to compose complex models.

Example:
```python
class ComplexNN(nn.Module):
    def __init__(self):
        super(ComplexNN, self).__init__()
        self.fc = SimpleNN()  # A neural network as a layer
        self.other_layer = nn.Linear(50, 10)

    def forward(self, x):
        x = self.fc(x)  # Forward pass through the SimpleNN
        x = self.other_layer(x)
        return x
```
This modular design helps in creating more complex architectures while keeping the code organized.

## 3.9 Model Training Mode vs. Evaluation Mode
Neural networks behave differently during training and evaluation. For example, dropout layers or batch normalization behave differently during training (when randomness or updating statistics is needed) versus evaluation (where we use the learned statistics).

- **Training mode**: By default, the model is in training mode. If you’re training, make sure to use:
  ```python
  model.train()  # Set the model to training mode
  ```

- **Evaluation mode**: Before evaluating your model on validation or test data, you should switch to evaluation mode:
  ```python
  model.eval()  # Set the model to evaluation mode
  ```
  This ensures that layers like dropout and batch normalization behave appropriately during evaluation.

## 3.10 Customizing Models
Since `nn.Module` is a class, you have full flexibility to customize it. You can define custom operations, layers, and even modify the forward pass logic.

Example of a custom model:
```python
class CustomModel(nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()
        self.fc1 = nn.Linear(10, 50)
        self.fc2 = nn.Linear(50, 1)

    def forward(self, x):
        # Custom logic: Apply ReLU only if the mean of the input is positive
        if x.mean() > 0:
            x = torch.relu(self.fc1(x))
        else:
            x = self.fc1(x)
        x = self.fc2(x)
        return x
```

This flexibility allows you to define any architecture and forward pass logic that suits your problem.

## Conclusion for Section 3
- The `nn.Module` class is the foundation of neural networks in PyTorch, allowing you to define and manage layers, forward passes, and parameters efficiently.
- You can build custom neural networks by subclassing `nn.Module`, and PyTorch automatically tracks and manages the model parameters.
- PyTorch provides various types of layers (fully connected, convolutional, recurrent, etc.), activation functions, and utilities to construct complex models.
- Understanding how to structure your neural networks and how the forward and backward passes work will help you build and train deep learning models effectively in PyTorch.
