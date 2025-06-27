
# Section 1: Tensors in PyTorch

Tensors are the primary data structure in PyTorch and are conceptually similar to NumPy arrays but with key advantages like support for automatic differentiation and the ability to leverage GPUs for computations. Here’s a deeper dive into understanding tensors:

## 1.1 What is a Tensor?
A tensor is a generalization of vectors and matrices to potentially higher dimensions. For example:
- A **scalar** is a 0D tensor.
- A **vector** is a 1D tensor.
- A **matrix** is a 2D tensor.
- A tensor can also have 3D or more dimensions, like a cube of numbers (3D tensor) or higher.

Tensors allow PyTorch to express multidimensional data structures like images, videos, and more, in an efficient manner.

## 1.2 Tensor Creation
There are various ways to create tensors in PyTorch:
- **From lists or arrays**: The simplest way is to create a tensor from Python lists or NumPy arrays.
  ```python
  import torch
  x = torch.tensor([1, 2, 3])  # 1D tensor (vector)
  y = torch.tensor([[1, 2], [3, 4]])  # 2D tensor (matrix)
  ```
  
- **Using built-in methods**: PyTorch provides a variety of functions for creating tensors initialized in different ways:
  - Zeros and Ones:
    ```python
    x = torch.zeros(3, 3)  # 3x3 tensor of zeros
    y = torch.ones(2, 2)   # 2x2 tensor of ones
    ```
  - Random tensors:
    ```python
    z = torch.rand(3, 3)  # 3x3 tensor of random numbers between 0 and 1
    ```

- **Tensor from NumPy**: PyTorch allows seamless conversion between NumPy arrays and tensors.
  ```python
  import numpy as np
  np_array = np.array([1, 2, 3])
  tensor_from_np = torch.from_numpy(np_array)
  ```

## 1.3 Tensor Operations
Like NumPy arrays, tensors support a wide variety of mathematical operations. These include addition, subtraction, matrix multiplication, element-wise operations, and more.

Example of basic tensor operations:
```python
x = torch.tensor([1, 2, 3])
y = torch.tensor([4, 5, 6])

# Element-wise operations
z = x + y  # Tensor([5, 7, 9])
w = x * y  # Tensor([4, 10, 18])

# Matrix multiplication
a = torch.tensor([[1, 2], [3, 4]])
b = torch.tensor([[5, 6], [7, 8]])
c = torch.mm(a, b)  # Matrix multiplication result
```

## 1.4 Tensor Shapes and Manipulation
Manipulating tensor dimensions (shapes) is common in deep learning when you need to reshape data for different layers (like in CNNs or RNNs).

- **Shape and Reshape**:
  ```python
  x = torch.rand(3, 4)  # 3 rows, 4 columns
  reshaped_x = x.view(12)  # Reshape to a 1D tensor with 12 elements
  reshaped_back = reshaped_x.view(3, 4)  # Back to original shape
  ```

- **Squeeze and Unsqueeze**:
  Sometimes tensors may have unnecessary dimensions (e.g., for batch processing).
  - `squeeze()`: Removes dimensions of size 1.
  - `unsqueeze()`: Adds a new dimension of size 1.
  ```python
  x = torch.tensor([1, 2, 3])  # Shape: (3,)
  x_unsqueezed = x.unsqueeze(0)  # Adds dimension at position 0, new shape: (1, 3)
  x_squeezed = x_unsqueezed.squeeze()  # Removes dimension 0, shape back to (3,)
  ```

## 1.5 Device Placement (CPU/GPU)
PyTorch tensors can be stored on either the CPU or the GPU. Operations on GPU tensors are significantly faster due to parallel processing power.

- **Moving tensors to GPU**:
  ```python
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  x = torch.tensor([1, 2, 3], device=device)  # Directly create on GPU
  ```

- **Moving between CPU and GPU**:
  If you want to move a tensor created on the CPU to the GPU (or vice versa), you can use `.to()` or `.cuda()` methods:
  ```python
  x_cpu = torch.tensor([1, 2, 3])
  x_gpu = x_cpu.to('cuda')  # Move to GPU
  x_cpu_again = x_gpu.to('cpu')  # Move back to CPU
  ```

## 1.6 Tensor Broadcasting
PyTorch follows broadcasting rules (similar to NumPy) for performing operations on tensors of different shapes. Broadcasting automatically expands the smaller tensor to match the shape of the larger one, without copying data.

For example:
```python
x = torch.tensor([1, 2, 3])
y = torch.tensor([[1], [2], [3]])  # (3, 1) shape
result = x + y  # Broadcasting will match x to (3, 3) shape
```
The operation will produce the result:
```
Tensor([[2, 3, 4],
        [3, 4, 5],
        [4, 5, 6]])
```

## 1.7 Gradient Tracking with Tensors
Tensors can track computation and store the history of operations to compute gradients (which are essential for backpropagation during training of neural networks). To enable this, use the `requires_grad=True` flag.

- **Gradient-enabled tensor**:
  ```python
  x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
  ```

This is critical for machine learning because during backpropagation, gradients are automatically calculated for tensors with `requires_grad=True`.

## 1.8 Common Tensor Functions
Some common PyTorch tensor functions include:
- **Sum/Mean**:
  ```python
  x = torch.tensor([[1, 2], [3, 4]])
  x.sum()  # Sum of all elements
  x.mean()  # Mean of all elements
  ```

- **Argmax/Argmin**: Finds the indices of the maximum and minimum values.
  ```python
  x = torch.tensor([1, 5, 3, 2])
  x.argmax()  # Returns 1 (index of the maximum value)
  ```

- **Item**: Extracts a Python number from a single-element tensor.
  ```python
  x = torch.tensor([10])
  x.item()  # Returns 10 (Python integer)
  ```

## Conclusion for Section 1
Tensors are the foundation of PyTorch and mastering their creation, manipulation, and operations is essential for any deep learning task. They allow for efficient numerical computations and seamless use of GPU for accelerated tasks, making PyTorch a powerful tool for both small and large-scale machine learning models.
