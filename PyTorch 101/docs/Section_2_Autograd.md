
# Section 2: Autograd (Automatic Differentiation) in PyTorch

PyTorch’s `autograd` module enables **automatic differentiation**, which is a key feature for efficiently training neural networks through gradient-based optimization methods like backpropagation. It does this by dynamically building a computational graph that tracks all operations on tensors with `requires_grad=True`.

Let’s dive deeper into how autograd works and the key concepts involved.

## 2.1 What is Automatic Differentiation?
Automatic differentiation (AD) computes the derivatives (gradients) of functions automatically. In PyTorch, this means when you perform operations on tensors, PyTorch creates a **computational graph** of these operations, enabling the automatic calculation of derivatives during backpropagation.

In machine learning, this is essential for optimizing model parameters during training. Given a loss function, PyTorch can automatically compute the gradients of the model parameters with respect to the loss, allowing the optimizer to adjust the parameters to minimize the loss.

## 2.2 How Autograd Works
PyTorch’s autograd works through **reverse-mode automatic differentiation**. Every time you apply an operation to a tensor that has `requires_grad=True`, PyTorch records this operation in a dynamic computation graph. When you call `.backward()`, PyTorch traverses this graph in reverse to compute the gradients of each tensor involved.

- **Dynamic Computational Graph (DCG)**: The graph is dynamic, meaning it is rebuilt from scratch on each forward pass. This allows for flexibility, as the structure of the graph is based on the operations performed during the forward pass, even if they change at runtime.

## 2.3 Tensor and requires_grad
When creating a tensor, you can specify whether PyTorch should track operations on that tensor by setting `requires_grad=True`. This tells PyTorch to build the computational graph for all operations involving that tensor.

```python
import torch

# Create a tensor with requires_grad=True
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
```
In this example, PyTorch will track every operation involving `x` and store the operations in the computational graph. When you call `backward()`, it will compute the gradients of `x` based on the operations performed on it.

## 2.4 Computational Graph and grad_fn
Every tensor that results from operations on tensors with `requires_grad=True` has a `.grad_fn` attribute, which points to the function that created the tensor. This is how PyTorch keeps track of the operations in the computational graph.

Example:
```python
x = torch.tensor([2.0, 3.0], requires_grad=True)
y = x + 2  # Perform an operation

# y has a grad_fn since it was created from x
print(y.grad_fn)  # Outputs: <AddBackward0 object at ...>
```
Here:
- The operation `y = x + 2` created a new tensor `y`, and `y.grad_fn` records the addition operation (`AddBackward0`). This is the node in the computational graph where PyTorch will later compute gradients for backpropagation.

If you create a tensor without setting `requires_grad=True`, it won’t have a `.grad_fn`, meaning PyTorch won’t track any operations on it.

## 2.5 Gradient Calculation with .backward()
The method `backward()` computes the gradients of the tensor it’s called on with respect to all the tensors that have `requires_grad=True` in the computational graph. Usually, you call `backward()` on the **loss tensor** during training to compute the gradients of the loss with respect to the model parameters.

Example:
```python
x = torch.tensor([2.0, 3.0], requires_grad=True)
y = x + 2
z = y * y * 3  # Some complex operation
z = z.mean()   # Average the result

# Compute the gradients of z w.r.t. x
z.backward()

# x.grad stores the gradient of z with respect to x
print(x.grad)  # Outputs: tensor([13., 18.])
```

Here’s what’s happening:

1.	The computational graph is built as we perform operations: first `y = x + 2`, then `z = y * y * 3`, and finally `z = z.mean()`.
2.	When `z.backward()` is called, PyTorch computes the gradient of z with respect to x by traversing the graph in reverse and applying the chain rule.
3.	The gradients are stored in `x.grad`.

## 2.6 Zeroing Gradients
One important thing to note is that PyTorch accumulates gradients in the `.grad` attribute. This means that if you don’t zero the gradients before the next backward pass, the new gradients will be added to the old ones.

Typically, you zero the gradients at the start of each iteration in the training loop:
```python
optimizer.zero_grad()  # Zero the gradients
loss.backward()  # Backpropagate the loss
```

## 2.7 Stopping Gradient Tracking
Sometimes, you may have tensors with `requires_grad=True` but want to stop tracking the gradients for certain operations (for example, during inference). You can achieve this using `torch.no_grad()` or by setting `requires_grad=False`.

- **Using `torch.no_grad()`**:
  ```python
  with torch.no_grad():
      result = model(input)  # No gradient computation
  ```
  This is useful in inference, where you don't need gradient tracking and want to save memory and computation.

- **Setting `requires_grad=False`**:
  ```python
  x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
  x.requires_grad_(False)  # Now, no gradient tracking for x
  ```

## 2.8 Detaching Tensors
If you want to stop tracking a tensor but continue using its value in further computations, you can detach it from the computational graph using `.detach()`. This creates a new tensor that shares the same data but doesn’t track gradients.

Example:
```python
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = x.detach()  # y has the same values but does not track gradients
```

## 2.9 Backpropagation through the Computational Graph
When training neural networks, the **loss** is computed by comparing the model’s output with the target values, and we need to compute the gradients of the loss with respect to the model parameters. This is done using the backpropagation algorithm, which computes gradients by traversing the computational graph backward and applying the chain rule.

Here’s how it works in a typical training loop:
1. **Forward pass**: Compute the output of the model given the input data.
2. **Compute loss**: Compare the model's output with the ground truth to compute the loss.
3. **Backward pass**: Call `loss.backward()` to compute the gradients of the loss with respect to the model parameters.
4. **Update parameters**: Use an optimizer (like SGD or Adam) to adjust the model parameters using the computed gradients.

Example of a simple training loop:
```python
model = SimpleNN()  # Some neural network
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
loss_fn = torch.nn.MSELoss()

for data, target in data_loader:
    optimizer.zero_grad()  # Zero the gradients
    output = model(data)   # Forward pass
    loss = loss_fn(output, target)  # Compute loss
    loss.backward()  # Backpropagate the loss
    optimizer.step()  # Update model parameters
```

## 2.10 Jacobian-Vector Product
In reverse-mode automatic differentiation, instead of calculating full Jacobians (matrices of partial derivatives), PyTorch computes **Jacobian-vector products**. This is much more efficient for large models, especially for neural networks, where the number of parameters can be huge.

## 2.11 Gradient Accumulation
By default, PyTorch accumulates gradients when calling `backward()`. This means that if you compute the gradients multiple times without resetting, they will be added up in the `.grad` attribute.

- **Example**:
  ```python
  x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
  y = x + 2
  z = y * y * 3
  z = z.mean()
  
  z.backward()  # Compute gradients
  z.backward()  # Compute again (gradients will accumulate)
  print(x.grad)  # Outputs: tensor([26., 36.])
  ```

## 2.12 Vector-Jacobian Product with .backward() and .grad_fn
`backward()` can also accept an argument, which allows for calculating the **Jacobian-vector product** (instead of just scalar backpropagation) for more advanced cases.

## Conclusion for Section 2
Autograd is one of the most important features of PyTorch, enabling automatic differentiation for tensors, allowing easy and efficient computation of gradients for optimizing models. It dynamically builds the computational graph and computes gradients during backpropagation, which are then used by optimizers to adjust the model parameters. Understanding autograd is crucial for anyone working with deep learning in PyTorch.
