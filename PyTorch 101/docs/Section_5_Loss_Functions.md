
# Section 5: Loss Functions in PyTorch

Loss functions, also known as **objective functions** or **cost functions**, are at the heart of machine learning models. They measure how well the model's predictions match the actual target values. PyTorch provides several built-in loss functions through the `torch.nn` module, each suitable for different types of tasks (regression, classification, etc.).

In this section, we’ll explore the key types of loss functions, their applications, and how they work in PyTorch.

## 5.1 What is a Loss Function?
A loss function quantifies the difference between the model’s prediction (`output`) and the true target (`label`). During training, the goal is to minimize this loss, which helps improve the accuracy or performance of the model.

Mathematically:
- Given a set of model parameters $\theta$, inputs $x$, and true target $y$, the loss function $L(\theta)$ computes how far the predicted output $\hat{y} = f(x; \theta)$ is from the true target $y$.
- During training, the optimizer adjusts the model parameters $\theta$ in the direction that minimizes the loss.

## 5.2 Common Loss Functions in PyTorch

Let’s dive deeper into the most commonly used loss functions provided by `torch.nn`.

### 1. **Mean Squared Error Loss (MSE Loss)**
- **Type**: Regression
- **Use Case**: For regression tasks where the goal is to predict continuous values.
- **Formula**:

  $\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$

  The Mean Squared Error (MSE) loss calculates the average of the squared differences between the predicted values  $\hat{y}_i$  and the actual values  $y_i$.
- **Usage**:
  ```python
  loss_fn = torch.nn.MSELoss()
  loss = loss_fn(output, target)
  ```

- **Example**:
  - Suppose you’re predicting house prices. The MSE loss measures the squared difference between the predicted price and the actual price.

### 2. **Cross-Entropy Loss**
- **Type**: Classification (Multi-class)
- **Use Case**: Used in classification tasks where the model outputs a probability distribution over several classes.
- **Formula**:

  $\text{CrossEntropyLoss} = - \sum_{i=1}^{C} y_i \log(\hat{y}_i)$

  Where  $y_i$  is the true label (as a one-hot vector) and  $\hat{y}_i$  is the predicted probability for class $i$.

- **Usage**:
  ```python
  loss_fn = torch.nn.CrossEntropyLoss()
  loss = loss_fn(output, target)
  ```

- **Key Points**:
  - The output in PyTorch’s CrossEntropyLoss should be raw logits (not passed through softmax), as the loss function internally applies log_softmax.
  - Target should be the class indices (not one-hot encoded).
- **Example**:
  - You are working on digit classification using the MNIST dataset, where you have 10 classes (digits 0-9). Cross-Entropy Loss is used to penalize wrong predictions, depending on how far off the predicted class probabilities are from the actual class.

### 3. **Binary Cross-Entropy Loss (BCE Loss)**
- **Type**: Binary Classification
- **Formula**:

  $\text{BCE} = - \frac{1}{n} \sum_{i=1}^{n} [y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i)]$

  Where  $\hat{y}_i$  is the predicted probability, and  $y_i$ is the true label (0 or 1).

- **Usage**:
  ```python
  loss_fn = torch.nn.BCELoss()
  loss = loss_fn(output, target)
  ```

- **Example**:
  - If you’re working on a spam detection model, you use BCE Loss to penalize incorrect spam/non-spam classifications.

### 4. **Binary Cross-Entropy with Logits Loss (BCEWithLogitsLoss)**
- **Type**: Binary Classification
-	**Use Case**: Similar to BCELoss, but this version is numerically more stable as it combines the sigmoid activation function and the binary cross-entropy loss into a single function.

- **Usage**:
  ```python
  loss_fn = torch.nn.BCEWithLogitsLoss()
  loss = loss_fn(output, target)
  ```
- **Key Points**:
  - The output should be raw logits (not passed through sigmoid), as BCEWithLogitsLoss internally applies the sigmoid function.
- **Example**:
  - This is useful for tasks like binary sentiment classification (positive/negative) in text analysis.

### 5. Negative Log-Likelihood Loss (NLL Loss)
- **Type**: Classification (Multi-class)
- **Use Case**: Used for classification tasks, typically when the model’s output is the result of log_softmax.

- **Formula**:
$\text{NLLLoss} = - \frac{1}{n} \sum_{i=1}^{n} \log(\hat{y}{i, c})$

Where  $\hat{y}{i, c}$  is the predicted probability for the correct class $c$.
- **Usage**:
```python
loss_fn = torch.nn.NLLLoss()
loss = loss_fn(output, target)  # output should be log-softmax probabilities
```
- **Key Points**
  - The output should already be passed through log_softmax, unlike CrossEntropyLoss, which applies it internally.

### 6. Huber Loss (Smooth L1 Loss)
- **Type**: Regression
- **Use Case**: A combination of MSE and MAE (Mean Absolute Error) loss, Huber loss is less sensitive to outliers than MSE. It uses MSE for small errors and switches to MAE for large errors.

- **Formula**:

$\text{Huber Loss} =
\begin{cases}
0.5 (y - \hat{y})^2 & \text{if} \, |y - \hat{y}| \leq \delta \\
\delta |y - \hat{y}| - 0.5 \delta^2 & \text{otherwise}
\end{cases}$

- **Usage**:
```python
loss_fn = torch.nn.SmoothL1Loss()
loss = loss_fn(output, target)
```

### 7. Kullback-Leibler Divergence Loss (KLDivLoss)
- **Type**: Distribution comparison
- **Use Case**: Measures how one probability distribution diverges from a reference distribution. Commonly used in training variational autoencoders or when dealing with probabilistic outputs.

- **Formula**:

$\text{KLDivLoss} = \sum_{i=1}^{C} p_i \log\left(\frac{p_i}{q_i}\right)$

Where $p_i$ is the true probability and $q_i$ is the predicted probability.

- **Usage**:
```python
loss_fn = torch.nn.KLDivLoss()
loss = loss_fn(output, target)  # output should be log probabilities
```

- **Example**:
  - Used in tasks involving comparing distributions, such as training autoencoders or models that output a distribution of probabilities.

## 5.3 Using Loss Functions in PyTorch

To use a loss function in PyTorch, follow these steps:

1.	Instantiate the loss function (e.g., `nn.MSELoss()` or `nn.CrossEntropyLoss()`).
2.	Pass the model’s output and the target values to the loss function to compute the loss.
3.	Backpropagate the loss to compute gradients using `.backward()`.

Example of a training loop using MSE loss:
```python
# Define model, optimizer, and loss function
model = SimpleNN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = torch.nn.MSELoss()

for data, target in data_loader:
    optimizer.zero_grad()  # Zero the gradients
    output = model(data)   # Forward pass
    loss = loss_fn(output, target)  # Compute loss
    loss.backward()  # Backward pass: compute gradients
    optimizer.step()  # Update parameters
```

## 5.4 Choosing the Right Loss Function

Choosing the correct loss function depends on the task:

- **Regression**: Use MSELoss, SmoothL1Loss, or HuberLoss.
- **Binary Classification**: Use BCELoss or BCEWithLogitsLoss.
- **Multi-class Classification**: Use CrossEntropyLoss or NLLLoss.
- **Distribution Comparison**: Use KLDivLoss.

## 5.5 Custom Loss Functions

If the built-in loss functions don’t meet your needs, you can define your own custom loss function by subclassing nn.Module and implementing the forward() method.

Example
```python
class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, output, target):
        loss = torch.mean((output - target) ** 2) + torch.sum(output)
        return loss

loss_fn = CustomLoss()
```
In this example, the custom loss function combines MSE loss with an additional term (sum of the outputs). You can design the loss to fit your specific problem.

## 5.6 Reduction in Loss Functions

Most PyTorch loss functions support the reduction parameter, which determines how the losses across the batch are combined. There are three options:

- mean (default): The mean of all losses.
- sum: The sum of all losses.
- none: No reduction; returns the individual losses for each sample.

Example:
```python
loss_fn = torch.nn.MSELoss(reduction='sum')  # Sum the losses instead of taking the mean
```

## 5.7 Regularization via Loss Functions

Loss functions can also incorporate regularization techniques like L2 regularization, which penalizes large weights and helps reduce overfitting.

Example:
```python
l2_lambda = 0.01
l2_reg = torch.tensor(0.0)
for param in model.parameters():
    l2_reg += torch.norm(param)
loss = loss_fn(output, target) + l2_lambda * l2_reg  # Add L2 regularization to the loss
```

## 5.8 Regularization and Weight Decay (L2 Regularization)

**Regularization** is a technique used to prevent overfitting by adding a penalty to the loss function. **L2 regularization**, also known as **weight decay**, penalizes large weights by adding the squared sum of all weights to the loss function.

### How L2 Regularization Works
The regularized loss function becomes:

$L(\theta) = L_0(\theta) + \lambda \sum_{j} \theta_j^2$

Where:
- $L_0(\theta)$ is the original loss function (e.g., cross-entropy or MSE).
- $\lambda$ (weight decay) is a hyperparameter that controls the amount of regularization.

L2 regularization discourages the model from fitting too closely to the training data, which can help improve generalization.

### Implementing Weight Decay in PyTorch
In PyTorch, weight decay can be added directly when creating an optimizer:
```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=0.001)
```
- The `weight_decay` parameter implements L2 regularization by adding $\lambda \sum_{j} \theta_j^2$ to the loss function.

**Note**: While weight decay is equivalent to L2 regularization, it's more efficient to use `weight_decay` directly in the optimizer rather than adding the term manually to the loss function.

## Conclusion for Section 5
Loss functions play a critical role in training neural networks by quantifying the difference between predictions and actual targets. PyTorch provides a variety of built-in loss functions for different types of tasks. Regularization, particularly L2 regularization (weight decay), helps prevent overfitting by penalizing large weights, promoting simpler models that generalize better. Understanding how to use and customize loss functions and regularization techniques is essential for effective deep learning training.
