
# Section 9: Saving and Loading Models in PyTorch

A crucial part of working with machine learning models is saving and loading models at various stages of the training process. PyTorch provides a flexible and efficient way to save and load both model weights and entire model objects using `torch.save()` and `torch.load()`. This capability allows you to:
- Save checkpoints during training.
- Load pre-trained models for fine-tuning or inference.
- Resume training from a saved checkpoint.
- Distribute models to be used on different platforms or machines.

Let’s explore the key concepts around saving and loading models in PyTorch, along with best practices.

## 9.1 Saving and Loading Model Weights (State Dict)

In PyTorch, model parameters (weights and biases) are stored in a dictionary called the **state dict**. The state dict is essentially a Python dictionary that maps each layer’s name to its corresponding tensor of parameters (e.g., weight matrices, bias vectors). Saving and loading just the state dict is the most common approach since it is lightweight and easy to transfer between different environments.

### Saving Model Weights
To save only the model’s learned parameters, you can use `torch.save()` to save the **state dict**.

Example:
```python
import torch

# Assume you have a model (instance of nn.Module)
model = MyModel()

# Save the state dict (weights) of the model to a file
torch.save(model.state_dict(), 'model_weights.pth')
```

In this example:

- `model.state_dict()`: Returns the state dict containing the model’s parameters.
- `torch.save()`: Saves the state dict to a file named `model_weights.pth`.

### Loading Model Weights
When loading a saved model’s weights, you need to first instantiate the model’s architecture and then load the saved weights into it using `load_state_dict()`.

Example:
```python
# Instantiate the model
model = MyModel()

# Load the saved state dict into the model
model.load_state_dict(torch.load('model_weights.pth'))

# Set the model to evaluation mode
model.eval()
```

In this example:

- `torch.load()`: Loads the state dict from the saved file.
- `model.load_state_dict()`: Loads the weights into the model.
- `model.eval()`: Sets the model to evaluation mode, which is important when doing inference (as it disables features like dropout and batch normalization updates).


## 9.2 Saving and Loading the Entire Model

If you want to save both the model’s architecture and the weights, you can save the entire model object. This includes the model class definition and its current state. However, this approach is less common because it makes it harder to load the model in different environments or when you modify the architecture.

### Saving the Entire Model
```python
# Save the entire model (including architecture and weights)
torch.save(model, 'entire_model.pth')
```

### Loading the Entire Model
```python
# Load the entire model
model = torch.load('entire_model.pth')
model.eval()  # Set the model to evaluation mode
```

This approach is useful when you want a quick and complete snapshot of the model. However, this method requires the exact model class definition to be available when loading the model.

## 9.3 Saving and Loading Checkpoints

When training models, especially over long periods, it is useful to save intermediate checkpoints. Checkpoints allow you to resume training from a certain point, avoiding the need to start from scratch if training is interrupted. You can save not only the model’s state dict but also other important information like the optimizer’s state, the current epoch, and the training loss.

### Saving a Checkpoint

You can store a variety of information in a checkpoint, including:

- The model’s state dict (weights).
- The optimizer’s state dict (optimizer configuration).
- The current epoch number.
- Any additional information, such as the current training loss or validation accuracy.

Example:

```python
# Save a checkpoint during training
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
}, 'checkpoint.pth')
```

### Loading from a Checkpoint

When loading from a checkpoint, you need to manually restore the model’s weights, optimizer state, and any other information saved in the checkpoint.

Example:

```python
# Load a checkpoint
checkpoint = torch.load('checkpoint.pth')

# Restore the model and optimizer states
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

# Restore the epoch and loss (optional)
epoch = checkpoint['epoch']
loss = checkpoint['loss']

# Set the model to evaluation or training mode depending on use case
model.eval()  # If using for inference
```

Checkpointing is critical when training large models or running training sessions over long periods, as it allows you to resume training if an interruption occurs (e.g., power failure, system crash).

## 9.4 Saving and Loading the Optimizer State

When resuming training from a saved checkpoint, it’s important to restore not only the model’s state but also the optimizer’s state. The optimizer’s state contains information about the learning rates, momentum, and other parameters that are essential for continuing training from the exact point it was left.

To save and load the optimizer state:
- **Save** the optimizer state along with the model.
- **Load** the optimizer state before resuming training.

Example:
```python
# Saving the optimizer state
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
}, 'checkpoint.pth')

# Loading the optimizer state
checkpoint = torch.load('checkpoint.pth')
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
```

## 9.5 Best Practices for Saving and Loading Models

### Best Practices:
1. **Save State Dict Instead of Full Model**: Saving and loading the state dict is more flexible and efficient than saving the entire model object. The state dict can be loaded into any model architecture, even if some details of the architecture have changed slightly.

2. **Use Checkpoints During Long Training**: For long training sessions, save checkpoints at regular intervals (e.g., after every few epochs). This allows you to resume training from the last checkpoint in case of interruptions.

3. **Set Model to `eval()` During Inference**: Always call `model.eval()` when using the model for inference. This ensures that layers like dropout and batch normalization behave correctly during inference.

4. **Save Optimizer State for Resuming Training**: If you plan to resume training from a checkpoint, save the optimizer’s state along with the model’s state. This allows the optimizer to pick up from where it left off, preserving learning rate schedules and momentum.

5. **Use Version Control for Model Architectures**: When saving checkpoints, make sure that the version of the model architecture is also stored or tracked using version control (such as Git). If you modify the model architecture later, ensure that the saved checkpoints are still compatible with the current architecture.

6. **Use Meaningful Filenames**: When saving models, use descriptive filenames that indicate the model version, epoch, or any other relevant information.
   ```bash
   model_epoch_10.pth
   model_final.pth
   ```

7. **Save Models After Every Epoch (or Interval)**: For long training sessions, it’s useful to save the model after every epoch or after certain intervals. This provides intermediate checkpoints that can be used in case of failures.

## 9.6 Cross-Device Compatibility

When saving models in PyTorch, the saved state dict is **device agnostic**, meaning that you can save a model trained on a GPU and load it back on a CPU (or vice versa). However, you need to ensure that you explicitly specify the device when loading the model.

Example: Loading a model trained on GPU to CPU
```python
# Load model trained on GPU onto CPU
model.load_state_dict(torch.load('model_weights.pth', map_location=torch.device('cpu')))
```

This is useful when you train a model on a GPU but want to perform inference or further processing on a machine with only a CPU.

## 9.7 Distributed Model Saving

When training models on multiple GPUs or in a distributed setting, each process will have a copy of the model. Typically, you save the model only from the main process (rank 0) to avoid saving multiple copies of the same model.

Example (in a distributed setup):
```python
if rank == 0:  # Save model only from the main process (rank 0)
    torch.save(model.state_dict(), 'distributed_model.pth')
```

This ensures that you don’t save redundant copies of the model.

---

## Conclusion for Section 9
- PyTorch provides flexible methods for saving and loading model weights, entire models, and optimizer states.
- The most common approach is to save the state dict, which contains the model’s parameters and can be loaded independently of the architecture.
- Checkpoints are useful for long-running training sessions, allowing you to resume training without starting from scratch.
- Always ensure that models are set to evaluation mode (`eval()`) during inference and that the optimizer state is saved when resuming training.
- Following best practices for saving and loading models ensures that you can effectively manage your models, restore training progress, and deploy models in different environments.
