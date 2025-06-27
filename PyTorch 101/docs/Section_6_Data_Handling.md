
# Section 6: Data Handling (Dataset and DataLoader) in PyTorch

In PyTorch, handling and loading data efficiently is crucial for training machine learning models, especially when dealing with large datasets. The `torch.utils.data` module provides two key components for data management: **Dataset** and **DataLoader**. These classes allow you to work with datasets easily by organizing them and feeding them into the model in an efficient and scalable manner.

Let’s dive deeper into these components and their usage.

## 6.1 What is a Dataset?
A **Dataset** in PyTorch is a class that defines how to access and preprocess the data. It acts as an interface to load the data in a structured way. PyTorch provides a base class, `torch.utils.data.Dataset`, which you can subclass to create your own dataset.

Each Dataset must define two key methods:
- **`__len__()`**: Returns the number of samples in the dataset.
- **`__getitem__(index)`**: Returns the sample at the given index, along with its corresponding label.

The Dataset class can be used for both in-memory datasets (e.g., NumPy arrays, Pandas dataframes) and datasets stored on disk (e.g., images, text files).

### Example: Custom Dataset
```python
import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Return the data and the corresponding label at index idx
        return self.data[idx], self.labels[idx]

# Example usage
data = torch.randn(100, 10)  # 100 samples, each with 10 features
labels = torch.randint(0, 2, (100,))  # 100 labels (binary classification)
dataset = CustomDataset(data, labels)
```

- `__len__()` returns the total number of samples in the dataset.
- `__getitem__()` retrieves a specific sample and its corresponding label based on the provided index.

## 6.2 Built-in Datasets
PyTorch provides several built-in datasets via the `torchvision` and `torchtext` libraries for common machine learning tasks such as image classification and natural language processing (NLP). These datasets come with predefined loaders, making it easier to get started with standard datasets.

Some common built-in datasets:

- **MNIST**: Handwritten digit dataset.
  ```python
  from torchvision import datasets
  from torchvision.transforms import ToTensor

  dataset = datasets.MNIST(root='data', train=True, download=True, transform=ToTensor())
  ```
- **CIFAR-10**: A dataset of 60,000 32x32 color images in 10 classes.
  ```python
  dataset = datasets.CIFAR10(root='data', train=True, download=True, transform=ToTensor())
  ```

- **ImageNet**: A large-scale dataset for image classification tasks (available through `torchvision`).


## 6.3 DataLoader
The **DataLoader** is an iterator that abstracts the process of loading and batching the data. It helps to efficiently load large datasets, shuffle data, and create mini-batches during training. The DataLoader handles parallelism using multiple worker processes, making data loading much faster.

### Key Parameters in `DataLoader`:
- **`dataset`**: The dataset object (custom or built-in).
- **`batch_size`**: Number of samples in each batch.
- **`shuffle`**: Whether to shuffle the data after each epoch.
- **`num_workers`**: Number of subprocesses to use for data loading.
- **`drop_last`**: Whether to drop the last incomplete batch.

### Example: Using DataLoader
```python
from torch.utils.data import DataLoader

# Create DataLoader for the dataset
data_loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

# Iterate through the DataLoader
for batch_data, batch_labels in data_loader:
    print(batch_data.size(), batch_labels.size())
```

Here, the DataLoader splits the dataset into batches of size 32 and shuffles the data after each epoch. The `num_workers=4` means it will use 4 subprocesses to load data in parallel.

## 6.4 Transforms
Transforms are functions that apply data preprocessing and augmentation techniques to each sample of a dataset. This is especially useful for image and text datasets, where you need to resize, normalize, or augment data before feeding it into the model.

In `torchvision`, you can use `transforms.Compose()` to chain multiple transforms.

### Example: Image Transformations
```python
from torchvision import transforms

# Define a series of transforms
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize to 128x128
    transforms.RandomHorizontalFlip(),  # Random horizontal flip
    transforms.ToTensor(),  # Convert image to tensor
    transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize tensor
])

# Apply transform to dataset
dataset = datasets.MNIST(root='data', train=True, download=True, transform=transform)
```

Here:

- `Resize` resizes the image to 128x128.
- `RandomHorizontalFlip` flips the image with a 50% probability, which serves as data augmentation.
- `ToTensor` converts the image into a PyTorch tensor.
- `Normalize` standardizes the image by scaling pixel values to have a mean of 0.5 and a standard deviation of 0.5.

## 6.5 Combining Dataset and DataLoader
Typically, in PyTorch, you define your custom dataset, apply the required transformations, and then feed the dataset into the DataLoader, which will handle batching, shuffling, and parallel loading. The combination of `Dataset` and `DataLoader` allows for scalable and efficient data handling.

### Example: Complete Data Pipeline
```python
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

# Define transformations
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Load dataset with transformations
dataset = datasets.CIFAR10(root='data', train=True, download=True, transform=transform)

# Create DataLoader
data_loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=2)

# Training loop example
for images, labels in data_loader:
    print(images.shape, labels.shape)
```

In this example, CIFAR-10 images are resized, normalized, and converted to tensors using transforms. The DataLoader then creates mini-batches of size 64, shuffles them, and loads them in parallel with 2 workers.

## 6.6 Sampling from a Dataset
PyTorch allows you to sample from datasets in custom ways using `torch.utils.data.Sampler`. By default, `DataLoader` shuffles data (if specified), but you can create custom sampling strategies.

- `SubsetRandomSampler`: Samples a random subset of the dataset.
```python
from torch.utils.data import SubsetRandomSampler

indices = list(range(len(dataset)))
sampler = SubsetRandomSampler(indices[:100])  # Sample first 100 indices randomly
data_loader = DataLoader(dataset, sampler=sampler, batch_size=32)
```

- `WeightedRandomSampler`: Samples elements based on assigned probabilities (useful for imbalanced datasets).
```python
from torch.utils.data import WeightedRandomSampler

class_sample_counts = [100, 400]  # Number of samples in each class
class_weights = 1. / torch.tensor(class_sample_counts, dtype=torch.float)
sample_weights = class_weights[labels]  # Assign weights based on class labels
sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
data_loader = DataLoader(dataset, sampler=sampler, batch_size=32)
```

## 6.7 Handling Large Datasets
When working with datasets too large to fit into memory, PyTorch allows you to load data lazily, i.e., load samples one at a time when they are needed.

### Example: Loading Data from Disk

For large image datasets, you can load images from disk on-the-fly in `__getitem__()`:

```python
from PIL import Image
import os

class LargeImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.image_filenames = os.listdir(image_dir)
        self.transform = transform

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_filenames[idx])
        image = Image.open(img_path)  # Load image from disk
        if self.transform:
            image = self.transform(image)
        return image

dataset = LargeImageDataset(image_dir='path_to_large_dataset', transform=transform)
data_loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
```

In this case, each image is loaded from disk in `__getitem__()` when needed, so you don’t need to load the entire dataset into memory.

## 6.8 DataLoader in Model Training
In practice, the `DataLoader` is a crucial component of the training loop in PyTorch. It handles the data loading in batches, allowing the model to process the data efficiently.

 Example of a training loop:
```python
for epoch in range(epochs):
    for data, target in data_loader:
        optimizer.zero_grad()  # Zero gradients
        output = model(data)   # Forward pass
        loss = loss_fn(output, target)  # Compute loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update model parameters
```

The `data_loader` feeds batches of data and corresponding targets to the model, ensuring efficient and scalable training.

## Conclusion for Section 6
The `Dataset` and `DataLoader` in PyTorch are fundamental tools for handling data efficiently. While the `Dataset` provides a structured way to access data and apply transformations, the `DataLoader` batches and loads the data in parallel, making it ready for training. Understanding how to customize `Dataset` and `DataLoader` and how to apply transformations and augmentations is essential for effective deep learning model training in PyTorch.
