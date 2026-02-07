# Deep Learning with PyTorch: Transfer Learning on Dog Breeds

This document serves as a comprehensive guide to understanding and recreating the Convolutional Neural Network (CNN) implementation found in `test.ipynb`. It focuses on pragmatic Deep Learning concepts using PyTorch, specifically **Transfer Learning** with a ResNet18 architecture.

---

## 1. Core Concepts: How Machines Learn

Before diving into code, it is crucial to understand the "Training Loop"—the repetitive cycle that allows a neural network to improve.

### The Training Cycle (The "Epoch")
A Neural Network is essentially a massive mathematical function with millions of adjustable parameters (weights). Training is the process of tuning these weights to minimize errors. This happens in four distinct steps for every batch of images:

#### A. The Forward Pass (Prediction)
*   **What it is**: The data travels "forward" through the network layers.
*   **The Action**: The image tensor enters the first layer. It undergoes matrix multiplications, convolutions (filtering), and activation functions (which add non-linearity).
*   **The Result**: The model outputs a vector of raw scores called **Logits**. For example, if we have 3 breeds, it might output `[2.1, -0.5, 4.3]`. Ideally, the highest number corresponds to the correct dog breed.
*   **Code**: `pred = model(X)`

#### B. Loss Calculation (The Scorecard)
*   **What it is**: Measuring how strictly "wrong" the model was.
*   **The Action**: We compare the model's prediction (`pred`) against the actual answer (`y`).
*   **The Mechanism**: We use **CrossEntropyLoss** (Standard for classification).
    *   It applies a specialized function (Softmax) to turn logits into probabilities (e.g., `[0.85, 0.05, 0.10]`).
    *   It then calculates the Negative Log Likelihood.
    *   If the model predicted the correct class with high confidence, Loss is near 0.
    *   If it was confident but wrong, Loss is very high.
*   **Code**: `loss = loss_fn(pred, y)`

#### C. Backpropagation (The "Blame Game")
*   **What it is**: The most critical mathematical step. It uses **Calculus (The Chain Rule)** to calculate gradients.
*   **The Action**: We travel *backwards* from the Loss through the network to the input.
*   **The Goal**: We determine exactly how much *every single weight* in the network contributed to the error.
    *   "If I increase Weight #402 slightly, does the Loss go up or down?"
    *   The **Gradient** is this slope/sensitivty.
*   **Code**: `loss.backward()`

#### D. Optimization (The Update)
*   **What it is**: Actually changing the weights to improve the model.
*   **The Action**: The optimizer inspects the gradients calculated during backpropagation and nudges every weight in the opposite direction of the gradient (to go "downhill" in terms of error).
*   **Learning Rate**: The size of the nudge. Too small = training takes forever. Too big = model overshoots the target.
*   **Code**: `optimizer.step()`

---

## 2. Advanced Mechanics: "Under the Hood"

The user often encounters terms like Dropout, BatchNorm, and Gradient Descent. Here is what is actually happening physically and mathematically.

### A. The Gradient & The Slope
Imagine you are standing on a misty mountain at night. Your goal is to reach the absolute bottom (Zero Loss/Perfect Accuracy).
*   **The Loss Landscape**: The "mountain". Every combination of network weights (coordinates) corresponds to a height (Error).
*   **The Gradient**: You tap the ground with your foot 360 degrees around you. You find the direction that goes *uphill* the steepest. That vector is the **Gradient**.
*   **Gradient Descent**: To reduce error, you take a step in the exact *opposite* direction of the gradient.
*   **Why "Down the Slope"?**: By consistently taking steps downhill, mathematically you will eventually reach a valley (a local minimum) where the model performs well.

### B. Why Reset Gradients? (`optimizer.zero_grad()`)
In PyTorch, when you call `loss.backward()`, it calculates the gradients and **adds** them to whatever is already stored in the `.grad` attribute of the weights.
*   **The Analogy**: Imagine a bucket used to collect rain (gradients) to measure how much it rained in the last hour.
*   **The Problem**: If you don't empty the bucket (`zero_grad()`) before the next hour, you are measuring old rain + new rain.
*   **The Result**: Your step direction would be a mix of the current batch's needs and the previous batch's needs, causing the model to move in weird, wrong directions. We must "empty the bucket" before every new calculation.

### C. Dropout (`nn.Dropout`)
A regularization technique to prevent **Overfitting** (memorizing the training data).
*   **How it works**: During training, we randomly "kill" (zero out) a percentage (e.g., 50%) of neurons in a layer for that specific step.
*   **The Effect**: No single neuron can rely on a specific feature (like "pointy ear") because that feature input might be turned off next time. The network is forced to learn redundant, robust features distributed across many neurons.
*   **Analogy**: A sports team where the coach randomly benches half the star players every game. The remaining players must learn to play every position to win.

### D. Batch Normalization (`nn.BatchNorm`)
This is often considered one of the most important innovations in modern Deep Learning. It is not just "making data normal"; it is a dynamic stabilization mechanism.

#### 1. The Problem: "Internal Covariate Shift"
Imagine you are trying to learn to play tennis.
*   **Day 1**: The ball used is standard yellow and bouncy. You adjust your muscle memory (weights).
*   **Day 2**: The ball is suddenly a heavy red medicine ball. You have to completely re-learn your swing.
*   **Day 3**: The ball is a light ping-pong ball. You are confused again.

In a deep network, Layer 10 is waiting for input from Layer 9. But as Layer 9 updates its weights during backpropagation, the *range* of values it outputs changes wildly (e.g., from range [0, 1] to [-50, 50]). Layer 10 has to waste time constantly re-adapting to this "moving target." This is **Internal Covariate Shift**.

#### 2. The Solution: Enforced Stability
BatchNorm forces the input to every layer to be mathematically consistent, regardless of what the previous layer did.
*   **Step 1 (Normalize)**: For the current batch (e.g., 32 images), calculate the Mean ($\mu$) and Variance ($\sigma^2$). Subtract the mean and divide by the standard deviation. Now, the data is strictly centered at 0 with a spread of 1.
*   **Step 2 (Recover functionality)**: What if the layer *needed* the data to be centered at 5? Forcing it to 0 might destroy information. So, BatchNorm introduces two **learnable parameters** per channel:
    *   **Scale ($\gamma$, Gamma)**: The network learns how much to stretch the data.
    *   **Shift ($\beta$, Beta)**: The network learns where to center the data.
    *   *Result*: The network controls its own distribution, rather than being at the mercy of previous layers.

#### 3. Why `model.eval()` matters
*   **Training**: We calculate mean/variance based on the *current batch* of 32 dogs. This noise actually helps regularization.
*   **Testing/Inference**: We might be predicting just *one* dog. We can't calculate a "batch mean" from one item.
*   **The Fix**: During training, PyTorch quietly keeps a "Running Average" of the global mean and variance across the entire dataset. When you call `model.eval()`, BatchNorm stops looking at the batch and uses those saved global statistics instead.

#### 4. The Benefits
*   **Speed**: We can use much higher learning rates (10x-100x) without the training exploding.
*   **Independence**: Weights are less sensitive to how they were initialized.
*   **Smoothing**: It makes the "Loss Landscape" much flatter, so the optimizer can glide down to the solution rather than navigating a rocky canyon.

---

## 3. Step-by-Step Guide to Creating `test.ipynb`

This section breaks down `test.ipynb` logically, explaining how to reconstruct it from scratch.

### Step 1: Imports and Setup
We need standard libraries for file handling and PyTorch for deep learning.

```python
from pathlib import Path
import json, random
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as T
from PIL import Image

# Reproducibility: Ensure we get the same random numbers every time we run
seed = 42
random.seed(seed)
torch.manual_seed(seed)

# Device Configuration: Use GPU (cuda/mps) if available, else CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device}")
```

### Step 2: The Dataset Class
**Concept**: PyTorch separates "Data Handling" from "Model Training". We create a class that knows how to read our specific files.

1.  **Inputs**: Paths to data, and a method to transform images.
2.  **Logic**:
    *   Read the `.jsonl` file.
    *   Build a `class_to_idx` dictionary. The network outputs numbers (0, 1, 2), effectively indices in an array. We must convert string labels ("Beagle") into these integers.

```python
class DogsDataset(Dataset):
    def __init__(self, jsonl_path, transform=None, class_to_idx=None):
        self.path = Path(jsonl_path)
        self.transform = transform
        # Load JSONL
        with self.path.open() as f:
            self.items = [json.loads(line) for line in f if line.strip()]

        # Create or assign Class Mapping
        if class_to_idx is None:
            # Sort to ensure deterministic order (Beagle=0, Poodle=1...)
            classes = sorted({x["class_name"] for x in self.items})
            self.class_to_idx = {c: i for i, c in enumerate(classes)}
        else:
            self.class_to_idx = class_to_idx
            
    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        # Important: Convert to RGB to handle grayscale/RGBA images correctly
        img = Image.open(item["image"]).convert("RGB")
        label = self.class_to_idx[item["class_name"]]
        
        if self.transform:
            img = self.transform(img)
            
        return img, label
```

### Step 3: Data Transformations (Augmentation)
**Concept**: Computers are literal. If we only train on perfect photos, the model fails on rotated or zoomed photos. **Data Augmentation** artificially creates "new" difficult training data during specific training epochs.

*   `RandomResizedCrop`: Forces model to look at details (ears, paws) not just the background.
*   `RandomHorizontalFlip`: Teaches invariance (a dog looking left is same as looking right).
*   `Normalization`: Math trick. Neural networks learn faster when input numbers are small (near 0) and standard (std dev 1). We use ImageNet statistics.

```python
image_size = 160 # Reduced from 224 for speed, standard ResNet is 224
stats = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

# Training gets augmentation (Harder for model = Better learning)
train_tf = T.Compose([
    T.RandomResizedCrop(image_size),
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    T.Normalize(*stats)
])

# Validation/Test gets standard resizing (We want a fair test)
eval_tf = T.Compose([
    T.Resize(int(image_size * 1.14)),
    T.CenterCrop(image_size),
    T.ToTensor(),
    T.Normalize(*stats)
])
```

### Step 4: Loading Data
Instantiate the datasets and wrap them in `DataLoader`. The DataLoader handles:
*   **Batching**: Stacking 16 images into one tensor (Batch Size).
*   **Shuffling**: Mixing data so the model doesn't memorize the order (Training only).

```python
train_ds = DogsDataset("data/splits/dogs_train.jsonl", transform=train_tf)
# Use train_ds.class_to_idx for others to ensure Schema Consistency (Label 0 is always the same breed)
val_ds = DogsDataset("data/splits/dogs_val.jsonl", transform=eval_tf, class_to_idx=train_ds.class_to_idx)
test_ds = DogsDataset("data/splits/dogs_test.jsonl", transform=eval_tf, class_to_idx=train_ds.class_to_idx)

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)
```

### Step 5: Transfer Learning (The Model)
**Concept**: Instead of training a brain from birth (Random Weights), we take a brain that has already gone to college (Pre-trained ResNet18).
*   **ResNet18**: A standard, high-performance CNN architecture.
*   **Pre-trained**: Trained on ImageNet (1.2M images, 1000 categories). It already knows edges, textures, eyes, fur.
*   **The Surgery**: We slice off the last layer (which outputs 1000 classes) and replace it with a new layer that outputs *our* number of dog breeds.

```python
weights = torchvision.models.ResNet18_Weights.DEFAULT
model = torchvision.models.resnet18(weights=weights)

# Replace the final Fully Connected (fc) layer
num_classes = len(train_ds.class_to_idx)
model.fc = nn.Linear(model.fc.in_features, num_classes)

model = model.to(device) # Move entire model to GPU/CPU
```

### Step 6: Training Configuration
*   **Loss Function**: `CrossEntropyLoss` (Standard for multi-class classification).
*   **Optimizer**: `AdamW`. A smarter version of Stochastic Gradient Descent. It adapts learning rates per-parameter.

```python
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4) # 3e-4 is a "Karpathy Constant" (very safe learning rate)
```

### Step 7: The Training Loop
We iterate through the dataset multiple times (`epochs`).

```python
def train_loop(dataloader, model, loss_fn, optimizer):
    model.train() # Enable Dropout/BatchNorm
    
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        
        # The 4 Steps Explained in Section 1
        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad() # RESET GRADIENTS

def test_loop(dataloader, model, loss_fn):
    model.eval() # Disable Dropout/BatchNorm
    test_loss, correct = 0, 0
    
    with torch.no_grad(): # SAFETY: No gradients needed here. Saves memory.
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            
            # Sum up correct predictions
            # pred.argmax(1) gets the index of the highest probability
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            
    print(f"Accuracy: {correct / len(dataloader.dataset):.1%}")
    return correct / len(dataloader.dataset)

# Run it
for t in range(10): # 10 Epochs
    print(f"Epoch {t+1}")
    train_loop(train_loader, model, loss_fn, optimizer)
    test_loop(val_loader, model, loss_fn)
```

### Step 8: Saving
Crucial Step: We not only save the model weights (`state_dict`) but also the `class_to_idx` map. Without the map, you have a model that tells you "It's class #42" but you won't know if #42 is a Pug or a Husky.

```python
torch.save({
    "state_dict": model.state_dict(),
    "class_to_idx": train_ds.class_to_idx
}, "resnet18_dogs_cpu.pth")
```

---

## 4. Guide to Inference (`predict.ipynb`)

Training is the "hard" part. Inference (using the model) is much simpler but has specific "gotchas" regarding architecture matching.

### Step 1: Loading the Checkpoint First
We need to load the saved file *before* building the model because the file tells us how many classes we have.
```python
# map_location='cpu' prevents errors if the model was trained on a GPU
checkpoint = torch.load("resnet18_dogs_cpu.pth", map_location=device)
```

### Step 2: Re-Building the Empty Shell
We must rebuild the exact same architecture we used in training.
1.  **Base Model**: Load standard ResNet18.
2.  **Surgery**: Just like in training, we must manually resize the final layer. If we forget this, `load_state_dict` will crash because of a shape mismatch (1000 vs 120).

```python
# 1. Base
model = torchvision.models.resnet18()

# 2. Get correct size
num_classes = len(checkpoint["class_to_idx"])

# 3. Resize
model.fc = nn.Linear(model.fc.in_features, num_classes)
```

### Step 3: Injecting the Muscle Memory
Now that the shell matches the saved data, we pour the weights in.
```python
model.load_state_dict(checkpoint["state_dict"])
model.eval() # CRITICAL: Freezes Dropout and BatchNorm
```

### Step 4: Preprocessing and Predicting
Before we can run an image through the model, we must re-define the transformation logic (Resizing/Normalization) so the image looks exactly like the training data.

```python
# 1. Re-define the Transform (Same as Validation)
eval_transform = T.Compose([
    T.Resize(int(160 * 1.14)),
    T.CenterCrop(160),
    T.ToTensor(),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])

def predict(image_path):
    # 2. Load and Transform
    img = Image.open(image_path).convert("RGB")
    tensor = eval_transform(img)
    
    # 3. Add Batch Dimension (The "Fake" Batch)
    # Neural networks expect [BatchSize, Channels, Height, Width]
    # This turns [3, 160, 160] -> [1, 3, 160, 160]
    tensor = tensor.unsqueeze(0).to(device)
    
    # 4. Forward Pass
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)
        
    return probs
```
