import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from classes.Izhikevich_SNN import IzhikevichSNN


# --------------------------
# 1. Poisson Encoding
# --------------------------
def poisson_encode(images, time_steps):
    """
    Convert images to Poisson spike trains over time.
    images: (batch, features)
    Returns: (time, batch, features)
    """
    images = images.unsqueeze(0).repeat(time_steps, 1, 1)  # (time, batch, features)
    return torch.bernoulli(images)


# --------------------------
# 4. Load MNIST
# --------------------------
batch_size = 1
transform = transforms.Compose([
    transforms.ToTensor(),  # [0,1]
    transforms.Lambda(lambda x: x.view(-1))  # flatten to (784,)
])

train_dataset = datasets.MNIST(root='../data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# --------------------------
# 5. Training Loop
# --------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = IzhikevichSNN(batch_size=batch_size).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

for epoch in range(1):
    correct = 0
    total = 0
    running_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # ---- Accuracy calculation ----
        preds = torch.argmax(output, dim=1)  # Shape: (batch,)
        correct += (preds == target).sum().item()
        total += target.size(0)

        if (batch_idx * batch_size) % 100 == 0 and batch_idx > 0:
            print(f"Epoch {epoch}, Batch {batch_idx * batch_size}, "
                  f"Loss: {loss.item():.4f}, Accuracy: {100 * correct / total:.2f}%")

    epoch_accuracy = 100 * correct / total
    print(f"Epoch {epoch} completed. Avg Loss: {running_loss / len(train_loader):.4f}, "
          f"Accuracy: {epoch_accuracy:.2f}%\n")
