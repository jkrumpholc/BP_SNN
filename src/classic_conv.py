import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# ---------- 1. Data Loading ----------
transform = transforms.Compose([
    transforms.ToTensor(),  # convert to [0, 1]
    transforms.Normalize((0.1307,), (0.3081,))  # standard MNIST normalization
])

train_dataset = torchvision.datasets.MNIST(root="../data/", train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST(root="../data/", train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=True)


# ---------- 2. Define CNN Model ----------
class MNISTCNN(nn.Module):
    def __init__(self):
        super(MNISTCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(64 * 14 * 14, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


# ---------- 3. Training Setup ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = MNISTCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ---------- 4. Training Loop ----------
num_epochs = 1
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_id, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
        if batch_id*train_loader.batch_size % 10000 < 120:
            accuracy = 100 * correct / total
            print(f"Epoch [{epoch + 1}/{num_epochs}] "
                  f"Batch [{batch_id*train_loader.batch_size}/{len(train_loader)*128}] - "
                  f"Loss: {total_loss:.4f}, Accuracy: {accuracy:.2f}%")

    accuracy = 100 * correct / total
    print(f"Epoch [{epoch + 1}/{num_epochs}] - Loss: {total_loss:.4f}, Accuracy: {accuracy:.2f}%")

# ---------- 5. Evaluation ----------
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

print(f"Test Accuracy: {100 * correct / total:.2f}%")
