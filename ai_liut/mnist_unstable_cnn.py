import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# Custom Dataset class
class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = self.load_dataset()

    def load_dataset(self):
        samples = []
        for class_folder in os.listdir(self.root_dir):
            class_path = os.path.join(self.root_dir, class_folder)
            if os.path.isdir(class_path):
                for sample_file in os.listdir(class_path):
                    sample_path = os.path.join(class_path, sample_file)
                    if sample_file.endswith(".npy"):
                        samples.append((sample_path, int(class_folder)))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_path, label = self.samples[idx]
        sample = np.load(sample_path)

        if self.transform:
            sample = self.transform(sample)

        return sample, label


# Transformation to convert data to PyTorch tensor
class ToTensor(object):
    def __call__(self, sample):
        # Swap channel axis and add a batch dimension
        sample = np.transpose(sample, (0, 1, 2))
        return torch.from_numpy(sample).float()


# CNN Model
# class CNN(nn.Module):
#     def __init__(self):
#         super(CNN, self).__init__()
#         self.conv1 = nn.Conv3d(1, 32, kernel_size=(10, 3, 3), padding=(0, 1, 1))
#         self.pool = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
#         self.fc1 = nn.Linear(64 * 7 * 7, 128)
#         self.fc2 = nn.Linear(128, 10)  # Assuming 10 classes for classification
#
#     def forward(self, x):
#         x = x.unsqueeze(1)  # Add channel dimension
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = x.view(-1, 64 * 7 * 7)
#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(10, 32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.global_avg_pooling = nn.AdaptiveAvgPool2d(1)

        self.fc1 = nn.Linear(64, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)  # Assuming 10 classes for MNIST

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.global_avg_pooling(x)
        x = x.view(x.size(0), -1)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x


# Training function
def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    iter = 0
    start_time = time.time()

    for inputs, labels in train_loader:

        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if iter % 100 == 0:
            print(f"iter {iter}, loss {loss.item():.4f}, time: {time.time() - start_time:.2f} seconds")
        iter += 1

        running_loss += loss.item()

    return running_loss / len(train_loader)


# Validation function
def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_loss = running_loss / len(val_loader)
    val_accuracy = correct / total
    return val_loss, val_accuracy


if __name__ == "__main__":
    # Set your dataset root paths
    train_root = "./processed_data/train"
    val_root = "./processed_data/val"

    # Define transformations
    transform = transforms.Compose([ToTensor()])

    # Create datasets and data loaders
    train_dataset = CustomDataset(root_dir=train_root, transform=transform)
    val_dataset = CustomDataset(root_dir=val_root, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    # Initialize the model, criterion, and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        print("Epoch " + str(epoch + 1) + " training.")
        train_loss = train(model, train_loader, criterion, optimizer, device)
        print("Epoch " + str(epoch + 1) + " validating.")
        val_loss, val_accuracy = validate(model, val_loader, criterion, device)

        print(f"Epoch [{epoch + 1}/{num_epochs}], "
              f"Train Loss: {train_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}, "
              f"Val Accuracy: {val_accuracy:.2%}")
