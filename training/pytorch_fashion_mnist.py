import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import numpy as np
import time
import os
from torchvision import datasets, transforms


# Function to load Fashion-MNIST data
def load_fashion_mnist():
    transform = transforms.Compose([transforms.ToTensor()])

    train_dataset = datasets.FashionMNIST(
        root="./data", train=True, download=True, transform=transform
    )
    test_dataset = datasets.FashionMNIST(
        root="./data", train=False, download=True, transform=transform
    )

    X_train = torch.stack(
        [train_dataset[i][0] for i in range(len(train_dataset))]
    ).view(-1, 28 * 28)
    y_train = torch.tensor([train_dataset[i][1] for i in range(len(train_dataset))])

    X_test = torch.stack([test_dataset[i][0] for i in range(len(test_dataset))]).view(
        -1, 28 * 28
    )
    y_test = torch.tensor([test_dataset[i][1] for i in range(len(test_dataset))])

    return X_train, y_train, X_test, y_test


class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Load Fashion-MNIST data
X_train, y_train, X_test, y_test = load_fashion_mnist()
X_train, y_train, X_test, y_test = (
    X_train.to(device),
    y_train.to(device),
    X_test.to(device),
    y_test.to(device),
)

train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

model = SimpleNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training
start_time = time.time()
epochs = 15
for epoch in range(epochs):
    model.train()
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

training_time = time.time() - start_time

# Accuracy
model.eval()
correct_train = 0
with torch.no_grad():
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        pred = output.argmax(dim=1, keepdim=True)
        correct_train += pred.eq(target.view_as(pred)).sum().item()
train_accuracy = correct_train / len(train_loader.dataset)

# torch.save(model.state_dict(), "pytorch_fashion_mnist_model.pth")

# Inference
correct_test = 0
start_time = time.time()
with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        pred = output.argmax(dim=1, keepdim=True)
        correct_test += pred.eq(target.view_as(pred)).sum().item()
inference_time = time.time() - start_time
test_accuracy = correct_test / len(test_loader.dataset)


# Count lines of code
def count_lines_of_code(file_path):
    with open(file_path, "r") as f:
        return sum(1 for line in f)


code_lines = count_lines_of_code(__file__)

# Prepare data for CSV
metrics_df = pd.DataFrame(
    {
        "Framework": ["PyTorch"],
        "Dataset": ["Fashion-MNIST"],
        "Training Accuracy": [train_accuracy],
        "Inference Accuracy": [test_accuracy],
        "Training Time": [training_time],
        "Prediction Time": [inference_time],
        "Number of Lines of Code": [code_lines],
    }
)

csv_file = "metrics.csv"
if os.path.exists(csv_file):
    metrics_df.to_csv(csv_file, mode="a", header=False, index=False)
else:
    metrics_df.to_csv(csv_file, mode="w", header=True, index=False)

print("Metrics saved to metrics.csv")
