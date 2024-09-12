import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import time
import os
from keras.datasets import cifar10


# Load CIFAR-10 dataset using keras
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
X_train = torch.tensor(X_train, dtype=torch.float32).reshape(-1, 32 * 32 * 3) / 255.0
X_test = torch.tensor(X_test, dtype=torch.float32).reshape(-1, 32 * 32 * 3) / 255.0
y_train = torch.tensor(y_train.flatten(), dtype=torch.int64)
y_test = torch.tensor(y_test.flatten(), dtype=torch.int64)


# Define the model
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(32 * 32 * 3, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleNN().to(device)

# Datasets and DataLoader
train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

# Optimizer and loss function
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training loop
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

# Accuracy calculation
model.eval()
correct_train = 0
with torch.no_grad():
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        pred = output.argmax(dim=1, keepdim=True)
        correct_train += pred.eq(target.view_as(pred)).sum().item()
train_accuracy = correct_train / len(train_loader.dataset)

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

# Save metrics
metrics_df = pd.DataFrame(
    {
        "Framework": ["PyTorch"],
        "Dataset": ["CIFAR-10"],
        "Training Accuracy": [train_accuracy],
        "Inference Accuracy": [test_accuracy],
        "Training Time": [training_time],
        "Prediction Time": [inference_time],
    }
)

csv_file = "metrics.csv"
if os.path.exists(csv_file):
    metrics_df.to_csv(csv_file, mode="a", header=False, index=False)
else:
    metrics_df.to_csv(csv_file, mode="w", header=True, index=False)

print("Metrics saved to metrics.csv")
