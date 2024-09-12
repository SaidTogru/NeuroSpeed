import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import numpy as np
import time
import os
import tensorflow as tf

# Load and preprocess IMDB dataset using TensorFlow
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.imdb.load_data(num_words=10000)

# Pad sequences to the same length as in TensorFlow (256)
X_train = tf.keras.preprocessing.sequence.pad_sequences(X_train, maxlen=256)
X_test = tf.keras.preprocessing.sequence.pad_sequences(X_test, maxlen=256)

# Convert TensorFlow data to PyTorch tensors
X_train, X_test = torch.tensor(X_train, dtype=torch.long), torch.tensor(
    X_test, dtype=torch.long
)
y_train, y_test = torch.tensor(y_train, dtype=torch.long), torch.tensor(
    y_test, dtype=torch.long
)


# Model Definition
class SimpleNN(nn.Module):
    def __init__(self, vocab_size):
        super(SimpleNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, 64)
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.embedding(x).mean(1)  # Average over words
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Define model, loss function, and optimizer
vocab_size = 10000  # Same as TensorFlow limit for IMDB vocab size
model = SimpleNN(vocab_size).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Load data into PyTorch DataLoader
train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

# Training the model
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

# Evaluate training accuracy
model.eval()
correct_train = 0
with torch.no_grad():
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        pred = output.argmax(dim=1, keepdim=True)
        correct_train += pred.eq(target.view_as(pred)).sum().item()
train_accuracy = correct_train / len(train_loader.dataset)

# torch.save(model.state_dict(), "pytorch_imdb_model.pth")

# Inference and evaluate test accuracy
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
        "Dataset": ["IMDB"],
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
