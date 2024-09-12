import numpy as np
import cupy as cp
import pandas as pd
import time
import os
from keras.datasets import cifar10

# Load CIFAR-10 dataset using keras
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
X_train, X_test = X_train.astype(cp.float32) / 255.0, X_test.astype(cp.float32) / 255.0
y_train, y_test = y_train.flatten(), y_test.flatten()

# Convert labels to cuPy arrays to match with other cuPy computations
y_train = cp.asarray(y_train)
y_test = cp.asarray(y_test)


# Simple neural network implementation
class SimpleNN:
    def __init__(self):
        self.weights1 = cp.random.randn(32 * 32 * 3, 128).astype(cp.float32) * 0.01
        self.bias1 = cp.zeros(128).astype(cp.float32)
        self.weights2 = cp.random.randn(128, 10).astype(cp.float32) * 0.01
        self.bias2 = cp.zeros(10).astype(cp.float32)

    def relu(self, z):
        return cp.maximum(0, z)

    def softmax(self, z):
        exps = cp.exp(z - cp.max(z, axis=1, keepdims=True))
        return exps / cp.sum(exps, axis=1, keepdims=True)

    def forward(self, X):
        X = cp.asarray(X)  # Ensure it's a CuPy array
        self.z1 = cp.dot(X, self.weights1) + self.bias1
        self.a1 = self.relu(self.z1)
        self.z2 = cp.dot(self.a1, self.weights2) + self.bias2
        return self.softmax(self.z2)

    def compute_loss(self, output, y):
        m = y.shape[0]
        log_likelihood = -cp.log(output[cp.arange(m), y])
        loss = cp.sum(log_likelihood) / m
        return loss

    def backward(self, X, y, output):
        X = cp.asarray(X)  # Ensure it's a CuPy array
        m = X.shape[0]
        dz2 = output
        dz2[cp.arange(m), y] -= 1
        dz2 /= m

        dw2 = cp.dot(self.a1.T, dz2)
        db2 = cp.sum(dz2, axis=0)

        dz1 = cp.dot(dz2, self.weights2.T) * (self.a1 > 0)
        dw1 = cp.dot(X.T, dz1)
        db1 = cp.sum(dz1, axis=0)

        self.weights1 -= 0.01 * dw1
        self.bias1 -= 0.01 * db1
        self.weights2 -= 0.01 * dw2
        self.bias2 -= 0.01 * db2


# Instantiate and train the model
model = SimpleNN()

# Training
epochs = 15
batch_size = 64
start_time = time.time()

for epoch in range(epochs):
    idx = np.random.permutation(X_train.shape[0])
    X_train, y_train = X_train[idx], y_train[idx]
    for i in range(0, X_train.shape[0], batch_size):
        X_batch = X_train[i : i + batch_size].reshape(-1, 32 * 32 * 3)
        y_batch = y_train[i : i + batch_size]
        output = model.forward(X_batch)
        loss = model.compute_loss(output, y_batch)
        model.backward(X_batch, y_batch, output)

training_time = time.time() - start_time


# Accuracy function (with cuPy arrays)
def accuracy(X, y):
    preds = model.forward(X.reshape(-1, 32 * 32 * 3)).argmax(axis=1)
    return cp.mean(preds == y)


train_accuracy = accuracy(X_train, y_train).get()  # Convert to NumPy
test_accuracy = accuracy(X_test, y_test).get()  # Convert to NumPy

# Inference
start_time = time.time()
test_accuracy = accuracy(X_test, y_test).get()  # Ensure baseline to NumPy conversion
inference_time = time.time() - start_time

# Prepare data for CSV
metrics_df = pd.DataFrame(
    {
        "Framework": ["Vanilla Python (baseline)"],
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
