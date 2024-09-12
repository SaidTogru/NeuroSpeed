import numpy as np
import cupy as cp
import pandas as pd
import time
import os
from keras.datasets import fashion_mnist

# Load Fashion-MNIST dataset
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

# Convert the data to NumPy float32 first, then CuPy arrays
X_train = cp.array(X_train.astype(np.float32).reshape(-1, 28 * 28) / 255.0)
X_test = cp.array(X_test.astype(np.float32).reshape(-1, 28 * 28) / 255.0)
y_train = cp.array(y_train.flatten())
y_test = cp.array(y_test.flatten())


# Simple neural network implementation
class SimpleNN:
    def __init__(self):
        self.weights1 = cp.random.randn(28 * 28, 128)
        self.bias1 = cp.random.randn(128)
        self.weights2 = cp.random.randn(128, 10)
        self.bias2 = cp.random.randn(10)

    def forward(self, X):
        # Ensure input is a CuPy array
        if not isinstance(X, cp.ndarray):
            X = cp.array(X)
        self.z1 = cp.dot(X, self.weights1) + self.bias1
        self.a1 = cp.maximum(0, self.z1)  # ReLU activation
        self.z2 = cp.dot(self.a1, self.weights2) + self.bias2
        return self.z2

    def compute_loss(self, output, y):
        m = y.shape[0]
        log_likelihood = -cp.log(output[cp.arange(m), y])
        loss = cp.sum(log_likelihood) / m
        return loss

    def backward(self, X, y, output):
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


# Initialize and train model
model = SimpleNN()

epochs = 15
batch_size = 64
start_time = time.time()
for epoch in range(epochs):
    for i in range(0, X_train.shape[0], batch_size):
        X_batch = X_train[i : i + batch_size]
        y_batch = y_train[i : i + batch_size]
        output = model.forward(X_batch)
        loss = model.compute_loss(output, y_batch)
        model.backward(X_batch, y_batch, output)

training_time = time.time() - start_time

# Accuracy on training set
correct_train = 0
output = model.forward(X_train)
pred = cp.argmax(output, axis=1)
correct_train = cp.sum(pred == y_train)
train_accuracy = correct_train / len(y_train)

# Inference and accuracy on test set
start_time = time.time()
output = model.forward(X_test)
inference_time = time.time() - start_time
pred = cp.argmax(output, axis=1)
correct_test = cp.sum(pred == y_test)
test_accuracy = correct_test / len(y_test)


# Count lines of code
def count_lines_of_code(file_path):
    with open(file_path, "r") as f:
        return sum(1 for line in f)


code_lines = count_lines_of_code(__file__)

# Prepare data for CSV
metrics_df = pd.DataFrame(
    {
        "Framework": ["Vanilla Python (baseline)"],
        "Dataset": ["Fashion-MNIST"],
        "Training Accuracy": [train_accuracy.get()],  # Use .get() to convert to NumPy
        "Inference Accuracy": [test_accuracy.get()],  # Use .get() to convert to NumPy
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
