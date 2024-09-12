import numpy as np
import cupy as cp
import pandas as pd
import time
import os
from keras.datasets import imdb
from keras.preprocessing import sequence

# Load IMDB dataset
max_features = 10000  # Number of words to consider as features
maxlen = 256  # Cut texts after this number of words (among top max_features most common words)
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=max_features)

X_train = sequence.pad_sequences(X_train, maxlen=maxlen).astype(cp.float32)
X_test = sequence.pad_sequences(X_test, maxlen=maxlen).astype(cp.float32)
y_train, y_test = cp.array(y_train), cp.array(y_test)


# Sigmoid function definition
def sigmoid(z):
    return 1 / (1 + cp.exp(-z))


# Simple neural network implementation
class SimpleNN:
    def __init__(self):
        # Adjust input size to match the dataset (256 input features)
        self.weights1 = cp.random.randn(maxlen, 128)  # Adjusted to (256, 128)
        self.bias1 = cp.random.randn(128)
        self.weights2 = cp.random.randn(
            128, 1
        )  # Single output for binary classification
        self.bias2 = cp.random.randn(1)

    def forward(self, X):
        # Ensure input is a CuPy array
        X = cp.array(X)
        self.z1 = cp.dot(X, self.weights1) + self.bias1
        self.a1 = cp.maximum(0, self.z1)  # ReLU activation
        self.z2 = cp.dot(self.a1, self.weights2) + self.bias2
        return sigmoid(self.z2)  # Use the custom sigmoid function

    def compute_loss(self, output, y):
        m = y.shape[0]
        loss = -cp.mean(y * cp.log(output) + (1 - y) * cp.log(1 - output))
        return loss

    def backward(self, X, y, output):
        m = X.shape[0]
        dz2 = output - y.reshape(-1, 1)  # Binary cross-entropy derivative
        dw2 = cp.dot(self.a1.T, dz2) / m
        db2 = cp.sum(dz2, axis=0) / m

        dz1 = cp.dot(dz2, self.weights2.T) * (self.a1 > 0)
        X = cp.array(X)  # Ensure X is a CuPy array
        dw1 = cp.dot(X.T, dz1) / m
        db1 = cp.sum(dz1, axis=0) / m

        # Update weights and biases
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
output = model.forward(X_train)
pred_train = (output > 0.5).astype(int)
train_accuracy = cp.mean(pred_train.flatten() == y_train)

# Inference and accuracy on test set
start_time = time.time()
output = model.forward(X_test)
inference_time = time.time() - start_time
pred_test = (output > 0.5).astype(int)
test_accuracy = cp.mean(pred_test.flatten() == y_test)


# Count lines of code
def count_lines_of_code(file_path):
    with open(file_path, "r") as f:
        return sum(1 for line in f)


code_lines = count_lines_of_code(__file__)

# Prepare data for CSV
metrics_df = pd.DataFrame(
    {
        "Framework": ["Vanilla Python (baseline)"],
        "Dataset": ["IMDB"],
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
