import cupy as cp
import pandas as pd
import numpy as np
import time
import os


# Function to load MNIST data from CSV files
def load_mnist_from_csv(train_csv, test_csv):
    # Load CSV files into pandas DataFrames
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    # Extract labels (first column) and features (remaining columns)
    X_train = (
        train_df.iloc[:, 1:].values.astype(np.float32) / 255.0
    )  # Normalize to [0, 1]
    y_train = train_df.iloc[:, 0].values.astype(np.int64)

    X_test = (
        test_df.iloc[:, 1:].values.astype(np.float32) / 255.0
    )  # Normalize to [0, 1]
    y_test = test_df.iloc[:, 0].values.astype(np.int64)

    return X_train, y_train, X_test, y_test


# Function to convert NumPy arrays to cuPy arrays
def to_gpu(array):
    return cp.array(array)


# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + cp.exp(-x))


# Derivative of the sigmoid function
def sigmoid_derivative(x):
    return x * (1 - x)


# Softmax function
def softmax(x):
    exps = cp.exp(x - cp.max(x))
    return exps / cp.sum(exps, axis=1, keepdims=True)


# Initialize weights
def initialize_weights(input_size, hidden_size, output_size):
    W1 = cp.random.randn(input_size, hidden_size) * 0.01
    W2 = cp.random.randn(hidden_size, output_size) * 0.01
    return W1, W2


# Forward propagation
def forward(X, W1, W2):
    Z1 = cp.dot(X, W1)
    A1 = sigmoid(Z1)
    Z2 = cp.dot(A1, W2)
    A2 = softmax(Z2)
    return Z1, A1, A2


# Backward propagation
def backward(X, y, W1, W2, Z1, A1, A2, learning_rate=0.01):
    m = X.shape[0]
    dZ2 = A2 - y
    dW2 = cp.dot(A1.T, dZ2) / m
    dZ1 = cp.dot(dZ2, W2.T) * sigmoid_derivative(A1)
    dW1 = cp.dot(X.T, dZ1) / m
    W1 -= learning_rate * dW1
    W2 -= learning_rate * dW2
    return W1, W2


# Predict function
def predict(X, W1, W2):
    _, _, A2 = forward(X, W1, W2)
    return cp.argmax(A2, axis=1)


# One-hot encode labels
def one_hot_encode(y, num_classes):
    one_hot = cp.zeros((y.size, num_classes))
    one_hot[cp.arange(y.size), y] = 1
    return one_hot


# Main function to train the vanilla neural network on MNIST using cuPy
def train_mnist_vanilla(X_train, y_train, X_test, y_test, epochs=15, batch_size=64):
    input_size = X_train.shape[1]
    hidden_size = 128
    output_size = 10

    # Initialize weights
    W1, W2 = initialize_weights(input_size, hidden_size, output_size)

    # One-hot encode labels
    y_train_one_hot = one_hot_encode(y_train, output_size)

    # Training loop
    for epoch in range(epochs):
        for i in range(0, X_train.shape[0], batch_size):
            X_batch = X_train[i : i + batch_size]
            y_batch = y_train_one_hot[i : i + batch_size]

            Z1, A1, A2 = forward(X_batch, W1, W2)
            W1, W2 = backward(X_batch, y_batch, W1, W2, Z1, A1, A2)

        # Calculate training accuracy
        train_preds = predict(X_train, W1, W2)
        train_accuracy = cp.mean(train_preds == y_train)
        print(f"Epoch {epoch+1}/{epochs} - Training Accuracy: {train_accuracy}")

    # Calculate test accuracy
    test_preds = predict(X_test, W1, W2)
    test_accuracy = cp.mean(test_preds == y_test)
    print(f"Test Accuracy: {test_accuracy}")

    return train_accuracy, test_accuracy, W1, W2


# Load MNIST data from CSV
train_csv = "data/mnist/mnist_train.csv"
test_csv = "data/mnist/mnist_test.csv"

X_train_np, y_train_np, X_test_np, y_test_np = load_mnist_from_csv(train_csv, test_csv)

# Convert to cuPy (GPU) arrays
X_train = to_gpu(X_train_np)
y_train = to_gpu(y_train_np)
X_test = to_gpu(X_test_np)
y_test = to_gpu(y_test_np)

# Train vanilla model with cuPy (GPU)
start_time = time.time()
train_accuracy, test_accuracy, W1, W2 = train_mnist_vanilla(
    X_train, y_train, X_test, y_test
)
training_time = time.time() - start_time

# Inference Time
start_time = time.time()
predict(X_test, W1, W2)
inference_time = time.time() - start_time


# Count lines of code
def count_lines_of_code(file_path):
    with open(file_path, "r") as f:
        return sum(1 for line in f)


code_lines = count_lines_of_code(__file__)

# Prepare data for CSV
metrics_df = pd.DataFrame(
    {
        "Framework": ["Vanilla Python (baseline)"],
        "Dataset": ["MNIST"],
        "Training Accuracy": [train_accuracy.get()],
        "Inference Accuracy": [test_accuracy.get()],
        "Training Time": [training_time],
        "Prediction Time": [inference_time],
        "Number of Lines of Code": [code_lines],
    }
)

# Check if CSV exists and append or create a new one
csv_file = "metrics.csv"
if os.path.exists(csv_file):
    # Append data to existing CSV
    metrics_df.to_csv(csv_file, mode="a", header=False, index=False)
else:
    # Create new CSV with header
    metrics_df.to_csv(csv_file, mode="w", header=True, index=False)

print("Metrics saved to metrics.csv")
