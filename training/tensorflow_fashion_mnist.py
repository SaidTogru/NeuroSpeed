import tensorflow as tf
import pandas as pd
import numpy as np
import time
import os

# Load Fashion-MNIST dataset
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
X_train, X_test = X_train.astype(np.float32) / 255.0, X_test.astype(np.float32) / 255.0
y_train, y_test = y_train.flatten(), y_test.flatten()

# Define the model
model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax"),
    ]
)

model.compile(
    optimizer="sgd", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

# Training
start_time = time.time()
history = model.fit(X_train, y_train, epochs=15, batch_size=64, verbose=1)
training_time = time.time() - start_time

# Evaluate
train_accuracy = history.history["accuracy"][-1]
start_time = time.time()
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
inference_time = time.time() - start_time


# Count lines of code
def count_lines_of_code(file_path):
    with open(file_path, "r") as f:
        return sum(1 for line in f)


code_lines = count_lines_of_code(__file__)

# Prepare data for CSV
metrics_df = pd.DataFrame(
    {
        "Framework": ["TensorFlow"],
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
