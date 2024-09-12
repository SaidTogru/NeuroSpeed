import tensorflow as tf
import pandas as pd
import numpy as np
import time
import os


# Function to load MNIST data from CSV files
def load_mnist_from_csv(train_csv, test_csv):
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)
    X_train = train_df.iloc[:, 1:].values.astype(np.float32) / 255.0
    y_train = train_df.iloc[:, 0].values.astype(np.int64)
    X_test = test_df.iloc[:, 1:].values.astype(np.float32) / 255.0
    y_test = test_df.iloc[:, 0].values.astype(np.int64)
    return X_train, y_train, X_test, y_test


# GPU check
if not tf.config.list_physical_devices("GPU"):
    print("GPU not available.")
else:
    print("GPU available.")

train_csv = "data/mnist/mnist_train.csv"
test_csv = "data/mnist/mnist_test.csv"
X_train, y_train, X_test, y_test = load_mnist_from_csv(train_csv, test_csv)
X_train = X_train.reshape(-1, 28, 28)
X_test = X_test.reshape(-1, 28, 28)

model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(10),
    ]
)

model.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

start_time = time.time()
history = model.fit(
    X_train, y_train, epochs=15, batch_size=64, validation_data=(X_test, y_test)
)
training_time = time.time() - start_time
train_accuracy = history.history["accuracy"][-1]

# model.save("tensorflow_mnist_model.h5")

start_time = time.time()
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
inference_time = time.time() - start_time


def count_lines_of_code(file_path):
    with open(file_path, "r") as f:
        return sum(1 for line in f)


code_lines = count_lines_of_code(__file__)

metrics_df = pd.DataFrame(
    {
        "Framework": ["TensorFlow"],
        "Dataset": ["MNIST"],
        "Training Accuracy": [train_accuracy],
        "Inference Accuracy": [test_acc],
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
