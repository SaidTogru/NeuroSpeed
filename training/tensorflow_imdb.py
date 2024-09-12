import tensorflow as tf
import pandas as pd
import numpy as np
import time
import os

# Load IMDB dataset
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.imdb.load_data(num_words=10000)
X_train = tf.keras.preprocessing.sequence.pad_sequences(X_train, maxlen=256)
X_test = tf.keras.preprocessing.sequence.pad_sequences(X_test, maxlen=256)

# Model Definition
model = tf.keras.Sequential(
    [
        tf.keras.layers.Embedding(input_dim=10000, output_dim=64, input_length=256),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(2, activation="softmax"),
    ]
)

model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
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
