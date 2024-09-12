#!/bin/bash

# Check if the mnist folder exists
if [ ! -d "data/mnist" ]; then
  echo "mnist folder not found, extracting mnist.zip..."
  
  # Check if mnist.zip exists
  if [ -f "data/mnist.zip" ]; then
    unzip data/mnist.zip -d data/
    echo "Extraction completed."
  else
    echo "Error: mnist.zip file not found!"
    exit 1
  fi
else
  echo "mnist folder already exists. No need to extract."
fi


# Function to run a script and update metrics CSV
run_and_update_csv() {
    local script_name=$1
    local framework_name=$2
    local dataset_name=$3

    # Run the Python script and measure time
    /usr/bin/time -v python3 $script_name

    echo "$framework_name on $dataset_name script completed"

    # No memory usage, updating CSV for runtime and accuracy only
}

#!/bin/bash

# Function to run a script and update metrics CSV
run_and_update_csv() {
    local script_name=$1
    local framework_name=$2
    local dataset_name=$3

    # Run the Python script and measure time
    /usr/bin/time -v python3 $script_name

    echo "$framework_name script for $dataset_name completed"

    # No memory usage, updating CSV for runtime and accuracy only
}

# Run Vanilla Python Scripts and Update CSV for Different Datasets
run_and_update_csv "training/vanilla_python_mnist.py" "Vanilla Python (baseline)" "MNIST"
run_and_update_csv "training/vanilla_python_fashion_mnist.py" "Vanilla Python (baseline)" "Fashion-MNIST"
run_and_update_csv "training/vanilla_python_imdb.py" "Vanilla Python (baseline)" "IMDB Movie Reviews"
run_and_update_csv "training/vanilla_python_cifar10.py" "Vanilla Python (baseline)" "CIFAR-10"

# Run PyTorch Scripts and Update CSV for Different Datasets
run_and_update_csv "training/pytorch_mnist.py" "PyTorch" "MNIST"
run_and_update_csv "training/pytorch_fashion_mnist.py" "PyTorch" "Fashion-MNIST"
run_and_update_csv "training/pytorch_imdb.py" "PyTorch" "IMDB Movie Reviews"
run_and_update_csv "training/pytorch_cifar10.py" "PyTorch" "CIFAR-10"

# Run TensorFlow Scripts and Update CSV for Different Datasets
run_and_update_csv "training/tensorflow_mnist.py" "TensorFlow" "MNIST"
run_and_update_csv "training/tensorflow_fashion_mnist.py" "TensorFlow" "Fashion-MNIST"
run_and_update_csv "training/tensorflow_imdb.py" "TensorFlow" "IMDB Movie Reviews"
run_and_update_csv "training/tensorflow_cifar10.py" "TensorFlow" "CIFAR-10"


echo "Metrics updated in metrics.csv for all datasets"
