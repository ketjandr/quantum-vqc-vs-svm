# Quantum VQC vs SVM Classification

Comparison of **Variational Quantum Classifier (VQC)** and **Support Vector Machine (SVM)** on the MNIST dataset.

## Project Structure

```
├── datasets/               # Place MNIST CSV files here (mnist_train.csv, mnist_test.csv)
├── src/
│   ├── svm_classifier.py   # SVM classification pipeline
│   ├── vqc_classifier.py   # VQC classification pipeline
│   └── plot_metrics.py     # Matplotlib accuracy metric plots
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

## Dataset

Download the MNIST dataset in CSV format and place the following files in the `datasets/` directory:

- `mnist_train.csv`
- `mnist_test.csv`

Each CSV should have a `label` column followed by 784 pixel columns.

## Usage

```bash
# Run SVM classifier
python src/svm_classifier.py

# Run VQC classifier
python src/vqc_classifier.py

# Generate accuracy metric comparison plots
python src/plot_metrics.py
```
