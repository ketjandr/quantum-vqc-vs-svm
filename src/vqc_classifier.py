"""
VQC Classification on MNIST
-----------------------------
Trains a Variational Quantum Classifier (VQC) on two selected digit classes
from the MNIST dataset using dimensionality reduction (TruncatedSVD + t‑SNE)
and evaluates accuracy metrics on the test set.  Profiles wall-clock time
with time.perf_counter() and peak memory with tracemalloc to support
heterogeneous runtime analysis.
"""

import time
import tracemalloc

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from skimage.transform import resize

from qiskit import Aer
from qiskit.utils import QuantumInstance
from qiskit.algorithms.optimizers import COBYLA
from qiskit.circuit.library import ZZFeatureMap, EfficientSU2
from qiskit_machine_learning.algorithms.classifiers import VQC

# ── Configuration ────────────────────────────────────────────────────────────
DATA_PATH = "./datasets/"
IMAGE_SIZE = 28
TARGET_RESOLUTION = (28, 28)
CLASS_1 = 0
CLASS_2 = 1
SEED = 12345
N_SVD_COMPONENTS = 10
N_TSNE_COMPONENTS = 2
COBYLA_MAXITER = 500
COBYLA_TOL = 0.001
SHOTS = 1024


# ── Helper functions ─────────────────────────────────────────────────────────
def load_data(path: str) -> tuple:
    """Load MNIST CSV; returns (features, labels) as numpy arrays."""
    raw = np.loadtxt(path, delimiter=",", skiprows=1)
    labels = raw[:, 0]
    features = raw[:, 1:]
    return features, labels


def resize_and_flatten(images, target_resolution: tuple) -> np.ndarray:
    """Reshape each flat image to 2‑D, resize, then flatten back."""
    return np.array([
        resize(img.reshape(IMAGE_SIZE, IMAGE_SIZE), target_resolution).flatten()
        for img in images
    ])


def reduce_dimensions(X_train, X_test):
    """Apply TruncatedSVD → t‑SNE to reduce feature dimensionality."""
    tsvd = TruncatedSVD(n_components=N_SVD_COMPONENTS)
    X_train_svd = tsvd.fit_transform(X_train)
    X_test_svd = tsvd.transform(X_test)

    np.random.seed(0)
    tsne_train = TSNE(n_components=N_TSNE_COMPONENTS)
    tsne_test = TSNE(n_components=N_TSNE_COMPONENTS)
    X_train_reduced = tsne_train.fit_transform(X_train_svd)
    X_test_reduced = tsne_test.fit_transform(X_test_svd)
    return X_train_reduced, X_test_reduced


def split_by_class(features, labels, class_1, class_2):
    """Split features into two arrays, one per class."""
    mask_a = labels == class_1
    mask_b = labels == class_2
    return features[mask_a], features[mask_b]


def normalize(arr, max_val: float = 100.0, offset: float = 1.0):
    """Scale array by *max_val* and shift by *offset*."""
    return arr / max_val + offset


# ── Main pipeline ────────────────────────────────────────────────────────────
def main():
    # Load & pre‑process
    train_features, train_labels = load_data(DATA_PATH + "mnist_train.csv")
    test_features, test_labels = load_data(DATA_PATH + "mnist_test.csv")

    train_features = resize_and_flatten(train_features, TARGET_RESOLUTION)
    test_features = resize_and_flatten(test_features, TARGET_RESOLUTION)

    # Dimensionality reduction
    train_reduced, test_reduced = reduce_dimensions(train_features, test_features)

    # Split classes
    zeros_train, ones_train = split_by_class(train_reduced, train_labels, CLASS_1, CLASS_2)
    zeros_test, ones_test = split_by_class(test_reduced, test_labels, CLASS_1, CLASS_2)

    # Normalize
    zeros_train = normalize(zeros_train)
    ones_train = normalize(ones_train)
    zeros_test = normalize(zeros_test)
    ones_test = normalize(ones_test)

    # Build quantum circuit components
    feature_dim = zeros_train.shape[1]

    feature_map = ZZFeatureMap(feature_dimension=feature_dim, reps=2, entanglement="full")
    feature_map.draw("mpl")

    var_circuit = EfficientSU2(feature_dim, reps=2)
    var_circuit.draw("mpl")
    plt.show()

    # Quantum backend & instance
    backend = Aer.get_backend("qasm_simulator")
    quantum_instance = QuantumInstance(
        backend,
        shots=SHOTS,
        seed_simulator=SEED,
        seed_transpiler=SEED,
        backend_options={"method": "statevector"},
    )

    # Build VQC
    optimizer = COBYLA(maxiter=COBYLA_MAXITER, tol=COBYLA_TOL)
    vqc = VQC(
        optimizer=optimizer,
        feature_map=feature_map,
        ansatz=var_circuit,
        quantum_instance=quantum_instance,
    )

    # Prepare training data
    X_train = np.concatenate((zeros_train, ones_train))
    y_train = np.concatenate((
        np.zeros(zeros_train.shape[0]),
        np.ones(ones_train.shape[0]),
    ))

    # Train
    tracemalloc.start()
    start = time.perf_counter()
    vqc.fit(X_train, y_train)
    training_time = time.perf_counter() - start
    _, peak_memory = tracemalloc.get_traced_memory()  # bytes
    tracemalloc.stop()
    print(f"Training Time: {training_time:.4f} seconds")
    print(f"Peak Memory:   {peak_memory / 1024:.2f} KB")

    # Prepare test data
    X_test = np.concatenate((zeros_test, ones_test))
    y_test = np.concatenate((
        np.zeros(zeros_test.shape[0]),
        np.ones(ones_test.shape[0]),
    ))

    # Evaluate
    y_pred = vqc.predict(X_test)
    print(f"Predicted labels: {y_pred}")
    print(f"Ground truth:     {y_test}")

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"Accuracy:  {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall:    {recall:.2f}")
    print(f"F1-Score:  {f1:.2f}")


if __name__ == "__main__":
    main()
