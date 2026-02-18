"""
Heterogeneous Runtime Benchmark
---------------------------------
Sweeps across 16–784 MNIST features and profiles training runtime
(time.perf_counter) and peak memory (tracemalloc) for both SVM and VQC
classifiers.  Results are saved to CSV for downstream scaling analysis.

Usage:
    python src/benchmark_runner.py              # full sweep, both models
    python src/benchmark_runner.py --model svm  # SVM only
    python src/benchmark_runner.py --model vqc  # VQC only
"""

import argparse
import csv
import math
import os
import time
import tracemalloc

import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from skimage.transform import resize

# ── Configuration ────────────────────────────────────────────────────────────
DATA_PATH = "./datasets/"
CLASS_1 = 0
CLASS_2 = 1
OUTPUT_DIR = "./results/"

# Feature counts correspond to image resolutions: 4x4, 6x6, ..., 28x28
FEATURE_COUNTS = [16, 36, 64, 100, 144, 196, 256, 324, 400, 484, 576, 676, 784]


# ── Data helpers ─────────────────────────────────────────────────────────────
def load_and_filter_pd(csv_path: str, classes: tuple):
    """Load CSV via pandas; return (features, labels) filtered to *classes*."""
    df = pd.read_csv(csv_path, delimiter=",")
    mask = df["label"].isin(classes)
    X = df.loc[mask].drop(columns=["label"]).values
    y = df.loc[mask]["label"].values
    return X, y


def resize_and_flatten(images, n_features: int):
    """Resize each 28×28 image to sqrt(n_features) × sqrt(n_features)."""
    side = int(math.isqrt(n_features))
    return np.array([
        resize(img.reshape(28, 28), (side, side)).flatten()
        for img in images
    ])


# ── SVM benchmark ────────────────────────────────────────────────────────────
def run_svm(X_train, y_train, X_test, y_test):
    """Train SVM, return dict of metrics + runtime + memory."""
    svm = SVC(kernel="poly", C=1.0, coef0=1.0, degree=4)

    tracemalloc.start()
    t0 = time.perf_counter()
    svm.fit(X_train, y_train)
    wall_time = time.perf_counter() - t0
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    y_pred = svm.predict(X_test)
    return {
        "wall_time_s": wall_time,
        "peak_memory_kb": peak_mem / 1024,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted"),
        "recall": recall_score(y_test, y_pred, average="weighted"),
        "f1": f1_score(y_test, y_pred, average="weighted"),
    }


# ── VQC benchmark (import lazily to keep SVM-only runs fast) ─────────────────
def run_vqc(X_train, y_train, X_test, y_test, n_features: int, seed=12345):
    """Train VQC, return dict of metrics + runtime + memory."""
    from sklearn.decomposition import TruncatedSVD
    from sklearn.manifold import TSNE
    from qiskit import Aer
    from qiskit.utils import QuantumInstance
    from qiskit.algorithms.optimizers import COBYLA
    from qiskit.circuit.library import ZZFeatureMap, EfficientSU2
    from qiskit_machine_learning.algorithms.classifiers import VQC

    # Dimensionality reduction  → 2 features for the quantum circuit
    n_svd = min(10, n_features - 1)
    tsvd = TruncatedSVD(n_components=n_svd)
    X_train_svd = tsvd.fit_transform(X_train)
    X_test_svd = tsvd.transform(X_test)

    np.random.seed(0)
    X_train_r = TSNE(n_components=2).fit_transform(X_train_svd)
    X_test_r = TSNE(n_components=2).fit_transform(X_test_svd)

    # Normalize
    X_train_r = X_train_r / 100.0 + 1.0
    X_test_r = X_test_r / 100.0 + 1.0

    feature_dim = 2
    feature_map = ZZFeatureMap(feature_dimension=feature_dim, reps=2, entanglement="full")
    var_circuit = EfficientSU2(feature_dim, reps=2)

    backend = Aer.get_backend("qasm_simulator")
    qi = QuantumInstance(
        backend, shots=1024,
        seed_simulator=seed, seed_transpiler=seed,
        backend_options={"method": "statevector"},
    )
    vqc = VQC(
        optimizer=COBYLA(maxiter=500, tol=0.001),
        feature_map=feature_map,
        ansatz=var_circuit,
        quantum_instance=qi,
    )

    tracemalloc.start()
    t0 = time.perf_counter()
    vqc.fit(X_train_r, y_train)
    wall_time = time.perf_counter() - t0
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    y_pred = vqc.predict(X_test_r)
    return {
        "wall_time_s": wall_time,
        "peak_memory_kb": peak_mem / 1024,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
    }


# ── Sweep & persist ─────────────────────────────────────────────────────────
def sweep(model: str):
    classes = (CLASS_1, CLASS_2)
    X_train_raw, y_train = load_and_filter_pd(DATA_PATH + "mnist_train.csv", classes)
    X_test_raw, y_test = load_and_filter_pd(DATA_PATH + "mnist_test.csv", classes)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUTPUT_DIR, f"{model}_benchmark.csv")
    fieldnames = ["n_features", "wall_time_s", "peak_memory_kb",
                  "accuracy", "precision", "recall", "f1"]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for nf in FEATURE_COUNTS:
            print(f"\n[{model.upper()}] n_features={nf}")
            X_train = resize_and_flatten(X_train_raw, nf)
            X_test = resize_and_flatten(X_test_raw, nf)

            if model == "svm":
                result = run_svm(X_train, y_train, X_test, y_test)
            else:
                result = run_vqc(X_train, y_train, X_test, y_test, nf)

            result["n_features"] = nf
            writer.writerow(result)
            print(f"  time={result['wall_time_s']:.4f}s  "
                  f"mem={result['peak_memory_kb']:.1f}KB  "
                  f"acc={result['accuracy']:.2f}")

    print(f"\nResults saved to {csv_path}")


# ── CLI ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Benchmark SVM / VQC training across feature counts")
    parser.add_argument("--model", choices=["svm", "vqc", "both"], default="both",
                        help="Which model(s) to benchmark (default: both)")
    args = parser.parse_args()

    models = ["svm", "vqc"] if args.model == "both" else [args.model]
    for m in models:
        sweep(m)


if __name__ == "__main__":
    main()
