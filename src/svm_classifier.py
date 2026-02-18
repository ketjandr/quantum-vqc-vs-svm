"""
SVM Classification on MNIST
----------------------------
Trains an SVM with a polynomial kernel on two selected digit classes
from the MNIST dataset and evaluates accuracy metrics on the test set.
"""

import time

import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from skimage.transform import resize

# ── Configuration ────────────────────────────────────────────────────────────
DATA_PATH = "./datasets/"
CLASS_1 = 0
CLASS_2 = 1
TARGET_RESOLUTION = (28, 28)


# ── Helper functions ─────────────────────────────────────────────────────────
def load_and_filter(csv_path: str, classes: tuple) -> tuple:
    """Load a CSV file and return features/labels filtered to *classes*."""
    df = pd.read_csv(csv_path, delimiter=",")
    mask = df["label"].isin(classes)
    X = df.loc[mask].drop(columns=["label"]).values
    y = df.loc[mask]["label"].values
    return X, y


def resize_and_flatten(images, target_resolution: tuple) -> list:
    """Reshape each flat image to 2‑D, resize, then flatten back."""
    return [
        resize(img.reshape(28, 28), target_resolution).flatten()
        for img in images
    ]


# ── Main pipeline ────────────────────────────────────────────────────────────
def main():
    classes = (CLASS_1, CLASS_2)

    # Load data
    X_train_raw, y_train = load_and_filter(DATA_PATH + "mnist_train.csv", classes)
    X_test_raw, y_test = load_and_filter(DATA_PATH + "mnist_test.csv", classes)

    # Pre‑process
    X_train = resize_and_flatten(X_train_raw, TARGET_RESOLUTION)
    X_test = resize_and_flatten(X_test_raw, TARGET_RESOLUTION)

    # Train SVM
    svm = SVC(kernel="poly", C=1.0, coef0=1.0, degree=4)

    start = time.time()
    svm.fit(X_train, y_train)
    training_time = time.time() - start
    print(f"Training Time: {training_time:.2f} seconds")

    # Evaluate
    y_pred = svm.predict(X_test)
    print(f"Predicted Labels: {y_pred}")

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")

    print(f"Accuracy:  {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall:    {recall:.2f}")
    print(f"F1-Score:  {f1:.2f}")


if __name__ == "__main__":
    main()
